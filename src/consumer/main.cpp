#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <nats.h>
#include <spdlog/spdlog.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "detection/v1/detection.pb.h"

namespace {
std::atomic<bool> g_running{true};
void on_signal(int) { g_running = false; }
}

int main() {
  // init spdlog otel sink
  edge::otel::init("mock-consumer", "0.1.0");
  spdlog::set_level(spdlog::level::info);

  const std::string nats_url =
      edge::getenv_or("NATS_URL", "nats://localhost:4222");
  constexpr const char* kStream = "CV_ALERTS";
  constexpr const char* kConsumer = "demo-consumer";
  constexpr const char* kSubject = "cv.alert";

  natsConnection* nc = nullptr;
  natsOptions* opts = nullptr;
  natsOptions_Create(&opts);
  natsOptions_SetURL(opts, nats_url.c_str());
  natsOptions_SetName(opts, "edge-cv-consumer");
  natsOptions_SetMaxReconnect(opts, -1);
  natsOptions_SetReconnectWait(opts, 2000);

  natsStatus s = natsConnection_Connect(&nc, opts);
  natsOptions_Destroy(opts);
  if (s != NATS_OK) {
    spdlog::error("nats.Connect: {}", natsStatus_GetText(s));
    return 1;
  }

  jsCtx* js = nullptr;
  jsOptions jsOpts;
  jsOptions_Init(&jsOpts);
  s = natsConnection_JetStream(&js, nc, &jsOpts);
  if (s != NATS_OK) {
    spdlog::error("JetStream: {}", natsStatus_GetText(s));
    natsConnection_Destroy(nc);
    return 1;
  }

  jsStreamInfo* si = nullptr;
  jsErrCode jerr{};
  for (int i = 0; i < 60; ++i) {
    s = js_GetStreamInfo(&si, js, kStream, nullptr, &jerr);
    if (s == NATS_OK) {
      spdlog::info("stream \"{}\" found", kStream);
      break;
    }
    spdlog::info("waiting for stream \"{}\" ... ({})", kStream, i + 1);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (si) jsStreamInfo_Destroy(si);
  if (s != NATS_OK) {
    spdlog::error("stream {} not found", kStream);
    jsCtx_Destroy(js);
    natsConnection_Destroy(nc);
    return 1;
  }

  jsConsumerConfig cfg;
  jsConsumerConfig_Init(&cfg);
  cfg.Durable = const_cast<char*>(kConsumer);
  cfg.AckPolicy = js_AckExplicit;
  cfg.FilterSubject = const_cast<char*>(kSubject);
  cfg.DeliverPolicy = js_DeliverNew;

  jsConsumerInfo* ci = nullptr;
  s = js_AddConsumer(&ci, js, kStream, &cfg, nullptr, &jerr);
  if (s != NATS_OK && !(s == NATS_ERR && jerr == JSConsumerNameExistErr)) {
    spdlog::warn("AddConsumer: {} (jerr={}) – continuing",
                 natsStatus_GetText(s), static_cast<int>(jerr));
  }
  if (ci) jsConsumerInfo_Destroy(ci);

  jsSubOptions so;
  jsSubOptions_Init(&so);
  so.Stream = const_cast<char*>(kStream);
  so.Consumer = const_cast<char*>(kConsumer);
  so.ManualAck = true;

  natsSubscription* sub = nullptr;
  s = js_PullSubscribe(&sub, js, kSubject, kConsumer, &jsOpts, &so, &jerr);
  if (s != NATS_OK || !sub) {
    spdlog::error("PullSubscribe: {} (jerr={})", natsStatus_GetText(s),
                  static_cast<int>(jerr));
    jsCtx_Destroy(js);
    natsConnection_Destroy(nc);
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  spdlog::info("consumer ready – listening on {} (protobuf payload)", kSubject);

  while (g_running) {
    natsMsgList list{};
    s = natsSubscription_Fetch(&list, sub, 1, 1000, &jerr);
    if (s == NATS_TIMEOUT) continue;
    if (s != NATS_OK) {
      spdlog::warn("Fetch: {} (jerr={})", natsStatus_GetText(s),
                   static_cast<int>(jerr));
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }

    for (int i = 0; i < list.Count; ++i) {
      natsMsg* msg = list.Msgs[i];

      detection::v1::Alert alert;
      if (!alert.ParseFromArray(natsMsg_GetData(msg),
                                natsMsg_GetDataLength(msg))) {
        spdlog::warn("failed to parse Alert protobuf ({} bytes)",
                     natsMsg_GetDataLength(msg));
        natsMsg_Ack(msg, nullptr);
        continue;
      }

      std::cout << "[ALERT] frame=" << alert.frame_id()
                << " dets=" << alert.detections_size()
                << " latency=" << alert.e2e_latency_ms() << "ms"
                << " hit=" << (alert.watchlist_hit() ? "yes" : "no");
      if (!alert.matched_label().empty()) {
        std::cout << " match=" << alert.matched_label();
      }
      std::cout << "\n";

      for (const auto& d : alert.detections()) {
        std::cout << "   → " << d.class_name()
                  << " " << d.confidence()
                  << "  [" << static_cast<int>(d.box().x1()) << ","
                  << static_cast<int>(d.box().y1()) << " - "
                  << static_cast<int>(d.box().x2()) << ","
                  << static_cast<int>(d.box().y2()) << "]\n";
      }

      natsMsg_Ack(msg, nullptr);
    }
    natsMsgList_Destroy(&list);
  }

  natsSubscription_Destroy(sub);
  jsCtx_Destroy(js);
  natsConnection_Destroy(nc);
  edge::otel::shutdown();
  return 0;
}
