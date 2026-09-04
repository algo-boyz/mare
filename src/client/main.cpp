#include <chrono>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <google/protobuf/util/time_util.h>
#include <spdlog/spdlog.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "detection/v1/ingest_service.grpc.pb.h"
#include "pipeline.hpp"

namespace {

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

detection::v1::Alert to_proto(const edge_cv::Alert& a, const std::string& source) {
  detection::v1::Alert out;
  out.set_frame_id(a.frame_id);
  *out.mutable_timestamp() = google::protobuf::util::TimeUtil::GetCurrentTime();
  out.set_watchlist_hit(a.watchlist_hit);
  out.set_matched_label(a.matched_label);
  out.set_e2e_latency_ms(a.e2e_latency_ms);
  out.set_source(source);

  for (const auto& d : a.detections) {
    auto* det = out.add_detections();
    det->set_class_id(d.class_id);
    det->set_class_name(d.class_name);
    det->set_confidence(d.confidence);
    auto* box = det->mutable_box();
    box->set_x1(d.box.x1);
    box->set_y1(d.box.y1);
    box->set_x2(d.box.x2);
    box->set_y2(d.box.y2);
  }
  return out;
}

}

int main(int argc, char** argv) {
  // init spdlog otel sink
  edge::otel::init("edge-client", "0.1.0");
  spdlog::set_level(spdlog::level::info);

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <source (0|video.mp4)> <model.onnx> [ingest_addr]\n";
    return 1;
  }

  const std::string source   = argv[1];
  const std::string model    = argv[2];
  const std::string ingest_addr =
      argc > 3 ? argv[3]
               : edge::getenv_or("INGEST_ADDR", "localhost:50052");

  auto channel = grpc::CreateChannel(ingest_addr,
                                     grpc::InsecureChannelCredentials());
  auto stub = detection::v1::IngestService::NewStub(channel);

  edge_cv::Pipeline::Config cfg;
  cfg.source = source;
  cfg.detector.model_path = model;
  // Docker/CPU images set EDGE_FORCE_CPU=1; also default CUDA off if unset
  {
    const char* force = std::getenv("EDGE_FORCE_CPU");
    if (force && force[0] == '1') {
      cfg.detector.use_cuda = false;
    }
  }
  cfg.queue_capacity = 2;
  cfg.print_latency = true;
  cfg.watchlist = {
      {"person", 0.55f},
      {"car", 0.50f},
      {"truck", 0.50f},
  };

  // On every alert → fire-and-forget grpc IngestAlert
  cfg.on_alert = [&](const edge_cv::Alert& alert) {
    if (alert.detections.empty() && !alert.watchlist_hit) return;

    detection::v1::IngestAlertRequest req;
    *req.mutable_alert() = to_proto(alert, source);

    detection::v1::IngestAlertResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() +
                     std::chrono::milliseconds(800));

    auto status = stub->IngestAlert(&ctx, req, &resp);
    if (!status.ok()) {
      spdlog::warn("ingest failed: {}", status.error_message());
      return;
    }
    if (resp.accepted()) {
      spdlog::info("alert frame={} dets={} → msg_id={}",
                   alert.frame_id, alert.detections.size(),
                   resp.message_id());
    } else if (resp.has_error()) {
      spdlog::warn("alert rejected: {}", resp.error().message());
    }
  };

  edge_cv::Pipeline pipeline(std::move(cfg));
  pipeline.start();

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  spdlog::info("edge_cv grpc client running → {}", ingest_addr);
  while (g_running && pipeline.is_running()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  pipeline.stop();
  spdlog::info("stopped");
  edge::otel::shutdown();
  return 0;
}
