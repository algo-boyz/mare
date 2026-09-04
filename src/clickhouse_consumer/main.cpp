#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <nats.h>
#include <spdlog/spdlog.h>

#include <clickhouse/client.h>
#include <clickhouse/columns/date.h>
#include <clickhouse/columns/numeric.h>
#include <clickhouse/columns/string.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "detection/v1/detection.pb.h"

namespace {

std::atomic<bool> g_running{true};
void on_signal(int) { g_running = false; }

struct DetectionRow {
  int64_t frame_id{0};
  int64_t ts_ns{0};
  std::string source;
  bool watchlist_hit{false};
  std::string matched_label;
  double e2e_latency_ms{0.0};
  int32_t class_id{-1};
  std::string class_name;
  float confidence{0.f};
  float x1{0.f}, y1{0.f}, x2{0.f}, y2{0.f};
  uint64_t nats_seq{0};
};

int64_t ts_to_ns(const google::protobuf::Timestamp& ts) {
  return static_cast<int64_t>(ts.seconds()) * 1'000'000'000LL +
         static_cast<int64_t>(ts.nanos());
}

std::vector<DetectionRow> alert_to_rows(const detection::v1::Alert& a,
                                        uint64_t seq) {
  std::vector<DetectionRow> rows;
  const int64_t ts_ns =
      a.has_timestamp()
          ? ts_to_ns(a.timestamp())
          : static_cast<int64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count());

  if (a.detections_size() == 0) {
    DetectionRow r;
    r.frame_id = a.frame_id();
    r.ts_ns = ts_ns;
    r.source = a.source();
    r.watchlist_hit = a.watchlist_hit();
    r.matched_label = a.matched_label();
    r.e2e_latency_ms = a.e2e_latency_ms();
    r.nats_seq = seq;
    rows.push_back(std::move(r));
    return rows;
  }

  rows.reserve(static_cast<size_t>(a.detections_size()));
  for (const auto& d : a.detections()) {
    DetectionRow r;
    r.frame_id = a.frame_id();
    r.ts_ns = ts_ns;
    r.source = a.source();
    r.watchlist_hit = a.watchlist_hit();
    r.matched_label = a.matched_label();
    r.e2e_latency_ms = a.e2e_latency_ms();
    r.class_id = d.class_id();
    r.class_name = d.class_name();
    r.confidence = d.confidence();
    r.x1 = d.box().x1();
    r.y1 = d.box().y1();
    r.x2 = d.box().x2();
    r.y2 = d.box().y2();
    r.nats_seq = seq;
    rows.push_back(std::move(r));
  }
  return rows;
}

void insert_batch(clickhouse::Client& ch,
                  const std::vector<DetectionRow>& rows) {
  if (rows.empty()) return;

  auto col_frame = std::make_shared<clickhouse::ColumnInt64>();
  auto col_ts = std::make_shared<clickhouse::ColumnDateTime64>(9);
  auto col_source = std::make_shared<clickhouse::ColumnString>();
  auto col_hit = std::make_shared<clickhouse::ColumnUInt8>();
  auto col_match = std::make_shared<clickhouse::ColumnString>();
  auto col_lat = std::make_shared<clickhouse::ColumnFloat64>();
  auto col_cid = std::make_shared<clickhouse::ColumnInt32>();
  auto col_cname = std::make_shared<clickhouse::ColumnString>();
  auto col_conf = std::make_shared<clickhouse::ColumnFloat32>();
  auto col_x1 = std::make_shared<clickhouse::ColumnFloat32>();
  auto col_y1 = std::make_shared<clickhouse::ColumnFloat32>();
  auto col_x2 = std::make_shared<clickhouse::ColumnFloat32>();
  auto col_y2 = std::make_shared<clickhouse::ColumnFloat32>();
  auto col_seq = std::make_shared<clickhouse::ColumnUInt64>();

  for (const auto& r : rows) {
    col_frame->Append(r.frame_id);
    col_ts->Append(r.ts_ns);
    col_source->Append(r.source);
    col_hit->Append(r.watchlist_hit ? 1 : 0);
    col_match->Append(r.matched_label);
    col_lat->Append(r.e2e_latency_ms);
    col_cid->Append(r.class_id);
    col_cname->Append(r.class_name);
    col_conf->Append(r.confidence);
    col_x1->Append(r.x1);
    col_y1->Append(r.y1);
    col_x2->Append(r.x2);
    col_y2->Append(r.y2);
    col_seq->Append(r.nats_seq);
  }

  clickhouse::Block block;
  block.AppendColumn("frame_id", col_frame);
  block.AppendColumn("ts", col_ts);
  block.AppendColumn("source", col_source);
  block.AppendColumn("watchlist_hit", col_hit);
  block.AppendColumn("matched_label", col_match);
  block.AppendColumn("e2e_latency_ms", col_lat);
  block.AppendColumn("class_id", col_cid);
  block.AppendColumn("class_name", col_cname);
  block.AppendColumn("confidence", col_conf);
  block.AppendColumn("x1", col_x1);
  block.AppendColumn("y1", col_y1);
  block.AppendColumn("x2", col_x2);
  block.AppendColumn("y2", col_y2);
  block.AppendColumn("nats_seq", col_seq);

  ch.Insert("cv_detections", block);
}

}

int main() {
  // init spdlog otel sink
  edge::otel::init("clickhouse-consumer", "0.1.0");
  
  spdlog::set_level(spdlog::level::info);

  const std::string nats_url =
      edge::getenv_or("NATS_URL", "nats://localhost:4222");
  const std::string ch_host =
      edge::getenv_or("CLICKHOUSE_HOST", "localhost");
  const int ch_port =
      std::stoi(edge::getenv_or("CLICKHOUSE_PORT", "9000"));
  const std::string ch_user =
      edge::getenv_or("CLICKHOUSE_USER", "default");
  const std::string ch_pass =
      edge::getenv_or("CLICKHOUSE_PASSWORD", "pass");
  const std::string ch_db =
      edge::getenv_or("CLICKHOUSE_DB", "default");

  constexpr const char* kStream = "CV_ALERTS";
  constexpr const char* kConsumer = "clickhouse-consumer";
  constexpr const char* kSubject = "cv.alert";

  clickhouse::ClientOptions opts;
  opts.SetHost(ch_host);
  opts.SetPort(ch_port);
  opts.SetUser(ch_user);
  opts.SetPassword(ch_pass);
  opts.SetDefaultDatabase(ch_db);

  std::unique_ptr<clickhouse::Client> ch;
  try {
    ch = std::make_unique<clickhouse::Client>(opts);
    ch->Execute(R"(
      CREATE TABLE IF NOT EXISTS cv_detections (
          frame_id         Int64,
          ts               DateTime64(9, 'UTC'),
          source           String,
          watchlist_hit    UInt8,
          matched_label    String,
          e2e_latency_ms   Float64,
          class_id         Int32,
          class_name       String,
          confidence       Float32,
          x1               Float32,
          y1               Float32,
          x2               Float32,
          y2               Float32,
          nats_seq         UInt64,
          ingested_at      DateTime64(3, 'UTC') DEFAULT now64(3)
      ) ENGINE = MergeTree()
      ORDER BY (source, ts, frame_id)
      TTL toDateTime(ts) + INTERVAL 90 DAY
    )");
    spdlog::info("ClickHouse table cv_detections ready");
  } catch (const std::exception& e) {
    spdlog::error("ClickHouse init: {}", e.what());
    return 1;
  }

  natsConnection* nc = nullptr;
  natsOptions* nopts = nullptr;
  natsOptions_Create(&nopts);
  natsOptions_SetURL(nopts, nats_url.c_str());
  natsOptions_SetName(nopts, "edge-clickhouse-consumer");
  natsOptions_SetMaxReconnect(nopts, -1);
  natsOptions_SetReconnectWait(nopts, 2000);

  natsStatus s = natsConnection_Connect(&nc, nopts);
  natsOptions_Destroy(nopts);
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
    spdlog::error("stream {} not ready", kStream);
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
  spdlog::info("clickhouse-consumer ready – {} → cv_detections", kSubject);

  std::vector<DetectionRow> batch;
  batch.reserve(64);
  auto last_flush = std::chrono::steady_clock::now();
  constexpr auto kFlushInterval = std::chrono::milliseconds(500);
  constexpr size_t kFlushSize = 32;

  auto flush = [&]() {
    if (batch.empty()) return;
    try {
      insert_batch(*ch, batch);
      spdlog::info("inserted {} rows into cv_detections", batch.size());
    } catch (const std::exception& e) {
      spdlog::error("ClickHouse insert: {}", e.what());
    }
    batch.clear();
    last_flush = std::chrono::steady_clock::now();
  };

  while (g_running) {
    natsMsgList list{};
    s = natsSubscription_Fetch(&list, sub, 8, 500, &jerr);
    if (s == NATS_TIMEOUT) {
      if (std::chrono::steady_clock::now() - last_flush > kFlushInterval) {
        flush();
      }
      continue;
    }
    if (s != NATS_OK) {
      spdlog::warn("Fetch: {} (jerr={})", natsStatus_GetText(s),
                   static_cast<int>(jerr));
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    for (int i = 0; i < list.Count; ++i) {
      natsMsg* msg = list.Msgs[i];

      detection::v1::Alert alert;
      if (!alert.ParseFromArray(natsMsg_GetData(msg),
                                natsMsg_GetDataLength(msg))) {
        spdlog::warn("bad protobuf payload ({} bytes)",
                     natsMsg_GetDataLength(msg));
        natsMsg_Ack(msg, nullptr);
        continue;
      }

      uint64_t seq = 0;
      jsMsgMetaData* meta = nullptr;
      if (natsMsg_GetMetaData(&meta, msg) == NATS_OK && meta) {
        seq = meta->Sequence.Stream;
        jsMsgMetaData_Destroy(meta);
      }

      auto rows = alert_to_rows(alert, seq);
      batch.insert(batch.end(), rows.begin(), rows.end());
      natsMsg_Ack(msg, nullptr);
    }
    natsMsgList_Destroy(&list);

    if (batch.size() >= kFlushSize) flush();
  }

  flush();

  natsSubscription_Destroy(sub);
  jsCtx_Destroy(js);
  natsConnection_Destroy(nc);
  edge::otel::shutdown();
  return 0;
}
