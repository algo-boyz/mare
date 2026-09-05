#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "audio/v1/transcript.pb.h"
#include "edge_audio/audio_pipeline.hpp"
#include "ingest/v1/ingest_service.grpc.pb.h"
#include "live_service.hpp"

namespace {

std::atomic<bool> g_running{true};
std::unique_ptr<grpc::Server> g_server;

void on_signal(int) {
  g_running = false;
  if (g_server) g_server->Shutdown();
}

audio::v1::Transcript to_proto(const edge_audio::TranscriptEvent& e) {
  audio::v1::Transcript out;
  out.set_source(e.source);
  out.set_text(e.text);
  out.set_is_final(e.is_final);
  out.set_confidence(e.confidence);
  out.set_language(e.language);
  out.set_speaker_id(e.speaker_id);
  out.set_e2e_latency_ms(e.e2e_latency_ms);

  auto to_ts = [](edge_audio::SystemTimePoint tp) {
    google::protobuf::Timestamp ts;
    auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp - secs);
    ts.set_seconds(secs.time_since_epoch().count());
    ts.set_nanos(static_cast<int32_t>(ns.count()));
    return ts;
  };

  *out.mutable_timestamp() = to_ts(e.timestamp);
  *out.mutable_audio_start() = to_ts(e.audio_start);
  *out.mutable_audio_end() = to_ts(e.audio_end);
  return out;
}

bool env_flag(const char* name, bool default_value = false) {
  const char* v = std::getenv(name);
  if (!v || !*v) return default_value;
  return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' ||
         v[0] == 'Y';
}

std::string env_or(const char* k, const std::string& def) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : def;
}

}  // namespace

int main() {
  edge::otel::init("edge-audio-client", "0.1.0");
  spdlog::set_level(spdlog::level::info);

  const std::string live_addr = edge::getenv_or("LIVE_ADDR", "0.0.0.0:50054");
  const std::string ingest_addr =
      edge::getenv_or("INGEST_ADDR", "localhost:50052");
  const std::string source = env_or("SOURCE", "mic");
  const bool ingest_finals = env_flag("INGEST_FINALS", true);
  const bool print_partial = env_flag("PRINT_PARTIAL", true);

  const std::string model_root = env_or(
      "SHERPA_MODEL_DIR",
      "models/sherpa-onnx-streaming-zipformer-en-2023-06-26");

  auto bus = std::make_shared<edge::TranscriptBroadcaster>();

  std::unique_ptr<audio::v1::AudioIngestService::Stub> ingest_stub;
  if (ingest_finals) {
    auto channel =
        grpc::CreateChannel(ingest_addr, grpc::InsecureChannelCredentials());
    ingest_stub = audio::v1::AudioIngestService::NewStub(channel);
    spdlog::info("finals → ingest at {}", ingest_addr);
  } else {
    spdlog::info("INGEST_FINALS=0 – live stream only");
  }

  edge::AudioLiveServiceImpl live_service(bus);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(live_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&live_service);
  g_server = builder.BuildAndStart();
  if (!g_server) {
    spdlog::error("failed to start AudioLiveService on {}", live_addr);
    return 1;
  }
  spdlog::info("AudioLiveService listening on {}", live_addr);

  edge_audio::AudioPipeline::Config cfg;
  cfg.source = source;
  cfg.print_partial = print_partial;
  cfg.emit_partial = true;

  cfg.transcriber.tokens =
      env_or("SHERPA_TOKENS", model_root + "/tokens.txt");
  cfg.transcriber.encoder = env_or(
      "SHERPA_ENCODER",
      model_root + "/encoder-epoch-99-avg-1-chunk-16-left-128.onnx");
  cfg.transcriber.decoder = env_or(
      "SHERPA_DECODER",
      model_root + "/decoder-epoch-99-avg-1-chunk-16-left-128.onnx");
  cfg.transcriber.joiner = env_or(
      "SHERPA_JOINER",
      model_root + "/joiner-epoch-99-avg-1-chunk-16-left-128.onnx");
  cfg.transcriber.num_threads = std::stoi(env_or("SHERPA_THREADS", "2"));
  cfg.transcriber.provider = env_or("SHERPA_PROVIDER", "cpu");

  cfg.on_transcript = [&](const edge_audio::TranscriptEvent& ev) {
    const auto msg = to_proto(ev);
    bus->publish(msg);  // partials + finals > live subscribers

    if (ev.is_final && ingest_stub) {
      audio::v1::IngestTranscriptRequest req;
      *req.mutable_transcript() = msg;
      audio::v1::IngestTranscriptResponse resp;
      grpc::ClientContext ctx;
      ctx.set_deadline(std::chrono::system_clock::now() +
                       std::chrono::milliseconds(500));
      auto status = ingest_stub->IngestTranscript(&ctx, req, &resp);
      if (!status.ok()) {
        spdlog::warn("final ingest failed: {}", status.error_message());
      } else if (resp.accepted()) {
        spdlog::info("final ingested msg_id={} text=\"{}\"",
                     resp.message_id(), ev.text.substr(0, 80));
      }
    }
  };

  std::unique_ptr<edge_audio::AudioPipeline> pipeline;
  try {
    pipeline = std::make_unique<edge_audio::AudioPipeline>(std::move(cfg));
    pipeline->start();
    spdlog::info("audio pipeline started source={}", source);
  } catch (const std::exception& e) {
    spdlog::error("failed to start audio pipeline: {}", e.what());
    if (g_server) g_server->Shutdown();
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::thread wait_thread([&] { g_server->Wait(); });
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  pipeline->stop();
  if (g_server) g_server->Shutdown();
  if (wait_thread.joinable()) wait_thread.join();

  spdlog::info("edge_audio_client stopped");
  edge::otel::shutdown();
  return 0;
}