#include <csignal>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <nats.h>
#include <spdlog/spdlog.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "publisher_service.hpp"

namespace {
natsConnection* g_nc = nullptr;
jsCtx* g_js = nullptr;
std::unique_ptr<grpc::Server> g_server;

void signal_handler(int) {
  spdlog::info("shutting down nats publisher...");
  if (g_server) g_server->Shutdown();
}
}

int main() {
  // init spdlog otel sink
  edge::otel::init("nats-publisher", "0.1.0");
  spdlog::set_level(spdlog::level::info);

  const std::string nats_url =
      edge::getenv_or("NATS_URL", "nats://localhost:4222");
  const std::string listen_addr =
      edge::getenv_or("GRPC_ADDR", "0.0.0.0:50051");
  constexpr const char* kStream = "CV_ALERTS";
  constexpr const char* kSubject = "cv.alert";

  natsOptions* opts = nullptr;
  natsOptions_Create(&opts);
  natsOptions_SetURL(opts, nats_url.c_str());
  natsOptions_SetName(opts, "edge-nats-publisher");
  natsOptions_SetMaxReconnect(opts, -1);
  natsOptions_SetReconnectWait(opts, 2000);

  natsStatus s = natsConnection_Connect(&g_nc, opts);
  natsOptions_Destroy(opts);
  if (s != NATS_OK) {
    spdlog::error("nats.Connect: {}", natsStatus_GetText(s));
    return 1;
  }

  jsOptions jsOpts;
  jsOptions_Init(&jsOpts);
  s = natsConnection_JetStream(&g_js, g_nc, &jsOpts);
  if (s != NATS_OK) {
    spdlog::error("JetStream: {}", natsStatus_GetText(s));
    natsConnection_Destroy(g_nc);
    return 1;
  }

  // Ensure stream exists
  jsStreamConfig sc;
  jsStreamConfig_Init(&sc);
  sc.Name = const_cast<char*>(kStream);
  const char* subjects[] = {kSubject};
  sc.Subjects = subjects;  // const char** in modern nats.c
  sc.SubjectsLen = 1;
  sc.Retention = js_LimitsPolicy;
  sc.Storage = js_FileStorage;
  sc.MaxMsgs = -1;
  sc.MaxBytes = -1;

  jsStreamInfo* si = nullptr;
  jsErrCode jerr{};
  s = js_AddStream(&si, g_js, &sc, nullptr, &jerr);
  if (s == NATS_OK) {
    spdlog::info("stream {} ready", kStream);
    if (si) jsStreamInfo_Destroy(si);
  } else if (s == NATS_ERR && jerr == JSStreamNameExistErr) {
    spdlog::info("stream {} already exists", kStream);
  } else {
    spdlog::error("AddStream: {} (jerr={})", natsStatus_GetText(s),
                  static_cast<int>(jerr));
    jsCtx_Destroy(g_js);
    natsConnection_Destroy(g_nc);
    return 1;
  }

  edge::NatsPublisherServiceImpl service(g_js);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  g_server = builder.BuildAndStart();
  if (!g_server) {
    spdlog::error("failed to start publisher grpc on {}", listen_addr);
    return 1;
  }
  spdlog::info("NATS Publisher grpc listening on {}", listen_addr);

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  g_server->Wait();

  if (g_js) jsCtx_Destroy(g_js);
  if (g_nc) natsConnection_Destroy(g_nc);
  edge::otel::shutdown();
  return 0;
}
