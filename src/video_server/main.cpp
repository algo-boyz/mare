#include <csignal>
#include <memory>
#include <string>

#include "absl/base/config.h"
#include "absl/base/options.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <spdlog/spdlog.h>

#include <clickhouse/client.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "video_service.hpp"

namespace {
std::unique_ptr<grpc::Server> g_server;

void signal_handler(int) {
  spdlog::info("shutting down video_server...");
  if (g_server) g_server->Shutdown();
}
}

int main() {
  edge::otel::init("video_server", "0.1.0");
  spdlog::set_level(spdlog::level::info);

  const std::string listen_addr =
      edge::getenv_or("GRPC_ADDR", "0.0.0.0:50053");
  const std::string video_root =
      edge::getenv_or("VIDEO_ROOT", "./assets");

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

  clickhouse::ClientOptions opts;
  opts.SetHost(ch_host);
  opts.SetPort(ch_port);
  opts.SetUser(ch_user);
  opts.SetPassword(ch_pass);
  opts.SetDefaultDatabase(ch_db);

  std::unique_ptr<clickhouse::Client> ch;
  try {
    ch = std::make_unique<clickhouse::Client>(opts);
    // lightweight ping
    ch->Execute("SELECT 1");
    spdlog::info("ClickHouse connected ({}:{}/{})", ch_host, ch_port, ch_db);
  } catch (const std::exception& e) {
    spdlog::error("ClickHouse connect failed: {}", e.what());
    return 1;
  }

  edge::AnnotatedVideoServiceImpl service(std::move(ch), video_root);

  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Larger message size for jpg frames
  builder.SetMaxSendMessageSize(16 * 1024 * 1024);
  builder.SetMaxReceiveMessageSize(4 * 1024 * 1024);

  g_server = builder.BuildAndStart();
  if (!g_server) {
    spdlog::error("failed to start annotated_video grpc on {}", listen_addr);
    return 1;
  }

  spdlog::info("AnnotatedVideo grpc listening on {} (VIDEO_ROOT={})",
               listen_addr, video_root);

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  g_server->Wait();
  edge::otel::shutdown();
  return 0;
}