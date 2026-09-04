#include <csignal>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>

#include "common/env.hpp"
#include "common/otel.hpp"
#include "ingest_service.hpp"

namespace {
std::unique_ptr<grpc::Server> g_server;

void signal_handler(int) {
  spdlog::info("shutting down ingest server...");
  if (g_server) g_server->Shutdown();
}
}

int main() {
  // init spdlog otel sink
  edge::otel::init("ingest-service", "0.1.0");
  spdlog::set_level(spdlog::level::info);

  const std::string listen_addr =
      edge::getenv_or("GRPC_ADDR", "0.0.0.0:50052");
  const std::string publisher_addr =
      edge::getenv_or("PUBLISHER_ADDR", "localhost:50051");

  auto channel = grpc::CreateChannel(publisher_addr,
                                     grpc::InsecureChannelCredentials());
  auto nats_stub = nats::v1::NatsPublisherService::NewStub(channel);

  edge::IngestServiceImpl service(std::move(nats_stub));

  grpc::ServerBuilder builder;
  builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  g_server = builder.BuildAndStart();
  if (!g_server) {
    spdlog::error("failed to start ingest grpc server on {}", listen_addr);
    return 1;
  }
  spdlog::info("Ingest grpc listening on {} (publisher at {})",
               listen_addr, publisher_addr);

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  g_server->Wait();
  edge::otel::shutdown();
  return 0;
}
