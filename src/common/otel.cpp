#include "common/otel.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifdef WITH_OTEL

// Switch from otlp_http to otlp_grpc
#include "opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_grpc_log_record_exporter_options.h"
#include "opentelemetry/logs/provider.h"
#include "opentelemetry/sdk/logs/batch_log_record_processor_factory.h"
#include "opentelemetry/sdk/logs/logger_provider_factory.h"
#include "opentelemetry/sdk/resource/resource.h"

namespace nostd     = opentelemetry::nostd;
namespace otlp      = opentelemetry::exporter::otlp;
namespace logs_sdk  = opentelemetry::sdk::logs;
namespace logs_api  = opentelemetry::logs;
namespace resource  = opentelemetry::sdk::resource;

namespace {

// Lock-free mapping from spdlog into OTel memory structures
template <typename Mutex>
class OTelSpdlogSink : public spdlog::sinks::base_sink<Mutex> {
 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    auto provider = logs_api::Provider::GetLoggerProvider();
    if (!provider) return;

    auto logger = provider->GetLogger("edge_spdlog");
    if (!logger) return;

    logs_api::Severity severity = logs_api::Severity::kInfo;
    switch (msg.level) {
      case spdlog::level::trace:    severity = logs_api::Severity::kTrace; break;
      case spdlog::level::debug:    severity = logs_api::Severity::kDebug; break;
      case spdlog::level::info:     severity = logs_api::Severity::kInfo; break;
      case spdlog::level::warn:     severity = logs_api::Severity::kWarn; break;
      case spdlog::level::err:      severity = logs_api::Severity::kError; break;
      case spdlog::level::critical: severity = logs_api::Severity::kFatal; break;
      default: break;
    }

    // Pushes into in-memory queue; worker thread flushes via grpc asynchronously
    std::string_view payload(msg.payload.data(), msg.payload.size());
    logger->EmitLogRecord(severity, payload);
  }

  void flush_() override {}
};

using OTelSpdlogSink_mt = OTelSpdlogSink<std::mutex>;

}

#endif  // WITH_OTEL

namespace edge::otel {
namespace {

std::once_flag g_init_flag;
#ifdef WITH_OTEL
std::shared_ptr<logs_sdk::LoggerProvider> g_provider;
#endif

std::string env_or(const char* key, const std::string& fallback) {
  if (const char* v = std::getenv(key); v && *v) return std::string{v};
  return fallback;
}

}

void init(const std::string& service_name,
          const std::string& service_version) {
#ifdef WITH_OTEL
  std::call_once(g_init_flag, [&] {
    auto res = resource::Resource::Create({
        {"service.name", service_name},
        {"service.version", service_version},
        {"deployment.environment", env_or("OTEL_ENVIRONMENT", "development")},
    });

    // Configure grpc Exporter Options
    otlp::OtlpGrpcLogRecordExporterOptions opts;
    // Default OTLP grpc port is 4317 (compared to HTTP 4318)
    opts.endpoint = env_or("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317");
    if (const char* logs_ep = std::getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT");
        logs_ep && *logs_ep) {
      opts.endpoint = logs_ep;
    }

    // Create grpc Exporter instance
    auto exporter = otlp::OtlpGrpcLogRecordExporterFactory::Create(opts);

    // Configure BatchProcessor options (determines async worker thread behavior)
    logs_sdk::BatchLogRecordProcessorOptions proc_opts;
    proc_opts.max_queue_size = 4096;                      // Maximum pending logs buffer
    proc_opts.schedule_delay_millis = std::chrono::milliseconds(200); // Flush interval
    proc_opts.max_export_batch_size = 512;                 // Max grpc message batch size

    // BatchLogRecordProcessor spawns a dedicated worker thread automatically!
    auto processor = logs_sdk::BatchLogRecordProcessorFactory::Create(
        std::move(exporter), proc_opts);

    g_provider = logs_sdk::LoggerProviderFactory::Create(
        std::move(processor), res);

    nostd::shared_ptr<logs_api::LoggerProvider> api_provider(g_provider);
    logs_api::Provider::SetLoggerProvider(api_provider);

    auto otel_sink = std::make_shared<OTelSpdlogSink_mt>();
    otel_sink->set_level(spdlog::level::info);

    auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console->set_level(spdlog::level::info);

    auto logger = std::make_shared<spdlog::logger>(
        "edge", spdlog::sinks_init_list{console, otel_sink});
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::warn);
    spdlog::set_default_logger(logger);

    spdlog::info("OpenTelemetry grpc logging initialised  service={}  endpoint={}",
                 service_name, opts.endpoint);
  });
#else
  (void)service_name;
  (void)service_version;
#endif
}

void shutdown() {
#ifdef WITH_OTEL
  if (g_provider) {
    // Gracefully signals the worker thread to send remaining grpc batches before exiting
    g_provider->ForceFlush();
    nostd::shared_ptr<logs_api::LoggerProvider> none;
    logs_api::Provider::SetLoggerProvider(none);
    g_provider.reset();
  }
#endif
  spdlog::shutdown();
}

}