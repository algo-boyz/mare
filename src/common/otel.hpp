#pragma once

#include <string>

namespace edge::otel {

// Init global LoggerProvider + install spdlog OTel sink.
// Safe to call multiple times subsequent calls are no-ops.
// When compiled without WITH_OTEL the function is no-op
void init(const std::string& service_name,
          const std::string& service_version = "0.1.0");

// Flush and tear down the provider
void shutdown();

}