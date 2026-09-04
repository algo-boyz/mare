#pragma once

#include <filesystem>
#include <string>

namespace edge {

inline std::string sanitize(const std::string& raw_source) {
  if (raw_source.empty()) return "unknown_src";
  
  // converts sample.mp4 to sample
  std::filesystem::path p(raw_source);
  return p.stem().string();
}

}