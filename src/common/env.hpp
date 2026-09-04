#pragma once

#include <cstdlib>
#include <string>

namespace edge {

inline std::string getenv_or(const char* key, const std::string& fallback) {
  if (const char* v = std::getenv(key); v && *v) {
    return std::string{v};
  }
  return fallback;
}

}
