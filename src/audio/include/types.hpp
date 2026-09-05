#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace edge_audio {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using SystemClock = std::chrono::system_clock;
using SystemTimePoint = SystemClock::time_point;

struct AudioChunk {
  std::vector<float> samples;   // mono float32
  float sample_rate{16000.f};
  SystemTimePoint capture_ts;   // wall clock of first sample
  int64_t sequence{0};
};

struct TranscriptEvent {
  std::string source;
  SystemTimePoint timestamp;        // when the event was produced
  SystemTimePoint audio_start;
  SystemTimePoint audio_end;
  std::string text;
  bool is_final{false};
  float confidence{0.f};
  std::string language;
  std::string speaker_id;
  double e2e_latency_ms{0.0};
};

}