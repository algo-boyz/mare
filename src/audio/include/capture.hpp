#pragma once

#include "types.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// PortAudio forward
typedef void PaStream;

namespace edge_audio {

// Thread-safe bounded queue of PCM chunks (drop-old when full for low latency).
class AudioQueue {
public:
  explicit AudioQueue(size_t capacity = 8);

  void push(AudioChunk chunk);
  std::optional<AudioChunk> pop(std::chrono::milliseconds timeout);
  void stop();
  bool stopped() const;

private:
  size_t cap_;
  std::queue<AudioChunk> q_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<bool> stopped_{false};
};

// Capture thread: PortAudio callback → AudioQueue with timestamps.
class AudioCapture {
public:
  struct Config {
    int device_index{-1};          // -1 = default input
    float sample_rate{16000.f};
    unsigned long frames_per_buffer{1024};
    size_t queue_capacity{8};
  };

  AudioCapture(Config cfg, std::shared_ptr<AudioQueue> queue);
  ~AudioCapture();

  AudioCapture(const AudioCapture&) = delete;
  AudioCapture& operator=(const AudioCapture&) = delete;

  void start();
  void stop();
  bool is_running() const { return running_.load(); }

  float sample_rate() const { return cfg_.sample_rate; }
  std::string device_name() const { return device_name_; }

private:
  static int pa_callback(const void* input, void* output,
                         unsigned long frame_count,
                         const void* time_info,
                         unsigned long status_flags,
                         void* user_data);

  Config cfg_;
  std::shared_ptr<AudioQueue> queue_;
  PaStream* stream_{nullptr};
  std::atomic<bool> running_{false};
  std::string device_name_{"unknown"};
  int64_t sequence_{0};
};

}