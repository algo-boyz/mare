#pragma once

#include "capture.hpp"
#include "transcriber.hpp"
#include "types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace edge_audio {

// Parallel audio modality:
// AudioCapture → AudioQueue → Transcriber (own thread) → TranscriptEvent callback.
// Never blocks the vision process loop.
class AudioPipeline {
public:
  using TranscriptCallback = std::function<void(const TranscriptEvent&)>;

  struct Config {
    AudioCapture::Config capture;
    Transcriber::Config transcriber;
    std::string source{"mic"};           // logical source id for correlation
    TranscriptCallback on_transcript;
    bool print_partial{true};
    bool emit_partial{true};             // also fire callback for partials
  };

  explicit AudioPipeline(Config cfg);
  ~AudioPipeline();

  AudioPipeline(const AudioPipeline&) = delete;
  AudioPipeline& operator=(const AudioPipeline&) = delete;

  void start();
  void stop();
  bool is_running() const { return running_.load(); }

private:
  void process_loop();

  Config cfg_;
  std::shared_ptr<AudioQueue> queue_;
  std::unique_ptr<AudioCapture> capture_;
  std::unique_ptr<Transcriber> transcriber_;

  std::thread process_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};

  // Utterance tracking
  SystemTimePoint utterance_start_;
  bool in_utterance_{false};
  std::string last_text_;
};

}