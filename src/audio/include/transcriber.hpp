#pragma once

#include "types.hpp"

#include <memory>
#include <string>

// Forward-declare to avoid pulling sherpa headers into public API
namespace sherpa_onnx {
namespace cxx {
class OnlineRecognizer;
class OnlineStream;
}
}

namespace edge_audio {

// Streaming STT wrapper around sherpa-onnx OnlineRecognizer.
// Load model once; feed PCM + sample rate; emit partial + final transcripts.
class Transcriber {
public:
  struct Config {
    std::string tokens;
    std::string encoder;
    std::string decoder;
    std::string joiner;
    int num_threads{2};
    std::string provider{"cpu"};   // cpu | cuda | coreml
    bool enable_endpoint{true};
    float rule1_min_trailing_silence{2.4f};
    float rule2_min_trailing_silence{1.2f};
    float rule3_min_utterance_length{20.f};
  };

  explicit Transcriber(const Config& cfg);
  ~Transcriber();

  Transcriber(const Transcriber&) = delete;
  Transcriber& operator=(const Transcriber&) = delete;

  // Feed mono float32 samples. Internally resamples if needed.
  void accept_waveform(float sample_rate, const float* samples, size_t n);

  // Decode ready frames. Returns true if a new hypothesis is available.
  bool decode();

  // Current hypothesis text (may be partial).
  std::string text() const;

  // True when endpoint (end-of-utterance) is detected.
  bool is_endpoint() const;

  // Reset stream after a final result was consumed.
  void reset();

  // Mark end of input (flush).
  void input_finished();

  double last_decode_ms() const { return last_decode_ms_; }

private:
  Config cfg_;
  std::unique_ptr<sherpa_onnx::cxx::OnlineRecognizer> recognizer_;
  std::unique_ptr<sherpa_onnx::cxx::OnlineStream> stream_;
  double last_decode_ms_{0.0};
};

}