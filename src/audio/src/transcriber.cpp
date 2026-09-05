#include "transcriber.hpp"

#include <chrono>
#include <stdexcept>

#include "sherpa-onnx/c-api/cxx-api.h"

namespace edge_audio {

Transcriber::Transcriber(const Config& cfg) : cfg_(cfg) {
  using namespace sherpa_onnx::cxx;

  OnlineRecognizerConfig config;
  config.model_config.transducer.encoder = cfg_.encoder;
  config.model_config.transducer.decoder = cfg_.decoder;
  config.model_config.transducer.joiner  = cfg_.joiner;
  config.model_config.tokens             = cfg_.tokens;
  config.model_config.num_threads        = cfg_.num_threads;
  config.model_config.provider           = cfg_.provider;
  config.model_config.debug              = false;

  config.enable_endpoint               = cfg_.enable_endpoint;
  config.rule1_min_trailing_silence    = cfg_.rule1_min_trailing_silence;
  config.rule2_min_trailing_silence    = cfg_.rule2_min_trailing_silence;
  config.rule3_min_utterance_length    = cfg_.rule3_min_utterance_length;

  recognizer_ = std::make_unique<OnlineRecognizer>(
      OnlineRecognizer::Create(config));
  if (!recognizer_ || !recognizer_->Get()) {
    throw std::runtime_error("Failed to create sherpa-onnx OnlineRecognizer");
  }
  stream_ = std::make_unique<OnlineStream>(recognizer_->CreateStream());
}

Transcriber::~Transcriber() = default;

void Transcriber::accept_waveform(float sample_rate, const float* samples,
                                  size_t n) {
  if (!stream_ || n == 0) return;
  stream_->AcceptWaveform(sample_rate, samples, static_cast<int>(n));
}

bool Transcriber::decode() {
  if (!recognizer_ || !stream_) return false;

  bool decoded = false;
  auto t0 = Clock::now();
  while (recognizer_->IsReady(stream_.get())) {
    recognizer_->Decode(stream_.get());
    decoded = true;
  }
  if (decoded) {
    last_decode_ms_ = std::chrono::duration<double, std::milli>(
                          Clock::now() - t0)
                          .count();
  }
  return decoded;
}

std::string Transcriber::text() const {
  if (!recognizer_ || !stream_) return {};
  auto result = recognizer_->GetResult(stream_.get());
  return result.text;
}

bool Transcriber::is_endpoint() const {
  if (!recognizer_ || !stream_) return false;
  return recognizer_->IsEndpoint(stream_.get());
}

void Transcriber::reset() {
  if (!recognizer_ || !stream_) return;
  recognizer_->Reset(stream_.get());
}

void Transcriber::input_finished() {
  if (!stream_) return;
  stream_->InputFinished();
}

}