#include "pipeline.hpp"

#include <iostream>

namespace edge_audio {

AudioPipeline::AudioPipeline(Config cfg) : cfg_(std::move(cfg)) {
  queue_ = std::make_shared<AudioQueue>(cfg_.capture.queue_capacity);
  transcriber_ = std::make_unique<Transcriber>(cfg_.transcriber);
  capture_ = std::make_unique<AudioCapture>(cfg_.capture, queue_);
}

AudioPipeline::~AudioPipeline() { stop(); }

void AudioPipeline::start() {
  if (running_.exchange(true)) return;
  stop_requested_ = false;
  capture_->start();
  process_thread_ = std::thread(&AudioPipeline::process_loop, this);
}

void AudioPipeline::stop() {
  if (!running_.exchange(false)) return;
  stop_requested_ = true;
  if (queue_) queue_->stop();
  if (capture_) capture_->stop();
  if (process_thread_.joinable()) process_thread_.join();
}

void AudioPipeline::process_loop() {
  using namespace std::chrono_literals;

  while (!stop_requested_) {
    auto chunk_opt = queue_->pop(50ms);
    if (!chunk_opt) continue;

    auto& chunk = *chunk_opt;

    if (!in_utterance_) {
      utterance_start_ = chunk.capture_ts;
      in_utterance_ = true;
    }

    transcriber_->accept_waveform(chunk.sample_rate, chunk.samples.data(),
                                  chunk.samples.size());
    transcriber_->decode();

    std::string text = transcriber_->text();
    auto now = SystemClock::now();

    // Partial update
    if (!text.empty() && text != last_text_) {
      if (cfg_.print_partial) {
        std::cout << "\r\033[K[audio partial] " << text << std::flush;
      }

      if (cfg_.emit_partial && cfg_.on_transcript) {
        TranscriptEvent ev;
        ev.source = cfg_.source;
        ev.timestamp = now;
        ev.audio_start = utterance_start_;
        ev.audio_end = chunk.capture_ts;
        ev.text = text;
        ev.is_final = false;
        ev.confidence = 0.f;
        ev.e2e_latency_ms =
            std::chrono::duration<double, std::milli>(now - chunk.capture_ts)
                .count();
        cfg_.on_transcript(ev);
      }
      last_text_ = text;
    }

    // Endpoint → final
    if (transcriber_->is_endpoint()) {
      if (!last_text_.empty()) {
        if (cfg_.print_partial) {
          std::cout << "\n[audio final] " << last_text_ << "\n";
        }

        if (cfg_.on_transcript) {
          TranscriptEvent ev;
          ev.source = cfg_.source;
          ev.timestamp = now;
          ev.audio_start = utterance_start_;
          ev.audio_end = chunk.capture_ts;
          ev.text = last_text_;
          ev.is_final = true;
          ev.confidence = 0.f;
          ev.e2e_latency_ms =
              std::chrono::duration<double, std::milli>(now - utterance_start_)
                  .count();
          cfg_.on_transcript(ev);
        }
      }
      transcriber_->reset();
      last_text_.clear();
      in_utterance_ = false;
    }
  }

  // Flush remaining
  transcriber_->input_finished();
  transcriber_->decode();
  auto final_text = transcriber_->text();
  if (!final_text.empty() && final_text != last_text_ && cfg_.on_transcript) {
    TranscriptEvent ev;
    ev.source = cfg_.source;
    ev.timestamp = SystemClock::now();
    ev.audio_start = utterance_start_;
    ev.audio_end = ev.timestamp;
    ev.text = final_text;
    ev.is_final = true;
    cfg_.on_transcript(ev);
  }

  running_ = false;
}

}