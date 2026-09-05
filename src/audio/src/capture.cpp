#include "capture.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "portaudio.h"

namespace edge_audio {

// ---------- AudioQueue ----------

AudioQueue::AudioQueue(size_t capacity) : cap_(capacity) {}

void AudioQueue::push(AudioChunk chunk) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (stopped_) return;
  while (q_.size() >= cap_) {
    q_.pop();  // drop-old
  }
  q_.push(std::move(chunk));
  cv_.notify_one();
}

std::optional<AudioChunk> AudioQueue::pop(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (!cv_.wait_for(lock, timeout, [&] { return stopped_ || !q_.empty(); })) {
    return std::nullopt;
  }
  if (q_.empty()) return std::nullopt;
  AudioChunk out = std::move(q_.front());
  q_.pop();
  return out;
}

void AudioQueue::stop() {
  std::lock_guard<std::mutex> lock(mtx_);
  stopped_ = true;
  cv_.notify_all();
}

bool AudioQueue::stopped() const { return stopped_.load(); }

// ---------- AudioCapture ----------

AudioCapture::AudioCapture(Config cfg, std::shared_ptr<AudioQueue> queue)
    : cfg_(std::move(cfg)), queue_(std::move(queue)) {}

AudioCapture::~AudioCapture() { stop(); }

int AudioCapture::pa_callback(const void* input, void* /*output*/,
                              unsigned long frame_count,
                              const void* /*time_info*/,
                              unsigned long /*status_flags*/,
                              void* user_data) {
  auto* self = static_cast<AudioCapture*>(user_data);
  if (!self || !self->running_.load() || !input) return paContinue;

  const float* samples = static_cast<const float*>(input);
  AudioChunk chunk;
  chunk.samples.assign(samples, samples + frame_count);
  chunk.sample_rate = self->cfg_.sample_rate;
  chunk.capture_ts = SystemClock::now();
  chunk.sequence = self->sequence_++;

  self->queue_->push(std::move(chunk));
  return paContinue;
}

void AudioCapture::start() {
  if (running_.exchange(true)) return;

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    running_ = false;
    throw std::runtime_error(std::string("Pa_Initialize failed: ") +
                             Pa_GetErrorText(err));
  }

  int device = cfg_.device_index;
  if (device < 0) {
    device = Pa_GetDefaultInputDevice();
  }
  if (const char* env = std::getenv("SHERPA_ONNX_MIC_DEVICE")) {
    device = std::atoi(env);
  }
  if (device == paNoDevice) {
    Pa_Terminate();
    running_ = false;
    throw std::runtime_error("No input audio device");
  }

  const PaDeviceInfo* info = Pa_GetDeviceInfo(device);
  device_name_ = info ? info->name : "unknown";

  if (const char* env = std::getenv("SHERPA_ONNX_MIC_SAMPLE_RATE")) {
    cfg_.sample_rate = static_cast<float>(std::atof(env));
  }

  PaStreamParameters params{};
  params.device = device;
  params.channelCount = 1;
  params.sampleFormat = paFloat32;
  params.suggestedLatency =
      info ? info->defaultLowInputLatency : 0.01;
  params.hostApiSpecificStreamInfo = nullptr;

  err = Pa_OpenStream(&stream_, &params, nullptr, cfg_.sample_rate,
                      cfg_.frames_per_buffer, paClipOff,
                      reinterpret_cast<PaStreamCallback*>(pa_callback), this);
  if (err != paNoError) {
    Pa_Terminate();
    running_ = false;
    throw std::runtime_error(std::string("Pa_OpenStream failed: ") +
                             Pa_GetErrorText(err));
  }

  err = Pa_StartStream(stream_);
  if (err != paNoError) {
    Pa_CloseStream(stream_);
    stream_ = nullptr;
    Pa_Terminate();
    running_ = false;
    throw std::runtime_error(std::string("Pa_StartStream failed: ") +
                             Pa_GetErrorText(err));
  }

  std::cerr << "[audio] capture started device=\"" << device_name_
            << "\" sr=" << cfg_.sample_rate << "\n";
}

void AudioCapture::stop() {
  if (!running_.exchange(false)) return;

  if (stream_) {
    Pa_StopStream(stream_);
    Pa_CloseStream(stream_);
    stream_ = nullptr;
  }
  Pa_Terminate();
  if (queue_) queue_->stop();
}

}