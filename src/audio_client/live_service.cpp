#include "live_service.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>

#include <spdlog/spdlog.h>

namespace edge {

TranscriptBroadcaster::ListenerId TranscriptBroadcaster::subscribe(ListenerFn fn) {
  std::lock_guard<std::mutex> lock(mtx_);
  const auto id = next_id_++;
  listeners_.emplace(id, std::move(fn));
  return id;
}

void TranscriptBroadcaster::unsubscribe(ListenerId id) {
  std::lock_guard<std::mutex> lock(mtx_);
  listeners_.erase(id);
}

void TranscriptBroadcaster::publish(const audio::v1::Transcript& t) {
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto& [id, fn] : listeners_) {
    try {
      fn(t);
    } catch (const std::exception& e) {
      spdlog::warn("broadcaster listener {}: {}", id, e.what());
    }
  }
}

size_t TranscriptBroadcaster::listener_count() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return listeners_.size();
}

AudioLiveServiceImpl::AudioLiveServiceImpl(
    std::shared_ptr<TranscriptBroadcaster> bus)
    : bus_(std::move(bus)) {}

grpc::Status AudioLiveServiceImpl::StreamPartials(
    grpc::ServerContext* context,
    const audio::v1::StreamPartialsRequest* request,
    grpc::ServerWriter<audio::v1::Transcript>* writer) {

  const std::string filter_source = request->source();
  const auto mode = request->mode();  // default STREAM_MODE_ALL

  auto accept = [mode](const audio::v1::Transcript& t) {
    switch (mode) {
      case audio::v1::STREAM_MODE_PARTIALS:
        return !t.is_final();
      case audio::v1::STREAM_MODE_FINALS:
        return t.is_final();
      case audio::v1::STREAM_MODE_ALL:
      default:
        return true;
    }
  };

  constexpr size_t kMaxQueue = 8;
  std::mutex q_mtx;
  std::condition_variable q_cv;
  std::deque<audio::v1::Transcript> queue;
  bool closed = false;

  auto id = bus_->subscribe([&](const audio::v1::Transcript& t) {
    if (!filter_source.empty() && t.source() != filter_source) return;
    if (!accept(t)) return;

    std::lock_guard<std::mutex> lock(q_mtx);
    if (closed) return;
    while (queue.size() >= kMaxQueue) queue.pop_front();
    queue.push_back(t);
    q_cv.notify_one();
  });

  spdlog::info(
      "StreamPartials connected source=\"{}\" mode={} listeners={}",
      filter_source, static_cast<int>(mode), bus_->listener_count());

  while (!context->IsCancelled()) {
    audio::v1::Transcript msg;
    {
      std::unique_lock<std::mutex> lock(q_mtx);
      q_cv.wait_for(lock, std::chrono::milliseconds(200), [&] {
        return closed || !queue.empty() || context->IsCancelled();
      });
      if (context->IsCancelled()) break;
      if (queue.empty()) continue;
      msg = std::move(queue.front());
      queue.pop_front();
    }
    if (!writer->Write(msg)) break;
  }

  {
    std::lock_guard<std::mutex> lock(q_mtx);
    closed = true;
  }
  bus_->unsubscribe(id);
  spdlog::info("StreamPartials disconnected listeners={}",
               bus_->listener_count());
  return grpc::Status::OK;
}

}  // namespace edge