#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <grpcpp/grpcpp.h>

#include "audio/v1/live_service.grpc.pb.h"
#include "audio/v1/transcript.pb.h"

namespace edge {

class TranscriptBroadcaster {
 public:
  using ListenerId = uint64_t;
  using ListenerFn = std::function<void(const audio::v1::Transcript&)>;

  ListenerId subscribe(ListenerFn fn);
  void unsubscribe(ListenerId id);
  void publish(const audio::v1::Transcript& t);
  size_t listener_count() const;

 private:
  mutable std::mutex mtx_;
  std::unordered_map<ListenerId, ListenerFn> listeners_;
  std::atomic<ListenerId> next_id_{1};
};

class AudioLiveServiceImpl final : public audio::v1::AudioLiveService::Service {
 public:
  explicit AudioLiveServiceImpl(std::shared_ptr<TranscriptBroadcaster> bus);

  grpc::Status StreamPartials(
      grpc::ServerContext* context,
      const audio::v1::StreamPartialsRequest* request,
      grpc::ServerWriter<audio::v1::Transcript>* writer) override;

 private:
  std::shared_ptr<TranscriptBroadcaster> bus_;
};

}  // namespace edge