#pragma once

#include <memory>
#include <string>

#include <nats.h>

#include "nats/v1/publisher_service.grpc.pb.h"

namespace edge {

class NatsPublisherServiceImpl final
    : public nats::v1::NatsPublisherService::Service {
 public:
  explicit NatsPublisherServiceImpl(jsCtx* js);
  ~NatsPublisherServiceImpl() override = default;

  grpc::Status PublishAlert(
      grpc::ServerContext* context,
      const nats::v1::PublishAlertRequest* request,
      nats::v1::PublishAlertResponse* response) override;

  grpc::Status PublishTranscript(
      grpc::ServerContext* context,
      const nats::v1::PublishTranscriptRequest* request,
      nats::v1::PublishTranscriptResponse* response) override;

 private:
  // Shared JetStream publish helper.
  // On success fills *out_seq and returns true.
  // On failure fills response error fields via the supplied setters and returns false.
  template <typename Response>
  bool publish_payload(const std::string& subject,
                       const std::string& payload,
                       Response* response,
                       uint64_t* out_seq);

  jsCtx* js_;  // non-owning
  static constexpr const char* kDefaultAlertSubject = "cv.alert";
  static constexpr const char* kDefaultTranscriptSubject = "audio.transcript";
};

}  // namespace edge