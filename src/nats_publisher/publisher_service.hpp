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

 private:
  jsCtx* js_;  // non-owning
  static constexpr const char* kDefaultSubject = "cv.alert";
};

}
