#pragma once

#include <memory>
#include <string>

#include "ingest/v1/ingest_service.grpc.pb.h"
#include "nats/v1/publisher_service.grpc.pb.h"

namespace edge {

class IngestServiceImpl final : public detection::v1::IngestService::Service {
 public:
  explicit IngestServiceImpl(
      std::shared_ptr<nats::v1::NatsPublisherService::Stub> nats_stub);

  grpc::Status IngestAlert(
      grpc::ServerContext* context,
      const detection::v1::IngestAlertRequest* request,
      detection::v1::IngestAlertResponse* response) override;

 private:
  std::shared_ptr<nats::v1::NatsPublisherService::Stub> nats_stub_;
  static std::string format_seq(uint64_t seq);
};

}
