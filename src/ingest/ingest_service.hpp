#pragma once

#include <memory>
#include <string>

#include "ingest/v1/ingest_service.grpc.pb.h"
#include "nats/v1/publisher_service.grpc.pb.h"

namespace edge {

class IngestServiceImpl final : public ingest::v1::IngestService::Service {
 public:
  explicit IngestServiceImpl(
      std::shared_ptr<nats::v1::NatsPublisherService::Stub> nats_stub);

  grpc::Status IngestAlert(
      grpc::ServerContext* context,
      const ingest::v1::IngestAlertRequest* request,
      ingest::v1::IngestAlertResponse* response) override;

  grpc::Status IngestTranscript(
      grpc::ServerContext* context,
      const ingest::v1::IngestTranscriptRequest* request,
      ingest::v1::IngestTranscriptResponse* response) override;

 private:
  std::shared_ptr<nats::v1::NatsPublisherService::Stub> nats_stub_;
  static std::string format_seq(uint64_t seq);
};

}  // namespace edge