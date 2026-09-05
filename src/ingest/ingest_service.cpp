#include "ingest_service.hpp"

#include <spdlog/spdlog.h>

namespace edge {

IngestServiceImpl::IngestServiceImpl(
    std::shared_ptr<nats::v1::NatsPublisherService::Stub> nats_stub)
    : nats_stub_(std::move(nats_stub)) {}

std::string IngestServiceImpl::format_seq(uint64_t seq) {
  return "js-" + std::to_string(seq);
}

grpc::Status IngestServiceImpl::IngestAlert(
    grpc::ServerContext* /*context*/,
    const ingest::v1::IngestAlertRequest* request,
    ingest::v1::IngestAlertResponse* response) {

  if (!request->has_alert()) {
    response->set_accepted(false);
    auto* err = response->mutable_error();
    err->set_code("INVALID_ARGUMENT");
    err->set_message("alert is required");
    return grpc::Status::OK;
  }

  nats::v1::PublishAlertRequest pub_req;
  *pub_req.mutable_alert() = request->alert();
  pub_req.set_subject("cv.alert");

  nats::v1::PublishAlertResponse pub_resp;
  grpc::ClientContext client_ctx;
  auto status = nats_stub_->PublishAlert(&client_ctx, pub_req, &pub_resp);

  if (!status.ok()) {
    spdlog::error("nats publisher call failed: {}", status.error_message());
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "nats publisher unavailable: " + status.error_message());
  }

  if (!pub_resp.published()) {
    std::string msg = "publish rejected";
    if (pub_resp.has_error()) msg = pub_resp.error().message();
    response->set_accepted(false);
    auto* err = response->mutable_error();
    err->set_code("PUBLISH_FAILED");
    err->set_message(msg);
    return grpc::Status::OK;
  }

  const auto& a = request->alert();
  spdlog::info("ingested frame={} dets={} latency={:.1f}ms → seq={}",
               a.frame_id(), a.detections_size(), a.e2e_latency_ms(),
               pub_resp.sequence());

  response->set_accepted(true);
  response->set_message_id(format_seq(pub_resp.sequence()));
  return grpc::Status::OK;
}

grpc::Status IngestServiceImpl::IngestTranscript(
    grpc::ServerContext* /*context*/,
    const ingest::v1::IngestTranscriptRequest* request,
    ingest::v1::IngestTranscriptResponse* response) {

  if (!request->has_transcript()) {
    response->set_accepted(false);
    auto* err = response->mutable_error();
    err->set_code("INVALID_ARGUMENT");
    err->set_message("transcript is required");
    return grpc::Status::OK;
  }

  nats::v1::PublishTranscriptRequest pub_req;
  *pub_req.mutable_transcript() = request->transcript();
  pub_req.set_subject("audio.transcript");

  nats::v1::PublishTranscriptResponse pub_resp;
  grpc::ClientContext client_ctx;
  auto status = nats_stub_->PublishTranscript(&client_ctx, pub_req, &pub_resp);

  if (!status.ok()) {
    spdlog::error("nats publisher call failed: {}", status.error_message());
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "nats publisher unavailable: " + status.error_message());
  }

  if (!pub_resp.published()) {
    std::string msg = "publish rejected";
    if (pub_resp.has_error()) msg = pub_resp.error().message();
    response->set_accepted(false);
    auto* err = response->mutable_error();
    err->set_code("PUBLISH_FAILED");
    err->set_message(msg);
    return grpc::Status::OK;
  }

  const auto& t = request->transcript();
  spdlog::info("ingested transcript source={} final={} latency={:.1f}ms "
               "text=\"{}\" → seq={}",
               t.source(), t.is_final(), t.e2e_latency_ms(),
               t.text().substr(0, 80), pub_resp.sequence());

  response->set_accepted(true);
  response->set_message_id(format_seq(pub_resp.sequence()));
  return grpc::Status::OK;
}

}  // namespace edge