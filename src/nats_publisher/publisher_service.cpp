#include "publisher_service.hpp"

#include <spdlog/spdlog.h>

#include "audio/v1/transcript.pb.h"
#include "common/v1/error.pb.h"
#include "detection/v1/detection.pb.h"

namespace edge {

NatsPublisherServiceImpl::NatsPublisherServiceImpl(jsCtx* js) : js_(js) {}

template <typename Response>
bool NatsPublisherServiceImpl::publish_payload(const std::string& subject,
                                               const std::string& payload,
                                               Response* response,
                                               uint64_t* out_seq) {
  jsPubOptions opts;
  jsPubOptions_Init(&opts);

  jsPubAck* ack = nullptr;
  jsErrCode jerr{};
  natsStatus s = js_Publish(&ack, js_, subject.c_str(),
                            payload.data(), static_cast<int>(payload.size()),
                            &opts, &jerr);

  if (s != NATS_OK) {
    spdlog::error("publish to {} failed: {} (jerr={})", subject,
                  natsStatus_GetText(s), static_cast<int>(jerr));
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("PUBLISH_FAILED");
    err->set_message(natsStatus_GetText(s));
    if (ack) jsPubAck_Destroy(ack);
    return false;
  }

  uint64_t seq = 0;
  if (ack) {
    seq = ack->Sequence;
    jsPubAck_Destroy(ack);
  }
  *out_seq = seq;
  return true;
}

// Explicit instantiations (both response types share the same shape)
template bool NatsPublisherServiceImpl::publish_payload(
    const std::string&, const std::string&,
    nats::v1::PublishAlertResponse*, uint64_t*);
template bool NatsPublisherServiceImpl::publish_payload(
    const std::string&, const std::string&,
    nats::v1::PublishTranscriptResponse*, uint64_t*);

grpc::Status NatsPublisherServiceImpl::PublishAlert(
    grpc::ServerContext* /*context*/,
    const nats::v1::PublishAlertRequest* request,
    nats::v1::PublishAlertResponse* response) {

  if (!request->has_alert()) {
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("INVALID_ARGUMENT");
    err->set_message("alert is required");
    return grpc::Status::OK;
  }

  const std::string subject = request->subject().empty()
                                  ? kDefaultAlertSubject
                                  : request->subject();

  std::string payload;
  if (!request->alert().SerializeToString(&payload)) {
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("SERIALIZE_FAILED");
    err->set_message("failed to serialize Alert protobuf");
    return grpc::Status::OK;
  }

  uint64_t seq = 0;
  if (!publish_payload(subject, payload, response, &seq)) {
    return grpc::Status::OK;
  }

  spdlog::info("published to {} seq={} frame={} dets={} bytes={}",
               subject, seq, request->alert().frame_id(),
               request->alert().detections_size(), payload.size());

  response->set_published(true);
  response->set_subject(subject);
  response->set_sequence(seq);
  return grpc::Status::OK;
}

grpc::Status NatsPublisherServiceImpl::PublishTranscript(
    grpc::ServerContext* /*context*/,
    const nats::v1::PublishTranscriptRequest* request,
    nats::v1::PublishTranscriptResponse* response) {

  if (!request->has_transcript()) {
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("INVALID_ARGUMENT");
    err->set_message("transcript is required");
    return grpc::Status::OK;
  }

  const std::string subject = request->subject().empty()
                                  ? kDefaultTranscriptSubject
                                  : request->subject();

  std::string payload;
  if (!request->transcript().SerializeToString(&payload)) {
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("SERIALIZE_FAILED");
    err->set_message("failed to serialize Transcript protobuf");
    return grpc::Status::OK;
  }

  uint64_t seq = 0;
  if (!publish_payload(subject, payload, response, &seq)) {
    return grpc::Status::OK;
  }

  const auto& t = request->transcript();
  spdlog::info("published to {} seq={} source={} final={} text=\"{}\" bytes={}",
               subject, seq, t.source(), t.is_final(),
               t.text().substr(0, 80), payload.size());

  response->set_published(true);
  response->set_subject(subject);
  response->set_sequence(seq);
  return grpc::Status::OK;
}

}  // namespace edge