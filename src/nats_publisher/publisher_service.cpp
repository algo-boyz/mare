#include "publisher_service.hpp"

#include <spdlog/spdlog.h>

#include "common/v1/error.pb.h"
#include "detection/v1/detection.pb.h"

namespace edge {

NatsPublisherServiceImpl::NatsPublisherServiceImpl(jsCtx* js) : js_(js) {}

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
                                  ? kDefaultSubject
                                  : request->subject();

  // Binary protobuf payload (smaller + faster than JSON)
  std::string payload;
  if (!request->alert().SerializeToString(&payload)) {
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("SERIALIZE_FAILED");
    err->set_message("failed to serialize Alert protobuf");
    return grpc::Status::OK;
  }

  jsPubOptions opts;
  jsPubOptions_Init(&opts);

  jsPubAck* ack = nullptr;
  jsErrCode jerr{};
  natsStatus s = js_Publish(&ack, js_, subject.c_str(),
                            payload.data(), static_cast<int>(payload.size()),
                            &opts, &jerr);

  if (s != NATS_OK) {
    spdlog::error("publish failed: {} (jerr={})", natsStatus_GetText(s),
                  static_cast<int>(jerr));
    response->set_published(false);
    auto* err = response->mutable_error();
    err->set_code("PUBLISH_FAILED");
    err->set_message(natsStatus_GetText(s));
    if (ack) jsPubAck_Destroy(ack);
    return grpc::Status::OK;
  }

  uint64_t seq = 0;
  if (ack) {
    seq = ack->Sequence;
    jsPubAck_Destroy(ack);
  }

  spdlog::info("published to {} seq={} frame={} dets={} bytes={}",
               subject, seq, request->alert().frame_id(),
               request->alert().detections_size(), payload.size());

  response->set_published(true);
  response->set_subject(subject);
  response->set_sequence(seq);
  return grpc::Status::OK;
}

}
