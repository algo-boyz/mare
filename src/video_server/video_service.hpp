#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <clickhouse/client.h>
#include <opencv2/core.hpp>

#include "detection/v1/annotated_video.grpc.pb.h"
#include "detection/v1/detection.pb.h"

namespace edge {

struct DetBox {
  int32_t     class_id{-1};
  std::string class_name;
  float       confidence{0.f};
  float       x1{0.f}, y1{0.f}, x2{0.f}, y2{0.f};
};

class AnnotatedVideoServiceImpl final
    : public detection::v1::AnnotatedVideoService::Service {
 public:
  AnnotatedVideoServiceImpl(std::unique_ptr<clickhouse::Client> ch,
                            std::string video_root);

  grpc::Status StreamAnnotatedVideo(
      grpc::ServerContext* context,
      const detection::v1::StreamAnnotatedVideoRequest* request,
      grpc::ServerWriter<detection::v1::AnnotatedFrame>* writer) override;

  grpc::Status DownloadAnnotatedVideo(
      grpc::ServerContext* context,
      const detection::v1::DownloadAnnotatedVideoRequest* request,
      detection::v1::DownloadAnnotatedVideoResponse* response) override;

 private:
  // Load all detections for the given src + frame range, grouped by frame_id
  std::unordered_map<int64_t, std::vector<DetBox>> load_detections(
      const std::string& source, int64_t start_frame, int64_t end_frame);

  static void draw_boxes(cv::Mat& frame, const std::vector<DetBox>& boxes);

  std::unique_ptr<clickhouse::Client> ch_;
  std::mutex ch_mutex_;  // Guard concurrent access to ch_
  std::string video_root_;
};

}