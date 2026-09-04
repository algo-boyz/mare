#include "video_service.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <mutex>
#include <sstream>

#include <clickhouse/columns/numeric.h>
#include <clickhouse/columns/string.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace edge {

namespace {

// Escapes single quotes and backslashes for string literals in SQL queries
std::string sanitize_sql_string(const std::string& input) {
  std::string escaped;
  escaped.reserve(input.size());
  for (char c : input) {
    if (c == '\'' || c == '\\') {
      escaped.push_back('\\');
    }
    escaped.push_back(c);
  }
  return escaped;
}

}  // namespace

AnnotatedVideoServiceImpl::AnnotatedVideoServiceImpl(
    std::unique_ptr<clickhouse::Client> ch, std::string video_root)
    : ch_(std::move(ch)), video_root_(std::move(video_root)) {}

std::unordered_map<int64_t, std::vector<DetBox>>
AnnotatedVideoServiceImpl::load_detections(const std::string& source,
                                           int64_t start_frame,
                                           int64_t end_frame) {
  std::unordered_map<int64_t, std::vector<DetBox>> out;

  // sanitize src prevent sqli
  const std::string safe_source = sanitize_sql_string(source);

  std::ostringstream sql;
  sql << "SELECT frame_id, class_id, class_name, confidence, x1, y1, x2, y2 "
      << "FROM cv_detections "
      << "WHERE source = '" << safe_source << "' "
      << "AND frame_id >= " << start_frame << " ";
  if (end_frame > 0) {
    sql << "AND frame_id < " << end_frame << " ";
  }
  sql << "ORDER BY frame_id";

  try {
    // Synchronize access to clickhouse::Client across gRPC worker threads
    std::lock_guard<std::mutex> lock(ch_mutex_);

    ch_->Select(sql.str(), [&](const clickhouse::Block& block) {
      if (block.GetColumnCount() == 0 || block.GetRowCount() == 0) {
        return;  // ignore empty progress block
      }
      auto col_fid  = block[0]->As<clickhouse::ColumnInt64>();
      auto col_cid  = block[1]->As<clickhouse::ColumnInt32>();
      auto col_name = block[2]->As<clickhouse::ColumnString>();
      auto col_conf = block[3]->As<clickhouse::ColumnFloat32>();
      auto col_x1   = block[4]->As<clickhouse::ColumnFloat32>();
      auto col_y1   = block[5]->As<clickhouse::ColumnFloat32>();
      auto col_x2   = block[6]->As<clickhouse::ColumnFloat32>();
      auto col_y2   = block[7]->As<clickhouse::ColumnFloat32>();

      for (size_t i = 0; i < block.GetRowCount(); ++i) {
        DetBox b;
        b.class_id   = col_cid->At(i);
        b.class_name = std::string(col_name->At(i));
        b.confidence = col_conf->At(i);
        b.x1         = col_x1->At(i);
        b.y1         = col_y1->At(i);
        b.x2         = col_x2->At(i);
        b.y2         = col_y2->At(i);

        out[col_fid->At(i)].push_back(std::move(b));
      }
    });
  } catch (const std::exception& e) {
    spdlog::error("ClickHouse query failed: {}", e.what());
    throw;
  }

  spdlog::info("loaded detections for {} frames (source={})", out.size(),
               source);
  return out;
}

void AnnotatedVideoServiceImpl::draw_boxes(cv::Mat& frame,
                                           const std::vector<DetBox>& boxes) {
  for (const auto& b : boxes) {
    const cv::Scalar color(0, 255, 0);  // green
    cv::rectangle(frame,
                  cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1)),
                  cv::Point(static_cast<int>(b.x2), static_cast<int>(b.y2)),
                  color, 2);

    std::ostringstream label;
    label << b.class_name << " " << std::fixed << std::setprecision(2)
          << b.confidence;

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseline);

    cv::rectangle(frame,
                  cv::Point(static_cast<int>(b.x1),
                            static_cast<int>(b.y1) - text_size.height - 4),
                  cv::Point(static_cast<int>(b.x1) + text_size.width,
                            static_cast<int>(b.y1)),
                  color, cv::FILLED);

    cv::putText(frame, label.str(),
                cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1) - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

grpc::Status AnnotatedVideoServiceImpl::StreamAnnotatedVideo(
    grpc::ServerContext* context,
    const detection::v1::StreamAnnotatedVideoRequest* request,
    grpc::ServerWriter<detection::v1::AnnotatedFrame>* writer) {

  if (request->source().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "source is required");
  }

  // Resolve path safely under VIDEO_ROOT
  fs::path video_path;
  if (fs::path(request->source()).is_absolute()) {
    video_path = request->source();
  } else {
    video_path = fs::path(video_root_) / request->source();
  }
  video_path = fs::weakly_canonical(video_path);

  // Basic path traversal protection
  if (!video_path.string().starts_with(fs::weakly_canonical(video_root_).string())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "src path escape");
  }

  if (!fs::exists(video_path)) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "video not found: " + video_path.string());
  }

  const int64_t start_frame = std::max<int64_t>(0, request->start_frame_id());
  const int64_t end_frame   = request->end_frame_id();
  const int quality = request->jpeg_quality() > 0 && request->jpeg_quality() <= 100
                          ? request->jpeg_quality()
                          : 85;

  // preload detections
  std::unordered_map<int64_t, std::vector<DetBox>> dets_by_frame;
  try {
    dets_by_frame = load_detections(request->source(), start_frame, end_frame);
  } catch (const std::exception& e) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string("ClickHouse error: ") + e.what());
  }

  cv::VideoCapture cap(video_path.string());
  if (!cap.isOpened()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "failed to open video: " + video_path.string());
  }

  spdlog::info("streaming annotated video {} (start={}, end={}, only_dets={})",
               video_path.string(), start_frame, end_frame,
               request->only_with_detections());

  int64_t frame_id = 0;
  cv::Mat frame;
  size_t emitted = 0;

  while (cap.read(frame)) {
    if (context->IsCancelled()) {
      spdlog::info("client cancelled stream after {} frames", emitted);
      break;
    }

    if (frame_id < start_frame) {
      ++frame_id;
      continue;
    }
    if (end_frame > 0 && frame_id >= end_frame) {
      break;
    }

    auto it = dets_by_frame.find(frame_id);
    const bool has_dets = it != dets_by_frame.end() && !it->second.empty();

    if (request->only_with_detections() && !has_dets) {
      ++frame_id;
      continue;
    }

    // Draw
    if (has_dets) {
      draw_boxes(frame, it->second);
    }

    // Encode
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", frame, buf, params)) {
      spdlog::warn("JPEG encode failed for frame {}", frame_id);
      ++frame_id;
      continue;
    }

    detection::v1::AnnotatedFrame out;
    out.set_frame_id(frame_id);
    out.set_width(frame.cols);
    out.set_height(frame.rows);
    out.set_jpeg(buf.data(), buf.size());
    out.set_detection_count(has_dets ? static_cast<int32_t>(it->second.size()) : 0);

    if (has_dets) {
      for (const auto& b : it->second) {
        auto* d = out.add_detections();
        d->set_class_id(b.class_id);
        d->set_class_name(b.class_name);
        d->set_confidence(b.confidence);
        auto* box = d->mutable_box();
        box->set_x1(b.x1);
        box->set_y1(b.y1);
        box->set_x2(b.x2);
        box->set_y2(b.y2);
      }
    }

    if (!writer->Write(out)) {
      spdlog::info("writer closed by client after {} frames", emitted);
      break;
    }

    ++emitted;
    ++frame_id;
  }

  spdlog::info("finished stream emitted {} annotated frames", emitted);
  return grpc::Status::OK;
}

grpc::Status AnnotatedVideoServiceImpl::DownloadAnnotatedVideo(
    grpc::ServerContext* context,
    const detection::v1::DownloadAnnotatedVideoRequest* request,
    detection::v1::DownloadAnnotatedVideoResponse* response) {

  if (request->source().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "src required");
  }

  // Resolve path VIDEO_ROOT
  fs::path video_path;
  if (fs::path(request->source()).is_absolute()) {
    video_path = request->source();
  } else {
    video_path = fs::path(video_root_) / request->source();
  }
  video_path = fs::weakly_canonical(video_path);

  if (!video_path.string().starts_with(fs::weakly_canonical(video_root_).string())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "source path escapes VIDEO_ROOT");
  }

  if (!fs::exists(video_path)) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        "video not found: " + video_path.string());
  }

  const int64_t start_frame = std::max<int64_t>(0, request->start_frame_id());
  const int64_t end_frame   = request->end_frame_id();

  // Load detections using mutex
  std::unordered_map<int64_t, std::vector<DetBox>> dets_by_frame;
  try {
    dets_by_frame = load_detections(request->source(), start_frame, end_frame);
  } catch (const std::exception& e) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string("ClickHouse error: ") + e.what());
  }

  cv::VideoCapture cap(video_path.string());
  if (!cap.isOpened()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "failed to open video: " + video_path.string());
  }

  double fps = request->fps() > 0 ? request->fps() : cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 30.0; // Fallback default FPS

  int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  // Output destination file in tmp dir
  fs::path out_path = fs::temp_directory_path() /
                      ("annotated_" + video_path.filename().string());

  cv::VideoWriter writer(out_path.string(),
                         cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264
                         fps,
                         cv::Size(width, height));

  if (!writer.isOpened()) {
    // use mp4 if avc1 isn't available
    writer.open(out_path.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps,
                cv::Size(width, height));
  }

  if (!writer.isOpened()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "failed to init VideoWriter for out file");
  }

  int64_t frame_id = 0;
  int64_t processed_frames = 0;
  cv::Mat frame;

  while (cap.read(frame)) {
    if (context->IsCancelled()) {
      spdlog::info("download request cancelled by client");
      break;
    }

    if (frame_id < start_frame) {
      ++frame_id;
      continue;
    }
    if (end_frame > 0 && frame_id >= end_frame) {
      break;
    }

    auto it = dets_by_frame.find(frame_id);
    if (it != dets_by_frame.end() && !it->second.empty()) {
      draw_boxes(frame, it->second);
    }

    writer.write(frame);
    ++processed_frames;
    ++frame_id;
  }

  writer.release();

  response->set_download_url(out_path.string());
  response->set_total_frames(processed_frames);
  response->set_duration_sec(fps > 0 ? static_cast<double>(processed_frames) / fps : 0.0);

  spdlog::info("generated downloaded video at {} ({} frames)", out_path.string(), processed_frames);
  return grpc::Status::OK;
}

}