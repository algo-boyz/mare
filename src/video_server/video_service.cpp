#include "video_service.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
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

float AnnotatedVideoServiceImpl::iou(const DetBox& a, const DetBox& b) {
  const float inter_x1 = std::max(a.x1, b.x1);
  const float inter_y1 = std::max(a.y1, b.y1);
  const float inter_x2 = std::min(a.x2, b.x2);
  const float inter_y2 = std::min(a.y2, b.y2);

  const float inter_w = std::max(0.f, inter_x2 - inter_x1);
  const float inter_h = std::max(0.f, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;

  const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  const float union_area = area_a + area_b - inter_area;

  if (union_area <= 0.f) return 0.f;
  return inter_area / union_area;
}

// ---------------------------------------------------------------------------
// Build tracks with IoU + direction/velocity gate
// ---------------------------------------------------------------------------
std::map<int, Track> AnnotatedVideoServiceImpl::build_tracks(
    const std::unordered_map<int64_t, std::vector<DetBox>>& dets_by_frame,
    float iou_threshold,
    int   max_misses) {

  // Collect sorted frame ids
  std::vector<int64_t> frame_ids;
  frame_ids.reserve(dets_by_frame.size());
  for (const auto& [fid, _] : dets_by_frame) {
    frame_ids.push_back(fid);
  }
  std::sort(frame_ids.begin(), frame_ids.end());

  std::map<int, Track> tracks;                 // track_id → waypoints
  std::map<int, DetBox> active_boxes;          // track_id → last box
  std::map<int, int>    misses;                // track_id → consecutive misses

  // last observed velocity (pixels per frame) per track
  // vx/vy == 0 means “no reliable motion yet”
  std::map<int, std::pair<float, float>> velocity;   // tid → {vx, vy}

  int next_track_id = 0;

  // Helper: box center
  auto center = [](const DetBox& b) -> std::pair<float, float> {
    return {(b.x1 + b.x2) * 0.5f, (b.y1 + b.y2) * 0.5f};
  };

  for (int64_t fid : frame_ids) {
    const auto& dets = dets_by_frame.at(fid);
    std::vector<bool> used(dets.size(), false);

    // --- 1. Collect candidates with IoU + direction gate ---
    struct MatchCandidate {
      int   tid;
      int   det_idx;
      float score;
    };
    std::vector<MatchCandidate> candidates;

    for (const auto& [tid, last_box] : active_boxes) {
      const auto [lx, ly] = center(last_box);
      const auto& [vx, vy] = velocity[tid];          // may be {0,0}
      const bool has_velocity = (vx != 0.f || vy != 0.f);

      for (size_t i = 0; i < dets.size(); ++i) {
        if (used[i]) continue;
        if (dets[i].class_id != last_box.class_id) continue;

        float score = iou(last_box, dets[i]);
        if (score <= iou_threshold) continue;

        // ----- Direction / velocity gate -----
        if (has_velocity) {
          const auto [cx, cy] = center(dets[i]);
          const float dx = cx - lx;
          const float dy = cy - ly;

          // Dot product with previous velocity
          const float dot = dx * vx + dy * vy;

          // Strongly opposite motion → reject
          // (tunable: 0.0 is pure opposite, negative values allow some angle)
          if (dot < -1.0f) {          // -1.0 is a good starting threshold
            continue;
          }

          // Optional soft bonus for continuing in the same direction
          // score *= (1.0f + 0.3f * std::max(0.f, dot / (std::hypot(dx,dy)*std::hypot(vx,vy) + 1e-6f)));
        }
        // ------------------------------------

        candidates.push_back({tid, static_cast<int>(i), score});
      }
    }

    // Greedy assignment sorted by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const MatchCandidate& a, const MatchCandidate& b) {
                return a.score > b.score;
              });

    std::unordered_set<int> matched_tids;
    for (const auto& c : candidates) {
      if (matched_tids.count(c.tid) || used[c.det_idx]) continue;

      used[c.det_idx] = true;
      matched_tids.insert(c.tid);

      const DetBox& matched = dets[c.det_idx];
      tracks[c.tid].push_back({fid, matched});

      // Update velocity (simple last-displacement)
      {
        const auto [lx, ly] = center(active_boxes[c.tid]);
        const auto [cx, cy] = center(matched);
        // We don’t know exact frame delta here (detections can be sparse),
        // so we just store the raw displacement. It is still directionally useful.
        velocity[c.tid] = {cx - lx, cy - ly};
      }

      active_boxes[c.tid] = matched;
      misses[c.tid] = 0;
    }

    // --- 2. Age unmatched tracks ---
    std::vector<int> to_remove;
    for (auto& [tid, last_box] : active_boxes) {
      if (matched_tids.count(tid)) continue;
      misses[tid]++;
      if (misses[tid] > max_misses) {
        to_remove.push_back(tid);
      }
    }
    for (int tid : to_remove) {
      active_boxes.erase(tid);
      misses.erase(tid);
      velocity.erase(tid);               // clean up
      // track stays in `tracks` so interpolation can still use the last waypoints
    }

    // --- 3. Start new tracks for unmatched detections ---
    for (size_t i = 0; i < dets.size(); ++i) {
      if (used[i]) continue;
      int tid = next_track_id++;
      tracks[tid].push_back({fid, dets[i]});
      active_boxes[tid] = dets[i];
      misses[tid] = 0;
      velocity[tid] = {0.f, 0.f};        // no velocity yet
    }
  }

  spdlog::info("built {} tracks from {} detection frames (max_misses={})",
               tracks.size(), frame_ids.size(), max_misses);
  return tracks;
}

// ---------------------------------------------------------------------------
// Produce boxes for a given frame (exact / interpolate / hold)
// ---------------------------------------------------------------------------
std::vector<DetBox> AnnotatedVideoServiceImpl::boxes_for_frame(
    int64_t frame_id,
    const std::map<int, Track>& tracks,
    int max_hold_frames) {

  std::vector<DetBox> out;

  for (const auto& [tid, waypoints] : tracks) {
    if (waypoints.empty()) continue;

    // waypoints are already sorted by frame_id
    auto it = std::lower_bound(
        waypoints.begin(), waypoints.end(), frame_id,
        [](const TrackWaypoint& wp, int64_t fid) {
          return wp.frame_id < fid;
        });

    if (it != waypoints.end() && it->frame_id == frame_id) {
      // Exact detection
      out.push_back(it->box);
      continue;
    }

    if (it == waypoints.begin()) {
      // Before first waypoint → nothing
      continue;
    }

    auto prev = std::prev(it);

    if (it != waypoints.end()) {
      // Between prev and *it → linear interpolation
      const int64_t f0 = prev->frame_id;
      const int64_t f1 = it->frame_id;
      if (f1 == f0) {
        out.push_back(prev->box);
        continue;
      }

      const float t = static_cast<float>(frame_id - f0) /
                      static_cast<float>(f1 - f0);

      DetBox interp;
      interp.class_id   = prev->box.class_id;
      interp.class_name = prev->box.class_name;
      interp.confidence = prev->box.confidence * (1.f - t) +
                          it->box.confidence   * t;
      interp.x1 = prev->box.x1 + t * (it->box.x1 - prev->box.x1);
      interp.y1 = prev->box.y1 + t * (it->box.y1 - prev->box.y1);
      interp.x2 = prev->box.x2 + t * (it->box.x2 - prev->box.x2);
      interp.y2 = prev->box.y2 + t * (it->box.y2 - prev->box.y2);

      // Optional: drop if the interpolated box is almost completely outside
      // (prevents weird “push-back” artefacts near the border)
      // You can tune the threshold or remove this check if you prefer.
      // const float img_w = ...; const float img_h = ...; // if known
      // if (interp.x2 < 5 || interp.y2 < 5 || interp.x1 > img_w-5 || ...) continue;

      out.push_back(interp);
    } else {
      // After last waypoint → hold for a limited number of frames
      const int64_t last_f = prev->frame_id;
      if (frame_id - last_f <= max_hold_frames) {
        out.push_back(prev->box);
      }
      // else track is considered dead
    }
  }

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

  if (!video_path.string().starts_with(fs::weakly_canonical(video_root_).string())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "src path escape");
  }

  if (!fs::exists(video_path)) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND,
                        "video not found: " + video_path.string());
  }

  const int64_t start_frame = std::max<int64_t>(0, request->start_frame_id());
  const int64_t end_frame   = request->end_frame_id();
  const int quality = (request->jpeg_quality() > 0 && request->jpeg_quality() <= 100)
                          ? request->jpeg_quality()
                          : 85;

  // Preload detections
  std::unordered_map<int64_t, std::vector<DetBox>> dets_by_frame;
  try {
    dets_by_frame = load_detections(request->source(), start_frame, end_frame);
  } catch (const std::exception& e) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string("ClickHouse error: ") + e.what());
  }

  // Build tracks (with aging)
  const auto tracks = build_tracks(dets_by_frame,
                                   /*iou_threshold=*/0.3f,
                                   /*max_misses=*/8);

  cv::VideoCapture cap(video_path.string());
  if (!cap.isOpened()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "failed to open video: " + video_path.string());
  }

  // Real-time pacing
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 1.0 || fps > 120.0) {
    fps = 25.0;   // safe fallback
  }
  const auto frame_duration =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / fps));
  auto next_frame_time = std::chrono::steady_clock::now();

  spdlog::info("streaming annotated video {} (start={}, end={}, only_dets={}, tracks={}, fps={:.2f})",
               video_path.string(), start_frame, end_frame,
               request->only_with_detections(), tracks.size(), fps);

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

    // Synthesize smoothed boxes
    std::vector<DetBox> boxes = boxes_for_frame(frame_id, tracks, /*max_hold=*/12);
    const bool has_dets = !boxes.empty();

    if (request->only_with_detections() && !has_dets) {
      ++frame_id;
      continue;
    }

    // Draw
    if (has_dets) {
      draw_boxes(frame, boxes);
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
    out.set_detection_count(static_cast<int32_t>(boxes.size()));

    for (const auto& b : boxes) {
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

    if (!writer->Write(out)) {
      spdlog::info("writer closed by client after {} frames", emitted);
      break;
    }

    ++emitted;
    ++frame_id;

    // Real-time pacing
    next_frame_time += frame_duration;
    std::this_thread::sleep_until(next_frame_time);
  }

  spdlog::info("finished stream emitted {} annotated frames", emitted);
  return grpc::Status::OK;
}

grpc::Status AnnotatedVideoServiceImpl::DownloadAnnotatedVideo(
    grpc::ServerContext* context,
    const detection::v1::DownloadAnnotatedVideoRequest* request,
    detection::v1::DownloadAnnotatedVideoResponse* response) {

  if (request->source().empty()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "src required");
  }

  // Resolve path under VIDEO_ROOT
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

  // Load detections
  std::unordered_map<int64_t, std::vector<DetBox>> dets_by_frame;
  try {
    dets_by_frame = load_detections(request->source(), start_frame, end_frame);
  } catch (const std::exception& e) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        std::string("ClickHouse error: ") + e.what());
  }

  // Build tracks
  const auto tracks = build_tracks(dets_by_frame,
                                   /*iou_threshold=*/0.3f,
                                   /*max_misses=*/8);

  cv::VideoCapture cap(video_path.string());
  if (!cap.isOpened()) {
    return grpc::Status(grpc::StatusCode::INTERNAL,
                        "failed to open video: " + video_path.string());
  }

  double fps = request->fps() > 0 ? request->fps() : cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0.0) fps = 30.0;

  int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  fs::path out_path = fs::temp_directory_path() /
                      ("annotated_" + video_path.filename().string());

  cv::VideoWriter writer(out_path.string(),
                         cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                         fps, cv::Size(width, height));

  if (!writer.isOpened()) {
    writer.open(out_path.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(width, height));
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

    // Synthesize smoothed boxes
    std::vector<DetBox> boxes = boxes_for_frame(frame_id, tracks, /*max_hold=*/12);
    if (!boxes.empty()) {
      draw_boxes(frame, boxes);
    }

    writer.write(frame);
    ++processed_frames;
    ++frame_id;
  }

  writer.release();

  response->set_download_url(out_path.string());
  response->set_total_frames(processed_frames);
  response->set_duration_sec(fps > 0 ? static_cast<double>(processed_frames) / fps : 0.0);

  spdlog::info("generated downloaded video at {} ({} frames, {} tracks)",
               out_path.string(), processed_frames, tracks.size());
  return grpc::Status::OK;
}

}
