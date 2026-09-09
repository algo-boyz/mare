#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

namespace edge_cv {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using DurationMs = std::chrono::duration<double, std::milli>;

struct BoundingBox {
    float x1{0.f};
    float y1{0.f};
    float x2{0.f};
    float y2{0.f};

    float width()  const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area()   const { return width() * height(); }
};

struct Detection {
    int         class_id{-1};
    std::string class_name;
    float       confidence{0.f};
    BoundingBox box;
    int         track_id{-1};         // assigned by tracker; -1 = untracked
    std::string ocr_text;             // filled by OCR stage when available
    float       ocr_confidence{0.f};  // OCR confidence
};

// Per-track memory for gating expensive secondary stages (plate OCR, etc.)
struct TrackState {
    int         track_id{-1};
    int         age{0};
    int         frames_since_ocr{999};
    std::string last_ocr_text;
    float       last_ocr_conf{0.f};
    bool        plate_confirmed{false};  // high-conf OCR obtained → skip future work
};

using TrackStateMap = std::unordered_map<int, TrackState>;

struct FrameMeta {
    int64_t   frame_id{0};
    TimePoint capture_ts;
    int       width{0};
    int       height{0};
};

struct Alert {
    int64_t                  frame_id{0};
    TimePoint                timestamp;
    std::vector<Detection>   detections;
    bool                     watchlist_hit{false};
    std::string              matched_label;   // "person:42"
    double                   e2e_latency_ms{0.0};
};

struct WatchlistEntry {
    std::string label;
    float       min_confidence{0.5f};
};

} // namespace edge_cv
