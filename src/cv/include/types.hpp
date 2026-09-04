#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

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
};

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

}
