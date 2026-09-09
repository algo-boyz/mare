#pragma once

#include "types.hpp"
#include "geo.hpp"

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace edge_cv {

// Lightweight IoU + class-aware tracker (ByteTrack-lite style).
// Assigns persistent track_ids across frames for the same object.
// Good enough for dashcam / parking footage without external deps.
class Tracker {
public:
    struct Config {
        float iou_threshold   = 0.3f;
        int   max_age         = 30;
        int   min_hits        = 2;
    };

    Tracker() : cfg_(Config{}) {}
    explicit Tracker(Config cfg) : cfg_(cfg) {}

    // Updates tracks with new detections (modifies dets in-place: sets track_id).
    // Returns the same detections with track_ids filled.
    std::vector<Detection> update(std::vector<Detection> dets);

    int num_tracks() const { return static_cast<int>(tracks_.size()); }

private:
    struct Track {
        int         id{-1};
        Detection   last;
        int         age{0};       // frames since last update
        int         hits{0};      // total successful associations
        bool        confirmed{false};
    };

    Config cfg_;
    int next_id_{1};
    std::vector<Track> tracks_;

    static float iou(const BoundingBox& a, const BoundingBox& b) {
        return geom::intersect(a, b);
    }
};

} // namespace edge_cv
