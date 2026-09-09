#include "tracker.hpp"

#include <limits>
#include <iostream>

namespace edge_cv {

std::vector<Detection> Tracker::update(std::vector<Detection> dets) {
    // 1. Age existing tracks
    for (auto& t : tracks_) {
        t.age++;
    }

    // 2. Greedy matching: for each detection, find best unmatched track of same class
    std::vector<bool> det_matched(dets.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);

    // Build cost matrix style: sort by confidence desc so high-conf first
    std::vector<size_t> order(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return dets[a].confidence > dets[b].confidence;
    });

    for (size_t di : order) {
        float best_iou = cfg_.iou_threshold;
        int   best_ti  = -1;

        for (size_t ti = 0; ti < tracks_.size(); ++ti) {
            if (track_matched[ti]) continue;
            if (tracks_[ti].last.class_id != dets[di].class_id) continue;

            float i = iou(tracks_[ti].last.box, dets[di].box);
            if (i > best_iou) {
                best_iou = i;
                best_ti  = static_cast<int>(ti);
            }
        }

        if (best_ti >= 0) {
            // Associate
            auto& t = tracks_[best_ti];
            t.last = dets[di];
            t.age  = 0;
            t.hits++;
            if (t.hits >= cfg_.min_hits) t.confirmed = true;
            dets[di].track_id = t.id;
            det_matched[di] = true;
            track_matched[best_ti] = true;
        }
    }

    // 3. Create new tracks for unmatched detections
    for (size_t di = 0; di < dets.size(); ++di) {
        if (det_matched[di]) continue;
        Track t;
        t.id   = next_id_++;
        t.last = dets[di];
        t.age  = 0;
        t.hits = 1;
        t.confirmed = (cfg_.min_hits <= 1);
        dets[di].track_id = t.id;
        tracks_.push_back(std::move(t));
    }

    // 4. Drop old tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const Track& t) { return t.age > cfg_.max_age; }),
        tracks_.end());

    return dets;
}

} // namespace edge_cv
