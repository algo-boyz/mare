#include "geo.hpp"
#include "postprocess.hpp"

#include <algorithm>
#include <cmath>

namespace edge_cv {


std::vector<Detection> nms(std::vector<Detection> dets, float threshold) {
    if (dets.empty()) return dets;

    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            // class aware detection
            if (dets[i].class_id != dets[j].class_id) continue;
            if (geom::intersect(dets[i].box, dets[j].box) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

std::vector<Detection> filter_detections(
    const std::vector<Detection>& dets,
    float min_conf,
    const std::vector<std::string>& keep_classes) {

    std::vector<Detection> out;
    out.reserve(dets.size());
    for (const auto& d : dets) {
        if (d.confidence < min_conf) continue;
        if (!keep_classes.empty()) {
            bool ok = false;
            for (const auto& c : keep_classes) {
                if (d.class_name == c) { ok = true; break; }
            }
            if (!ok) continue;
        }
        out.push_back(d);
    }
    return out;
}

bool check_watchlist(const std::vector<Detection>& dets,
                     const std::vector<WatchlistEntry>& watchlist,
                     std::string& matched_label) {
    matched_label.clear();
    for (const auto& d : dets) {
        for (const auto& w : watchlist) {
            if (d.class_name == w.label && d.confidence >= w.min_confidence) {
                matched_label = d.class_name + ":" + w.label;
                return true;
            }
        }
    }
    return false;
}

Alert make_alert(int64_t frame_id,
                 TimePoint capture_ts,
                 const std::vector<Detection>& dets,
                 bool watchlist_hit,
                 const std::string& matched_label,
                 TimePoint now) {
    Alert a;
    a.frame_id       = frame_id;
    a.timestamp      = now;
    a.detections     = dets;
    a.watchlist_hit  = watchlist_hit;
    a.matched_label  = matched_label;
    a.e2e_latency_ms = DurationMs(now - capture_ts).count();
    return a;
}

}
