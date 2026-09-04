#pragma once

#include "types.hpp"

#include <vector>
#include <string>

namespace edge_cv {

// non max suppression aka greedy detection
// operates in-place on sorted (by conf desc) list
std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh);

// filter to class of interest
std::vector<Detection> filter_detections(
    const std::vector<Detection>& dets,
    float min_conf = 0.4f,
    const std::vector<std::string>& keep_classes = {"person", "car", "truck", "bus", "motorcycle"});

// mock correlation returning label if match found
bool check_watchlist(const std::vector<Detection>& dets,
                     const std::vector<WatchlistEntry>& watchlist,
                     std::string& matched_label);

Alert make_alert(int64_t frame_id,
                 TimePoint capture_ts,
                 const std::vector<Detection>& dets,
                 bool watchlist_hit,
                 const std::string& matched_label,
                 TimePoint now);

}