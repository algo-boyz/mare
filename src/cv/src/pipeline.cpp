#include "pipeline.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>

namespace edge_cv {

namespace {

bool is_vehicle(const std::string& name) {
    return name == "car" || name == "truck" || name == "bus" ||
           name == "motorcycle" || name == "vehicle";
}

bool any_of_class(const std::vector<Detection>& dets, const std::string& cls) {
    return std::any_of(dets.begin(), dets.end(),
                       [&](const Detection& d) { return d.class_name == cls; });
}

bool any_vehicle(const std::vector<Detection>& dets) {
    return std::any_of(dets.begin(), dets.end(),
                       [](const Detection& d) { return is_vehicle(d.class_name); });
}

} // namespace

Pipeline::Pipeline(Config cfg) : cfg_(std::move(cfg)) {
    queue_ = std::make_shared<FrameQueue>(cfg_.queue_capacity);
    detector_ = std::make_unique<Detector>(cfg_.detector);
    tracker_ = std::make_unique<Tracker>(cfg_.tracker);
    ocr_stage_ = std::make_unique<PlateStage>(cfg_.plates);
    capture_ = std::make_unique<Capture>(cfg_.source, queue_,
                                         cfg_.target_width, cfg_.target_height);
}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (running_.exchange(true)) return;

    stop_requested_ = false;
    capture_->start();
    process_thread_ = std::thread(&Pipeline::process_loop, this);
}

void Pipeline::stop() {
    if (!running_.exchange(false)) return;
    stop_requested_ = true;
    queue_->stop();
    capture_->stop();
    if (process_thread_.joinable()) process_thread_.join();
}

void Pipeline::process_loop() {
    using namespace std::chrono_literals;

    while (!stop_requested_) {
        auto item = queue_->pop(100ms);
        if (!item) continue;

        auto& [frame, meta] = *item;

        // 1. Primary detection (always)
        auto raw_dets = detector_->infer(frame);

        // 2. Class / confidence filter (vehicles + person)
        auto dets = filter_detections(raw_dets, 0.40f);

        // Cheap presence flags for gating
        const bool has_person  = any_of_class(dets, "person");
        const bool has_vehicle = any_vehicle(dets);

        // 3. Tracking → assign persistent track_ids
        dets = tracker_->update(std::move(dets));

        // 4. Secondary OCR stage – only when vehicles are present
        //    (PlateStage itself further gates per-track)
        if (has_vehicle) {
            ocr_stage_->process(frame, dets, track_states_);
        } else {
            // Still age track states so counters stay consistent
            for (auto& [id, ts] : track_states_) {
                ++ts.frames_since_ocr;
                ++ts.age;
            }
        }

        // Future extension point:
        // if (has_person) {
        //     person_stage_->process(frame, dets, track_states_);
        // }

        // Prune stale tracks (not seen for a while)
        // Simple: drop tracks older than ~5 seconds @ 30 fps
        constexpr int kMaxTrackAge = 150;
        for (auto it = track_states_.begin(); it != track_states_.end(); ) {
            if (it->second.age > kMaxTrackAge) {
                it = track_states_.erase(it);
            } else {
                ++it;
            }
        }

        // 5. Watchlist (class or plate text)
        std::string matched;
        bool hit = check_watchlist(dets, cfg_.watchlist, matched);

        auto now = Clock::now();
        Alert alert = make_alert(meta.frame_id, meta.capture_ts,
                                 dets, hit, matched, now);

        if (cfg_.print_latency) {
            double infer_ms = detector_->last_infer_ms();
            double plate_ms = ocr_stage_->last_ms();
            int    plate_n  = ocr_stage_->last_processed_count();

            std::cout << std::fixed << std::setprecision(1)
                      << "[Frame " << std::setw(5) << alert.frame_id << "] "
                      << "e2e=" << std::setw(6) << alert.e2e_latency_ms << " ms  "
                      << "infer=" << std::setw(5) << infer_ms << " ms  "
                      << "plate=" << std::setw(5) << plate_ms << " ms"
                      << " (n=" << plate_n << ")  "
                      << "dets=" << dets.size()
                      << " tracks=" << tracker_->num_tracks()
                      << " states=" << track_states_.size();
            if (hit) {
                std::cout << "  HIT: " << matched << " **";
            }
            std::cout << "\n";

            for (const auto& d : dets) {
                std::cout << "    → ";
                if (d.track_id >= 0) {
                    std::cout << "#" << d.track_id << " ";
                }
                std::cout << d.class_name
                          << " " << std::setprecision(2) << d.confidence
                          << "  [" << static_cast<int>(d.box.x1) << ","
                          << static_cast<int>(d.box.y1) << " - "
                          << static_cast<int>(d.box.x2) << ","
                          << static_cast<int>(d.box.y2) << "]";
                if (!d.ocr_text.empty()) {
                    std::cout << "  plate=" << d.ocr_text
                              << " (" << std::setprecision(2) << d.ocr_confidence << ")";
                    // Indicate whether this came from cache
                    auto it = track_states_.find(d.track_id);
                    if (it != track_states_.end() && it->second.plate_confirmed) {
                        std::cout << " [confirmed]";
                    }
                }
                std::cout << "\n";
            }
        }

        if (cfg_.on_alert) {
            cfg_.on_alert(alert);
        }
    }
    running_ = false;
}

} // namespace edge_cv
