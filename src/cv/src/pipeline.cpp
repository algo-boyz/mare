#include "pipeline.hpp"

#include <iostream>
#include <iomanip>

namespace edge_cv {

Pipeline::Pipeline(Config cfg) : cfg_(std::move(cfg)) {
    queue_ = std::make_shared<FrameQueue>(cfg_.queue_capacity);
    detector_ = std::make_unique<Detector>(cfg_.detector);
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

        auto raw_dets = detector_->infer(frame);

        auto dets = filter_detections(raw_dets, 0.40f);

        std::string matched;
        bool hit = check_watchlist(dets, cfg_.watchlist, matched);

        auto now = Clock::now();
        Alert alert = make_alert(meta.frame_id, meta.capture_ts,
                                 dets, hit, matched, now);

        if (cfg_.print_latency) {
            double infer_ms = detector_->last_infer_ms();
            std::cout << std::fixed << std::setprecision(1)
                      << "[Frame " << std::setw(5) << alert.frame_id << "] "
                      << "e2e=" << std::setw(6) << alert.e2e_latency_ms << " ms  "
                      << "infer=" << std::setw(5) << infer_ms << " ms  "
                      << "dets=" << dets.size();
            if (hit) {
                std::cout << "  HIT: " << matched << " **";
            }
            std::cout << "\n";

            for (const auto& d : dets) {
                std::cout << "    → " << d.class_name
                          << " " << std::setprecision(2) << d.confidence
                          << "  [" << static_cast<int>(d.box.x1) << ","
                          << static_cast<int>(d.box.y1) << " - "
                          << static_cast<int>(d.box.x2) << ","
                          << static_cast<int>(d.box.y2) << "]\n";
            }
        }

        if (cfg_.on_alert) {
            cfg_.on_alert(alert);
        }
    }
    running_ = false;
}

}
