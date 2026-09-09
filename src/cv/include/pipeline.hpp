#pragma once

#include "capture.hpp"
#include "detector.hpp"
#include "postprocess.hpp"
#include "tracker.hpp"
#include "plate.hpp"
#include "types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace edge_cv {

// e2e pipeline: capture thread > FrameQueue > processing thread > alert callback
// process: infer > filter > track > gated OCR stage > watchlist > alert
//
// Secondary stages (plate, future person attributes, etc.) are gated:
// they only consume cycles when the prerequisite class is present and
// the track has not already been satisfactorily identified.
class Pipeline {
public:
    using AlertCallback = std::function<void(const Alert&)>;

    struct Config {
        std::string              source{"0"};          // cam idx / mp4 path
        Detector::Config         detector;
        Tracker::Config          tracker;
        PlateStage::Config       plates;
        size_t                   queue_capacity{2};
        std::vector<WatchlistEntry> watchlist;
        AlertCallback            on_alert;
        bool                     print_latency{true};
        int                      target_width{0};      // native
        int                      target_height{0};
    };

    explicit Pipeline(Config cfg);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    void start();
    void stop();

    bool is_running() const { return running_.load(); }

private:
    void process_loop();

    Config cfg_;
    std::shared_ptr<FrameQueue> queue_;
    std::unique_ptr<Capture>    capture_;
    std::unique_ptr<Detector>   detector_;
    std::unique_ptr<Tracker>    tracker_;
    std::unique_ptr<PlateStage> ocr_stage_;

    // Persistent per-track memory for gating secondary stages
    TrackStateMap track_states_;

    std::thread process_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
};

} // namespace edge_cv
