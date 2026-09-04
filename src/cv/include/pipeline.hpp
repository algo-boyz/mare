#pragma once

#include "capture.hpp"
#include "detector.hpp"
#include "postprocess.hpp"
#include "types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace edge_cv {

// e2e pipeline: capture thread > FrameQueue > processing thread > alert callback
// pop frame > infer > filter > watchlist > alert > log
// processing thread owns detector
class Pipeline {
public:
    using AlertCallback = std::function<void(const Alert&)>;

    struct Config {
        std::string              source{"0"};          // cam idx / mp4 path
        Detector::Config         detector;
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

    std::thread process_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
};

}
