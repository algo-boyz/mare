#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace edge_cv {

// thread-safe bounded frame queue 
// keeps latest frame when full (drop-old) to min latency
class FrameQueue {
public:
    explicit FrameQueue(size_t capacity = 2);

    // non-blocking push
    void push(cv::Mat frame, FrameMeta meta);

    // blocking pop with timeout
    std::optional<std::pair<cv::Mat, FrameMeta>> pop(std::chrono::milliseconds timeout);

    void stop();
    bool stopped() const;

private:
    size_t cap_;
    std::queue<std::pair<cv::Mat, FrameMeta>> q_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_{false};
};

// capture thread reads VideoCapture src & pushes onto FrameQueue
class Capture {
public:
    Capture(std::string source, std::shared_ptr<FrameQueue> queue,
            int target_width = 0, int target_height = 0);

    ~Capture();

    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;

    void start();
    void stop();

    bool is_running() const { return running_.load(); }

    // last known frame size after any resize
    int width()  const { return width_; }
    int height() const { return height_; }

private:
    void capture_loop();

    std::string source_;
    std::shared_ptr<FrameQueue> queue_;
    int target_width_;
    int target_height_;

    cv::VideoCapture cap_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    int width_{0};
    int height_{0};
    int64_t frame_counter_{0};
};

}
