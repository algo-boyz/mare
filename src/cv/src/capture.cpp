#include "capture.hpp"

#include <iostream>
#include <chrono>
#include <opencv2/imgproc.hpp>

namespace edge_cv {

FrameQueue::FrameQueue(size_t capacity) : cap_(capacity) {}

void FrameQueue::push(cv::Mat frame, FrameMeta meta) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (stopped_) return;

    // drop-old
    while (q_.size() >= cap_) {
        q_.pop();
    }
    q_.emplace(std::move(frame), std::move(meta));
    cv_.notify_one();
}

std::optional<std::pair<cv::Mat, FrameMeta>>
FrameQueue::pop(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!cv_.wait_for(lock, timeout, [this] {
            return !q_.empty() || stopped_;
        })) {
        return std::nullopt;  // timeout
    }
    if (stopped_ && q_.empty()) return std::nullopt;

    auto item = std::move(q_.front());
    q_.pop();
    return item;
}

void FrameQueue::stop() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    cv_.notify_all();
}

bool FrameQueue::stopped() const {
    return stopped_.load();
}

Capture::Capture(std::string source, std::shared_ptr<FrameQueue> queue,
                 int target_width, int target_height)
    : source_(std::move(source)),
      queue_(std::move(queue)),
      target_width_(target_width),
      target_height_(target_height) {}

Capture::~Capture() {
    stop();
}

void Capture::start() {
    if (running_.exchange(true)) return;  // already on

    bool is_camera = false;
    try {
        int cam_idx = std::stoi(source_);
        is_camera = true;
#if defined(__linux__)
        cap_.open(cam_idx, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            cap_.open(cam_idx);  // fallback
        }
#elif defined(__APPLE__)
        cap_.open(cam_idx, cv::CAP_AVFOUNDATION);
        if (!cap_.isOpened()) {
            cap_.open(cam_idx);
        }
#else
        cap_.open(cam_idx);
#endif
    } catch (...) {
        cap_.open(source_);
    }

    if (!cap_.isOpened()) {
        running_ = false;
        throw std::runtime_error(
            "Failed to open: " + source_ + "\n");
    }

    if (is_camera) {
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        // min internal buffer
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }

    width_  = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    if (target_width_ > 0 && target_height_ > 0) {
        width_  = target_width_;
        height_ = target_height_;
    }

    stop_requested_ = false;
    thread_ = std::thread(&Capture::capture_loop, this);
}

void Capture::stop() {
    if (!running_.exchange(false)) return;
    stop_requested_ = true;
    if (thread_.joinable()) thread_.join();
    if (cap_.isOpened()) cap_.release();
}

void Capture::capture_loop() {
    cv::Mat frame;
    while (!stop_requested_) {
        if (!cap_.read(frame) || frame.empty()) {
            // end of file / cam err
            if (source_.find('.') != std::string::npos) {
                break;
            }
            // pause & retry
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        cv::Mat to_push;
        if (target_width_ > 0 && target_height_ > 0 &&
            (frame.cols != target_width_ || frame.rows != target_height_)) {
            cv::resize(frame, to_push, {target_width_, target_height_});
        } else {
            to_push = frame;
        }

        FrameMeta meta;
        meta.frame_id   = frame_counter_++;
        meta.capture_ts = Clock::now();
        meta.width      = to_push.cols;
        meta.height     = to_push.rows;

        queue_->push(std::move(to_push), meta);
    }
    queue_->stop();
    running_ = false;
}

}
