#pragma once

#include "types.hpp"

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace edge_cv {

// yolo style detector using onnx runtime.
// model input is float32 of size 1x3xHxW
class Detector {
public:
    struct Config {
        std::string model_path;
        int         input_width  = 640;
        int         input_height = 640;
        float       conf_thresh  = 0.35f;
        float       iou_thresh   = 0.45f;
        bool        use_cuda     = true;
        int         num_threads  = 4;
    };

    explicit Detector(const Config& cfg);
    ~Detector() = default;

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // inference in original image coordinates
    std::vector<Detection> infer(const cv::Mat& bgr_frame);

    // timing of last inference ms
    double last_infer_ms() const { return last_infer_ms_; }

    const std::vector<std::string>& class_names() const { return class_names_; }

private:
    void load_model();
    cv::Mat preprocess(const cv::Mat& bgr, float& scale, int& pad_x, int& pad_y);
    std::vector<Detection> postprocess(const float* output,
                                       size_t output_size,
                                       float scale, int pad_x, int pad_y,
                                       int orig_w, int orig_h);

    Config cfg_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;

    std::vector<std::string> class_names_;  // coco 80

    double last_infer_ms_{0.0};
};

}
