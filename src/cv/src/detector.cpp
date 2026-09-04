#include "geo.hpp"
#include "detector.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace edge_cv {

namespace {

// coco 80
const std::vector<std::string> kCocoNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

}

Detector::Detector(const Config& cfg)
    : cfg_(cfg),
      env_(ORT_LOGGING_LEVEL_WARNING, "edge_cv_detector"),
      class_names_(kCocoNames) {
    load_model();
}

void Detector::load_model() {
    session_options_.SetIntraOpNumThreads(cfg_.num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (cfg_.use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_opts);
            std::cout << "[Detector] CUDA requested\n";
        } catch (const std::exception& e) {
            std::cerr << "[Detector] CUDA unavailable, fallback to CPU: "
                      << e.what() << "\n";
        }
    }

    session_ = std::make_unique<Ort::Session>(env_, cfg_.model_path.c_str(), session_options_);

    // Input / output names
    size_t num_inputs = session_->GetInputCount();
    size_t num_outputs = session_->GetOutputCount();

    input_names_.resize(num_inputs);
    output_names_.resize(num_outputs);
    input_names_cstr_.resize(num_inputs);
    output_names_cstr_.resize(num_outputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_[i] = name.get();
        input_names_cstr_[i] = input_names_[i].c_str();
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_[i] = name.get();
        output_names_cstr_[i] = output_names_[i].c_str();
    }

    std::cout << "[Detector] Model loaded: " << cfg_.model_path
              << "  inputs=" << num_inputs << " outputs=" << num_outputs << "\n";
}

cv::Mat Detector::preprocess(const cv::Mat& bgr, float& scale, int& pad_x, int& pad_y) {
    // letterbox resize to width x height
    const int w = bgr.cols;
    const int h = bgr.rows;
    scale = std::min(static_cast<float>(cfg_.input_width) / w,
                     static_cast<float>(cfg_.input_height) / h);
    const int new_w = static_cast<int>(std::round(w * scale));
    const int new_h = static_cast<int>(std::round(h * scale));
    pad_x = (cfg_.input_width  - new_w) / 2;
    pad_y = (cfg_.input_height - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, {new_w, new_h});

    cv::Mat padded(cfg_.input_height, cfg_.input_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // BGR > RGB, HWC > CHW, /255  (1×3×H×W float32 buffer)
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    const int H = cfg_.input_height;
    const int W = cfg_.input_width;
    cv::Mat chw(1, 3 * H * W, CV_32FC1);   // single block
    float* dst = chw.ptr<float>();

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    for (int c = 0; c < 3; ++c) {
        std::memcpy(dst + c * H * W, channels[c].ptr<float>(), H * W * sizeof(float));
    }
    return chw;
}

std::vector<Detection> Detector::postprocess(const float* output,
                                             size_t /*output_size*/,
                                             float scale, int pad_x, int pad_y,
                                             int orig_w, int orig_h) {
    // yolo out [1, 84, 8400] transposed or [1, 8400, 84]
    const int num_classes = 80;
    const int num_proposals = 8400;

    std::vector<Detection> candidates;
    candidates.reserve(256);

    for (int i = 0; i < num_proposals; ++i) {
        // layout
        float cx = output[0 * num_proposals + i];
        float cy = output[1 * num_proposals + i];
        float w  = output[2 * num_proposals + i];
        float h  = output[3 * num_proposals + i];

        // match best class
        float best_score = 0.f;
        int   best_cls   = -1;
        for (int c = 0; c < num_classes; ++c) {
            float s = output[(4 + c) * num_proposals + i];
            if (s > best_score) {
                best_score = s;
                best_cls   = c;
            }
        }
        if (best_score < cfg_.conf_thresh) continue;

        // convert to xyxy letterbox space then undo letterbox
        float x1 = (cx - w * 0.5f - pad_x) / scale;
        float y1 = (cy - h * 0.5f - pad_y) / scale;
        float x2 = (cx + w * 0.5f - pad_x) / scale;
        float y2 = (cy + h * 0.5f - pad_y) / scale;

        // clip
        x1 = std::clamp(x1, 0.f, static_cast<float>(orig_w - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(orig_h - 1));
        x2 = std::clamp(x2, 0.f, static_cast<float>(orig_w - 1));
        y2 = std::clamp(y2, 0.f, static_cast<float>(orig_h - 1));

        Detection d;
        d.class_id   = best_cls;
        d.class_name = (best_cls >= 0 && best_cls < static_cast<int>(class_names_.size()))
                           ? class_names_[best_cls]
                           : "unknown";
        d.confidence = best_score;
        d.box        = {x1, y1, x2, y2};
        candidates.push_back(std::move(d));
    }

    // sort by confidence desc then NMS
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            if (candidates[i].class_id != candidates[j].class_id) continue;
            if (geom::intersect(candidates[i].box, candidates[j].box) > cfg_.iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

std::vector<Detection> Detector::infer(const cv::Mat& bgr_frame) {
    if (bgr_frame.empty()) return {};

    auto t0 = Clock::now();

    float scale = 1.f;
    int pad_x = 0;
    int pad_y = 0;
    cv::Mat input_chw = preprocess(bgr_frame, scale, pad_x, pad_y);

    // ort tensor (zero-copy from ocv data)
    std::vector<int64_t> input_shape = {1, 3, cfg_.input_height, cfg_.input_width};
    size_t input_tensor_size = 1 * 3 * cfg_.input_height * cfg_.input_width;

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        reinterpret_cast<float*>(input_chw.data),
        input_tensor_size,
        input_shape.data(),
        input_shape.size());

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names_cstr_.data(), &input_tensor, 1,
                                 output_names_cstr_.data(), output_names_cstr_.size());

    const float* out_data = outputs[0].GetTensorData<float>();

    auto result = postprocess(out_data, 0, scale, pad_x, pad_y,
                              bgr_frame.cols, bgr_frame.rows);

    auto t1 = Clock::now();
    last_infer_ms_ = DurationMs(t1 - t0).count();
    return result;
}

}
