#pragma once

#include "types.hpp"
#include "detector.hpp"

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace edge_cv {

// Secondary stage: given vehicle detections, find plates and run OCR.
// Designed so the plate detector can be a second YOLO / RT-DETR ONNX
// and the OCR can be a lightweight sequence model (or placeholder).
//
// For Phase 2:
//  - optional dedicated plate detector ONNX (falls back to heuristic ROI)
//  - simple OCR placeholder that can be swapped for a real ONNX LPRNet /
//    fast-plate-ocr model later.
class PlateStage {
public:
    struct Config {
        std::string plate_detector_model;   // empty = heuristic ROI on vehicle
        std::string plate_ocr_model;        // empty = mock / placeholder OCR
        float       min_vehicle_conf  = 0.45f;
        float       min_plate_conf    = 0.30f;
        int         max_vehicles     = 8;   // limit secondary load
        bool        enabled          = true;
    };

    explicit PlateStage(const Config& cfg);
    ~PlateStage() = default;

    // Runs on the full frame + current detections.
    // Mutates vehicle detections in-place: fills ocr_text / ocr_confidence
    // and may append extra "license_plate" detections.
    void process(const cv::Mat& bgr_frame, std::vector<Detection>& dets);

    double last_ms() const { return last_ms_; }

private:
    Config cfg_;
    std::unique_ptr<Detector> plate_det_;   // optional secondary detector
    double last_ms_{0.0};

    // Heuristic: bottom-center region of a vehicle box is a good plate prior
    static BoundingBox heuristic_plate_roi(const BoundingBox& vehicle);

    // Placeholder OCR – returns a fake plate string for demo / wiring.
    // Replace with real ONNX LPRNet / CRNN call later.
    std::pair<std::string, float> run_ocr(const cv::Mat& plate_crop);
};

} // namespace edge_cv
