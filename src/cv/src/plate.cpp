#include "plate.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>

namespace edge_cv {

namespace {

bool is_vehicle(const std::string& name) {
    return name == "car" || name == "truck" || name == "bus" ||
           name == "motorcycle" || name == "vehicle";
}

// Very lightweight "OCR": look for high-contrast rectangular regions and
// invent a plausible plate string based on crop hash. This keeps the
// cascade wiring live until a real LPRNet / fast-plate-ocr ONNX is dropped in.
std::string mock_plate_from_crop(const cv::Mat& crop) {
    if (crop.empty()) return {};
    // Simple perceptual hash of mean intensity + size
    cv::Scalar m = cv::mean(crop);
    uint32_t h = static_cast<uint32_t>(m[0] * 7 + m[1] * 13 + m[2] * 19);
    h ^= static_cast<uint32_t>(crop.cols * 31u + crop.rows * 17u);
    // Produce something that looks like a US plate
    static const char alpha[] = "ABCDEFGHJKLMNPRSTUVWXYZ";
    static const char digit[] = "0123456789";
    std::string s;
    s += alpha[(h >> 0)  % 23];
    s += alpha[(h >> 5)  % 23];
    s += digit[(h >> 10) % 10];
    s += digit[(h >> 14) % 10];
    s += alpha[(h >> 18) % 23];
    s += alpha[(h >> 23) % 23];
    return s;
}

} // namespace

PlateStage::PlateStage(const Config& cfg) : cfg_(cfg) {
    if (!cfg_.plate_detector_model.empty()) {
        Detector::Config dcfg;
        dcfg.model_path   = cfg_.plate_detector_model;
        dcfg.conf_thresh  = cfg_.min_plate_conf;
        dcfg.use_cuda     = true;   // will fall back inside Detector
        dcfg.input_width  = 640;
        dcfg.input_height = 640;
        try {
            plate_det_ = std::make_unique<Detector>(dcfg);
            std::cout << "[PlateStage] Loaded dedicated plate detector: "
                      << cfg_.plate_detector_model << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[PlateStage] Failed to load plate detector, "
                      << "falling back to heuristic ROI: " << e.what() << "\n";
            plate_det_.reset();
        }
    } else {
        std::cout << "[PlateStage] No plate detector model – using heuristic ROI\n";
    }
    if (cfg_.plate_ocr_model.empty()) {
        std::cout << "[PlateStage] No OCR model – using mock plate strings "
                  << "(swap in real LPRNet ONNX later)\n";
    }
}

BoundingBox PlateStage::heuristic_plate_roi(const BoundingBox& v) {
    // Typical rear plate sits in lower 25-40% of vehicle box, centered.
    float w = v.width();
    float h = v.height();
    float px = v.x1 + w * 0.25f;
    float py = v.y1 + h * 0.62f;
    float pw = w * 0.50f;
    float ph = h * 0.18f;
    return {px, py, px + pw, py + ph};
}

std::pair<std::string, float> PlateStage::run_ocr(const cv::Mat& plate_crop) {
    if (plate_crop.empty() || plate_crop.cols < 20 || plate_crop.rows < 8) {
        return {"", 0.f};
    }
    // TODO: replace with real ONNX LPRNet / fast-plate-ocr inference
    // For now emit a deterministic mock so the rest of the pipeline
    // (OSD, alerts, ClickHouse, watchlist) can be exercised end-to-end.
    std::string text = mock_plate_from_crop(plate_crop);
    float conf = 0.82f + 0.15f * (static_cast<float>(plate_crop.cols % 10) / 10.f);
    return {text, std::min(conf, 0.99f)};
}

void PlateStage::process(const cv::Mat& bgr_frame, std::vector<Detection>& dets) {
    if (!cfg_.enabled || bgr_frame.empty()) return;

    auto t0 = Clock::now();

    // Collect vehicle indices (limit work)
    std::vector<size_t> vehicles;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (is_vehicle(dets[i].class_name) &&
            dets[i].confidence >= cfg_.min_vehicle_conf) {
            vehicles.push_back(i);
        }
    }
    if (vehicles.size() > static_cast<size_t>(cfg_.max_vehicles)) {
        // Keep highest confidence
        std::partial_sort(vehicles.begin(),
                          vehicles.begin() + cfg_.max_vehicles,
                          vehicles.end(),
                          [&](size_t a, size_t b) {
                              return dets[a].confidence > dets[b].confidence;
                          });
        vehicles.resize(cfg_.max_vehicles);
    }

    std::vector<Detection> extra_plates;

    for (size_t vi : vehicles) {
        auto& veh = dets[vi];
        BoundingBox plate_box;

        if (plate_det_) {
            // Crop vehicle with padding and run secondary detector
            int x1 = std::max(0, static_cast<int>(veh.box.x1) - 8);
            int y1 = std::max(0, static_cast<int>(veh.box.y1) - 8);
            int x2 = std::min(bgr_frame.cols, static_cast<int>(veh.box.x2) + 8);
            int y2 = std::min(bgr_frame.rows, static_cast<int>(veh.box.y2) + 8);
            if (x2 <= x1 || y2 <= y1) continue;

            cv::Mat crop = bgr_frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
            auto plate_dets = plate_det_->infer(crop);

            // Take highest-conf plate-like detection
            float best = 0.f;
            BoundingBox best_box;
            for (const auto& pd : plate_dets) {
                // Accept any class or specifically "license_plate" if model has it
                if (pd.confidence > best) {
                    best = pd.confidence;
                    best_box = pd.box;
                    // remap to full-frame coords
                    best_box.x1 += x1;
                    best_box.y1 += y1;
                    best_box.x2 += x1;
                    best_box.y2 += y1;
                }
            }
            if (best < cfg_.min_plate_conf) {
                plate_box = heuristic_plate_roi(veh.box);
            } else {
                plate_box = best_box;
            }
        } else {
            plate_box = heuristic_plate_roi(veh.box);
        }

        // Crop plate ROI (clamp)
        int px1 = std::max(0, static_cast<int>(plate_box.x1));
        int py1 = std::max(0, static_cast<int>(plate_box.y1));
        int px2 = std::min(bgr_frame.cols, static_cast<int>(plate_box.x2));
        int py2 = std::min(bgr_frame.rows, static_cast<int>(plate_box.y2));
        if (px2 - px1 < 16 || py2 - py1 < 8) continue;

        cv::Mat plate_crop = bgr_frame(cv::Rect(px1, py1, px2 - px1, py2 - py1)).clone();

        auto [text, conf] = run_ocr(plate_crop);
        if (text.empty()) continue;

        // Attach to the vehicle detection
        veh.ocr_text       = text;
        veh.ocr_confidence = conf;

        // Also emit a dedicated license_plate detection for OSD / alerts
        Detection pd;
        pd.class_id       = 1000;               // synthetic
        pd.class_name     = "license_plate";
        pd.confidence     = conf;
        pd.box            = {static_cast<float>(px1), static_cast<float>(py1),
                             static_cast<float>(px2), static_cast<float>(py2)};
        pd.track_id       = veh.track_id;       // inherit parent track
        pd.ocr_text     = text;
        pd.ocr_confidence = conf;
        extra_plates.push_back(std::move(pd));
    }

    // Append plate detections
    dets.insert(dets.end(),
                std::make_move_iterator(extra_plates.begin()),
                std::make_move_iterator(extra_plates.end()));

    auto t1 = Clock::now();
    last_ms_ = DurationMs(t1 - t0).count();
}

} // namespace edge_cv
