#pragma once

#include "types.hpp"

namespace edge_cv {
namespace geom {

// intersection over union
inline float intersect(const BoundingBox& a, const BoundingBox& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float uni = a.area() + b.area() - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

}
}