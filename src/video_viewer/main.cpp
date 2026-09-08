#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include "common/env.hpp"
#include "detection/v1/annotated_video.grpc.pb.h"

// Minimal header-only HTTP server (we'll fetch it in CMake)
#include "httplib.h"

namespace {

std::mutex g_frame_mu;
std::vector<uchar> g_latest_jpeg;
std::atomic<bool> g_running{true};
std::atomic<int64_t> g_frame_id{-1};
std::atomic<int> g_det_count{0};

void grpc_pull_loop(const std::string& server, const std::string& source) {
  auto channel = grpc::CreateChannel(server, grpc::InsecureChannelCredentials());
  auto stub = detection::v1::AnnotatedVideoService::NewStub(channel);

  detection::v1::StreamAnnotatedVideoRequest req;
  req.set_source(source);
  req.set_only_with_detections(false);
  req.set_jpeg_quality(80);

  while (g_running) {
    grpc::ClientContext ctx;
    auto reader = stub->StreamAnnotatedVideo(&ctx, req);

    detection::v1::AnnotatedFrame frame;
    while (g_running && reader->Read(&frame)) {
      std::vector<uchar> jpeg(frame.jpeg().begin(), frame.jpeg().end());
      {
        std::lock_guard lock(g_frame_mu);
        g_latest_jpeg = std::move(jpeg);
      }
      g_frame_id = frame.frame_id();
      g_det_count = frame.detection_count();
    }

    auto status = reader->Finish();
    if (!status.ok()) {
      spdlog::warn("gRPC stream ended: {} – reconnecting in 2s", status.error_message());
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string grpc_addr = edge::getenv_or("VIDEO_SERVER_ADDR", "video_server:50053");
  const std::string source    = (argc > 1) ? argv[1] : edge::getenv_or("VIDEO_SOURCE", "sample.mp4");
  const int http_port         = std::stoi(edge::getenv_or("HTTP_PORT", "8080"));

  spdlog::info("video_viewer starting – gRPC={} source={} http=:{}", grpc_addr, source, http_port);

  std::thread puller(grpc_pull_loop, grpc_addr, source);

  httplib::Server svr;

  // Live MJPEG stream
  svr.Get("/stream", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Cache-Control", "no-cache, no-store, must-revalidate");
    res.set_header("Pragma", "no-cache");
    res.set_header("Connection", "close");
    res.set_content_provider(
        "multipart/x-mixed-replace; boundary=frame",
        [](size_t /*offset*/, httplib::DataSink& sink) {
          while (g_running) {
            std::vector<uchar> jpeg;
            {
              std::lock_guard lock(g_frame_mu);
              if (g_latest_jpeg.empty()) {
                // wait a bit for first frame
              } else {
                jpeg = g_latest_jpeg;
              }
            }

            if (!jpeg.empty()) {
              std::string header =
                  "--frame\r\n"
                  "Content-Type: image/jpeg\r\n"
                  "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";

              if (!sink.write(header.data(), header.size())) return false;
              if (!sink.write(reinterpret_cast<const char*>(jpeg.data()), jpeg.size())) return false;
              if (!sink.write("\r\n", 2)) return false;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 fps max
          }
          return true;
        });
  });

  // Simple status page
  svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
    std::string html = R"(
<!DOCTYPE html>
<html><head><title>MARE Annotated Stream</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto}</style></head>
<body>
  <h2>MARE Annotated Video</h2>
  <p>frame <span id="fid">-</span> · detections <span id="dets">-</span></p>
  <img src="/stream" alt="stream">
  <script>
    setInterval(async () => {
      try {
        const r = await fetch('/status');
        const j = await r.json();
        document.getElementById('fid').textContent = j.frame_id;
        document.getElementById('dets').textContent = j.detections;
      } catch(e){}
    }, 500);
  </script>
</body></html>)";
    res.set_content(html, "text/html");
  });

  svr.Get("/status", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(
        "{\"frame_id\":" + std::to_string(g_frame_id.load()) +
        ",\"detections\":" + std::to_string(g_det_count.load()) + "}",
        "application/json");
  });

  spdlog::info("MJPEG endpoint ready → http://0.0.0.0:{}/stream", http_port);
  svr.listen("0.0.0.0", http_port);

  g_running = false;
  if (puller.joinable()) puller.join();
  return 0;
}