#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <filesystem>
#include <vector>
#include <deque>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace fs = std::filesystem;

//std::string folder_path = "INSERT_FOLDER_PATH";
//std::string model_path = "INSERT_MODEL_PATH";

std::string folder_path = "C:/Users/johnn/Downloads/testing_footage/";
std::string model_path = "C:/src/lane_tracker/lane_tracker/models/ultra-fast-lane-det-culane.onnx";

// ---------------------------------------------------------------------------
// Top-of-file tuning: hood/dashboard exclusion
// ---------------------------------------------------------------------------
// How many pixels UP FROM THE BOTTOM of the frame count as hood/dashboard, not road.
// Points in that band are dropped before drawing OR classification. Absolute pixel
// count, not a fraction of frame height - recheck this if your footage resolution changes.
int HOOD_EXCLUSION_HEIGHT_PX = 220;

// Whether the hood exclusion line is drawn on screen at all.
bool SHOW_HOOD_EXCLUSION_LINE = true; // temporarily on - diagnosing far-lane edge-curving (6.1)

// ---------------------------------------------------------------------------
// Top-of-file tuning: paint detection
// ---------------------------------------------------------------------------
// How much brighter the line has to be than the asphalt beside it to count as "paint".
int PAINT_CONTRAST_THRESHOLD = 25; //40

// "-" search band: fixed HORIZONTAL span around each point (half-width and probe spacing).
int PAINT_SEARCH_HALF_WIDTH_PX = 10;
int PAINT_SEARCH_STEP_PX = 4;

// ---------------------------------------------------------------------------
// Top-of-file tuning: solid/dashed classification
// ---------------------------------------------------------------------------
// Show the full "SOLID  n=8  paint=84%  trans=3" breakdown instead of just "SOLID" - flip
// this on while tuning the thresholds below so you can see the actual numbers driving
// each call, flip it off for a clean display otherwise.
bool SHOW_LINE_TYPE_DIAGNOSTICS = true; // temporarily on - diagnosing far-lane edge-curving (6.1)

// Temporary: logs argmax bin vs softmax-expectation bin for the far-left/far-right lane
// channels at the horizon-side row anchors, to test whether tracked-point edge-curving
// comes from truncation bias in the expected-value decode. Remove once 6.1 is resolved.
bool DEBUG_LOG_DECODER_BIAS = true;

// A lane is called SOLID if painted_fraction is above this AND transitions is at or below this.
float SOLID_MIN_PAINTED_FRACTION = 0.70f;
int   SOLID_MAX_TRANSITIONS = 8; // effectively "don't care" right now - see comment at the check itself

// A lane is called DASHED if transitions is at or above this AND painted_fraction falls in this range.
int   DASHED_MIN_TRANSITIONS = 0; // effectively "don't care" right now - see comment at the check itself
float DASHED_MIN_PAINTED_FRACTION = 0.15f;
float DASHED_MAX_PAINTED_FRACTION = 0.75f;

// Pre-processes frame: Applies CULane aspect ratio crop, resizes, and normalizes
int preProcessFrame(const cv::Mat& src_frame, std::vector<float>& input_tensor_values, int target_w, int target_h) {
    int raw_w = src_frame.cols;
    int raw_h = src_frame.rows;

    int crop_h = static_cast<int>(raw_w * (static_cast<float>(target_h) / static_cast<float>(target_w)));
    int crop_y = std::max(0, raw_h - crop_h);
    cv::Rect roi(0, crop_y, raw_w, crop_h);
    cv::Mat cropped_frame = src_frame(roi);

    cv::Mat rgb_frame, resized_frame;
    cv::cvtColor(cropped_frame, rgb_frame, cv::COLOR_BGR2RGB);
    cv::resize(rgb_frame, resized_frame, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);

    input_tensor_values.resize(1 * 3 * target_h * target_w);

    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std_dev[3] = { 0.229f, 0.224f, 0.225f };
    int image_area = target_h * target_w;

    for (int i = 0; i < target_h; ++i) {
        const cv::Vec3b* row_ptr = resized_frame.ptr<cv::Vec3b>(i);
        for (int j = 0; j < target_w; ++j) {
            int pixel_index = i * target_w + j;
            input_tensor_values[0 * image_area + pixel_index] = ((row_ptr[j][0] / 255.0f) - mean[0]) / std_dev[0];
            input_tensor_values[1 * image_area + pixel_index] = ((row_ptr[j][1] / 255.0f) - mean[1]) / std_dev[1];
            input_tensor_values[2 * image_area + pixel_index] = ((row_ptr[j][2] / 255.0f) - mean[2]) / std_dev[2];
        }
    }

    return crop_y;
}

// ---------------------------------------------------------------------------
// Cross-lane outlier rejection (load-bearing for line-type classification, not just
// cosmetic: a single point that snapped onto the next lane over will corrupt a
// paint/gap read for every segment touching it).
// ---------------------------------------------------------------------------

struct FittedLane {
    bool valid = false;
    double A = 0, B = 0, C = 0; // x = A*y^2 + B*y + C
    double predict(double y) const { return A * y * y + B * y + C; }
};

FittedLane fitQuadratic(const std::vector<cv::Point>& pts) {
    FittedLane result;
    int n = static_cast<int>(pts.size());
    if (n < 4) return result;

    cv::Mat M(n, 3, CV_64F);
    cv::Mat X(n, 1, CV_64F);
    for (int i = 0; i < n; ++i) {
        double y = static_cast<double>(pts[i].y);
        M.at<double>(i, 0) = y * y;
        M.at<double>(i, 1) = y;
        M.at<double>(i, 2) = 1.0;
        X.at<double>(i, 0) = static_cast<double>(pts[i].x);
    }

    cv::Mat coeffs;
    if (!cv::solve(M, X, coeffs, cv::DECOMP_SVD)) return result;

    result.A = coeffs.at<double>(0);
    result.B = coeffs.at<double>(1);
    result.C = coeffs.at<double>(2);
    result.valid = true;
    return result;
}

std::vector<cv::Point> rejectCrossLaneOutliers(const std::vector<cv::Point>& pts,
    float min_reject_px, float mad_multiplier, FittedLane* out_fit) {

    if (pts.size() < 5) {
        if (out_fit) *out_fit = fitQuadratic(pts);
        return pts; // too few points to tell a real trend from an outlier
    }

    FittedLane fit = fitQuadratic(pts);
    if (!fit.valid) {
        if (out_fit) *out_fit = fit;
        return pts;
    }

    std::vector<double> residuals(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        residuals[i] = std::abs(pts[i].x - fit.predict(pts[i].y));
    }

    std::vector<double> sorted_res = residuals;
    std::sort(sorted_res.begin(), sorted_res.end());
    double median_res = sorted_res[sorted_res.size() / 2];
    double threshold = std::max(static_cast<double>(min_reject_px), mad_multiplier * median_res);

    std::vector<cv::Point> inliers;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (residuals[i] <= threshold) inliers.push_back(pts[i]);
    }

    FittedLane refit = fitQuadratic(inliers);
    if (out_fit) *out_fit = refit.valid ? refit : fit;

    return inliers;
}

// ---------------------------------------------------------------------------
// Line-type classification (solid vs dashed)
// ---------------------------------------------------------------------------

enum class LineType { UNKNOWN, SOLID, DASHED };

const char* lineTypeToString(LineType t) {
    switch (t) {
    case LineType::SOLID:  return "SOLID";
    case LineType::DASHED: return "DASHED";
    default:                return "?";
    }
}

// PAINT_CONTRAST_THRESHOLD, PAINT_SEARCH_HALF_WIDTH_PX, PAINT_SEARCH_STEP_PX are declared
// at the top of the file for easy tuning.

// Horizontal, not tilted to the local curve direction - matches the row-anchor convention
// (each anchor is already a horizontal slice of the road). Reports the exact span it
// scanned via out_seg_a/out_seg_b for the debug overlay.
bool isPaintSample(const cv::Mat& gray, cv::Point2f pt, cv::Point2f* out_seg_a, cv::Point2f* out_seg_b) {
    int x = static_cast<int>(std::round(pt.x));
    int y = static_cast<int>(std::round(pt.y));
    if (x < 2 || y < 2 || x >= gray.cols - 2 || y >= gray.rows - 2) return false;

    if (out_seg_a) *out_seg_a = cv::Point2f(pt.x - PAINT_SEARCH_HALF_WIDTH_PX, pt.y);
    if (out_seg_b) *out_seg_b = cv::Point2f(pt.x + PAINT_SEARCH_HALF_WIDTH_PX, pt.y);

    auto sampleMaxBrightness = [&](cv::Point2f center, int half_win) {
        int cx = static_cast<int>(std::round(center.x));
        int cy = static_cast<int>(std::round(center.y));
        int max_val = 0;
        for (int dy = -half_win; dy <= half_win; ++dy) {
            for (int dx = -half_win; dx <= half_win; ++dx) {
                int sx = cx + dx, sy = cy + dy;
                if (sx < 0 || sy < 0 || sx >= gray.cols || sy >= gray.rows) continue;
                max_val = std::max(max_val, static_cast<int>(gray.at<uchar>(sy, sx)));
            }
        }
        return max_val;
        };

    int line_brightness = 0;
    for (int off = -PAINT_SEARCH_HALF_WIDTH_PX; off <= PAINT_SEARCH_HALF_WIDTH_PX; off += PAINT_SEARCH_STEP_PX) {
        line_brightness = std::max(line_brightness, sampleMaxBrightness(cv::Point2f(pt.x + off, pt.y), 3));
    }

    // MIN, not average, of the two side reads. The far-left lane's left-side probe is the
    // one most likely to land off the actual road surface (shoulder, guardrail, barrier -
    // all commonly brighter than asphalt). Averaging let a single contaminated bright side
    // drag the reference up, which shrinks measured contrast and reads as "no paint" even
    // directly on real paint. Taking the min means a clean reading on either side is
    // trusted, instead of a bad one on one side poisoning the whole reference.
    int bg_offset = PAINT_SEARCH_HALF_WIDTH_PX + 30;
    int bg_brightness = std::min(
        sampleMaxBrightness(cv::Point2f(pt.x + bg_offset, pt.y), 5),
        sampleMaxBrightness(cv::Point2f(pt.x - bg_offset, pt.y), 5)
    );

    return (line_brightness - bg_brightness) > PAINT_CONTRAST_THRESHOLD;
}

struct LineTypeResult {
    LineType type = LineType::UNKNOWN;
    int n_points = 0;
    int n_samples = 0;
    float painted_fraction = 0.0f;
    int transitions = 0;
};

// Walks a point chain (either this frame's real detections, or a memory-coasted synthetic
// chain - see main()), sampling paint/gap every few pixels, and classifies from the
// resulting on/off pattern.
LineTypeResult classifyLineType(const cv::Mat& gray, std::vector<cv::Point> pts,
    std::vector<std::tuple<cv::Point2f, cv::Point2f, bool>>* debug_samples) {

    LineTypeResult result;
    result.n_points = static_cast<int>(pts.size());
    if (pts.size() < 3) return result;

    std::sort(pts.begin(), pts.end(), [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });

    std::vector<bool> samples;
    const float STEP = 3.0f;

    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        cv::Point2f p0(pts[i].x, pts[i].y);
        cv::Point2f p1(pts[i + 1].x, pts[i + 1].y);
        float seg_len = static_cast<float>(cv::norm(p1 - p0));
        if (seg_len < 1.0f) continue;

        int n_steps = std::max(1, static_cast<int>(seg_len / STEP));

        for (int s = 0; s < n_steps; ++s) {
            float t = static_cast<float>(s) / static_cast<float>(n_steps);
            cv::Point2f sample_pt(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t);
            cv::Point2f seg_a, seg_b;
            bool is_paint = isPaintSample(gray, sample_pt, &seg_a, &seg_b);
            samples.push_back(is_paint);
            if (debug_samples) debug_samples->push_back({ seg_a, seg_b, is_paint });
        }
    }

    result.n_samples = static_cast<int>(samples.size());
    if (samples.size() < 10) return result; // not enough to say anything - stays UNKNOWN

    int transitions = 0;
    int painted_count = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (samples[i]) painted_count++;
        if (i > 0 && samples[i] != samples[i - 1]) transitions++;
    }
    result.transitions = transitions;
    result.painted_fraction = static_cast<float>(painted_count) / static_cast<float>(samples.size());

    if (result.painted_fraction > SOLID_MIN_PAINTED_FRACTION && transitions <= SOLID_MAX_TRANSITIONS) { //bro icl transitions dont matter at all, sometimes the dashed lines have less transitons cuz its all red lmao
        result.type = LineType::SOLID;
    }
    else if (transitions >= DASHED_MIN_TRANSITIONS && result.painted_fraction > DASHED_MIN_PAINTED_FRACTION && result.painted_fraction < DASHED_MAX_PAINTED_FRACTION) { //once again, sometimes the dashes have less transitions
        result.type = LineType::DASHED;
    }
    return result;
}

// Rolling majority vote so one bad frame (shadow, occlusion, transient misdetection)
// doesn't flip the displayed classification.
struct LineTypeHistory {
    std::deque<LineType> recent;
    static const size_t WINDOW = 15;

    void push(LineType t) {
        if (t == LineType::UNKNOWN) return;
        recent.push_back(t);
        if (recent.size() > WINDOW) recent.pop_front();
    }

    LineType stable() const {
        if (recent.empty()) return LineType::UNKNOWN;
        int solid_votes = static_cast<int>(std::count(recent.begin(), recent.end(), LineType::SOLID));
        int dashed_votes = static_cast<int>(std::count(recent.begin(), recent.end(), LineType::DASHED));
        if (solid_votes == 0 && dashed_votes == 0) return LineType::UNKNOWN;
        return (solid_votes >= dashed_votes) ? LineType::SOLID : LineType::DASHED;
    }
};

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LaneTrackerEnv");
    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(11);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading ONNX model..." << std::endl;
    std::wstring model_path_w(model_path.begin(), model_path.end());

    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(env, model_path_w.c_str(), session_options);
        std::cout << "\xE2\x9C\x94 ONNX Runtime Session running on 11 cores!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: Failed to load ONNX model. " << e.what() << std::endl;
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session->GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();
    Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();

    int model_input_h = static_cast<int>(input_dims[2]);
    int model_input_w = static_cast<int>(input_dims[3]);

    auto output_name_ptr = session->GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();

    // Temporary (6.1 diagnosis, plan prerequisite): the decode loop below hardcodes
    // num_bins=201/num_anchors=18 - verify those against what the loaded model actually
    // reports before trusting any bias measurement taken from that decode loop.
    {
        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        std::cout << "[decbias] actual output shape: [";
        for (size_t i = 0; i < output_dims.size(); ++i) {
            std::cout << output_dims[i] << (i + 1 < output_dims.size() ? "," : "");
        }
        std::cout << "] (code assumes [1,201,18,4])" << std::endl;
    }

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    const std::vector<int> culane_row_anchors = {
        121, 131, 141, 150, 160, 170, 180, 189, 199,
        209, 219, 228, 238, 248, 258, 267, 277, 287
    };

    const int num_lanes = 4;

    // --- Confidence gate (per-row-anchor detection quality) ---
    const float MAX_BACKGROUND_PROB = 0.40f;
    const float MIN_PEAK_SPATIAL_PROB = 0.05f;

    // --- Cross-lane outlier rejection ---
    const float MIN_REJECT_PX = 15.0f;
    const float MAD_MULTIPLIER = 3.0f;

    // --- Line-type diagnostics ---
    const bool DEBUG_SHOW_PAINT_SAMPLES = true;
    // Debug bars are drawn semi-transparent so the road stays visible underneath. Bumped
    // up for more punch in a screenshot/video - lower this back toward 0.5-0.6 if it
    // starts obscuring the road too much for your taste.
    const double DEBUG_OVERLAY_ALPHA = 0.8;
    // The actual scanned width (PAINT_SEARCH_HALF_WIDTH_PX) stays what it is - this only
    // narrows what gets DRAWN, so the visual is cleaner without changing what's checked.
    const int DEBUG_LINE_VISUAL_HALF_WIDTH_PX = 5;

    // --- Palette (BGR, OpenCV convention) - bolder/more saturated than the first muted
    // pass, but still coherent tones rather than neon primaries.
    const cv::Scalar COLOR_TRACKING_DOT(230, 150, 30);   // vivid azure blue for all raw tracking points
    const cv::Scalar COLOR_DOT_OUTLINE(20, 20, 20);      // dark halo for contrast against bright road
    const cv::Scalar COLOR_PAINT(70, 220, 90);           // vivid emerald green - paint found
    const cv::Scalar COLOR_GAP(50, 50, 220);             // vivid red - no paint found
    const cv::Scalar COLOR_HOOD_LINE(200, 90, 200);      // magenta-leaning mauve
    const cv::Scalar LANE_LABEL_COLORS[4] = {
        cv::Scalar(220, 140, 60),  // lane 0 (far left)  - bold azure
        cv::Scalar(70, 200, 110),  // lane 1 (host left) - bold green
        cv::Scalar(50, 180, 240),  // lane 2 (host right)- bold amber
        cv::Scalar(80, 80, 230),   // lane 3 (far right) - bold red
    };

    std::vector<std::string> video_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".MP4" || entry.path().extension() == ".mp4") {
            video_files.push_back(entry.path().string());
        }
    }

    if (video_files.empty()) {
        std::cerr << "Error: No MP4 files found in " << folder_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<float> input_tensor_values;
    cv::namedWindow("UFLD Tracker", cv::WINDOW_AUTOSIZE);

    std::vector<LineTypeHistory> line_type_history(num_lanes);

    for (size_t file_idx = 0; file_idx < video_files.size(); ++file_idx) {
        std::string video_path = video_files[file_idx];
        cv::VideoCapture cap(video_path);

        if (!cap.isOpened()) continue;
        std::cout << "\nPlaying [" << file_idx + 1 << "/" << video_files.size() << "]: " << video_path << std::endl;

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Real-time playback pacing: compare the video's own timeline against wall-clock
        // time elapsed since this file started, and skip decode-only (no inference, no
        // drawing) whenever the video has fallen behind. This replaces a fixed 1-in-N
        // skip with something that adapts automatically to whatever the machine can
        // actually keep up with frame to frame - "behind" skips, "ahead or on time"
        // processes normally, with no artificial cap on how many frames in a row either
        // case can happen. Reset per file so a slow previous clip can't bleed lag into
        // the next one's start.
        double source_fps = cap.get(cv::CAP_PROP_FPS);
        if (source_fps <= 1.0) source_fps = 30.0; // sane fallback if the container doesn't report a usable fps
        auto video_start_time = std::chrono::steady_clock::now();
        long long frame_index = 0;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            double video_time_sec = static_cast<double>(frame_index) / source_fps;
            frame_index++;

            double elapsed_wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - video_start_time).count();

            if (video_time_sec < elapsed_wall_sec) {
                // Behind schedule - this frame's timestamp has already passed in real
                // time. Skip straight to grabbing the next frame with none of the
                // expensive work below; nothing about the tracking/classification
                // pipeline itself runs differently on a frame that IS processed.
                continue;
            }
            // At or ahead of schedule - process this frame normally, exactly as before.

            // Temporary (6.1 diagnosis): frame_index increments on skipped frames too (see
            // above), so gating debug logging on frame_index is unreliable on a Debug build
            // that's falling behind - most frames get skipped and which ones survive isn't
            // correlated with frame_index's value. Count only frames that actually reach here.
            static long long processed_frame_count = 0;
            processed_frame_count++;

            int crop_y = preProcessFrame(frame, input_tensor_values, model_input_w, model_input_h);
            int crop_h = frame.rows - crop_y;
            int hood_cutoff_y = frame.rows - HOOD_EXCLUSION_HEIGHT_PX;

            std::vector<int64_t> current_input_dims = { 1, 3, model_input_h, model_input_w };
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                current_input_dims.data(), current_input_dims.size()
            );

            std::vector<Ort::Value> output_tensors;
            try {
                output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
            }
            catch (const std::exception& e) {
                return -1;
            }

            float* raw_output_data = output_tensors[0].GetTensorMutableData<float>();

            int num_bins = 201;
            int num_anchors = 18;
            std::vector<std::vector<cv::Point>> detected_lanes(num_lanes);

            float scale_x = static_cast<float>(frame.cols) / static_cast<float>(model_input_w);
            float scale_y = static_cast<float>(crop_h) / static_cast<float>(model_input_h);

            for (int l = 0; l < num_lanes; ++l) {
                for (int a = 0; a < num_anchors; ++a) {

                    float* bin_scores = raw_output_data + (a * num_lanes) + l;
                    auto binValue = [&](int b) { return bin_scores[b * num_anchors * num_lanes]; };

                    float max_logit = binValue(0);
                    for (int b = 1; b < num_bins; ++b) {
                        max_logit = std::max(max_logit, binValue(b));
                    }

                    std::vector<float> exp_all(num_bins);
                    float sum_exp_all = 0.0f;
                    for (int b = 0; b < num_bins; ++b) {
                        exp_all[b] = std::exp(binValue(b) - max_logit);
                        sum_exp_all += exp_all[b];
                    }

                    float p_background = exp_all[200] / sum_exp_all;

                    float max_spatial_prob = 0.0f;
                    int argmax_bin = 0;
                    for (int b = 0; b < 200; ++b) {
                        float p = exp_all[b] / sum_exp_all;
                        if (p > max_spatial_prob) { max_spatial_prob = p; argmax_bin = b; }
                    }

                    bool confident = (p_background < MAX_BACKGROUND_PROB) && (max_spatial_prob > MIN_PEAK_SPATIAL_PROB);
                    if (!confident) continue;

                    float sum_exp_spatial = sum_exp_all - exp_all[200];
                    float expected_x = 0.0f;
                    for (int b = 0; b < 200; ++b) {
                        expected_x += static_cast<float>(b) * (exp_all[b] / sum_exp_spatial);
                    }

                    // Bin-to-pixel mapping matches the official UFLD reference decoder: divide
                    // by griding_num-1 (199), not griding_num (200), and index bins from 1.
                    // The old /200.0f version compressed far-bin points inward by ~8-14 native
                    // px (confirmed via [decbias] logging, 6.1 Step 2) - small but real and
                    // always in the same (inward) direction, independent of footage.
                    float model_x = (expected_x + 1.0f) * (static_cast<float>(model_input_w - 1) / 199.0f) - 1.0f;
                    float model_y = static_cast<float>(culane_row_anchors[a]);

                    // Temporary (6.1 diagnosis): compare argmax bin vs the softmax-expectation
                    // bin for the far-left/far-right channels at the horizon-side anchors - Step 1
                    // truncation-bias check. Not yet confirmed as of this pass (669-sample run
                    // showed no consistent inward-pulling direction for the well-sampled far-right
                    // channel). See plan §6.1.
                    if (DEBUG_LOG_DECODER_BIAS && (l == 0 || l == num_lanes - 1) && a < 5) {
                        float gap_bins = static_cast<float>(argmax_bin) - expected_x;
                        std::cout << "[decbias] frame=" << processed_frame_count
                                   << " lane=" << l << (l == 0 ? "(far-left)" : "(far-right)")
                                   << " anchor_y=" << culane_row_anchors[a]
                                   << " argmax=" << argmax_bin
                                   << " expected=" << expected_x
                                   << " gap_bins=" << gap_bins
                                   << std::endl;
                    }

                    int native_x = static_cast<int>(model_x * scale_x);
                    int native_y = static_cast<int>(model_y * scale_y) + crop_y;

                    if (native_y >= hood_cutoff_y) continue; // never trust/sample points on the hood

                    detected_lanes[l].push_back(cv::Point(native_x, native_y));
                }
            }

            // Cross-lane outlier rejection - foundational for reliable line-type reads
            std::vector<FittedLane> lane_fits(num_lanes);
            for (int l = 0; l < num_lanes; ++l) {
                detected_lanes[l] = rejectCrossLaneOutliers(detected_lanes[l], MIN_REJECT_PX, MAD_MULTIPLIER, &lane_fits[l]);
            }

            // Brightness test uses max(B,G,R) - equivalent to HSV's Value channel - instead
            // of standard grayscale luminance. For white/gray paint this is nearly
            // identical to luminance (channels are already similar), but for yellow paint
            // (high R+G, low B) it reads meaningfully brighter than luminance does, which
            // is the leading suspect for why the far-left line specifically underperforms
            // even when tracking is accurate: it's very likely the yellow center divider,
            // and plain grayscale undersells yellow's contrast against gray asphalt.
            cv::Mat channels[3];
            cv::split(frame, channels);
            cv::Mat gray = cv::max(cv::max(channels[0], channels[1]), channels[2]);

            // No more per-lane position memory: it was extrapolating the fitted quadratic
            // across the ENTIRE detection crop regardless of how narrow a range it was
            // actually fit on, which is exactly what produced the runaway lines - a
            // quadratic diverges fast outside the range it was fit against, and outer
            // lanes (fewer, sparser points) hit that worst. Back to: fresh detection only,
            // nothing drawn on a frame that doesn't have enough real points.
            cv::Mat overlay = frame.clone();

            std::vector<std::vector<cv::Point>> points_for_display(num_lanes);
            std::vector<LineTypeResult> results(num_lanes);
            std::vector<LineType> stables(num_lanes);

            for (int l = 0; l < num_lanes; ++l) {
                points_for_display[l] = detected_lanes[l];

                std::vector<std::tuple<cv::Point2f, cv::Point2f, bool>> debug_samples;
                results[l] = classifyLineType(gray, points_for_display[l],
                    DEBUG_SHOW_PAINT_SAMPLES ? &debug_samples : nullptr);
                line_type_history[l].push(results[l].type);
                stables[l] = line_type_history[l].stable();

                if (DEBUG_SHOW_PAINT_SAMPLES) {
                    for (const auto& s : debug_samples) {
                        cv::Point2f seg_a, seg_b; bool is_paint;
                        std::tie(seg_a, seg_b, is_paint) = s;
                        // Draw a NARROWER bar than what was actually checked - the check
                        // itself is untouched (PAINT_SEARCH_HALF_WIDTH_PX), this only
                        // affects the visual for a cleaner look.
                        cv::Point2f mid((seg_a.x + seg_b.x) * 0.5f, seg_a.y);
                        cv::Point2f draw_a(mid.x - DEBUG_LINE_VISUAL_HALF_WIDTH_PX, mid.y);
                        cv::Point2f draw_b(mid.x + DEBUG_LINE_VISUAL_HALF_WIDTH_PX, mid.y);
                        cv::line(overlay, draw_a, draw_b, is_paint ? COLOR_PAINT : COLOR_GAP, 5, cv::LINE_AA);
                    }
                }
            }

            if (DEBUG_SHOW_PAINT_SAMPLES) {
                // Note: unchanged pixels in `overlay` are identical to `frame`, so this
                // blend is a no-op everywhere except where bars were actually drawn -
                // alpha*bar_color + (1-alpha)*original there, alpha*F + (1-alpha)*F = F
                // (unchanged) everywhere else.
                cv::addWeighted(overlay, DEBUG_OVERLAY_ALPHA, frame, 1.0 - DEBUG_OVERLAY_ALPHA, 0.0, frame);
            }

            // Pass 2: opaque dots + labels on top of the now-blended frame, always fully
            // legible regardless of the overlay underneath.
            for (int l = 0; l < num_lanes; ++l) {
                for (size_t p = 0; p < points_for_display[l].size(); ++p) {
                    cv::circle(frame, points_for_display[l][p], 5, COLOR_DOT_OUTLINE, -1, cv::LINE_AA);
                    cv::circle(frame, points_for_display[l][p], 4, COLOR_TRACKING_DOT, -1, cv::LINE_AA);
                }

                if (!points_for_display[l].empty()) {
                    char label[128];
                    if (SHOW_LINE_TYPE_DIAGNOSTICS) {
                        snprintf(label, sizeof(label), "%s  n=%d  paint=%.0f%%  trans=%d",
                            lineTypeToString(stables[l]), results[l].n_points,
                            results[l].painted_fraction * 100.0f, results[l].transitions);
                    }
                    else {
                        snprintf(label, sizeof(label), "%s", lineTypeToString(stables[l]));
                    }
                    cv::Point label_pos = points_for_display[l].back() + cv::Point(12, 0);
                    // Drop-shadow style: dark pass first, then color on top, for legibility
                    // against whatever the road happens to look like underneath.
                    cv::putText(frame, label, label_pos + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(20, 20, 20), 3, cv::LINE_AA);
                    cv::putText(frame, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.65, LANE_LABEL_COLORS[l], 2, cv::LINE_AA);
                }
            }

            if (SHOW_HOOD_EXCLUSION_LINE) {
                cv::line(frame, cv::Point(0, hood_cutoff_y), cv::Point(frame.cols, hood_cutoff_y), COLOR_HOOD_LINE, 1, cv::LINE_AA);
                cv::putText(frame, "Hood exclusion line", cv::Point(10, hood_cutoff_y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HOOD_LINE, 1, cv::LINE_AA);
            }

            // Temporary (6.1 diagnosis, plan Step 0): dump a handful of frames with a
            // populated far-lane channel to disk so the tracked-point overlay can be
            // checked against actual paint pixels without needing to watch the live window.
            if (DEBUG_LOG_DECODER_BIAS) {
                static int dumped_frames = 0;
                bool far_lane_populated = points_for_display[0].size() >= 4 ||
                                           points_for_display[num_lanes - 1].size() >= 4;
                if (dumped_frames < 6 && far_lane_populated) {
                    std::string dump_path = "C:/Users/johnn/AppData/Local/Temp/claude/C--src-lane-tracker/670b6ff1-c000-425a-98a3-77b2fdfed6e5/scratchpad/frame_dump_" + std::to_string(dumped_frames) + ".png";
                    cv::imwrite(dump_path, frame);
                    std::cout << "[decbias] dumped " << dump_path << std::endl;
                    dumped_frames++;
                }
            }

            cv::imshow("UFLD Tracker", frame);

            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 27) return 0;
        }
    }

    std::cout << "Successfully processed all available files." << std::endl;
    cv::destroyAllWindows();
    return 0;
}