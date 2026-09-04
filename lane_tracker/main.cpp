// dml_provider_factory.h drags in <d3d12.h> -> <windows.h>, which #defines max/min unless
// told not to - that clobbers every std::max/std::min call in this file (cv::opencv.hpp
// included right below uses them too). Must be defined before any of those headers.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
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

std::string folder_path = "C:/Users/johnn/Downloads/testing_footage/original_dashcam/";
std::string model_path = "C:/src/lane_tracker/lane_tracker/models/ultra-fast-lane-det-culane.onnx";

// ---------------------------------------------------------------------------
// Hood/dashboard exclusion: no manual pixel constant anymore - see
// calibrateHoodCutoff() below preProcessFrame. Computed once per video and gates both
// tracking and classification exactly like the old constant did - same single cutoff, same
// two consumers, just measured per-video instead of hand-typed.
// ---------------------------------------------------------------------------

// Where the model's input band is anchored. The band's height is locked to the model's
// aspect ratio (raw_w * 288/800), so the only free choice is where its BOTTOM edge sits.
// Anchoring it to the frame bottom spends the lowest rows of the network's input - and
// every CULane row anchor that lands in them - on car bodywork. Measured on 1336x750
// highway footage: 4 of 18 row anchors fell at or below the hood line and were discarded
// on every frame, while 75 rows of plainly visible road sat ABOVE the topmost anchor with
// nothing to predict them (topmost anchor row 471, vanishing point row 396). Anchoring to
// the hood's leading edge instead slides the whole anchor ladder up onto road.
// Set false to restore the previous frame-bottom anchoring.
bool CROP_ANCHOR_TO_HOOD = true;

// ---------------------------------------------------------------------------
// Far-field gate: the horizon-side twin of the hood cutoff.
// ---------------------------------------------------------------------------
// The 18 CULane row anchors are fixed in the MODEL's coordinates, so the only thing this
// code can choose is where the ladder lands (CROP_ANCHOR_TO_HOOD above). It cannot make
// the ladder shorter. Wherever the ladder is placed, its topmost rungs can end up close
// enough to the vanishing point that a dashed line's gaps project to under a pixel and
// the line reads as continuous - measured on original_dashcam, points ~29 rows below the
// horizon pushed the DASHED median painted fraction from 0.486 to 0.552 and cost 1.8
// points of line-type accuracy. Those points are also where the tracked position is
// least trustworthy, since a whole lane's worth of road collapses into a few rows.
//
// So: measure the vanishing point per clip and stop tracking short of it.
bool  ENABLE_FAR_GATE = true;

// How far up the road tracking is allowed to reach, measured in the road's own ruler:
// the width of the ego lane, in pixels, at that row. For a flat road that width is
// A * (row - vanishing point), where A is lane width over camera height - the focal
// length cancels out completely, so A is a property of the mounting, not the lens.
// Measured A is 2.79 on the highway camera and 2.54-2.88 across the four
// original_dashcam clips, and extrapolating each clip's lane width back to zero
// reproduces its separately measured vanishing point to within a row or two. Frame
// height, which this margin used to be a fraction of, is not a property of the road at
// all: the same 3% meant a 63px-wide lane on one camera and a 92px-wide lane on the
// other, for no reason connected to either road.
//
// The limit is self-selecting. Highway chains reach a lane width of 101px; the
// original_dashcam chains stop at 207px and never come near it. So one number can bind
// on the camera that over-reaches and stay inert on the one that does not, without
// being told which is which.
//
// 63 reproduces the reach the frame-height margin gave on the highway clip, and the
// evidence does not support pulling the ego channels in any further: measured
// paint-on-line rates hold flat on both of them down to a lane width of 80px. The
// far-left and far-right channels DO fall apart below about 160px - 45% and 40% of
// their samples land on paint there, against a 60%+ plateau further down the road - so
// 160 is the setting to reach for if those two need shortening. It costs about 4 points
// of solid-line recall on the ego channels, which share the limit.
float MIN_LANE_WIDTH_PX = 63.0f;

// Fallback only, for footage where the ego fits never yield a usable A.
float HORIZON_MARGIN_FRAC = 0.03f;   // of frame height, below the measured horizon

// A outside this range means the two ego fits were not measuring a lane: a lane change
// or a fork mid-calibration, or one channel locked onto a barrier. Any dashcam that can
// see its own lane lands near 2.2-3.1, so well outside that is a bad fit rather than an
// unusual car, and the frame-height fallback is the safer answer.
// How wide to search sideways for the painted line, as a fraction of the ego lane width
// at that row - so the search covers the same slice of road everywhere instead of the
// same number of pixels. A flat PAINT_SEARCH_HALF_WIDTH_PX is a third of a lane far away
// and narrower than the stripe itself close up, which is what made real paint on a solid
// line read as gap in the near field (paint-call rate 96.6% at the far end of a chain,
// 57.0% at the near end, on stretches known to be solid).
//
// 0.035 puts the search at about +/-26px where the lane is 740px wide, comfortably wider
// than a 0.15m stripe there, and at about +/-4px where the lane is 120px wide. It does not
// cost dashed lines anything: a dash gap has no paint within reach at any width. Measured
// against a flat search, with the near-end vote exclusion removed in the same step: all
// four known-solid stretches on the highway score higher (83.8/78.9/92.8/96.2% against
// 81.6/67.9/91.7/88.7%), false SOLID on the two dashed channels falls, and in-domain
// accuracy returns to 97.7% from 95.3%.
//
// Set to 0 to restore the flat search.
float PAINT_SEARCH_WIDTH_FRAC = 0.035f;
int   PAINT_SEARCH_MIN_HALF_PX = 3;    // never so narrow that a 1px wobble misses
int   PAINT_SEARCH_MAX_HALF_PX = 30;   // never so wide it can reach a neighbouring line

// Clear distance between the edge of the line search and the background probe. This used
// to be folded into a single 40px offset from the tracked point; it is separate now
// because the search width moves and the background must stay outside it. Held at 30 to
// keep the background reference itself unchanged - it is the delicate term in this
// function (see the note on clipped probes below), and pinning it while the search width
// varied is what attributed the improvement to the search rather than to the reference.
int   PAINT_BG_GAP_PX = 30;

// Set once per clip from the same calibration that fixes the far gate, and read by
// isPaintSample, which has no other route to the road's scale. Zero until the clip has
// calibrated, which is what keeps its first seconds on the flat search.
float g_lane_px_per_row = 0.0f;
int   g_horizon_y = 0;

float LANE_SLOPE_MIN = 1.5f;
float LANE_SLOPE_MAX = 5.0f;

// Both places that set the gate agree on this, including the one that runs on a clip
// whose scale was already measured on an earlier visit.
int farCutoffRow(int horizon_y, float lane_px_per_row, int frame_h) {
    if (lane_px_per_row >= LANE_SLOPE_MIN && lane_px_per_row <= LANE_SLOPE_MAX)
        return horizon_y + static_cast<int>(MIN_LANE_WIDTH_PX / lane_px_per_row);
    return horizon_y + static_cast<int>(HORIZON_MARGIN_FRAC * static_cast<float>(frame_h));
}

// How many per-frame vanishing-point estimates to collect before locking one in. The
// estimate is a median, so this only has to be enough to outvote the frames where a
// curve or a lane change skews a single reading.
int HORIZON_CALIB_SAMPLES = 90;

// Whether anchoring the band to the hood is the right call for THIS camera.
// The anchor ladder is a fixed 166 of the model's 288 input rows, which works out to
// 166 * raw_w / 800 native rows - a property of the camera, not of the road. The road
// band it has to cover is horizon..hood. When the two are comparable the ladder can sit
// on the road with nothing wasted, and anchoring to the hood is strictly better. When the
// ladder is much taller than the road band it cannot fit either way, and anchoring to the
// hood spends the excess ABOVE the horizon, where points are kept and wrong, instead of
// below the hood, where they were simply gated away. Measured: highway 277 vs 267 rows
// (ratio 1.04, anchoring helps a lot); original_dashcam 398 vs 215 (ratio 1.85, anchoring
// costs 1.5 points of line-type accuracy). 1.3 sits in the gap between those two cases.
float LADDER_FIT_MAX_RATIO = 1.3f;

// ---------------------------------------------------------------------------
// Classification window: which PART of the chain gets a vote on line type.
// ---------------------------------------------------------------------------
// Both ends are counted now, so this is inert. It is kept as a tunable because the
// mechanism was load-bearing until the paint sampler stopped being scale-blind, and
// because the shape of the failure it worked around is worth recording.
//
// The near end used to be excluded. Approaching the camera the line term fell (213 -> 194
// on the highway clip) while the background held, so real paint on a solid line read as
// gap: over stretches known to be solid, the paint-call rate was 96.6% at the far end of
// the chain against 57.0% at the near end, which dragged solid lines under
// SOLID_MIN_PAINTED_FRACTION. Excluding the near fifth bought 8 points of solid-line
// recall on the highway and cost 2.4 points of in-domain accuracy.
//
// It is not needed any more, and it was never the right fix. That collapse was the search
// for the line being a flat +/-10px wide while the painted stripe near the camera is
// wider than that - see PAINT_SEARCH_WIDTH_FRAC. With the search scaled to the road, the
// near end reads correctly, and excluding it only throws away good samples: measured with
// the window off and the scaled search on, every known-solid stretch on the highway scores
// HIGHER than it did with the window on, and in-domain accuracy returns to 97.7%.
//
// The far end was excluded too, on the theory that unresolvable dashes read as paint there
// - they do, 76.7% against a 41.5% chain average - but replaying every labelled chain on
// both cameras shows it changes almost no verdicts, because the far end is too small a
// share of the samples to push a dashed line over the SOLID cut.
//
// If either is ever re-enabled: the fraction is of the CHAIN's own extent, not of the
// horizon-to-hood band. That distinction is load-bearing - a band-relative version was
// tried and cost 7 points of in-domain accuracy, because on that camera the chain only
// occupies the near part of the road band, so 'drop the near quarter of the band' threw
// away the good end of the chain. Whatever is excluded is still sampled and still drawn;
// only the vote is restricted.
float CLASSIFY_TRIM_FAR = 0.0f;     // fraction of the chain not counted at the horizon end
float CLASSIFY_TRIM_NEAR = 0.0f;    // fraction of the chain not counted at the hood end

// Whether the hood exclusion line is drawn on screen at all.
bool SHOW_HOOD_EXCLUSION_LINE = true; // temporarily on - diagnosing far-lane edge-curving (6.1)

// Logs when the temporal-consistency check (see LaneTemporalState below) suppresses a lane
// for looking like a sudden jump - rare events, cheap to leave on.
bool SHOW_TEMPORAL_REJECTIONS = false;

// Diagnostic-only: processes every frame as fast as possible instead of matching the video's
// real-time playback speed. Only useful for offline analysis (verifying a change against every
// frame, reliably, across a full run) - never for live/interactive viewing, since it blows
// through the actual video's timing. Leave false outside an active diagnostic pass.
bool DISABLE_REALTIME_PACING = false;

// ---------------------------------------------------------------------------
// Top-of-file tuning: temporal consistency
// ---------------------------------------------------------------------------
// How much a lane's fitted MEAN position (see fitsAgree) is allowed to move between one
// trusted frame and the next, per frame of gap, before it's treated as a suspicious jump
// rather than normal motion. Picked the same way ABS_REJECT_PX=9 was picked - well above the
// clean baseline, not just above its median: real mean-displacement distribution measured on
// this footage is p50=1.7px, p75=3.9px, p90=8.7px, p95=15.7px, p97=23.3px, p98=33.4px. 30px
// sits just under p98 (roughly 8x the p75 baseline, comparable margin to ABS_REJECT_PX's own
// ~5x), so under a percent of genuinely clean frames should ever cross it, while a confirmed
// vehicle-edge capture event measured mean displacements of 23-78px sustained across ~8
// frames - most of that event still clears this bar. Scales linearly with the frame-index gap
// between the current frame and the last trusted one (pacing can skip frames).
double TEMPORAL_BASE_THRESHOLD_PX = 30.0;
// Beyond this many frames of gap since the last trusted fit, there's too much real time
// elapsed to judge anything against it (a genuine lane change, a long occlusion) - accept and
// re-bootstrap rather than holding a stale reference forever.
long long TEMPORAL_MAX_GAP_FRAMES = 10;

// ---------------------------------------------------------------------------
// Top-of-file tuning: playback controls
// ---------------------------------------------------------------------------
// How far one arrow-key press moves through the footage. Seeks run over the WHOLE folder as
// a single timeline, not just the clip currently playing - see the clip table built in main().
double SEEK_STEP_SEC = 10.0;

// Which arrow does which. Set to +1.0 for the conventional mapping (Right = forward through
// the footage, Left = back); set to -1.0 to swap them.
double SEEK_RIGHT_DIRECTION = +1.0;

// waitKeyEx codes for the arrow keys. Windows' highgui backend reports them in the high
// bytes; the GTK/Qt values are listed too so this keeps working if the project is ever built
// on a non-Windows backend. Deliberately NOT matching the bare values 81/83 that some
// backends use - those collide with ASCII 'Q' and 'S'.
bool isSeekBackKey(int k) { return k == 2424832 || k == 65361; }    // Left
bool isSeekForwardKey(int k) { return k == 2555904 || k == 65363; } // Right

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
float SOLID_MIN_PAINTED_FRACTION = 0.75f; //0.7 was slightly low, caused close dashed lines to be misclassified as solid
// A rate, not a count. An absolute cap silently changes meaning whenever the number of
// samples in a chain changes - it had to be rescaled from 8 to 4 when the classification
// window was introduced, and would need rescaling again for any camera whose chains are
// longer or shorter. A solid line has near-zero transitions however long it is; a dashed
// line has two per dash cycle, so both scale with chain length and a rate does not drift.
//
// This gate is a backstop, not a discriminator. Measured on chains that clear the SOLID
// painted-fraction cut, the transition rate of true-dashed and true-solid chains overlaps
// almost completely (highway medians 0.043 against 0.029, 90th percentiles 0.080 against
// 0.078), so any setting tight enough to reject those dashed chains rejects a comparable
// share of real solids: 0.06 costs 5 points of solid recall to gain 1 of dashed. 0.11 is
// above both distributions and rejects ~2% of each, which leaves it doing the one job it
// is actually good at - catching a closely-spaced dashed line, whose short dash cycle
// puts its rate far above anything a real solid line reaches.
float SOLID_MAX_TRANSITION_RATE = 0.11f;   // transitions per counted sample
int   SOLID_MIN_TRANSITION_ALLOWANCE = 3;  // floor, so a short chain can still read SOLID
// enough painted_fraction to clear the SOLID bar on paint alone, and transitions is what still
// tells them apart from a real solid. Was briefly set to 1000 to test removing it; put back.

// A lane is called DASHED if transitions is at or above this AND painted_fraction falls in this range.
int   DASHED_MIN_TRANSITIONS = 0; // effectively "don't care" right now - see comment at the check itself
float DASHED_MIN_PAINTED_FRACTION = 0.15f;
float DASHED_MAX_PAINTED_FRACTION = 0.75f;

// Pre-processes frame: Applies CULane aspect ratio crop, resizes, and normalizes.
// road_bottom_y is where usable road ends (the hood's leading edge); pass <=0 to keep the
// old frame-bottom anchoring. The crop's HEIGHT is written to out_crop_h: the decoder needs
// it to map row anchors back to native pixels, and it is no longer (raw_h - crop_y) now
// that the band can sit clear of the frame bottom.
int preProcessFrame(const cv::Mat& src_frame, std::vector<float>& input_tensor_values,
                    int target_w, int target_h, int road_bottom_y, int* out_crop_h) {
    int raw_w = src_frame.cols;
    int raw_h = src_frame.rows;

    int crop_h = static_cast<int>(raw_w * (static_cast<float>(target_h) / static_cast<float>(target_w)));
    if (crop_h > raw_h) crop_h = raw_h;   // very wide/short source: cannot honour the aspect

    // See CROP_ANCHOR_TO_HOOD at the top of the file for why the bottom edge is placed here.
    int bottom = raw_h;
    if (CROP_ANCHOR_TO_HOOD && road_bottom_y > 0 && road_bottom_y <= raw_h) bottom = road_bottom_y;

    int crop_y = bottom - crop_h;
    if (crop_y < 0) crop_y = 0;
    if (crop_y + crop_h > raw_h) crop_y = raw_h - crop_h;

    if (out_crop_h) *out_crop_h = crop_h;
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
// Automatic hood/dashboard boundary detection
// ---------------------------------------------------------------------------
// UFLD was trained on CULane, which never frames a hood, so its row anchors still land on
// whatever is physically at the bottom of THIS camera's frame - usually the hood, not road.
// This replaces the old hand-typed pixel constant with a per-video measurement, but plays
// the exact same role: one cutoff, gating both tracking and classification identically.
//
// Validated against two different real vehicles with the SAME parameters, no retuning
// between them: a textured/painted hood (1920x1080) and a black, mirror-reflective hood
// (1338x750, different shape entirely). Scans ONE wide horizontal band, not per-column -
// narrow bands got hijacked by local features (a lane stripe, a reflection, an OSD overlay)
// on both vehicles tested, and per-column curve-fitting was tried and found LESS robust
// on the reflective hood, not more, so it was dropped in favor of this simpler version.
// Finds the FIRST genuine edge encountered coming from the road side (top of the search
// band downward), not the single strongest edge in the range - the strongest edge is
// reliably a real-but-wrong vehicle feature further into the hood (a cowl trim seam, a
// wiper rest position), not the boundary itself.
int findFirstProminentPeak(const std::vector<float>& row_energy, int search_lo, int search_hi,
    int window, float min_prominence_ratio) {
    for (int y = search_lo + window; y < search_hi - window; ++y) {
        float center = row_energy[y];
        float local_max = center;
        for (int k = -window; k <= window; ++k) local_max = std::max(local_max, row_energy[y + k]);
        if (center != local_max) continue; // not a local max over this window

        float left_floor = row_energy[y - window];
        for (int k = -window; k < 0; ++k) left_floor = std::min(left_floor, row_energy[y + k]);
        float right_floor = row_energy[y + 1];
        for (int k = 1; k <= window; ++k) right_floor = std::min(right_floor, row_energy[y + k]);
        float local_floor = std::min(left_floor, right_floor);

        if (local_floor > 0.0f && (center / local_floor) >= min_prominence_ratio) return y;
    }
    return -1; // no confident edge - caller applies a generic fallback
}

// Runs once per opened video file, on its own throwaway VideoCapture (never seeks or shares
// state with the real playback capture, so it can't disturb frame_index or the pacing
// clock). Returns a native_y cutoff - points with native_y >= this are treated as hood, same
// as the old constant - or -1 if this file couldn't be confidently calibrated (e.g. a
// degenerate clip), in which case the caller applies a generic fallback fraction.
int calibrateHoodCutoff(const std::string& video_path) {
    const int   NUM_CALIB_FRAMES = 20;
    const int   FRAME_STRIDE = 3;
    const float BAND_FRAC = 0.6f;   // central 60% of columns - dodges corner OSD overlays
    const float SEARCH_LO_FRAC = 0.67f;  // boundary must be in the bottom third
    const float SEED_FRAC = 0.05f;  // bottom 5% is unconditionally past any real boundary
    const int   PEAK_WINDOW = 10;    // local-max window, in rows
    const float MIN_PROMINENCE_RATIO = 1.8f;   // how much a peak must clear its local floor by

    cv::VideoCapture calib(video_path);
    if (!calib.isOpened()) return -1;

    std::vector<cv::Mat> frames;
    cv::Mat f;
    int i = 0;
    while (i < NUM_CALIB_FRAMES * FRAME_STRIDE && (int)frames.size() < NUM_CALIB_FRAMES) {
        calib >> f;
        if (f.empty()) break;
        if (i % FRAME_STRIDE == 0) frames.push_back(f.clone());
        ++i;
    }
    if (frames.empty()) return -1;

    int h = frames[0].rows, w = frames[0].cols;
    int x0 = static_cast<int>(w * (1.0f - BAND_FRAC) / 2.0f), x1 = w - x0;

    std::vector<float> row_energy(h, 0.0f);
    for (auto& fr : frames) {
        cv::Mat gray, sob;
        cv::cvtColor(fr, gray, cv::COLOR_BGR2GRAY);
        cv::Sobel(gray(cv::Rect(x0, 0, x1 - x0, h)), sob, CV_32F, 0, 1, 3);
        cv::Mat abs_sob = cv::abs(sob), row_mean;
        cv::reduce(abs_sob, row_mean, 1, cv::REDUCE_AVG, CV_32F);
        for (int y = 0; y < h; ++y) row_energy[y] += row_mean.at<float>(y);
    }
    for (auto& v : row_energy) v /= static_cast<float>(frames.size());

    int seed_rows = std::max(1, static_cast<int>(h * SEED_FRAC));
    int search_lo = static_cast<int>(h * SEARCH_LO_FRAC);
    int search_hi = h - seed_rows;

    return findFirstProminentPeak(row_energy, search_lo, search_hi, PEAK_WINDOW, MIN_PROMINENCE_RATIO);
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

// ---------------------------------------------------------------------------
// Temporal consistency (catches a SUDDEN, TRANSIENT whole-lane jump - e.g. a passing
// vehicle's edge or a crosswalk stripe briefly captured - that looks internally
// self-consistent within a single frame's own points, so nothing evaluated one frame at a
// time (this file's outlier rejection included) has any basis to reject it. Comparing against
// recent history is the only way to notice a lane's position jumped instead of evolved.
//
// Known, measured limitation: this does NOT catch a SLOWLY-EVOLVING false lock (e.g. a
// parked car the vehicle is gradually approaching) - that changes smoothly enough frame to
// frame to never look like a jump at all, so there is no signal here to act on. Not fixed by
// this mechanism; would need information this single check doesn't have (e.g. an object
// detector).
// ---------------------------------------------------------------------------

// Trusted = the most recent frame whose fit was accepted (either it agreed with the previous
// trusted fit, or two consecutive frames agreed with EACH OTHER on a new position - see
// temporalConsistencyCheck). Pending = the most recent frame that DISAGREED with trusted,
// kept around only to check whether the NEXT frame repeats it. min_y/max_y are the y-range
// each fit was actually built from - required so comparisons never evaluate a stored fit
// outside its own observed range (a quadratic diverges fast past where it was fit - see the
// "No more per-lane position memory" comment below for the prior bug this guards against).
struct LaneTemporalState {
    bool has_trusted = false;
    FittedLane trusted_fit;
    int trusted_min_y = 0, trusted_max_y = 0;
    long long trusted_frame_index = -1;

    bool has_pending = false;
    FittedLane pending_fit;
    int pending_min_y = 0, pending_max_y = 0;
    long long pending_frame_index = -1;
};

// Whether `fit` (observed over [min_y,max_y]) is consistent with `ref_fit` (observed over
// [ref_min_y,ref_max_y]) to within base_threshold_px * gap - gap-scaled because the pacing
// loop can skip frames, so "the previous trusted frame" may be several video-frames back, and
// more time passing means more genuine motion is expected. Only ever compares inside the
// overlap of both ranges; returns true (fail open, don't reject) if the ranges don't overlap
// at all - no basis to judge, so don't punish it.
bool fitsAgree(const FittedLane& fit, int min_y, int max_y,
    const FittedLane& ref_fit, int ref_min_y, int ref_max_y,
    long long gap, double base_threshold_px) {
    int ov_lo = std::max(min_y, ref_min_y);
    int ov_hi = std::min(max_y, ref_max_y);
    if (ov_lo > ov_hi) return true;

    // MEAN, not max, displacement across the overlap - two independently-fit quadratics from
    // small, noisy point sets can disagree a lot right at the edge of their shared range even
    // when the real line barely moved (curvature error amplifies fastest far from the data's
    // center), so max_disp is a much noisier signal than mean_disp for this comparison.
    double sum_disp = 0.0;
    int n_samples = 0;
    for (int y = ov_lo; y <= ov_hi; y += 5) {
        sum_disp += std::abs(fit.predict(y) - ref_fit.predict(y));
        n_samples++;
    }
    double mean_disp = sum_disp / n_samples;
    return mean_disp <= base_threshold_px * static_cast<double>(gap);
}

// Returns true if this frame's fit should be trusted (drawn/tracked normally), false if it
// should be suppressed as a likely transient jump. Updates `state` either way. max_gap_frames
// bounds how far back "recent" can mean - past that, there's too much real time elapsed to
// judge anything (a genuine lane change, a long occlusion), so it accepts and re-bootstraps
// rather than holding a stale reference forever.
bool temporalConsistencyCheck(LaneTemporalState& state, const FittedLane& cur_fit,
    int cur_min_y, int cur_max_y, long long processed_frame_count,
    double base_threshold_px, long long max_gap_frames) {

    if (!state.has_trusted) {
        state.has_trusted = true;
        state.trusted_fit = cur_fit;
        state.trusted_min_y = cur_min_y;
        state.trusted_max_y = cur_max_y;
        state.trusted_frame_index = processed_frame_count;
        return true;
    }

    long long gap = processed_frame_count - state.trusted_frame_index;
    bool agrees_with_trusted = (gap > max_gap_frames) ||
        fitsAgree(cur_fit, cur_min_y, cur_max_y, state.trusted_fit, state.trusted_min_y, state.trusted_max_y, gap, base_threshold_px);

    if (agrees_with_trusted) {
        state.trusted_fit = cur_fit;
        state.trusted_min_y = cur_min_y;
        state.trusted_max_y = cur_max_y;
        state.trusted_frame_index = processed_frame_count;
        state.has_pending = false;
        return true;
    }

    // Disagrees with trusted history - but if it agrees with the LAST disagreeing frame
    // instead, that's two consecutive frames agreeing on something new: treat it as a genuine
    // change (a real lane change, a sharp curve), not another bad frame, and promote it.
    if (state.has_pending) {
        long long pending_gap = processed_frame_count - state.pending_frame_index;
        bool agrees_with_pending = (pending_gap > max_gap_frames) ||
            fitsAgree(cur_fit, cur_min_y, cur_max_y, state.pending_fit, state.pending_min_y, state.pending_max_y, pending_gap, base_threshold_px);
        if (agrees_with_pending) {
            state.trusted_fit = cur_fit;
            state.trusted_min_y = cur_min_y;
            state.trusted_max_y = cur_max_y;
            state.trusted_frame_index = processed_frame_count;
            state.has_pending = false;
            return true;
        }
    }

    state.has_pending = true;
    state.pending_fit = cur_fit;
    state.pending_min_y = cur_min_y;
    state.pending_max_y = cur_max_y;
    state.pending_frame_index = processed_frame_count;
    return false;
}

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

// Iterative reject-refit: fit a quadratic, find the single worst-residual point, and if it's
// beyond abs_reject_px, drop just that one point and refit - repeat, bounded by max_removals
// and a floor of 4 surviving points (fitQuadratic's own minimum). This replaced a one-shot
// version whose threshold was `max(min_reject_px, mad_multiplier * median_residual)`: measured
// directly (residual logging against real footage, see project memory) that when 1-2 points are
// genuinely bad, the least-squares fit itself gets dragged toward them, which inflates the
// median residual of the WHOLE frame and therefore the threshold too - self-defeating exactly
// when it mattered most (79% of real isolated single-frame point-jumps measured were missed
// this way, because the inflated threshold ended up looser than the very outlier it needed to
// catch). Using a fixed absolute cutoff instead of a self-relative one avoids that; refitting
// after each removal means the fit itself de-leverages instead of staying dragged.
// abs_reject_px is picked well above the clean-footage baseline (measured p75 ~1.7px, so this
// is 5-6x that) and well below every measured genuine jump (18-37px), so it should not touch
// legitimately-tracked points - max_removals caps how much a single bad frame (e.g. a bridge
// underpass or a stretch with no real line - not what this is meant to fix) can be pruned down.
// Verified 2026-08-24 on real (non-fisheye) dashcam footage, same-run A/B against the old
// formula on identical frames: catches 100% of the cases the old formula missed (123/123),
// zero false rejections on frames the old formula considered fully clean (0/5642), and overall
// mean surviving points per lane dropped by under 1% (-0.9%) - the accuracy cost this technique
// trades for that catch rate is close to the noise floor of the measurement itself.
std::vector<cv::Point> rejectCrossLaneOutliers(const std::vector<cv::Point>& pts,
    float abs_reject_px, int max_removals, FittedLane* out_fit) {

    if (pts.size() < 5) {
        if (out_fit) *out_fit = fitQuadratic(pts);
        return pts; // too few points to tell a real trend from an outlier
    }

    const size_t MIN_SURVIVING = 4; // fitQuadratic's own floor
    std::vector<cv::Point> working = pts;
    int removed = 0;

    while (working.size() > MIN_SURVIVING && removed < max_removals) {
        FittedLane fit = fitQuadratic(working);
        if (!fit.valid) break;

        size_t worst_idx = 0;
        double worst_res = -1.0;
        for (size_t i = 0; i < working.size(); ++i) {
            double r = std::abs(working[i].x - fit.predict(working[i].y));
            if (r > worst_res) { worst_res = r; worst_idx = i; }
        }

        if (worst_res <= abs_reject_px) break; // remaining shape is coherent - stop pruning

        working.erase(working.begin() + worst_idx);
        removed++;
    }

    FittedLane final_fit = fitQuadratic(working);
    if (!final_fit.valid) final_fit = fitQuadratic(pts);
    if (out_fit) *out_fit = final_fit;

    return working;
}

// ---------------------------------------------------------------------------
// Shape-based point correction: instead of just dropping a point that doesn't fit the shape
// the REST of a lane's points describe (`--^---` doesn't look like a line at that one spot),
// use the other points to predict where it should have been and correct it there - keeps the
// point COUNT intact (line-type classification needs enough samples to say anything), instead
// of shrinking it the way plain deletion does.
// ---------------------------------------------------------------------------

// Exhaustively tries every 4-point subset of `pts` as a candidate quadratic (lanes only ever
// have a handful of points - 18 anchors at most - so this is at most a few thousand cheap fits,
// never a performance concern) and returns whichever candidate has the most INLIERS (points
// within inlier_px of it). This is more robust to MULTIPLE simultaneously-bad points than
// fitting to everything at once and removing the single worst offender: a fit built from all
// points (including several bad ones) can itself get dragged toward the bad ones, which is
// exactly the trap a plain "remove more points" idea falls into (see project memory) - a
// minimal-subset consensus fit never lets the bad points influence which candidate wins.
FittedLane bestConsensusFit(const std::vector<cv::Point>& pts, float inlier_px, std::vector<bool>* out_is_inlier) {
    FittedLane best;
    int best_inlier_count = -1;
    int n = static_cast<int>(pts.size());

    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            for (int c = b + 1; c < n; ++c)
                for (int d = c + 1; d < n; ++d) {
                    FittedLane candidate = fitQuadratic({ pts[a], pts[b], pts[c], pts[d] });
                    if (!candidate.valid) continue;

                    int inliers = 0;
                    for (const auto& p : pts) {
                        if (std::abs(p.x - candidate.predict(p.y)) <= inlier_px) inliers++;
                    }
                    if (inliers > best_inlier_count) {
                        best_inlier_count = inliers;
                        best = candidate;
                    }
                }

    if (best.valid) {
        // Refit using ALL inliers of the winning candidate, not just its founding 4 points,
        // for a more accurate final curve.
        std::vector<cv::Point> inlier_pts;
        for (const auto& p : pts) {
            if (std::abs(p.x - best.predict(p.y)) <= inlier_px) inlier_pts.push_back(p);
        }
        FittedLane refit = fitQuadratic(inlier_pts);
        if (refit.valid) best = refit;
    }

    if (out_is_inlier) {
        out_is_inlier->assign(n, false);
        if (best.valid) {
            for (int i = 0; i < n; ++i) {
                (*out_is_inlier)[i] = std::abs(pts[i].x - best.predict(pts[i].y)) <= inlier_px;
            }
        }
    }
    return best;
}

// Corrects (does not delete) up to max_corrections points that disagree with the consensus
// fit found above - replaces each disagreeing point's x with the fit's own prediction at that
// same y. If MORE than max_corrections points disagree, there is no longer a clear majority to
// trust a correction FROM - guessing a shape from a fit that might itself be built on a
// minority would be exactly the "confidently wrong" failure mode this whole file has been
// fighting, so this falls back to the plain, already-verified drop-based
// rejectCrossLaneOutliers instead of imputing anything in that case (the user's own caution:
// "be careful with this if multiple points are being noisy").
std::vector<cv::Point> correctOrRejectOutliers(const std::vector<cv::Point>& pts,
    float inlier_px, int max_corrections, FittedLane* out_fit) {

    if (pts.size() < 5) {
        // Too few points for a 4-point-subset consensus to mean anything against.
        return rejectCrossLaneOutliers(pts, inlier_px, max_corrections, out_fit);
    }

    std::vector<bool> is_inlier;
    FittedLane consensus = bestConsensusFit(pts, inlier_px, &is_inlier);
    if (!consensus.valid) {
        return rejectCrossLaneOutliers(pts, inlier_px, max_corrections, out_fit);
    }

    int n_outliers = 0;
    for (bool inl : is_inlier) if (!inl) n_outliers++;

    if (n_outliers == 0) {
        if (out_fit) *out_fit = consensus;
        return pts;
    }
    if (n_outliers > max_corrections) {
        return rejectCrossLaneOutliers(pts, inlier_px, max_corrections, out_fit);
    }

    std::vector<cv::Point> corrected = pts;
    for (size_t i = 0; i < corrected.size(); ++i) {
        if (!is_inlier[i]) {
            corrected[i].x = static_cast<int>(std::lround(consensus.predict(corrected[i].y)));
        }
    }
    if (out_fit) *out_fit = consensus;
    return corrected;
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

    // Falls back to the flat search, step included, whenever the clip has no measured
    // scale yet or the scaling is turned off - so PAINT_SEARCH_HALF_WIDTH_PX and
    // PAINT_SEARCH_STEP_PX still describe exactly what happens on that path.
    int search_half = PAINT_SEARCH_HALF_WIDTH_PX;
    int search_step = PAINT_SEARCH_STEP_PX;
    if (PAINT_SEARCH_WIDTH_FRAC > 0.0f && g_lane_px_per_row > 0.0f && pt.y > g_horizon_y) {
        float lane_w = g_lane_px_per_row * (pt.y - static_cast<float>(g_horizon_y));
        search_half = std::min(PAINT_SEARCH_MAX_HALF_PX,
            std::max(PAINT_SEARCH_MIN_HALF_PX,
                     static_cast<int>(PAINT_SEARCH_WIDTH_FRAC * lane_w)));
        // Same number of probes across the span however wide it is, so a wider search
        // means probes further apart rather than proportionally more work.
        search_step = std::max(2, search_half / 3);
    }

    if (out_seg_a) *out_seg_a = cv::Point2f(pt.x - search_half, pt.y);
    if (out_seg_b) *out_seg_b = cv::Point2f(pt.x + search_half, pt.y);

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

    // Used only as the fallback when both background probes are clipped - see the comment at
    // the background reference below. Deliberately NOT used for the normal path: swapping the
    // background from a max to a median everywhere would lower the reference on every sample,
    // including the ones sitting in a dash gap, which inflates their contrast and starts
    // reading gaps as paint. Matching statistics on both terms is what keeps gaps as gaps.
    auto sampleMedianBrightness = [&](cv::Point2f center, int half_win) {
        int cx = static_cast<int>(std::round(center.x));
        int cy = static_cast<int>(std::round(center.y));
        std::vector<uchar> vals;
        vals.reserve((2 * half_win + 1) * (2 * half_win + 1));
        for (int dy = -half_win; dy <= half_win; ++dy) {
            for (int dx = -half_win; dx <= half_win; ++dx) {
                int sx = cx + dx, sy = cy + dy;
                if (sx < 0 || sy < 0 || sx >= gray.cols || sy >= gray.rows) continue;
                vals.push_back(gray.at<uchar>(sy, sx));
            }
        }
        if (vals.empty()) return 255; // no pixels to judge from - stay conservative
        std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
        return static_cast<int>(vals[vals.size() / 2]);
        };

    int line_brightness = 0;
    for (int off = -search_half; off <= search_half; off += search_step) {
        line_brightness = std::max(line_brightness, sampleMaxBrightness(cv::Point2f(pt.x + off, pt.y), 3));
    }

    // MIN, not average, of the two side reads. The far-left lane's left-side probe is the
    // one most likely to land off the actual road surface (shoulder, guardrail, barrier -
    // all commonly brighter than asphalt). Averaging let a single contaminated bright side
    // drag the reference up, which shrinks measured contrast and reads as "no paint" even
    // directly on real paint. Taking the min means a clean reading on either side is
    // trusted, instead of a bad one on one side poisoning the whole reference.
    //
    // ...but min() only rescues the case where ONE side is contaminated. When BOTH sides
    // read blown-out white the reference pins at ~255, contrast goes to zero or below, and
    // real paint underneath reads as a gap. That is what speckles the far-left solid line
    // here: a sunlit concrete gutter runs alongside it, clipping the left probe on
    // essentially every sample, so the min is really just the right probe - and that one
    // clips too whenever it lands on a "50" pavement marking, the hood, or a light-coloured
    // car. Measured over 21275 real samples: on far-left GAP samples the right probe's
    // median was 255, versus 148 (asphalt) on far-left PAINT samples. The samples were not
    // marginal - contrast is strongly bimodal, gaps near 0 and paint near 100 - so this was
    // never a threshold that needed lowering, it was a reference that had stopped measuring
    // road.
    //
    // So: a probe pinned at the sensor ceiling is not a reading of the road, and is dropped.
    // Note what this does NOT change - if exactly one probe is clipped, min() already
    // returned the other one, so the result is bit-identical to before. The behaviour only
    // differs when BOTH are clipped, where the old code was guaranteed to answer "no paint".
    // That is why this leaves dashed lines alone: measured on the same 21275 samples, the
    // far-left solid went 78.8% -> 91.9% painted while the two dashed channels moved +0.0%
    // and +0.1%. A variant that spread more probes per side scored better on the far-left
    // (96.6%) but pushed the dashed channels up +5.3% and +4.7% - i.e. it started bridging
    // dash gaps, which is the "close dashes read as SOLID" failure - so it was rejected.
    const int BG_SATURATED = 250;
    int bg_offset = search_half + PAINT_BG_GAP_PX;
    int bg_left = sampleMaxBrightness(cv::Point2f(pt.x - bg_offset, pt.y), 5);
    int bg_right = sampleMaxBrightness(cv::Point2f(pt.x + bg_offset, pt.y), 5);

    bool left_usable = bg_left < BG_SATURATED;
    bool right_usable = bg_right < BG_SATURATED;

    int bg_brightness;
    if (left_usable && right_usable) bg_brightness = std::min(bg_left, bg_right);
    else if (left_usable)            bg_brightness = bg_left;
    else if (right_usable)           bg_brightness = bg_right;
    else {
        // Both sides clipped. A max over a blown-out window says nothing, but the window is
        // usually only PARTLY blown out - the edge of a marking, the lip of the hood - so a
        // median still finds the road pixels in it. Falling back to that is strictly better
        // than the old behaviour, which was to accept 255 and call real paint a gap.
        bg_brightness = std::min(
            sampleMedianBrightness(cv::Point2f(pt.x - bg_offset, pt.y), 5),
            sampleMedianBrightness(cv::Point2f(pt.x + bg_offset, pt.y), 5)
        );
    }

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

    // The window that gets a vote - see CLASSIFY_TRIM_FAR/NEAR at the top of the file.
    // Computed here, applied after sampling: the whole chain is still sampled and every
    // sample is still handed to the caller for drawing, so the excluded end looks exactly
    // like the rest of the line on screen. It is only barred from voting.
    int  vote_lo = 0, vote_hi = 0;
    bool windowed = false;
    if ((CLASSIFY_TRIM_FAR > 0.0f || CLASSIFY_TRIM_NEAR > 0.0f) && pts.size() >= 6) {
        int y_far = pts.front().y, y_near = pts.back().y;   // sorted by y, so far end first
        float span = static_cast<float>(y_near - y_far);
        if (span > 1.0f) {
            vote_lo = y_far + static_cast<int>(CLASSIFY_TRIM_FAR * span);
            vote_hi = y_near - static_cast<int>(CLASSIFY_TRIM_NEAR * span);
            windowed = true;
        }
    }

    std::vector<bool> all_paint;
    std::vector<int>  all_y;
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
            all_paint.push_back(is_paint);
            all_y.push_back(static_cast<int>(sample_pt.y));
            if (debug_samples) debug_samples->push_back({ seg_a, seg_b, is_paint });
        }
    }

    // Keep only the samples inside the voting window. Falls back to the whole chain when
    // the window would leave too little to judge from, since classifying a compromised
    // span beats not classifying at all.
    std::vector<bool> samples;
    samples.reserve(all_paint.size());
    for (size_t i = 0; i < all_paint.size(); ++i) {
        if (!windowed || (all_y[i] >= vote_lo && all_y[i] <= vote_hi))
            samples.push_back(all_paint[i]);
    }
    if (samples.size() < 10) samples = all_paint;

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

    int max_transitions = std::max(SOLID_MIN_TRANSITION_ALLOWANCE,
        static_cast<int>(SOLID_MAX_TRANSITION_RATE * static_cast<float>(samples.size())));
    if (result.painted_fraction > SOLID_MIN_PAINTED_FRACTION && transitions <= max_transitions) { //bro icl transitions dont matter at all, sometimes the dashed lines have less transitons cuz its all red lmao
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

    // --- DirectML execution provider (GPU) ---
    // DisableMemPattern()/ORT_SEQUENTIAL are documented hard requirements for the DML EP,
    // not tuning knobs - DML manages its own memory patterns and doesn't support ORT's
    // parallel executor. If no DirectML-capable device is found (or the vendored
    // DirectML.dll/providers_shared.dll aren't sitting next to the exe), this falls back
    // to CPU with the thread/opt-level settings above untouched, rather than failing to
    // start - see [[project-onnxruntime-vcpkg-broken]] for why this stack vendors its own
    // onnxruntime instead of using vcpkg's.
    bool using_dml = false;
    try {
        session_options.DisableMemPattern();
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        const OrtDmlApi* dml_api = nullptr;
        Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi(
            "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml_api)));
        Ort::ThrowOnError(dml_api->SessionOptionsAppendExecutionProvider_DML(session_options, 0));
        using_dml = true;
    }
    catch (const std::exception& e) {
        std::cerr << "DirectML EP unavailable (" << e.what() << ") - falling back to CPU." << std::endl;
    }

    std::cout << "Loading ONNX model..." << std::endl;
    std::wstring model_path_w(model_path.begin(), model_path.end());

    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(env, model_path_w.c_str(), session_options);
        std::cout << "\xE2\x9C\x94 ONNX Runtime Session running on "
                   << (using_dml ? "DirectML (GPU)" : "CPU, 11 cores") << "!" << std::endl;
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

    // --- Per-lane shape outlier rejection (iterative reject-refit - see function comment) ---
    const float ABS_REJECT_PX = 9.0f;
    const int MAX_REMOVALS = 2;

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

    // ---------------------------------------------------------------------------
    // Seeking: one continuous timeline across all the clips
    // ---------------------------------------------------------------------------
    // A dashcam chops one continuous drive into fixed-length files, so "skip back 10s" from
    // 4s into a clip has to land 6s before the END of the previous one - a per-file seek
    // can't express that. Measuring every clip's duration once up front turns the folder
    // into a single virtual timeline: seeks are computed in global seconds and then mapped
    // back to (which clip, how far into it). Durations come from container metadata, so this
    // costs one cheap open per file at startup and no decoding.
    struct ClipInfo {
        std::string path;
        double fps = 30.0;
        double duration_sec = 0.0;
        double global_start_sec = 0.0;
        bool hood_measured = false;  // cached so jumping between clips doesn't re-run
        int  hood_cutoff_y = 0;      // calibrateHoodCutoff (~60 frames) on every jump
        int  hood_raw_y = 0;         // the SAME measurement without PERMANENT_OFFSET_PX
        bool horizon_measured = false;
        int  horizon_y = 0;          // vanishing point, from ego-lane fits (see far gate)
        float lane_px_per_row = 0.0f; // A: ego lane width gained per row below the horizon
        int  crop_bottom_y = 0;      // 0 = frame bottom; set once the ladder fit is known
    };

    std::vector<ClipInfo> clips;
    double timeline_total_sec = 0.0;
    for (const auto& p : video_files) {
        cv::VideoCapture probe(p);
        if (!probe.isOpened()) continue;
        ClipInfo ci;
        ci.path = p;
        ci.fps = probe.get(cv::CAP_PROP_FPS);
        if (ci.fps <= 1.0) ci.fps = 30.0;
        double n_frames = probe.get(cv::CAP_PROP_FRAME_COUNT);
        if (n_frames > 1.0) {
            ci.duration_sec = n_frames / ci.fps;
        }
        else {
            // Container didn't report a frame count - ask the demuxer to jump to the end and
            // report where it landed. Still metadata-only, no decode.
            probe.set(cv::CAP_PROP_POS_AVI_RATIO, 1.0);
            double end_ms = probe.get(cv::CAP_PROP_POS_MSEC);
            ci.duration_sec = (end_ms > 0.0) ? end_ms / 1000.0 : 0.0;
        }
        ci.global_start_sec = timeline_total_sec;
        timeline_total_sec += ci.duration_sec;
        if (ci.duration_sec <= 0.0) {
            std::cout << "[seek] warning: could not determine duration of " << p
                       << " - seeks will not be able to land inside it" << std::endl;
        }
        clips.push_back(ci);
    }
    if (clips.empty()) {
        std::cerr << "Error: none of the MP4 files in " << folder_path << " could be opened." << std::endl;
        return -1;
    }
    std::cout << "[seek] " << clips.size() << " clip(s), " << timeline_total_sec
               << "s total. Left/Right arrows skip " << SEEK_STEP_SEC << "s, q/ESC quits." << std::endl;

    // Global seconds -> (clip index, offset within that clip). Clamps into range, so seeking
    // past either end parks at the boundary instead of falling off it. Clips whose duration
    // is unknown (0) are stepped over rather than landed in.
    auto globalToClip = [&clips, timeline_total_sec](double global_sec, size_t& out_idx, double& out_offset) {
        global_sec = std::max(0.0, std::min(global_sec, timeline_total_sec - 0.001));
        for (size_t i = 0; i < clips.size(); ++i) {
            bool last = (i + 1 == clips.size());
            if (global_sec < clips[i].global_start_sec + clips[i].duration_sec || last) {
                out_idx = i;
                out_offset = std::max(0.0, global_sec - clips[i].global_start_sec);
                return;
            }
        }
        out_idx = 0;
        out_offset = 0.0;
        };

    // Pacing compares the video's own timeline against wall-clock elapsed since a fixed
    // origin. After a jump that origin has to move with it, or the very next frame looks
    // wildly "behind schedule" and the pacer throws away frames trying to catch up to a
    // position it never actually played.
    auto pacingOriginFor = [](double offset_sec) {
        return std::chrono::steady_clock::now()
            - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(offset_sec));
        };

    cv::Mat frame;
    std::vector<float> input_tensor_values;
    cv::namedWindow("UFLD Tracker", cv::WINDOW_AUTOSIZE);

    std::vector<LineTypeHistory> line_type_history(num_lanes);
    std::vector<LaneTemporalState> lane_temporal(num_lanes);

    // A jump can land in a different clip, so the file loop can't be a plain forward for() -
    // it has to be able to be sent backwards or skip ahead. pending_seek_offset carries the
    // landing position into the next iteration when a seek crosses a clip boundary.
    size_t file_idx = 0;
    bool arrived_via_seek = false;
    double pending_seek_offset = 0.0;

    while (file_idx < clips.size()) {
        std::string video_path = clips[file_idx].path;
        cv::VideoCapture cap(video_path);

        if (!cap.isOpened()) { file_idx++; arrived_via_seek = false; pending_seek_offset = 0.0; continue; }
        std::cout << "\nPlaying [" << file_idx + 1 << "/" << clips.size() << "]: " << video_path << std::endl;
        std::fill(lane_temporal.begin(), lane_temporal.end(), LaneTemporalState{}); // reset per file - a new file's first frame has no relevant history
        if (arrived_via_seek) {
            // A jump is a scene discontinuity, so the rolling solid/dashed vote must not carry
            // across it - 15 frames' worth of votes from somewhere else would mislabel the
            // landing point. Only done for seeks: natural clip-to-clip playback is continuous
            // footage, and its existing carry-over behaviour is left exactly as it was.
            for (auto& h : line_type_history) h = LineTypeHistory{};
        }

        // Permanent calibration offset: measured ONCE, by comparing the auto-detected
        // boundary against the last known-good hand-tuned constant this project shipped
        // with (HOOD_EXCLUSION_HEIGHT_PX=220, i.e. cutoff=860 on this camera's 1080-tall
        // frame) on the reference footage in original_dashcam/. auto-detection there
        // measured 819 - so PERMANENT_OFFSET_PX = 860 - 819 = 41. This is now a fixed
        // constant, not recomputed per video, specifically so tracking on this camera's
        // footage matches the last GitHub-pushed behavior exactly (same anchors admitted,
        // same cutoff row), while still adapting to genuinely different footage through the
        // calibrateHoodCutoff() measurement it's added on top of.
        const int PERMANENT_OFFSET_PX = 41;

        // Cached per clip: seeking back and forth across a boundary would otherwise re-run
        // this ~60-frame measurement on every crossing, for a result that cannot change.
        if (!clips[file_idx].hood_measured) {
            int measured = calibrateHoodCutoff(video_path);
            int raw_edge;
            if (measured < 0) {
                int fallback_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                if (fallback_h <= 1) fallback_h = 1080; // last-resort guard if container metadata is unreadable
                measured = static_cast<int>(fallback_h * 0.78f);
                raw_edge = measured;
                std::cout << "[hood] " << video_path << ": no confident boundary found - using fallback hood_cutoff_y=" << measured << std::endl;
            }
            else {
                // Two different jobs, so two different rows. PERMANENT_OFFSET_PX is a
                // correction for THIS detector being conservative on the original_dashcam
                // camera; it is a camera-specific fudge, not a geometric property, so it is
                // kept only on the point gate where it was calibrated. The crop is anchored
                // to the detector's own reading, which was checked by eye against two
                // timestamps of the 750-row highway footage and lands exactly on the hood's
                // leading edge there.
                raw_edge = measured;
                measured += PERMANENT_OFFSET_PX;
                std::cout << "[hood] " << video_path << ": hood_cutoff_y=" << measured << " (auto-detected " << raw_edge << " + " << PERMANENT_OFFSET_PX << "px permanent offset)" << std::endl;
            }
            clips[file_idx].hood_measured = true;
            clips[file_idx].hood_cutoff_y = measured;
            clips[file_idx].hood_raw_y = raw_edge;
        }
        int hood_cutoff_y = clips[file_idx].hood_cutoff_y;
        int hood_raw_y = clips[file_idx].hood_raw_y;

        // Far gate state for this clip. Until enough samples are in, far_cutoff_y stays 0
        // and the gate is inert - the first ~3 seconds run exactly as they did before, and
        // are also what the estimate is built from.
        std::vector<int> horizon_samples;
        std::vector<double> scale_samples;   // ego lane width gained per row, per frame
        int far_cutoff_y = 0;
        // 0 until the horizon is known, which means the frame-bottom anchoring the project
        // shipped with. The first ~3 seconds of a clip therefore behave exactly as before,
        // and are what the decision is made from.
        int crop_bottom_y = clips[file_idx].crop_bottom_y;
        g_lane_px_per_row = 0.0f;
        g_horizon_y = 0;
        if (clips[file_idx].horizon_measured && ENABLE_FAR_GATE) {
            int gate_frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            if (gate_frame_h <= 1) gate_frame_h = 1080;
            far_cutoff_y = farCutoffRow(clips[file_idx].horizon_y,
                                        clips[file_idx].lane_px_per_row, gate_frame_h);
            g_lane_px_per_row = clips[file_idx].lane_px_per_row;
            g_horizon_y = clips[file_idx].horizon_y;
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Real-time playback pacing: compare the video's own timeline against wall-clock
        // time elapsed since this file started, and skip decode-only (no inference, no
        // drawing) whenever the video has fallen behind. This replaces a fixed 1-in-N
        // skip with something that adapts automatically to whatever the machine can
        // actually keep up with frame to frame - "behind" skips, "ahead or on time"
        // processes normally, with no artificial cap on how many frames in a row either
        // case can happen. Reset per file so a slow previous clip can't bleed lag into
        // the next one's start.
        double source_fps = clips[file_idx].fps;
        long long frame_index = 0;

        // Land at the seek target, if we got here by seeking. Read the position back from the
        // demuxer rather than trusting the requested time: seeking snaps to a keyframe, and
        // frame_index has to describe where we ACTUALLY are or every derived timestamp below
        // (pacing, the on-screen clock, the next seek's origin) drifts by the snap distance.
        if (arrived_via_seek && pending_seek_offset > 0.0) {
            cap.set(cv::CAP_PROP_POS_MSEC, pending_seek_offset * 1000.0);
            double landed = cap.get(cv::CAP_PROP_POS_FRAMES);
            frame_index = (landed > 0.0) ? static_cast<long long>(landed)
                                          : static_cast<long long>(pending_seek_offset * source_fps);
        }
        arrived_via_seek = false;
        pending_seek_offset = 0.0;

        auto video_start_time = pacingOriginFor(static_cast<double>(frame_index) / source_fps);
        bool jumped_to_other_clip = false;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            double video_time_sec = static_cast<double>(frame_index) / source_fps;
            frame_index++;

            double elapsed_wall_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - video_start_time).count();

            if (!DISABLE_REALTIME_PACING && video_time_sec < elapsed_wall_sec) {
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

            auto frame_perf_start = std::chrono::steady_clock::now();

            int crop_h = 0;
            int crop_y = preProcessFrame(frame, input_tensor_values, model_input_w, model_input_h,
                                         crop_bottom_y, &crop_h);

            std::vector<int64_t> current_input_dims = { 1, 3, model_input_h, model_input_w };
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                current_input_dims.data(), current_input_dims.size()
            );

            std::vector<Ort::Value> output_tensors;
            auto infer_start = std::chrono::steady_clock::now();
            try {
                output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
            }
            catch (const std::exception& e) {
                return -1;
            }
            auto infer_end = std::chrono::steady_clock::now();

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
                    if (far_cutoff_y > 0 && native_y < far_cutoff_y) continue; // too near the horizon

                    detected_lanes[l].push_back(cv::Point(native_x, native_y));
                }
            }

            // Vanishing point for the far gate. Least-squares x = m*y + c over each ego
            // channel's whole chain, then intersect the two lines. A least-squares fit over
            // 14-16 points shrugs off the couple of noisy far points that a chord through
            // the chain's extremes would be entirely at the mercy of. Only runs while the
            // estimate is still being built, so it costs nothing afterwards.
            if (ENABLE_FAR_GATE && !clips[file_idx].horizon_measured &&
                detected_lanes[1].size() >= 5 && detected_lanes[2].size() >= 5) {
                auto fitLine = [](const std::vector<cv::Point>& pts, double& m, double& c) {
                    double sy = 0, sx = 0, syy = 0, sxy = 0;
                    double n = static_cast<double>(pts.size());
                    for (const auto& q : pts) {
                        sy += q.y; sx += q.x; syy += static_cast<double>(q.y) * q.y;
                        sxy += static_cast<double>(q.x) * q.y;
                    }
                    double den = n * syy - sy * sy;
                    if (std::abs(den) < 1e-9) return false;
                    m = (n * sxy - sx * sy) / den;
                    c = (sx - m * sy) / n;
                    return true;
                    };
                double m1, c1, m2, c2;
                if (fitLine(detected_lanes[1], m1, c1) && fitLine(detected_lanes[2], m2, c2) &&
                    std::abs(m1 - m2) > 1e-6) {
                    double vy = (c2 - c1) / (m1 - m2);
                    // The same two fits read the other way: how fast the ego lane widens
                    // per row is the road's scale at this camera.
                    scale_samples.push_back(m2 - m1);
                    // A sane vanishing point is above every tracked point and inside the frame.
                    if (vy > 0 && vy < static_cast<double>(frame.rows)) {
                        horizon_samples.push_back(static_cast<int>(vy));
                    }
                }
                if (static_cast<int>(horizon_samples.size()) >= HORIZON_CALIB_SAMPLES) {
                    std::nth_element(horizon_samples.begin(),
                                     horizon_samples.begin() + horizon_samples.size() / 2,
                                     horizon_samples.end());
                    int h_y = horizon_samples[horizon_samples.size() / 2];
                    clips[file_idx].horizon_y = h_y;
                    clips[file_idx].horizon_measured = true;

                    // The road's ruler, from the same fits over the same window. Median
                    // for the same reason the vanishing point is one: a single curve or
                    // lane change must not be able to move it.
                    float lane_A = 0.0f;
                    if (!scale_samples.empty()) {
                        std::nth_element(scale_samples.begin(),
                                         scale_samples.begin() + scale_samples.size() / 2,
                                         scale_samples.end());
                        lane_A = static_cast<float>(scale_samples[scale_samples.size() / 2]);
                    }
                    clips[file_idx].lane_px_per_row = lane_A;
                    g_lane_px_per_row = lane_A;
                    g_horizon_y = h_y;
                    far_cutoff_y = farCutoffRow(h_y, lane_A, frame.rows);
                    bool gate_by_scale = (lane_A >= LANE_SLOPE_MIN && lane_A <= LANE_SLOPE_MAX);
                    std::cout << "[horizon] " << video_path << ": vanishing point y=" << h_y
                              << ", ego lane widens " << lane_A << "px per row, far gate at y="
                              << far_cutoff_y;
                    if (gate_by_scale) std::cout << " (where the ego lane narrows to "
                                                 << MIN_LANE_WIDTH_PX << "px)" << std::endl;
                    else std::cout << " (no usable lane scale - fell back to "
                                   << HORIZON_MARGIN_FRAC << " x " << frame.rows << " rows)" << std::endl;

                    // Now that the road band is known, decide where the band's bottom edge
                    // belongs. See LADDER_FIT_MAX_RATIO. Changing it mid-clip moves every
                    // tracked point at once, which is exactly the kind of jump the temporal
                    // check exists to reject, so the smoothing state is reset with it.
                    float ladder_h = 166.0f * static_cast<float>(frame.cols) / 800.0f;
                    int   road_band = hood_raw_y - h_y;
                    if (CROP_ANCHOR_TO_HOOD && road_band > 0 &&
                        ladder_h <= LADDER_FIT_MAX_RATIO * static_cast<float>(road_band)) {
                        clips[file_idx].crop_bottom_y = hood_raw_y;
                        std::cout << "[crop] ladder " << static_cast<int>(ladder_h) << " rows vs road band "
                                  << road_band << " (ratio " << (ladder_h / road_band)
                                  << ") - anchoring the band to the hood edge at y=" << hood_raw_y << std::endl;
                    }
                    else {
                        clips[file_idx].crop_bottom_y = frame.rows;
                        std::cout << "[crop] ladder " << static_cast<int>(ladder_h) << " rows vs road band "
                                  << road_band << " (ratio " << (road_band > 0 ? ladder_h / road_band : 0.0f)
                                  << ") - ladder does not fit, keeping the frame-bottom anchor" << std::endl;
                    }
                    if (crop_bottom_y != clips[file_idx].crop_bottom_y) {
                        crop_bottom_y = clips[file_idx].crop_bottom_y;
                        for (auto& h : line_type_history) h = LineTypeHistory{};
                        for (auto& t : lane_temporal) t = LaneTemporalState{};
                    }
                }
            }

            // Cross-lane outlier correction - foundational for reliable line-type reads.
            // correctOrRejectOutliers tries to fix a bad point in place (from a
            // majority-consensus shape) rather than just dropping it, falling back to plain
            // deletion when there's no clear majority to correct from - see its own comment.
            std::vector<FittedLane> lane_fits(num_lanes);
            for (int l = 0; l < num_lanes; ++l) {
                detected_lanes[l] = correctOrRejectOutliers(detected_lanes[l], ABS_REJECT_PX, MAX_REMOVALS, &lane_fits[l]);
            }

            // Temporal consistency: suppress a lane's ENTIRE current-frame detection if its
            // fit jumped from recent history more than normal motion would explain - catches a
            // sudden, transient whole-lane capture (see LaneTemporalState comment above for
            // what this does and does not catch). Runs after outlier rejection so it judges
            // the same trusted fit everything else downstream uses.
            for (int l = 0; l < num_lanes; ++l) {
                if (!lane_fits[l].valid || detected_lanes[l].empty()) continue;
                int cur_min_y = detected_lanes[l].front().y, cur_max_y = detected_lanes[l].front().y;
                for (const auto& p : detected_lanes[l]) {
                    cur_min_y = std::min(cur_min_y, p.y);
                    cur_max_y = std::max(cur_max_y, p.y);
                }

                bool consistent = temporalConsistencyCheck(lane_temporal[l], lane_fits[l], cur_min_y, cur_max_y,
                    processed_frame_count, TEMPORAL_BASE_THRESHOLD_PX, TEMPORAL_MAX_GAP_FRAMES);

                if (!consistent) {
                    if (SHOW_TEMPORAL_REJECTIONS) {
                        std::cout << "[temporal] frame=" << processed_frame_count << " lane=" << l
                                   << " rejected as a sudden jump from recent history" << std::endl;
                    }
                    detected_lanes[l].clear();
                }
            }

            // Brute-force off-road suppression: a median wall/guardrail on a highway, or a
            // curb on a regular road, sitting just outside the road edge sometimes gets
            // picked up as if it were another lane line, in the outermost channel (0 or 3).
            // Real lane paint doesn't exist beyond the solid line that marks the actual road
            // edge, so once a lane's inner neighbor (1 or 2) has settled on SOLID, anything
            // detected further outboard is treated as off-road and dropped outright - no
            // exceptions, no per-frame existence voting. Gated on the already-smoothed
            // history (line_type_history's majority-vote stable()), not a fresh per-frame
            // classification, so this doesn't flicker on/off frame to frame - a per-frame
            // confidence/existence check was tried before for a similar problem and didn't
            // hold up. Verified 2026-08-24 against real (non-fisheye) dashcam footage: fires
            // only when a neighbor line has genuinely settled SOLID, not spuriously.
            if (line_type_history[1].stable() == LineType::SOLID) detected_lanes[0].clear();
            if (line_type_history[2].stable() == LineType::SOLID) detected_lanes[3].clear();

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
                    // Neutral spine along the whole chain, drawn under the paint bars so
                    // they cover it. The bars now span the whole chain too, so this only
                    // shows through between them, joining up a line whose samples are
                    // spaced further apart than the bars are wide.
                    for (size_t p = 0; p + 1 < points_for_display[l].size(); ++p) {
                        cv::line(overlay, points_for_display[l][p], points_for_display[l][p + 1],
                                 COLOR_TRACKING_DOT, 3, cv::LINE_AA);
                    }
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

            // Perf check: total per-frame processing time vs. just the session->Run() slice
            // of it, so "is GPU worth it" is measured instead of assumed - if inference isn't
            // most of the frame, moving it to DML won't move the needle much on its own.
            // First 20 processed frames are skipped: DML compiles its kernels on first use,
            // which would otherwise dominate (and mislead) the first reported window.
            {
                static double sum_total_ms = 0.0, sum_infer_ms = 0.0;
                static int window_count = 0;
                if (processed_frame_count > 20) {
                    double total_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - frame_perf_start).count();
                    double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
                    sum_total_ms += total_ms;
                    sum_infer_ms += infer_ms;
                    window_count++;
                    if (window_count >= 60) {
                        std::cout << "[perf] avg/frame over last " << window_count << " frames: total="
                                   << (sum_total_ms / window_count) << "ms  inference="
                                   << (sum_infer_ms / window_count) << "ms ("
                                   << (100.0 * sum_infer_ms / sum_total_ms) << "%)" << std::endl;
                        sum_total_ms = 0.0; sum_infer_ms = 0.0; window_count = 0;
                    }
                }
            }

            cv::imshow("UFLD Tracker", frame);

            // waitKeyEx, not waitKey: the arrow keys are extended codes that live above the
            // low byte, so the old `(char)cv::waitKey(1)` truncated them away entirely.
            int key = cv::waitKeyEx(1);
            if (key == 'q' || key == 27) return 0;

            double seek_delta = 0.0;
            if (isSeekForwardKey(key))   seek_delta = SEEK_STEP_SEC * SEEK_RIGHT_DIRECTION;
            else if (isSeekBackKey(key)) seek_delta = -SEEK_STEP_SEC * SEEK_RIGHT_DIRECTION;

            if (seek_delta != 0.0) {
                // video_time_sec is this frame's position within the clip; adding the clip's
                // own start turns it into a position on the whole-folder timeline, which is
                // the only frame of reference in which "10 seconds earlier" is well defined
                // when 10 seconds earlier is in a different file.
                double global_now = clips[file_idx].global_start_sec + video_time_sec;
                size_t target_idx = file_idx;
                double target_offset = 0.0;
                globalToClip(global_now + seek_delta, target_idx, target_offset);

                std::cout << "[seek] " << (seek_delta > 0 ? "+" : "") << seek_delta << "s -> clip "
                           << target_idx + 1 << "/" << clips.size() << " at " << target_offset
                           << "s (global " << (clips[target_idx].global_start_sec + target_offset)
                           << "s of " << timeline_total_sec << "s)" << std::endl;

                if (target_idx == file_idx) {
                    // Same clip: seek in place, no need to reopen or re-measure anything.
                    cap.set(cv::CAP_PROP_POS_MSEC, target_offset * 1000.0);
                    double landed = cap.get(cv::CAP_PROP_POS_FRAMES);
                    frame_index = (landed > 0.0) ? static_cast<long long>(landed)
                                                  : static_cast<long long>(target_offset * source_fps);
                    video_start_time = pacingOriginFor(static_cast<double>(frame_index) / source_fps);
                    std::fill(lane_temporal.begin(), lane_temporal.end(), LaneTemporalState{});
                    for (auto& h : line_type_history) h = LineTypeHistory{};
                }
                else {
                    // Different clip: hand the landing position to the outer loop, which
                    // reopens there and applies it.
                    arrived_via_seek = true;
                    pending_seek_offset = target_offset;
                    file_idx = target_idx;
                    jumped_to_other_clip = true;
                    break;
                }
            }
        }

        if (jumped_to_other_clip) continue; // file_idx already points at the clip we jumped to
        file_idx++;
    }

    std::cout << "Successfully processed all available files." << std::endl;
    cv::destroyAllWindows();
    return 0;
}