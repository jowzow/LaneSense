#pragma once

// Turn-signal decision logic + the serial link to the ESP32 firmware.
//
// This is the ONLY place that knows the wire protocol the firmware speaks
// (see led_firmware/src/main.cpp): "<DIRECTION>,<PATTERN>\n". A future
// lane-detection decision script talks the same protocol independently --
// this file doesn't need to change for that, and the firmware doesn't
// either.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <string>

#include "dual_logger.h"

enum class Direction { NONE, LEFT, RIGHT };
enum class Pattern { OFF, STEADY, BLINK };

// Distance (meters) from the upcoming maneuver point at which the signal
// starts as STEADY ("upcoming") vs. switches to BLINK ("imminent").
constexpr double STEADY_DISTANCE_M = 300.0;
constexpr double BLINK_DISTANCE_M = 100.0;

// If no GPS fix arrives for this long while a signal is active, force it
// off rather than let it hold a stale "turn now" pattern indefinitely.
constexpr unsigned long STALE_FIX_TIMEOUT_MS = 3000;

// Maps a Google Routes API maneuver string (e.g. "TURN_LEFT", "FORK_RIGHT")
// to a signal side. Maneuvers with no explicit side (STRAIGHT, MERGE,
// ON_RAMP/OFF_RAMP, DEPART, ...) map to NONE -- there's nothing to signal.
Direction maneuverToDirection(const std::string& maneuver);

// Pure decision: given the direction of the upcoming maneuver and the
// distance remaining to it, what should the signal show?
Pattern distanceToPattern(double remaining_m);

// Owns the serial link to the ESP32 and the last-sent state, so callers
// don't need to track either. One instance per run.
class LedSignalLink {
public:
    ~LedSignalLink();

    // Opens the port with DTR/RTS control disabled (most ESP32 boards,
    // including the Feather V2, wire DTR/RTS to EN/BOOT for auto-reset --
    // opening the port can still reset the board once; this gives it time
    // to finish booting before we trust anything it sends back).
    bool connect(const std::string& port_name, DWORD baud_rate, DualLogger& log);

    // Call whenever a fresh GPS fix produces a new (direction, remaining_m)
    // for the upcoming maneuver. Only writes to serial when the resulting
    // (direction, pattern) actually changes. Also resets the stale-fix timer.
    void onFix(Direction direction, double remaining_m, DualLogger& log);

    // Call every loop tick regardless of whether a fix arrived. Forces the
    // signal OFF (once) if too long has passed since the last onFix() call.
    void checkStaleFix(DualLogger& log);

private:
    void send(Direction direction, Pattern pattern, DualLogger& log);

    HANDLE handle_ = nullptr;
    Direction lastDirection_ = Direction::NONE;
    Pattern lastPattern_ = Pattern::OFF;
    unsigned long lastFixTickMs_ = 0;
    bool everFixed_ = false;
};
