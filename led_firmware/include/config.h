#pragma once

// All the "we don't know yet / we'll tune this on the car" knobs live here.
// Nothing outside this file should need to change to answer those questions.

// ---- Hardware ----
constexpr int DATA_PIN = 27;
constexpr int NUM_LEDS = 30;
constexpr uint8_t BRIGHTNESS = 50;  // 0-255. Current-draw safety margin -- raise deliberately, not by accident.

// ---- Serial ----
constexpr unsigned long SERIAL_BAUD = 115200;  // must match platformio.ini monitor_speed and the host script

// ---- Pixel mapping ----
// Physical mounting on the wheel hasn't happened yet, so this mapping is a guess:
// index 0 is the DIN end of the strip (nearest GPIO27), split into two contiguous
// halves. Once mounted, send "TEST,WALK" to light pixels 0->29 one at a time and
// watch which physical LED lights first, then adjust these four numbers to match
// reality. No other code in this file (or the host side) needs to change.
constexpr int LEFT_START = 0;
constexpr int LEFT_COUNT = 3;
constexpr int RIGHT_START = 3;
constexpr int RIGHT_COUNT = 3;

// ---- Appearance ----
// Same color both directions -- which side lights up is what conveys direction,
// same as a real turn signal. Amber.
constexpr uint8_t COLOR_R = 255;
constexpr uint8_t COLOR_G = 90;
constexpr uint8_t COLOR_B = 0;

constexpr unsigned long BLINK_INTERVAL_MS = 400;  // on/off half-period for BLINK mode
constexpr unsigned long WALK_STEP_MS = 150;       // step time for the TEST,WALK diagnostic

// ---- Safety ----
// If no valid command arrives for this long while a pattern is active, force OFF.
// Covers the host process dying or the USB link dropping without a clean NONE,OFF.
constexpr unsigned long COMMAND_TIMEOUT_MS = 5000;
