// Serial-driven turn-signal renderer. This file only knows how to render a
// (direction, pattern) pair onto the strip -- it has no idea why a turn is
// coming up. That decision lives entirely on the host side (GPS/routing today,
// lane detection later), which is the point: a future decision-logic script
// can drive this exact protocol without any firmware changes.
//
// Protocol (one command per line, comma-separated, newline-terminated):
//   <DIRECTION>,<PATTERN>   DIRECTION = LEFT | RIGHT | NONE
//                           PATTERN   = OFF | STEADY | BLINK
//   TEST,WALK               lights pixels 0..29 one at a time, for confirming
//                            physical mounting orientation
// Every command gets one reply line: "OK ..." or "ERR ...".
//
// Serial is read non-blocking (char-by-char via Serial.available()) so a
// command is never missed while a blink or walk step is in progress -- no
// delay() anywhere in loop().

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include "config.h"

Adafruit_NeoPixel strip(NUM_LEDS, DATA_PIN, NEO_GRB + NEO_KHZ800);

enum class Direction { NONE, LEFT, RIGHT };
enum class Pattern { OFF, STEADY, BLINK, WALK };

static Direction currentDirection = Direction::NONE;
static Pattern currentPattern = Pattern::OFF;

static bool blinkOn = true;
static unsigned long blinkTimer = 0;

static int walkIndex = 0;
static unsigned long walkTimer = 0;

static unsigned long lastCommandMs = 0;
static bool needsRender = true;

static const int LINE_BUF_SIZE = 32;
static char lineBuf[LINE_BUF_SIZE];
static int lineLen = 0;

static void applyPixels(bool lightOn) {
  strip.clear();
  if (lightOn) {
    int start = (currentDirection == Direction::LEFT) ? LEFT_START
              : (currentDirection == Direction::RIGHT) ? RIGHT_START
              : -1;
    int count = (currentDirection == Direction::LEFT) ? LEFT_COUNT
              : (currentDirection == Direction::RIGHT) ? RIGHT_COUNT
              : 0;
    for (int i = start; i < start + count; i++) {
      strip.setPixelColor(i, strip.Color(COLOR_R, COLOR_G, COLOR_B));
    }
  }
  strip.show();
}

// Applies a freshly-parsed, valid (direction, pattern) command: resets blink
// phase so the change is visible immediately instead of waiting out a
// half-finished blink interval.
static void setState(Direction dir, Pattern pat) {
  currentDirection = dir;
  currentPattern = pat;
  blinkOn = true;
  blinkTimer = millis();
  needsRender = true;
}

static bool parseDirection(const char* tok, Direction& out) {
  if (strcmp(tok, "LEFT") == 0) { out = Direction::LEFT; return true; }
  if (strcmp(tok, "RIGHT") == 0) { out = Direction::RIGHT; return true; }
  if (strcmp(tok, "NONE") == 0) { out = Direction::NONE; return true; }
  return false;
}

static bool parsePattern(const char* tok, Pattern& out) {
  if (strcmp(tok, "OFF") == 0) { out = Pattern::OFF; return true; }
  if (strcmp(tok, "STEADY") == 0) { out = Pattern::STEADY; return true; }
  if (strcmp(tok, "BLINK") == 0) { out = Pattern::BLINK; return true; }
  return false;
}

static void handleLine(char* line) {
  // strtok mutates the buffer in place (writes '\0' over each comma), so grab
  // a copy up front -- otherwise the ERR message below would only show the
  // first token instead of the full line that was actually received.
  char original[LINE_BUF_SIZE];
  strncpy(original, line, LINE_BUF_SIZE - 1);
  original[LINE_BUF_SIZE - 1] = '\0';

  char* tok1 = strtok(line, ",");
  char* tok2 = tok1 ? strtok(nullptr, ",") : nullptr;

  if (tok1 && tok2 && strcmp(tok1, "TEST") == 0 && strcmp(tok2, "WALK") == 0) {
    currentPattern = Pattern::WALK;
    walkIndex = 0;
    walkTimer = millis();
    lastCommandMs = millis();
    Serial.println("OK TEST WALK");
    return;
  }

  Direction dir;
  Pattern pat;
  if (tok1 && tok2 && parseDirection(tok1, dir) && parsePattern(tok2, pat)) {
    setState(dir, pat);
    lastCommandMs = millis();
    Serial.print("OK ");
    Serial.print(tok1);
    Serial.print(" ");
    Serial.println(tok2);
    return;
  }

  Serial.print("ERR unknown command: ");
  Serial.println(original);
}

static void pollSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      // Accept CR, LF, or CRLF as the line ending -- terminal/monitor tools
      // differ on which they send. The second byte of a CRLF pair finds
      // lineLen == 0 and is harmlessly ignored here.
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuf[lineLen++] = c;
    } else {
      // line too long for the buffer -- drop it rather than act on a truncated command
      lineLen = 0;
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  strip.clear();
  strip.show();
  lastCommandMs = millis();
}

void loop() {
  pollSerial();
  unsigned long now = millis();

  // Watchdog: if we're mid-pattern and haven't heard from the host in a
  // while (process died, USB dropped), don't sit there blinking forever.
  if (currentPattern != Pattern::OFF && (now - lastCommandMs > COMMAND_TIMEOUT_MS)) {
    currentDirection = Direction::NONE;
    currentPattern = Pattern::OFF;
    needsRender = true;
    Serial.println("ERR command timeout, forcing OFF");
  }

  if (currentPattern == Pattern::WALK) {
    if (now - walkTimer >= WALK_STEP_MS) {
      walkTimer = now;
      strip.clear();
      strip.setPixelColor(walkIndex, strip.Color(COLOR_R, COLOR_G, COLOR_B));
      strip.show();
      walkIndex++;
      if (walkIndex >= NUM_LEDS) {
        currentPattern = Pattern::OFF;
        needsRender = true;
      }
    }
  } else if (currentPattern == Pattern::BLINK) {
    if (now - blinkTimer >= BLINK_INTERVAL_MS) {
      blinkTimer = now;
      blinkOn = !blinkOn;
      needsRender = true;
    }
  }

  if (needsRender) {
    bool lightOn = (currentPattern == Pattern::STEADY) ||
                   (currentPattern == Pattern::BLINK && blinkOn);
    applyPixels(lightOn);
    needsRender = false;
  }
}
