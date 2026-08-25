#include "led_signal.h"

static const char* directionToken(Direction d) {
    switch (d) {
        case Direction::LEFT: return "LEFT";
        case Direction::RIGHT: return "RIGHT";
        default: return "NONE";
    }
}

static const char* patternToken(Pattern p) {
    switch (p) {
        case Pattern::STEADY: return "STEADY";
        case Pattern::BLINK: return "BLINK";
        default: return "OFF";
    }
}

Direction maneuverToDirection(const std::string& maneuver) {
    if (maneuver.find("LEFT") != std::string::npos) return Direction::LEFT;
    if (maneuver.find("RIGHT") != std::string::npos) return Direction::RIGHT;
    return Direction::NONE;
}

Pattern distanceToPattern(double remaining_m) {
    if (remaining_m <= BLINK_DISTANCE_M) return Pattern::BLINK;
    if (remaining_m <= STEADY_DISTANCE_M) return Pattern::STEADY;
    return Pattern::OFF;
}

LedSignalLink::~LedSignalLink() {
    if (handle_) CloseHandle(handle_);
}

bool LedSignalLink::connect(const std::string& port_name, DWORD baud_rate, DualLogger& log) {
    std::string full_name = "\\\\.\\" + port_name;
    handle_ = CreateFileA(full_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
        log << "ERROR: Failed to open LED port " << port_name << ". Error code: " << GetLastError() << std::endl;
        handle_ = nullptr;
        return false;
    }

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(handle_, &dcb)) { CloseHandle(handle_); handle_ = nullptr; return false; }
    dcb.BaudRate = baud_rate;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fDtrControl = DTR_CONTROL_DISABLE;
    dcb.fRtsControl = RTS_CONTROL_DISABLE;
    if (!SetCommState(handle_, &dcb)) { CloseHandle(handle_); handle_ = nullptr; return false; }

    COMMTIMEOUTS timeouts = {};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    SetCommTimeouts(handle_, &timeouts);

    log << "LED link: opened " << port_name << ", waiting for the ESP32 to finish booting..." << std::endl;
    // Opening the port can still pulse the board's auto-reset line depending on the
    // driver, even with DTR/RTS control disabled above. Give it time to boot rather
    // than sending a command into the boot ROM's garbage output.
    Sleep(2000);

    return true;
}

void LedSignalLink::send(Direction direction, Pattern pattern, DualLogger& log) {
    if (!handle_) return;

    std::string line = std::string(directionToken(direction)) + "," + patternToken(pattern) + "\n";
    DWORD written = 0;
    if (!WriteFile(handle_, line.c_str(), static_cast<DWORD>(line.size()), &written, nullptr)) {
        log << "ERROR: LED write failed. Error code: " << GetLastError() << std::endl;
        return;
    }

    char reply[128] = {};
    DWORD read = 0;
    if (ReadFile(handle_, reply, sizeof(reply) - 1, &read, nullptr) && read > 0) {
        log << "LED <- " << line.substr(0, line.size() - 1) << "  LED -> " << std::string(reply, read) << std::endl;
    } else {
        log << "LED <- " << line.substr(0, line.size() - 1) << "  (no ack)" << std::endl;
    }
}

void LedSignalLink::onFix(Direction direction, double remaining_m, DualLogger& log) {
    everFixed_ = true;
    lastFixTickMs_ = GetTickCount();

    Pattern pattern = (direction == Direction::NONE) ? Pattern::OFF : distanceToPattern(remaining_m);

    if (direction != lastDirection_ || pattern != lastPattern_) {
        send(direction, pattern, log);
        lastDirection_ = direction;
        lastPattern_ = pattern;
    }
}

void LedSignalLink::checkStaleFix(DualLogger& log) {
    if (!everFixed_) return;
    if (lastPattern_ == Pattern::OFF) return;
    if (GetTickCount() - lastFixTickMs_ < STALE_FIX_TIMEOUT_MS) return;

    log << "LED link: no GPS fix for " << STALE_FIX_TIMEOUT_MS << "ms, forcing signal off." << std::endl;
    send(Direction::NONE, Pattern::OFF, log);
    lastDirection_ = Direction::NONE;
    lastPattern_ = Pattern::OFF;
}
