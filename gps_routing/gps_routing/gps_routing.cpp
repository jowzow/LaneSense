// Merges the two pieces: real GPS fixes now drive the route progress tracker,
// replacing the simulated movement. Origin is your actual current position (from
// the first GPS fix) instead of a hardcoded coordinate - destination is still
// hardcoded for now, since typing/picking a destination is a separate feature.
//
// Include order matters here: curl.h needs to come before windows.h, or you can
// hit a winsock redefinition compile error. WIN32_LEAN_AND_MEAN and NOMINMAX
// keep windows.h from pulling in extra stuff that conflicts with curl/std::.

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <iostream>
#include <fstream> // Added for file logging
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using json = nlohmann::json;

#include "dual_logger.h"
#include "led_signal.h"

static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* out) {
    size_t total = size * nmemb;
    out->append(static_cast<char*>(contents), total);
    return total;
}

double haversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const double R = 6371000.0;
    double phi1 = lat1 * M_PI / 180.0;
    double phi2 = lat2 * M_PI / 180.0;
    double dphi = (lat2 - lat1) * M_PI / 180.0;
    double dlambda = (lon2 - lon1) * M_PI / 180.0;
    double a = std::sin(dphi / 2) * std::sin(dphi / 2)
        + std::cos(phi1) * std::cos(phi2) * std::sin(dlambda / 2) * std::sin(dlambda / 2);
    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
    return R * c;
}

struct RouteStep {
    double start_lat, start_lng;
    double end_lat, end_lng;
    std::string maneuver;
    std::string instructions;
    double distance_m = 0;
};

std::vector<RouteStep> fetchRouteSteps(const std::string& api_key,
    double origin_lat, double origin_lng, double dest_lat, double dest_lng, DualLogger& log) {

    std::vector<RouteStep> steps_out;

    json request_body = {
        {"origin", {{"location", {{"latLng", {{"latitude", origin_lat}, {"longitude", origin_lng}}}}}}},
        {"destination", {{"location", {{"latLng", {{"latitude", dest_lat}, {"longitude", dest_lng}}}}}}},
        {"travelMode", "DRIVE"},
        {"routingPreference", "TRAFFIC_AWARE"}
    };
    std::string body_str = request_body.dump();

    CURL* curl = curl_easy_init();
    if (!curl) return steps_out;

    std::string response_str;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("X-Goog-Api-Key: " + api_key).c_str());
    headers = curl_slist_append(headers,
        "X-Goog-FieldMask: routes.legs.steps.navigationInstruction,routes.legs.steps.distanceMeters,"
        "routes.legs.steps.startLocation,routes.legs.steps.endLocation");

    curl_easy_setopt(curl, CURLOPT_URL, "https://routes.googleapis.com/directions/v2:computeRoutes");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        log << "ERROR (curl): " << curl_easy_strerror(res) << std::endl;
        return steps_out;
    }

    json response = json::parse(response_str, nullptr, false);
    if (response.is_discarded() || !response.contains("routes") || response["routes"].empty()) {
        log << "ERROR: No route in response:\n" << response_str << std::endl;
        return steps_out;
    }

    for (auto& step : response["routes"][0]["legs"][0]["steps"]) {
        RouteStep s;
        s.start_lat = step["startLocation"]["latLng"]["latitude"];
        s.start_lng = step["startLocation"]["latLng"]["longitude"];
        s.end_lat = step["endLocation"]["latLng"]["latitude"];
        s.end_lng = step["endLocation"]["latLng"]["longitude"];
        s.maneuver = step["navigationInstruction"].value("maneuver", "UNKNOWN");
        s.instructions = step["navigationInstruction"].value("instructions", "");
        s.distance_m = step.value("distanceMeters", 0);
        steps_out.push_back(s);
    }
    return steps_out;
}

// --- GPS reading (from the standalone test, unchanged) ---

HANDLE openSerialPort(const std::string& port_name, DWORD baud_rate, DualLogger& log) {
    std::string full_name = "\\\\.\\" + port_name;
    HANDLE h = CreateFileA(full_name.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        log << "ERROR: Failed to open " << port_name << ". Error code: " << GetLastError() << std::endl;
        return nullptr;
    }

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(h, &dcb)) { CloseHandle(h); return nullptr; }
    dcb.BaudRate = baud_rate;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    if (!SetCommState(h, &dcb)) { CloseHandle(h); return nullptr; }

    COMMTIMEOUTS timeouts = {};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    SetCommTimeouts(h, &timeouts);

    return h;
}

bool parseGGA(const std::string& line, double& lat, double& lon, int& fix_quality) {
    if (line.size() < 6) return false;
    std::string tag = line.substr(0, 6);
    if (tag != "$GPGGA" && tag != "$GNGGA") return false;

    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    if (fields.size() < 7) return false;

    fix_quality = fields[6].empty() ? 0 : std::stoi(fields[6]);
    if (fix_quality == 0) return false;
    if (fields[2].empty() || fields[4].empty()) return false;

    double raw_lat = std::stod(fields[2]);
    int lat_deg = static_cast<int>(raw_lat / 100);
    double lat_min = raw_lat - (lat_deg * 100);
    lat = lat_deg + (lat_min / 60.0);
    if (fields[3] == "S") lat = -lat;

    double raw_lon = std::stod(fields[4]);
    int lon_deg = static_cast<int>(raw_lon / 100);
    double lon_min = raw_lon - (lon_deg * 100);
    lon = lon_deg + (lon_min / 60.0);
    if (fields[5] == "W") lon = -lon;

    return true;
}

bool pollForFix(HANDLE serial, std::string& line_buffer, double& lat, double& lng) {
    char read_buf[256];
    DWORD bytes_read;
    if (!ReadFile(serial, read_buf, sizeof(read_buf), &bytes_read, nullptr) || bytes_read == 0) {
        return false;
    }

    for (DWORD i = 0; i < bytes_read; ++i) {
        char c = read_buf[i];
        if (c == '\n') {
            if (!line_buffer.empty() && line_buffer.back() == '\r') line_buffer.pop_back();
            int fix_quality;
            bool got_fix = parseGGA(line_buffer, lat, lng, fix_quality);
            line_buffer.clear();
            if (got_fix) return true;
        }
        else if (c != '\r') {
            line_buffer += c;
        }
    }
    return false;
}

int main() {
    // Initialize our logger - it will create or append to this file in the directory you run from
    DualLogger log("gps_walk_log.txt");

    const std::string API_KEY = "api_key_here";
    const std::string port_name = "COM3";     // GPS receiver <-- CHANGE to match Device Manager
    const std::string led_port_name = "COM4"; // ESP32 (led_firmware/platformio.ini upload_port) <-- CHANGE to match Device Manager

    log << "\n--- Starting new run ---\n";

    double dest_lat = 37.2761, dest_lng = -121.8267; //valley coords

    HANDLE serial = openSerialPort(port_name, 9600, log);
    if (!serial) return -1;

    LedSignalLink ledLink;
    if (!ledLink.connect(led_port_name, 115200, log)) {
        log << "Continuing without LED signaling (ESP32 link failed to open)." << std::endl;
    }

    std::string line_buffer;
    double current_lat = 0, current_lng = 0;

    log << "Waiting for first GPS fix before requesting a route..." << std::endl;
    while (!pollForFix(serial, line_buffer, current_lat, current_lng)) {
        // keep polling
    }
    log << "Got first fix: " << current_lat << ", " << current_lng << "\n" << std::endl;

    std::vector<RouteStep> steps = fetchRouteSteps(API_KEY, current_lat, current_lng, dest_lat, dest_lng, log);
    if (steps.empty()) {
        log << "Failed to get route - fix that before continuing." << std::endl;
        CloseHandle(serial);
        return -1;
    }
    log << "Got route with " << steps.size() << " steps.\n" << std::endl;

    size_t current_step = 0;
    const double ARRIVAL_RADIUS_M = 25.0;

    while (current_step < steps.size()) {
        if (pollForFix(serial, line_buffer, current_lat, current_lng)) {
            const RouteStep& step = steps[current_step];
            double remaining_m = haversineMeters(current_lat, current_lng, step.end_lat, step.end_lng);

            log << "Step " << current_step << "/" << steps.size() - 1
                << " [" << step.maneuver << "] "
                << static_cast<int>(remaining_m) << "m remaining - " << step.instructions << std::endl;

            // The maneuver at the point we're approaching (end of the current step) is
            // described by the NEXT step, not this one -- steps[i].maneuver is how you
            // entered step i. Confirmed against gps_walk_log.txt: DEPART at step 0, then
            // ">>> Reached maneuver point. Next: TURN_LEFT" matching steps[1].maneuver.
            bool hasNext = (current_step + 1) < steps.size();
            Direction upcomingDirection = hasNext ? maneuverToDirection(steps[current_step + 1].maneuver) : Direction::NONE;
            ledLink.onFix(upcomingDirection, remaining_m, log);

            if (remaining_m < ARRIVAL_RADIUS_M) {
                current_step++;
                if (current_step < steps.size()) {
                    log << ">>> Reached maneuver point. Next: " << steps[current_step].maneuver
                        << " - " << steps[current_step].instructions << "\n" << std::endl;
                }
            }
        }
        ledLink.checkStaleFix(log);
    }

    log << "\nArrived." << std::endl;
    CloseHandle(serial);
    return 0;
}