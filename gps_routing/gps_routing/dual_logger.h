#pragma once

#include <iostream>
#include <fstream>
#include <string>

// Writes to both the console (std::cout) and a text file simultaneously.
class DualLogger {
    std::ofstream log_file;
public:
    DualLogger(const std::string& filename) {
        // ios::app appends to the file so you don't overwrite previous runs
        log_file.open(filename, std::ios::out | std::ios::app);
    }

    template <typename T>
    DualLogger& operator<<(const T& x) {
        std::cout << x;
        if (log_file.is_open()) {
            log_file << x;
            log_file.flush(); // Force write to disk immediately (crash protection)
        }
        return *this;
    }

    // Special overload to handle std::endl
    DualLogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
        std::cout << manip;
        if (log_file.is_open()) {
            log_file << manip;
            log_file.flush();
        }
        return *this;
    }
};
