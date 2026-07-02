// data_transmitter.hpp
//
// C++ translation of the Python DataTransmitter / SharedMemoryManager module.
//
// Dependencies:
//   - libzmq + cppzmq        (sudo apt install libzmq3-dev, header-only cppzmq)
//   - OpenCV                 (sudo apt install libopencv-dev)
//   - nlohmann/json          (sudo apt install nlohmann-json3-dev)
//   - POSIX shared memory    (shm_open/mmap, part of librt/libpthread on Linux)
//
// Link with: -lzmq -lopencv_core -lopencv_imgcodecs -lpthread -lrt

#pragma once

#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <iostream>
#include <cstring>
#include <memory>

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────
namespace params {
    constexpr int H = 480;
    constexpr int W = 848;
    constexpr int C = 3;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Memory Manager
// ─────────────────────────────────────────────────────────────────────────────
class SharedMemoryManager {
public:
    SharedMemoryManager(const std::string& name, size_t size, bool create)
        : name_(name), size_(size), create_(create) {
        open();
    }

    ~SharedMemoryManager() {
        try {
            shutdown();
        } catch (...) {
            // swallow, mirroring Python's __del__ best-effort cleanup
        }
    }

    // Non-copyable, movable
    SharedMemoryManager(const SharedMemoryManager&) = delete;
    SharedMemoryManager& operator=(const SharedMemoryManager&) = delete;

    SharedMemoryManager(SharedMemoryManager&& other) noexcept { *this = std::move(other); }
    SharedMemoryManager& operator=(SharedMemoryManager&& other) noexcept {
        if (this != &other) {
            shutdown();
            name_ = std::move(other.name_);
            size_ = other.size_;
            create_ = other.create_;
            fd_ = other.fd_;
            ptr_ = other.ptr_;
            other.fd_ = -1;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // ── Shutdown block ───────────────────────────────────────────────────────
    void close() {
        if (ptr_ != nullptr) {
            munmap(ptr_, size_);
            ptr_ = nullptr;
        }
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void unlink() {
        if (!create_) return;
        shm_unlink(("/" + name_).c_str()); // ignore errors, mirrors Python's broad except
    }

    void shutdown() {
        close();
        unlink();
    }

    // ── Accessors ────────────────────────────────────────────────────────────
    unsigned char* buf() {
        if (ptr_ == nullptr) {
            throw std::runtime_error("SharedMemory '" + name_ + "' is not open.");
        }
        return static_cast<unsigned char*>(ptr_);
    }

    size_t size() const { return size_; }

private:
    std::string name_;
    size_t size_;
    bool create_;
    int fd_ = -1;
    void* ptr_ = nullptr;

    void open() {
        if (create_) {
            create_or_replace();
        } else {
            attach();
        }
    }

    void create_or_replace() {
        std::string posix_name = "/" + name_;
        fd_ = shm_open(posix_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
        if (fd_ == -1 && errno == EEXIST) {
            // Stale segment from a previous crashed run - unlink and retry,
            // mirroring the Python FileExistsError fallback.
            shm_unlink(posix_name.c_str());
            fd_ = shm_open(posix_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
        }
        if (fd_ == -1) {
            throw std::runtime_error("Failed to create shared memory: " + name_);
        }
        if (ftruncate(fd_, static_cast<off_t>(size_)) == -1) {
            throw std::runtime_error("Failed to size shared memory: " + name_);
        }
        map();
    }

    void attach() {
        std::string posix_name = "/" + name_;
        bool attached = false;
        while (!attached) {
            fd_ = shm_open(posix_name.c_str(), O_RDWR, 0666);
            if (fd_ != -1) {
                attached = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        map();
    }

    void map() {
        ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            throw std::runtime_error("mmap failed for shared memory: " + name_);
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Data transmitter
// ─────────────────────────────────────────────────────────────────────────────
class DataTransmitter {
public:
    enum class Mode { Sender, Receiver };

    DataTransmitter(Mode mode, int device_id, const std::string& topic, int port = 6000)
        : mode_(mode),
          device_id_(device_id),
          port_(port + device_id),
          topic_(topic),
          nbytes_(static_cast<size_t>(params::H) * params::W * params::C) {
        if (mode_ == Mode::Sender) {
            setup_zmq_sender();
            setup_shm_sender();
        } else {
            setup_zmq_receiver();
            setup_shm_receiver();
        }
    }

    ~DataTransmitter() {
        try {
            shutdown();
        } catch (...) {
        }
    }

    DataTransmitter(const DataTransmitter&) = delete;
    DataTransmitter& operator=(const DataTransmitter&) = delete;

    // ── Send block (sender mode only) ───────────────────────────────────────
    void send_frame(const cv::Mat& frame) {
        require(Mode::Sender);
        size_t n = frame.total() * frame.elemSize();
        if (n != nbytes_) {
            throw std::runtime_error("Frame size does not match shared memory buffer size");
        }
        std::memcpy(shm_->buf(), frame.data, n);
    }

    void send_skeleton_data(const std::vector<std::array<double, 3>>& skeleton,
                             const std::vector<double>& confidence) {
        require(Mode::Sender);

        json skel_json = json::array();
        for (const auto& pt : skeleton) {
            skel_json.push_back({pt[0], pt[1], pt[2]});
        }
        json conf_json = confidence;

        std::string msg = topic_ + "_" + std::to_string(device_id_) + "; " +
                           skel_json.dump() + "; " + conf_json.dump();

        zmq::message_t zmsg(msg.size());
        std::memcpy(zmsg.data(), msg.data(), msg.size());
        socket_->send(zmsg, zmq::send_flags::none);
    }

    void send_rula_score(const std::array<int, 2>& score) {
        require(Mode::Sender);

        json score_json = json::array();
        score_json.push_back({score[0], score[1]});

        std::string msg = topic_ + "_" + std::to_string(device_id_) + "; " +
                           score_json.dump();

        zmq::message_t zmsg(msg.size());
        std::memcpy(zmsg.data(), msg.data(), msg.size());
        socket_->send(zmsg, zmq::send_flags::none);
    }

    // ── Receive block (receiver mode only) ──────────────────────────────────
    cv::Mat receive_raw_frame() {
        require(Mode::Receiver);
        cv::Mat frame(params::H, params::W, CV_8UC3);
        std::memcpy(frame.data, shm_->buf(), nbytes_);
        return frame;
    }

    std::string receive_packed_msg() {
        require(Mode::Receiver);
        zmq::message_t zmsg;
        auto result = socket_->recv(zmsg, zmq::recv_flags::none);
        (void)result;
        return std::string(static_cast<char*>(zmsg.data()), zmsg.size());
    }

    // Returns a base64 data-URI JPEG, equivalent to Python's receive_frames()
    std::string receive_frame_b64() {
        require(Mode::Receiver);
        return cv2_to_b64(receive_raw_frame());
    }

    // skeleton, confidence
    std::pair<std::vector<std::array<double, 3>>, std::vector<double>> receive_skeleton_data() {
        require(Mode::Receiver);
        std::string packed = receive_packed_msg();

        size_t first  = packed.find("; ");
        size_t second = packed.find("; ", first + 2);
        std::string skeleton_packed   = sanitize_nan(packed.substr(first + 2, second - (first + 2)));
        std::string confidence_packed = sanitize_nan(packed.substr(second + 2));

        json skel_json = json::parse(skeleton_packed);
        json conf_json = json::parse(confidence_packed);

        std::vector<std::array<double, 3>> skeleton;
        for (const auto& pt : skel_json) {
            skeleton.push_back({
                json_to_double(pt[0]),
                json_to_double(pt[1]),
                json_to_double(pt[2])
            });
        }

        std::vector<double> confidence;
        for (const auto& v : conf_json) {
            confidence.push_back(json_to_double(v));
        }

        return {skeleton, confidence};
    }

    std::array<int, 2> receive_rula_score() {
        require(Mode::Receiver);
        std::string packed = receive_packed_msg();

        size_t first  = packed.find("; ");
        size_t second = packed.find("; ", first + 2);
        std::string score_packed   = sanitize_nan(packed.substr(first + 2, second - (first + 2)));

        json score_json = json::parse(score_packed);
        std::array<int, 2> score = {json_to_int(score_json[0][0]), json_to_int(score_json[0][1])};

        return score;
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────
    void shutdown() {
        if (socket_) {
            socket_->close();
            socket_.reset();
        }
        if (shm_) {
            shm_->shutdown();
            shm_.reset();
        }
    }

private:
    Mode mode_;
    int device_id_;
    int port_;
    std::string topic_;
    size_t nbytes_;

    zmq::context_t ctx_{1};
    std::unique_ptr<zmq::socket_t> socket_;
    std::unique_ptr<SharedMemoryManager> shm_;

    void require(Mode expected) {
        if (mode_ != expected) {
            throw std::runtime_error("DataTransmitter method called in wrong mode");
        }
    }

    // ── ZeroMQ setup ─────────────────────────────────────────────────────────
    void setup_zmq_sender() {
        try {
            socket_ = std::make_unique<zmq::socket_t>(ctx_, zmq::socket_type::pub);
            int linger = 0;
            socket_->set(zmq::sockopt::linger, linger);
            socket_->set(zmq::sockopt::sndhwm, 1);
            socket_->bind("tcp://*:" + std::to_string(port_));
        } catch (const std::exception& e) {
            // mirrors Python's broad "except Exception: pass"
            std::cerr << "ZMQ sender setup failed: " << e.what() << std::endl;
            socket_.reset();
        }
    }

    void setup_zmq_receiver() {
        socket_ = std::make_unique<zmq::socket_t>(ctx_, zmq::socket_type::sub);
        socket_->set(zmq::sockopt::conflate, 1);
        std::string filter = topic_ + "_" + std::to_string(device_id_);
        socket_->set(zmq::sockopt::subscribe, filter);
        socket_->connect("tcp://localhost:" + std::to_string(port_));
    }

    // ── Shared memory setup ──────────────────────────────────────────────────
    void setup_shm_sender() {
        shm_ = std::make_unique<SharedMemoryManager>(
            "shared_image" + std::to_string(device_id_), nbytes_, /*create=*/true);
    }

    void setup_shm_receiver() {
        shm_ = std::make_unique<SharedMemoryManager>(
            "shared_image" + std::to_string(device_id_), nbytes_, /*create=*/false);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    static std::string base64_encode(const std::vector<uchar>& data) {
        static const char* table =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(((data.size() + 2) / 3) * 4);
        size_t i = 0;
        while (i + 3 <= data.size()) {
            unsigned int n = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
            out += table[(n >> 18) & 0x3F];
            out += table[(n >> 12) & 0x3F];
            out += table[(n >> 6) & 0x3F];
            out += table[n & 0x3F];
            i += 3;
        }
        size_t rem = data.size() - i;
        if (rem == 1) {
            unsigned int n = data[i] << 16;
            out += table[(n >> 18) & 0x3F];
            out += table[(n >> 12) & 0x3F];
            out += "==";
        } else if (rem == 2) {
            unsigned int n = (data[i] << 16) | (data[i + 1] << 8);
            out += table[(n >> 18) & 0x3F];
            out += table[(n >> 12) & 0x3F];
            out += table[(n >> 6) & 0x3F];
            out += "=";
        }
        return out;
    }

    static std::string cv2_to_b64(const cv::Mat& img) {
        std::vector<uchar> buffer;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
        bool ok = cv::imencode(".jpg", img, buffer, params);
        if (!ok) return "";
        return "data:image/jpeg;base64," + base64_encode(buffer);
    }

    static std::string sanitize_nan(std::string s) {
        // word-boundary-aware replace: NaN not preceded/followed by alnum/_
        static const std::string from = "NaN";
        static const std::string to   = "null";
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            bool left_ok  = (pos == 0)              || !std::isalnum(s[pos - 1]);
            bool right_ok = (pos + 3 >= s.size())   || !std::isalnum(s[pos + 3]);
            if (left_ok && right_ok) {
                s.replace(pos, 3, to);
                pos += to.size();
            } else {
                pos += from.size();
            }
        }
        return s;
    }

    static double json_to_double(const json& v) {
        if (v.is_null()) return std::numeric_limits<double>::quiet_NaN();
        return v.get<double>();
    }

    static int json_to_int(const json& v) {
        if (v.is_null()) return std::numeric_limits<int>::quiet_NaN();
        return v.get<int>();
    }
};
