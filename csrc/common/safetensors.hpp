// MASt3R Runtime - Safetensors format parser
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
//
// Minimal safetensors parser with no external dependencies.
// Format spec: https://huggingface.co/docs/safetensors/

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mast3r {
namespace safetensors {

// Supported data types
enum class DType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL
};

// Get byte size per element
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:
        case DType::I32:
            return 4;
        case DType::F16:
        case DType::BF16:
        case DType::I16:
            return 2;
        case DType::I64:
            return 8;
        case DType::I8:
        case DType::U8:
        case DType::BOOL:
            return 1;
        default:
            return 0;
    }
}

// Parse dtype from string
inline DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "I16") return DType::I16;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "BOOL") return DType::BOOL;
    throw std::runtime_error("Unknown dtype: " + s);
}

// Tensor metadata
struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    size_t data_offset_start;
    size_t data_offset_end;

    size_t num_elements() const {
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    size_t size_bytes() const {
        return num_elements() * dtype_size(dtype);
    }
};

// Minimal JSON parser for safetensors header
class HeaderParser {
public:
    explicit HeaderParser(const std::string& json) : json_(json), pos_(0) {}

    std::unordered_map<std::string, TensorInfo> parse() {
        std::unordered_map<std::string, TensorInfo> tensors;

        skip_whitespace();
        expect('{');

        while (pos_ < json_.size()) {
            skip_whitespace();
            if (peek() == '}') break;

            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();

            if (key == "__metadata__") {
                // Skip metadata object
                skip_value();
            } else {
                TensorInfo info = parse_tensor_info(key);
                tensors[key] = info;
            }

            skip_whitespace();
            if (peek() == ',') {
                pos_++;
            }
        }

        return tensors;
    }

private:
    const std::string& json_;
    size_t pos_;

    char peek() const {
        return pos_ < json_.size() ? json_[pos_] : '\0';
    }

    void expect(char c) {
        if (peek() != c) {
            throw std::runtime_error(
                std::string("Expected '") + c + "' at position " + std::to_string(pos_)
            );
        }
        pos_++;
    }

    void skip_whitespace() {
        while (pos_ < json_.size() && std::isspace(json_[pos_])) {
            pos_++;
        }
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (pos_ < json_.size() && json_[pos_] != '"') {
            if (json_[pos_] == '\\' && pos_ + 1 < json_.size()) {
                pos_++;  // Skip escape char
            }
            result += json_[pos_++];
        }
        expect('"');
        return result;
    }

    int64_t parse_number() {
        std::string num_str;
        while (pos_ < json_.size() &&
               (std::isdigit(json_[pos_]) || json_[pos_] == '-')) {
            num_str += json_[pos_++];
        }
        return std::stoll(num_str);
    }

    void skip_value() {
        skip_whitespace();
        char c = peek();

        if (c == '{') {
            // Skip object
            int depth = 1;
            pos_++;
            while (pos_ < json_.size() && depth > 0) {
                if (json_[pos_] == '{') depth++;
                else if (json_[pos_] == '}') depth--;
                else if (json_[pos_] == '"') {
                    pos_++;
                    while (pos_ < json_.size() && json_[pos_] != '"') {
                        if (json_[pos_] == '\\') pos_++;
                        pos_++;
                    }
                }
                pos_++;
            }
        } else if (c == '[') {
            // Skip array
            int depth = 1;
            pos_++;
            while (pos_ < json_.size() && depth > 0) {
                if (json_[pos_] == '[') depth++;
                else if (json_[pos_] == ']') depth--;
                pos_++;
            }
        } else if (c == '"') {
            parse_string();
        } else {
            // Skip number/bool/null
            while (pos_ < json_.size() &&
                   json_[pos_] != ',' && json_[pos_] != '}' && json_[pos_] != ']') {
                pos_++;
            }
        }
    }

    TensorInfo parse_tensor_info(const std::string& name) {
        TensorInfo info;
        info.name = name;

        expect('{');

        while (pos_ < json_.size() && peek() != '}') {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();

            if (key == "dtype") {
                info.dtype = parse_dtype(parse_string());
            } else if (key == "shape") {
                info.shape = parse_int_array();
            } else if (key == "data_offsets") {
                auto offsets = parse_int_array();
                if (offsets.size() >= 2) {
                    info.data_offset_start = static_cast<size_t>(offsets[0]);
                    info.data_offset_end = static_cast<size_t>(offsets[1]);
                }
            } else {
                skip_value();
            }

            skip_whitespace();
            if (peek() == ',') pos_++;
        }

        expect('}');
        return info;
    }

    std::vector<int64_t> parse_int_array() {
        std::vector<int64_t> result;
        expect('[');
        skip_whitespace();

        while (peek() != ']') {
            result.push_back(parse_number());
            skip_whitespace();
            if (peek() == ',') {
                pos_++;
                skip_whitespace();
            }
        }

        expect(']');
        return result;
    }
};

// Safetensors file reader
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string& path) : path_(path) {
        load_header();
    }

    // Get all tensor names
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        names.reserve(tensors_.size());
        for (const auto& [name, _] : tensors_) {
            names.push_back(name);
        }
        return names;
    }

    // Get tensor info
    const TensorInfo& tensor_info(const std::string& name) const {
        return tensors_.at(name);
    }

    // Check if tensor exists
    bool has_tensor(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    // Load tensor data as float32 (converts from F16/BF16 if needed)
    std::vector<float> load_tensor_f32(const std::string& name) const {
        const auto& info = tensors_.at(name);

        std::ifstream file(path_, std::ios::binary);
        file.seekg(static_cast<std::streamoff>(data_start_ + info.data_offset_start));

        size_t num_elements = info.num_elements();
        std::vector<float> result(num_elements);

        if (info.dtype == DType::F32) {
            file.read(reinterpret_cast<char*>(result.data()),
                      static_cast<std::streamsize>(num_elements * sizeof(float)));
        } else if (info.dtype == DType::F16) {
            std::vector<uint16_t> f16_data(num_elements);
            file.read(reinterpret_cast<char*>(f16_data.data()),
                      static_cast<std::streamsize>(num_elements * sizeof(uint16_t)));
            for (size_t i = 0; i < num_elements; i++) {
                result[i] = f16_to_f32(f16_data[i]);
            }
        } else if (info.dtype == DType::BF16) {
            std::vector<uint16_t> bf16_data(num_elements);
            file.read(reinterpret_cast<char*>(bf16_data.data()),
                      static_cast<std::streamsize>(num_elements * sizeof(uint16_t)));
            for (size_t i = 0; i < num_elements; i++) {
                result[i] = bf16_to_f32(bf16_data[i]);
            }
        } else {
            throw std::runtime_error("Unsupported dtype for float conversion");
        }

        return result;
    }

    // Load raw tensor bytes
    std::vector<uint8_t> load_tensor_raw(const std::string& name) const {
        const auto& info = tensors_.at(name);
        size_t size = info.data_offset_end - info.data_offset_start;

        std::ifstream file(path_, std::ios::binary);
        file.seekg(static_cast<std::streamoff>(data_start_ + info.data_offset_start));

        std::vector<uint8_t> result(size);
        file.read(reinterpret_cast<char*>(result.data()),
                  static_cast<std::streamsize>(size));

        return result;
    }

    size_t num_tensors() const { return tensors_.size(); }

private:
    std::string path_;
    size_t data_start_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;

    void load_header() {
        std::ifstream file(path_, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + path_);
        }

        // Read header size (8 bytes, little-endian)
        uint64_t header_size = 0;
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

        // Read header JSON
        std::string header_json(header_size, '\0');
        file.read(header_json.data(), static_cast<std::streamsize>(header_size));

        // Parse header
        HeaderParser parser(header_json);
        tensors_ = parser.parse();

        // Data starts after header
        data_start_ = 8 + header_size;
    }

    // IEEE 754 half-precision to single-precision
    static float f16_to_f32(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) {
                // Zero
                uint32_t result = sign;
                float f;
                std::memcpy(&f, &result, sizeof(f));
                return f;
            }
            // Denormalized
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            exp = exp + 127 - 15;
        } else if (exp == 31) {
            // Inf or NaN
            exp = 255;
        } else {
            exp = exp + 127 - 15;
        }

        uint32_t result = sign | (exp << 23) | (mant << 13);
        float f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }

    // BFloat16 to single-precision (just shift)
    static float bf16_to_f32(uint16_t h) {
        uint32_t result = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &result, sizeof(f));
        return f;
    }
};

// Multi-file safetensors loader (for DUNE split files)
class MultiSafetensorsFile {
public:
    MultiSafetensorsFile() = default;

    // Add a file to the collection
    void add_file(const std::string& path) {
        files_.emplace_back(path);
        // Merge tensor info
        for (const auto& name : files_.back().tensor_names()) {
            file_map_[name] = files_.size() - 1;
        }
    }

    // Add multiple files from directory
    void add_directory(const std::string& dir_path) {
        // Common safetensor file names
        std::vector<std::string> names = {"encoder.safetensors", "decoder.safetensors",
                                           "unified.safetensors", "model.safetensors"};
        for (const auto& name : names) {
            std::string full_path = dir_path + "/" + name;
            std::ifstream test(full_path);
            if (test.good()) {
                add_file(full_path);
            }
        }
    }

    bool has_tensor(const std::string& name) const {
        return file_map_.find(name) != file_map_.end();
    }

    std::vector<float> load_tensor_f32(const std::string& name) const {
        auto it = file_map_.find(name);
        if (it == file_map_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return files_[it->second].load_tensor_f32(name);
    }

    const TensorInfo& tensor_info(const std::string& name) const {
        auto it = file_map_.find(name);
        if (it == file_map_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return files_[it->second].tensor_info(name);
    }

    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : file_map_) {
            names.push_back(name);
        }
        return names;
    }

    size_t num_tensors() const { return file_map_.size(); }
    size_t num_files() const { return files_.size(); }

private:
    std::vector<SafetensorsFile> files_;
    std::unordered_map<std::string, size_t> file_map_;  // tensor name -> file index
};

}  // namespace safetensors
}  // namespace mast3r
