#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <cuda_fp16.h>
#include "src/utils/string_utils.h"
#include "src/utils/macro.h"
enum Device
{
    CPU_PINNED,
    CPU,
    GPU
};

enum DataType
{
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return DataType::FP32;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return DataType::FP16;
    }
    else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return DataType::INT32;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return DataType::INT8;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return DataType::BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return DataType::BYTES;
    }
    else {
        return DataType::UNSUPPORTED;
    }
}
template<typename T>
class TensorWrapper;

struct Tensor {
    Device              location;
    DataType            dtype;
    std::vector<int>    shape;

    Tensor() = default;

    Tensor(const Device location_, const DataType dtype_, const std::vector<int> shape_):
        location(location_), dtype(dtype_), shape(shape_) {}
    virtual int size() const {
        if (shape.size() == 0) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }
    template<typename T>
    TensorWrapper<T>* as() {
        return static_cast<TensorWrapper<T>*>(this);
    }

    std::string DeviceString() const
    {
        static const std::unordered_map<Device, std::string> devicetring{
            {CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}};
        return devicetring.at(location);
    }

    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"},
            {INT32,"INT32"},
            {FP16, "FP16"},
            {FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str());
    }
};

template<typename T>
class TensorWrapper: public Tensor {
public:
    T* data;

    TensorWrapper(Device location, DataType dtype, std::vector<int> shape):
    	Tensor(location, dtype, shape){}
    
    TensorWrapper(Device location, DataType dtype, std::vector<int> shape, T* data):
    	Tensor(location, dtype, shape),
    	data(data){
            DataType in_dtype = getTensorType<T>();
            LLM_CHECK_WITH_INFO(in_dtype == dtype, "when build TensorWrapper, the passed in data type should be same as dtype in params");
        }
    
    virtual int size() const {
        if (data == nullptr || shape.size() == 0) {
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }

    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"},
            {FP16, "FP16"},
            {FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str(),
                    data);
    }    
};