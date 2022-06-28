/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
#define TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include "tensorflow/stream_executor/data_type.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {

// Allows to represent a value that is either a host scalar or a scalar stored
// on the GPU device.
// See also the specialization for ElemT=void below.
template <typename ElemT>
class HostOrDeviceScalar {
 public:
  // Not marked as explicit because when using this constructor, we usually want
  // to set this to a compile-time constant.
  HostOrDeviceScalar(ElemT value) : value_(value), is_pointer_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_0(mht_0_v, 202, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  explicit HostOrDeviceScalar(const DeviceMemory<ElemT>& pointer)
      : pointer_(pointer), is_pointer_(true) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_1(mht_1_v, 207, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");

    CHECK_EQ(1, pointer.ElementCount());
  }

  bool is_pointer() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_2(mht_2_v, 214, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "is_pointer");
 return is_pointer_; }
  const DeviceMemory<ElemT>& pointer() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_3(mht_3_v, 218, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "pointer");

    CHECK(is_pointer());
    return pointer_;
  }
  const ElemT& value() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_4(mht_4_v, 225, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "value");

    CHECK(!is_pointer());
    return value_;
  }

 private:
  union {
    ElemT value_;
    DeviceMemory<ElemT> pointer_;
  };
  bool is_pointer_;
};

// Specialization for wrapping a dynamically-typed value (via type erasure).
template <>
class HostOrDeviceScalar<void> {
 public:
  using DataType = dnn::DataType;

  // Constructors not marked as explicit because when using this constructor, we
  // usually want to set this to a compile-time constant.

  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(float value)
      : float_(value), is_pointer_(false), dtype_(DataType::kFloat) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_5(mht_5_v, 252, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(double value)
      : double_(value), is_pointer_(false), dtype_(DataType::kDouble) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_6(mht_6_v, 258, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(Eigen::half value)
      : half_(value), is_pointer_(false), dtype_(DataType::kHalf) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_7(mht_7_v, 264, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(int8 value)
      : int8_(value), is_pointer_(false), dtype_(DataType::kInt8) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_8(mht_8_v, 270, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(int32 value)
      : int32_(value), is_pointer_(false), dtype_(DataType::kInt32) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_9(mht_9_v, 276, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(std::complex<float> value)
      : complex_float_(value),
        is_pointer_(false),
        dtype_(DataType::kComplexFloat) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_10(mht_10_v, 284, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(std::complex<double> value)
      : complex_double_(value),
        is_pointer_(false),
        dtype_(DataType::kComplexDouble) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_11(mht_11_v, 292, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");
}
  template <typename T>
  explicit HostOrDeviceScalar(const DeviceMemory<T>& pointer)
      : pointer_(pointer),
        is_pointer_(true),
        dtype_(dnn::ToDataType<T>::value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_12(mht_12_v, 300, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar");

    CHECK_EQ(1, pointer.ElementCount());
  }
  // Construct from statically-typed version.
  template <typename T, typename std::enable_if<!std::is_same<T, void>::value,
                                                int>::type = 0>
  // NOLINTNEXTLINE google-explicit-constructor
  HostOrDeviceScalar(const HostOrDeviceScalar<T>& other) {
    if (other.is_pointer()) {
      *this = HostOrDeviceScalar(other.pointer());
    } else {
      *this = HostOrDeviceScalar(other.value());
    }
  }

  bool is_pointer() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_13(mht_13_v, 318, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "is_pointer");
 return is_pointer_; }
  template <typename T>
  const DeviceMemory<T>& pointer() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_14(mht_14_v, 323, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "pointer");

    CHECK(is_pointer());
    CHECK(dtype_ == dnn::ToDataType<T>::value);
    return pointer_;
  }
  template <typename T>
  const T& value() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_15(mht_15_v, 332, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "value");

    CHECK(!is_pointer());
    CHECK(dtype_ == dnn::ToDataType<T>::value);
    return value_impl<T>();
  }
  const DeviceMemoryBase& opaque_pointer() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_16(mht_16_v, 340, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "opaque_pointer");

    CHECK(is_pointer());
    return pointer_;
  }
  const void* opaque_value() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_17(mht_17_v, 347, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "opaque_value");

    CHECK(!is_pointer());
    switch (dtype_) {
      case DataType::kFloat:
        return &float_;
      case DataType::kDouble:
        return &double_;
      case DataType::kHalf:
        return &half_;
      case DataType::kInt8:
        return &int8_;
      case DataType::kInt32:
        return &int32_;
      case DataType::kComplexFloat:
        return &complex_float_;
      case DataType::kComplexDouble:
        return &complex_double_;
      default:
        return nullptr;
    }
  }
  DataType data_type() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_18(mht_18_v, 371, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "data_type");
 return dtype_; }

 private:
  template <typename T>
  const T& value_impl() const;

  union {
    float float_;
    double double_;
    Eigen::half half_;
    int8 int8_;
    int32 int32_;
    std::complex<float> complex_float_;
    std::complex<double> complex_double_;
    DeviceMemoryBase pointer_;
  };
  bool is_pointer_;
  DataType dtype_;
};

template <>
inline const float& HostOrDeviceScalar<void>::value_impl<float>() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_19(mht_19_v, 395, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<float>");

  return float_;
}

template <>
inline const double& HostOrDeviceScalar<void>::value_impl<double>() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_20(mht_20_v, 403, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<double>");

  return double_;
}

template <>
inline const Eigen::half& HostOrDeviceScalar<void>::value_impl<Eigen::half>()
    const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_21(mht_21_v, 412, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<Eigen::half>");

  return half_;
}

template <>
inline const int8& HostOrDeviceScalar<void>::value_impl<int8>() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_22(mht_22_v, 420, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<int8>");

  return int8_;
}

template <>
inline const int32& HostOrDeviceScalar<void>::value_impl<int32>() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_23(mht_23_v, 428, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<int32>");

  return int32_;
}

template <>
inline const std::complex<float>&
HostOrDeviceScalar<void>::value_impl<std::complex<float>>() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_24(mht_24_v, 437, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<std::complex<float>>");

  return complex_float_;
}

template <>
inline const std::complex<double>&
HostOrDeviceScalar<void>::value_impl<std::complex<double>>() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSstream_executorPShost_or_device_scalarDTh mht_25(mht_25_v, 446, "", "./tensorflow/stream_executor/host_or_device_scalar.h", "HostOrDeviceScalar<void>::value_impl<std::complex<double>>");

  return complex_double_;
}

}  // namespace stream_executor
#endif  // TENSORFLOW_STREAM_EXECUTOR_HOST_OR_DEVICE_SCALAR_H_
