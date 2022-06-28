/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_TYPES_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh() {
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


#include <map>
#include <set>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Variant;

// MemoryType is used to describe whether input or output Tensors of
// an OpKernel should reside in "Host memory" (e.g., CPU memory) or
// "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
// devices).
enum MemoryType {
  DEVICE_MEMORY = 0,
  HOST_MEMORY = 1,
};

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType {
 public:
  DeviceType(const char* type)  // NOLINT(runtime/explicit)
      : type_(type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_0(mht_0_v, 228, "", "./tensorflow/core/framework/types.h", "DeviceType");
}

  explicit DeviceType(StringPiece type) : type_(type.data(), type.size()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_1(mht_1_v, 233, "", "./tensorflow/core/framework/types.h", "DeviceType");
}

  const char* type() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_2(mht_2_v, 238, "", "./tensorflow/core/framework/types.h", "type");
 return type_.c_str(); }
  const std::string& type_string() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_3(mht_3_v, 242, "", "./tensorflow/core/framework/types.h", "type_string");
 return type_; }

  bool operator<(const DeviceType& other) const;
  bool operator==(const DeviceType& other) const;
  bool operator!=(const DeviceType& other) const { return !(*this == other); }

 private:
  std::string type_;
};
std::ostream& operator<<(std::ostream& os, const DeviceType& d);

// Convenient constants that can be passed to a DeviceType constructor
TF_EXPORT extern const char* const DEVICE_DEFAULT;     // "DEFAULT"
TF_EXPORT extern const char* const DEVICE_CPU;         // "CPU"
TF_EXPORT extern const char* const DEVICE_GPU;         // "GPU"
TF_EXPORT extern const char* const DEVICE_TPU;         // "TPU"
TF_EXPORT extern const char* const DEVICE_TPU_SYSTEM;  // "TPU_SYSTEM"

template <typename Device>
struct DeviceName {};

template <>
struct DeviceName<Eigen::ThreadPoolDevice> {
  static const std::string value;
};

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
template <>
struct DeviceName<Eigen::GpuDevice> {
  static const std::string value;
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;

typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;

typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;
typedef gtl::InlinedVector<std::pair<DeviceType, int32>, 4>
    PrioritizedDeviceTypeVector;

// Convert the enums to strings for errors:
std::string DataTypeString(DataType dtype);
std::string DeviceTypeString(const DeviceType& device_type);
std::string DataTypeSliceString(const DataTypeSlice dtypes);
inline std::string DataTypeVectorString(const DataTypeVector& dtypes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_4(mht_4_v, 294, "", "./tensorflow/core/framework/types.h", "DataTypeVectorString");

  return DataTypeSliceString(dtypes);
}

// DataTypeSet represents a set of DataType values as a simple and efficient
// bit mask.  Note that DataTypeSet cannot represent all DataType values; it
// cannot represent any of the DT_*_REF values.
class DataTypeSet {
 private:
  const uint32 mask_;

  static constexpr uint32 kNumBits = 32;

 public:
  constexpr DataTypeSet(const DataTypeSet& other) : mask_(other.mask_) {}
  explicit constexpr DataTypeSet(uint32 mask) : mask_(mask) {}

  constexpr bool Contains(DataType dt) const {
    return (static_cast<uint32>(dt) < kNumBits) &&
           ((mask_ >> static_cast<uint32>(dt)) & 1u) != 0u;
  }

  class Iterator {
    const DataTypeSet& set_;
    uint32 pos_;

   public:
    Iterator(const DataTypeSet& set, uint32 pos) : set_(set), pos_(pos) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_5(mht_5_v, 324, "", "./tensorflow/core/framework/types.h", "Iterator");

      DCHECK_LE(pos, kNumBits);
    }
    DataType operator*() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_6(mht_6_v, 330, "", "./tensorflow/core/framework/types.h", "*");
 return static_cast<DataType>(pos_); }
    Iterator& operator++() {
      ++pos_;
      DCHECK_LE(pos_, kNumBits);
      if (pos_ < kNumBits) {
        uint32 remaining_mask = set_.mask_ >> pos_;
        if (remaining_mask != 0u) {
          pos_ += ctz_uint32(remaining_mask);
        }
      }
      DCHECK_LE(pos_, kNumBits);
      return *this;
    }
    bool operator==(const Iterator& other) const { return pos_ == other.pos_; }
    bool operator!=(const Iterator& other) const { return !(*this == other); }
    size_t operator-(const Iterator& other) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_7(mht_7_v, 348, "", "./tensorflow/core/framework/types.h", "-");

      return this->pos_ - other.pos_;
    }
  };

  static uint32 ctz_uint32(uint32 x) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_8(mht_8_v, 356, "", "./tensorflow/core/framework/types.h", "ctz_uint32");

    DCHECK_NE(x, 0u);
#ifdef __GNUC__
    return __builtin_ctz(x);
#else
    uint32 n = 0u;
    while ((x & 1u) == 0u) {
      x >>= 1;
      ++n;
    }
    return n;
#endif
  }

  static uint32 clz_uint32(uint32 x) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_9(mht_9_v, 373, "", "./tensorflow/core/framework/types.h", "clz_uint32");

    DCHECK_NE(x, 0u);
#ifdef __GNUC__
    return __builtin_clz(x);
#else
    uint32 n = 0u;
    while ((x >> (kNumBits - 1u)) == 0u) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }

  Iterator begin() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_10(mht_10_v, 390, "", "./tensorflow/core/framework/types.h", "begin");

    // The begin position is the index of the first bit set to 1 in the entire
    // bit mask. If there are no bits set to 1, then the index is 0.
    if (mask_ != 0) {
      return Iterator(*this, ctz_uint32(mask_));
    }
    // The set is empty.
    return Iterator(*this, 0);
  }

  Iterator end() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_11(mht_11_v, 403, "", "./tensorflow/core/framework/types.h", "end");

    // The end position is the index of the highest bit that is set, plus 1.
    // If there are no bits set to 1, then the index is 0.
    if (mask_ != 0) {
      return Iterator(*this, kNumBits - clz_uint32(mask_));
    }
    // The set is empty.
    return Iterator(*this, 0);
  }

  size_t size() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_12(mht_12_v, 416, "", "./tensorflow/core/framework/types.h", "size");

#if defined(__GNUC__)
    return __builtin_popcount(mask_);
#else
    size_t n = 0;
    uint32 x = mask_;
    while (x > 0) {
      n += x & 1u;
      x >>= 1;
    }
    return n;
#endif
  }

  constexpr DataTypeSet operator|(const DataTypeSet& other) const {
    return DataTypeSet(mask_ | other.mask_);
  }
};

// If "sp" names a valid type, store it in "*dt" and return true.  Otherwise,
// return false.
bool DataTypeFromString(StringPiece sp, DataType* dt);

constexpr inline DataTypeSet ToSet(DataType dt) {
  return DataTypeSet(1u << static_cast<uint32>(dt));
}

// DT_FLOAT + kDataTypeRefOffset == DT_FLOAT_REF, etc.
enum { kDataTypeRefOffset = 100 };
inline bool IsRefType(DataType dtype) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_13(mht_13_v, 448, "", "./tensorflow/core/framework/types.h", "IsRefType");

  return dtype > static_cast<DataType>(kDataTypeRefOffset);
}
inline DataType MakeRefType(DataType dtype) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_14(mht_14_v, 454, "", "./tensorflow/core/framework/types.h", "MakeRefType");

  DCHECK(!IsRefType(dtype));
  return static_cast<DataType>(dtype + kDataTypeRefOffset);
}
inline DataType RemoveRefType(DataType dtype) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_15(mht_15_v, 461, "", "./tensorflow/core/framework/types.h", "RemoveRefType");

  DCHECK(IsRefType(dtype));
  return static_cast<DataType>(dtype - kDataTypeRefOffset);
}
inline DataType BaseType(DataType dtype) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_16(mht_16_v, 468, "", "./tensorflow/core/framework/types.h", "BaseType");

  return IsRefType(dtype) ? RemoveRefType(dtype) : dtype;
}

// Returns true if the actual type is the same as or ref of the expected type.
inline bool TypesCompatible(DataType expected, DataType actual) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_17(mht_17_v, 476, "", "./tensorflow/core/framework/types.h", "TypesCompatible");

  return expected == actual || expected == BaseType(actual);
}

// Does not include _ref types.
constexpr DataTypeSet kAllTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_UINT8) |
    ToSet(DT_INT16) | ToSet(DT_UINT16) | ToSet(DT_INT8) | ToSet(DT_STRING) |
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_INT64) |
    ToSet(DT_BOOL) | ToSet(DT_QINT8) | ToSet(DT_QUINT8) | ToSet(DT_QINT16) |
    ToSet(DT_QUINT16) | ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_RESOURCE) |
    ToSet(DT_VARIANT) | ToSet(DT_UINT32) | ToSet(DT_UINT64) |
    ToSet(DT_BFLOAT16);
inline const DataTypeSet& AllTypes() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_18(mht_18_v, 492, "", "./tensorflow/core/framework/types.h", "AllTypes");
 return kAllTypes; }

#if !defined(IS_MOBILE_PLATFORM) || defined(SUPPORT_SELECTIVE_REGISTRATION)

// Types that support '<' and '>'.
constexpr DataTypeSet kRealNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_INT64) |
    ToSet(DT_UINT8) | ToSet(DT_INT16) | ToSet(DT_INT8) | ToSet(DT_UINT16) |
    ToSet(DT_HALF) | ToSet(DT_UINT32) | ToSet(DT_UINT64) | ToSet(DT_BFLOAT16);
inline const DataTypeSet RealNumberTypes() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_19(mht_19_v, 504, "", "./tensorflow/core/framework/types.h", "RealNumberTypes");
 return kRealNumberTypes; }

// Return the list of all numeric types.
// Includes complex and quantized types.
// NOTE: On Android, we only include the float and int32 types for now.
const DataTypeSet kNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT64) | ToSet(DT_INT32) |
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_QINT8) |
    ToSet(DT_QUINT8) | ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_UINT32) |
    ToSet(DT_UINT64) | ToSet(DT_BFLOAT16);
inline const DataTypeSet& NumberTypes() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_20(mht_20_v, 518, "", "./tensorflow/core/framework/types.h", "NumberTypes");
 return kNumberTypes; }

constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
                                        ToSet(DT_QINT32);
inline const DataTypeSet& QuantizedTypes() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_21(mht_21_v, 526, "", "./tensorflow/core/framework/types.h", "QuantizedTypes");
 return kQuantizedTypes; }

// Types that support '<' and '>', including quantized types.
const DataTypeSet kRealAndQuantizedTypes =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_INT64) |
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
    ToSet(DT_QINT8) | ToSet(DT_QUINT8) | ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
    ToSet(DT_QINT32) | ToSet(DT_HALF) | ToSet(DT_BFLOAT16);
inline const DataTypeSet& RealAndQuantizedTypes() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_22(mht_22_v, 537, "", "./tensorflow/core/framework/types.h", "RealAndQuantizedTypes");

  return kRealAndQuantizedTypes;
}

#elif defined(__ANDROID_TYPES_FULL__)

constexpr DataTypeSet kRealNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_INT64) | ToSet(DT_HALF);
inline DataTypeSet RealNumberTypes() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_23(mht_23_v, 548, "", "./tensorflow/core/framework/types.h", "RealNumberTypes");
 return kRealNumberTypes; }

constexpr DataTypeSet kNumberTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_INT64) | ToSet(DT_QINT8) |
    ToSet(DT_QUINT8) | ToSet(DT_QINT32) | ToSet(DT_HALF);
inline DataTypeSet NumberTypes() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_24(mht_24_v, 556, "", "./tensorflow/core/framework/types.h", "NumberTypes");
 return kNumberTypes; }

constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
                                        ToSet(DT_QINT32);
inline DataTypeSet QuantizedTypes() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_25(mht_25_v, 564, "", "./tensorflow/core/framework/types.h", "QuantizedTypes");
 return kQuantizedTypes; }

constexpr DataTypeSet kRealAndQuantizedTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_INT64) | ToSet(DT_QINT8) |
    ToSet(DT_QUINT8) | ToSet(DT_QINT16) | ToSet(DT_QUINT16) | ToSet(DT_QINT32) |
    ToSet(DT_HALF);
inline DataTypeSet RealAndQuantizedTypes() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_26(mht_26_v, 573, "", "./tensorflow/core/framework/types.h", "RealAndQuantizedTypes");
 return kRealAndQuantizedTypes; }

#else  // defined(IS_MOBILE_PLATFORM) && !defined(__ANDROID_TYPES_FULL__)

constexpr DataTypeSet kRealNumberTypes = ToSet(DT_FLOAT) | ToSet(DT_INT32);
inline DataTypeSet RealNumberTypes() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_27(mht_27_v, 581, "", "./tensorflow/core/framework/types.h", "RealNumberTypes");
 return kRealNumberTypes; }

constexpr DataTypeSet kNumberTypes = ToSet(DT_FLOAT) | ToSet(DT_INT32) |
                                     ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                     ToSet(DT_QINT32);
inline DataTypeSet NumberTypes() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_28(mht_28_v, 589, "", "./tensorflow/core/framework/types.h", "NumberTypes");
 return kNumberTypes; }

constexpr DataTypeSet kQuantizedTypes = ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
                                        ToSet(DT_QINT16) | ToSet(DT_QUINT16) |
                                        ToSet(DT_QINT32);
inline DataTypeSet QuantizedTypes() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_29(mht_29_v, 597, "", "./tensorflow/core/framework/types.h", "QuantizedTypes");
 return kQuantizedTypes; }

constexpr DataTypeSet kRealAndQuantizedTypes =
    ToSet(DT_FLOAT) | ToSet(DT_INT32) | ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
    ToSet(DT_QINT16) | ToSet(DT_QUINT16) | ToSet(DT_QINT32);
inline DataTypeSet RealAndQuantizedTypes() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_30(mht_30_v, 605, "", "./tensorflow/core/framework/types.h", "RealAndQuantizedTypes");
 return kRealAndQuantizedTypes; }

#endif  // defined(IS_MOBILE_PLATFORM)

// Validates type T for whether it is a supported DataType.
template <class T>
struct IsValidDataType;

// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                 \
  template <>                                           \
  struct DataTypeToEnum<TYPE> {                         \
    static DataType v() { return ENUM; }                \
    static DataType ref() { return MakeRefType(ENUM); } \
    static constexpr DataType value = ENUM;             \
  };                                                    \
  template <>                                           \
  struct IsValidDataType<TYPE> {                        \
    static constexpr bool value = true;                 \
  };                                                    \
  template <>                                           \
  struct EnumToDataType<ENUM> {                         \
    typedef TYPE Type;                                  \
  }

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32, DT_INT32);
MATCH_TYPE_AND_ENUM(uint32, DT_UINT32);
MATCH_TYPE_AND_ENUM(uint16, DT_UINT16);
MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16, DT_INT16);
MATCH_TYPE_AND_ENUM(int8, DT_INT8);
MATCH_TYPE_AND_ENUM(tstring, DT_STRING);
MATCH_TYPE_AND_ENUM(complex64, DT_COMPLEX64);
MATCH_TYPE_AND_ENUM(complex128, DT_COMPLEX128);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);
MATCH_TYPE_AND_ENUM(qint8, DT_QINT8);
MATCH_TYPE_AND_ENUM(quint8, DT_QUINT8);
MATCH_TYPE_AND_ENUM(qint16, DT_QINT16);
MATCH_TYPE_AND_ENUM(quint16, DT_QUINT16);
MATCH_TYPE_AND_ENUM(qint32, DT_QINT32);
MATCH_TYPE_AND_ENUM(bfloat16, DT_BFLOAT16);
MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
MATCH_TYPE_AND_ENUM(ResourceHandle, DT_RESOURCE);
MATCH_TYPE_AND_ENUM(Variant, DT_VARIANT);

template <>
struct DataTypeToEnum<long> {
  static DataType v() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_31(mht_31_v, 669, "", "./tensorflow/core/framework/types.h", "v");
 return value; }
  static DataType ref() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_32(mht_32_v, 673, "", "./tensorflow/core/framework/types.h", "ref");
 return MakeRefType(value); }
  static constexpr DataType value = sizeof(long) == 4 ? DT_INT32 : DT_INT64;
};
template <>
struct IsValidDataType<long> {
  static constexpr bool value = true;
};
template <>
struct EnumToDataType<DT_INT64> {
  typedef int64_t Type;
};

template <>
struct DataTypeToEnum<unsigned long> {
  static DataType v() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_33(mht_33_v, 690, "", "./tensorflow/core/framework/types.h", "v");
 return value; }
  static DataType ref() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_34(mht_34_v, 694, "", "./tensorflow/core/framework/types.h", "ref");
 return MakeRefType(value); }
  static constexpr DataType value =
      sizeof(unsigned long) == 4 ? DT_UINT32 : DT_UINT64;
};
template <>
struct IsValidDataType<unsigned long> {
  static constexpr bool value = true;
};
template <>
struct EnumToDataType<DT_UINT64> {
  typedef tensorflow::uint64 Type;
};

template <>
struct DataTypeToEnum<long long> {
  static DataType v() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_35(mht_35_v, 712, "", "./tensorflow/core/framework/types.h", "v");
 return DT_INT64; }
  static DataType ref() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_36(mht_36_v, 716, "", "./tensorflow/core/framework/types.h", "ref");
 return MakeRefType(DT_INT64); }
  static constexpr DataType value = DT_INT64;
};
template <>
struct IsValidDataType<long long> {
  static constexpr bool value = true;
};

template <>
struct DataTypeToEnum<unsigned long long> {
  static DataType v() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_37(mht_37_v, 729, "", "./tensorflow/core/framework/types.h", "v");
 return DT_UINT64; }
  static DataType ref() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_38(mht_38_v, 733, "", "./tensorflow/core/framework/types.h", "ref");
 return MakeRefType(DT_UINT64); }
  static constexpr DataType value = DT_UINT64;
};
template <>
struct IsValidDataType<unsigned long long> {
  static constexpr bool value = true;
};

#undef MATCH_TYPE_AND_ENUM

// All types not specialized are marked invalid.
template <class T>
struct IsValidDataType {
  static constexpr bool value = false;
};

// Extra validity checking; not part of public API.
static_assert(IsValidDataType<int64_t>::value, "Incorrect impl for int64");
static_assert(IsValidDataType<int32>::value, "Incorrect impl for int32");

// TODO(jeff): Maybe unify this with Tensor::CanUseDMA, or the underlying
// is_simple<T> in tensor.cc (and possible choose a more general name?)
constexpr DataTypeSet kDataTypesCanUseMemcpy =
    ToSet(DT_FLOAT) | ToSet(DT_DOUBLE) | ToSet(DT_INT32) | ToSet(DT_UINT32) |
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_INT16) | ToSet(DT_INT8) |
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128) | ToSet(DT_INT64) |
    ToSet(DT_UINT64) | ToSet(DT_BOOL) | ToSet(DT_QINT8) | ToSet(DT_QUINT8) |
    ToSet(DT_QINT16) | ToSet(DT_QUINT16) | ToSet(DT_QINT32) |
    ToSet(DT_BFLOAT16) | ToSet(DT_HALF);
inline bool DataTypeCanUseMemcpy(DataType dt) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_39(mht_39_v, 765, "", "./tensorflow/core/framework/types.h", "DataTypeCanUseMemcpy");

  return kDataTypesCanUseMemcpy.Contains(dt);
}

// Returns true iff 'dt' is a real, non-quantized floating point type.
constexpr DataTypeSet kDataTypeIsFloating =
    ToSet(DT_HALF) | ToSet(DT_BFLOAT16) | ToSet(DT_FLOAT) | ToSet(DT_DOUBLE);
inline bool DataTypeIsFloating(DataType dt) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_40(mht_40_v, 775, "", "./tensorflow/core/framework/types.h", "DataTypeIsFloating");

  return kDataTypeIsFloating.Contains(dt);
}

// Returns true iff 'dt' is a complex type.
constexpr DataTypeSet kDataTypeIsComplex =
    ToSet(DT_COMPLEX64) | ToSet(DT_COMPLEX128);
inline bool DataTypeIsComplex(DataType dt) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_41(mht_41_v, 785, "", "./tensorflow/core/framework/types.h", "DataTypeIsComplex");

  return kDataTypeIsComplex.Contains(dt);
}

inline bool DataTypeIsQuantized(DataType dt) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_42(mht_42_v, 792, "", "./tensorflow/core/framework/types.h", "DataTypeIsQuantized");

  return kQuantizedTypes.Contains(dt);
}

// Is the dtype nonquantized integral?
constexpr DataTypeSet kDataTypeIsInteger =
    ToSet(DT_INT8) | ToSet(DT_UINT8) | ToSet(DT_INT16) | ToSet(DT_UINT16) |
    ToSet(DT_INT32) | ToSet(DT_UINT32) | ToSet(DT_INT64) | ToSet(DT_UINT64);
inline bool DataTypeIsInteger(DataType dt) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_43(mht_43_v, 803, "", "./tensorflow/core/framework/types.h", "DataTypeIsInteger");

  return kDataTypeIsInteger.Contains(dt);
}

// Is the dtype a signed integral type?
constexpr DataTypeSet kDataTypeIsSigned =
    ToSet(DT_INT8) | ToSet(DT_INT16) | ToSet(DT_INT32) | ToSet(DT_INT64);
inline bool DataTypeIsSigned(DataType dt) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_44(mht_44_v, 813, "", "./tensorflow/core/framework/types.h", "DataTypeIsSigned");

  return kDataTypeIsSigned.Contains(dt);
}

// Is the dtype an unsigned integral type?
constexpr DataTypeSet kDataTypeIsUnsigned =
    ToSet(DT_UINT8) | ToSet(DT_UINT16) | ToSet(DT_UINT32) | ToSet(DT_UINT64);
inline bool DataTypeIsUnsigned(DataType dt) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_45(mht_45_v, 823, "", "./tensorflow/core/framework/types.h", "DataTypeIsUnsigned");

  return kDataTypeIsUnsigned.Contains(dt);
}

// Returns a 0 on failure
int DataTypeSize(DataType dt);

// Returns HOST_MEMORY if `dtype` is always on host or is a DT_INT32,
// DEVICE_MEMORY otherwise.
MemoryType MTypeFromDType(const DataType dtype);

// Returns HOST_MEMORY if `dtype` is always on host, DEVICE_MEMORY otherwise.
// The reason we have MTypeFromDType() and MTypeFromDTypeIntsOnDevice(): for
// GPUs, we would like to keep int operations on host for performance concerns.
// But for TPUs (and other devices), int operations are placed on device.
MemoryType MTypeFromDTypeIntsOnDevice(const DataType dtype);

// Types that always sit on host: DT_STRING, DT_STRING_REF, DT_RESOURCE.
// For DT_RESOURCE, the handle always sits on host (even if the underlying
// object has device-allocated resources).
bool DataTypeAlwaysOnHost(DataType dt);

// FullType implementation.

// Reference container for a type definition. These values are usually interned.
// These containers admit a notion of ordering for efficient access. The
// ordering has no semantic otherwise.
struct TypeRef {
  std::shared_ptr<FullTypeDef> full_type;

  bool operator==(const TypeRef& other) const {
    // TODO(mdan): This should be more efficient.
    return full_type->SerializeAsString() ==
           other.full_type->SerializeAsString();
  }
  bool operator<(const TypeRef& other) const {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStypesDTh mht_46(mht_46_v, 861, "", "./tensorflow/core/framework/types.h", "operator<");

    return full_type->SerializeAsString() <
           other.full_type->SerializeAsString();
  }
};

struct TypeHasher {
  std::size_t operator()(const TypeRef& k) const {
    return std::hash<std::string>()(k.full_type->SerializeAsString());
  }
};

// Maps a legacy DType proto enum to an equivalent FullType ID.
void map_dtype_to_tensor(const DataType& dtype, FullTypeDef& t);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TYPES_H_
