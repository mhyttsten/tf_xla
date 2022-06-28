/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TYPES_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TYPES_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh() {
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


#include <array>
#include <cstddef>
#include <cstdint>

#include "fp16.h"  // from @FP16

namespace tflite {
namespace gpu {

// TODO(akulik): make these types Google-style compliant.

using HalfBits = uint16_t;

class alignas(2) half {
 public:
  HalfBits bits;

  half() = default;

  half(const half& f) : bits(f.bits) {}

  explicit half(float other) { bits = fp16_ieee_from_fp32_value(other); }

  void operator=(float f) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/delegates/gpu/common/types.h", "=");
 *this = half(f); }

  operator float() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_1(mht_1_v, 216, "", "./tensorflow/lite/delegates/gpu/common/types.h", "float");
 return fp16_ieee_to_fp32_value(bits); }
};

template <typename T>
struct alignas(sizeof(T)) Vec4 {
  union {
    struct {
      T x, y, z, w;
    };
    std::array<T, 4> data_;
  };

  Vec4() : Vec4(T(0.0f)) {}

  template <typename S>
  Vec4(S x_, S y_, S z_, S w_) : x(x_), y(y_), z(z_), w(w_) {}
  explicit Vec4(T v) : x(v), y(v), z(v), w(v) {}

  template <typename S>
  explicit Vec4(S v) : x(v), y(v), z(v), w(v) {}

  Vec4(const Vec4& f) : x(f.x), y(f.y), z(f.z), w(f.w) {}

  template <typename S>
  Vec4(const Vec4<S>& f) : x(f.x), y(f.y), z(f.z), w(f.w) {}

  Vec4& operator=(const Vec4& other) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_2(mht_2_v, 245, "", "./tensorflow/lite/delegates/gpu/common/types.h", "=");

    x = other.x;
    y = other.y;
    z = other.z;
    w = other.w;
    return *this;
  }

  static constexpr int size() { return 4; }

  T& operator[](size_t n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_3(mht_3_v, 258, "", "./tensorflow/lite/delegates/gpu/common/types.h", "lambda");
 return data_[n]; }
  T operator[](size_t n) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_4(mht_4_v, 262, "", "./tensorflow/lite/delegates/gpu/common/types.h", "lambda");
 return data_[n]; }

  bool operator==(const Vec4& value) const {
    return data_[0] == value[0] && data_[1] == value[1] &&
           data_[2] == value[2] && data_[3] == value[3];
  }
  bool operator!=(const Vec4& value) const {
    return !(this->operator==(value));
  }
};

template <typename T>
struct alignas(sizeof(T)) Vec3 {
  union {
    struct {
      T x, y, z;
    };
    std::array<T, 3> data_;
  };

  Vec3() : Vec3(T(0.0f)) {}

  template <typename S>
  constexpr Vec3(S x_, S y_, S z_) : x(x_), y(y_), z(z_) {}
  explicit Vec3(T v) : x(v), y(v), z(v) {}

  template <typename S>
  explicit Vec3(S v) : x(v), y(v), z(v) {}

  Vec3(const Vec3& f) : x(f.x), y(f.y), z(f.z) {}

  template <typename S>
  Vec3(const Vec3<S>& f) : x(f.x), y(f.y), z(f.z) {}

  Vec3& operator=(const Vec3& other) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_5(mht_5_v, 299, "", "./tensorflow/lite/delegates/gpu/common/types.h", "=");

    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  static constexpr int size() { return 3; }

  T& operator[](size_t n) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_6(mht_6_v, 311, "", "./tensorflow/lite/delegates/gpu/common/types.h", "lambda");
 return data_[n]; }
  T operator[](size_t n) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_7(mht_7_v, 315, "", "./tensorflow/lite/delegates/gpu/common/types.h", "lambda");
 return data_[n]; }
  bool operator==(const Vec3& value) const {
    return data_[0] == value[0] && data_[1] == value[1] && data_[2] == value[2];
  }
  bool operator!=(const Vec3& value) const {
    return !(this->operator==(value));
  }
};

template <typename T>
struct alignas(sizeof(T)) Vec2 {
  union {
    struct {
      T x, y;
    };
    std::array<T, 2> data_;
  };

  Vec2() : Vec2(T(0.0f)) {}

  template <typename S>
  Vec2(S x_, S y_) : x(x_), y(y_) {}
  explicit Vec2(T v) : x(v), y(v) {}

  template <typename S>
  explicit Vec2(S v) : x(v), y(v) {}

  Vec2(const Vec2& f) : x(f.x), y(f.y) {}

  template <typename S>
  Vec2(const Vec2<S>& f) : x(f.x), y(f.y) {}

  Vec2& operator=(const Vec2& other) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_8(mht_8_v, 350, "", "./tensorflow/lite/delegates/gpu/common/types.h", "=");

    x = other.x;
    y = other.y;
    return *this;
  }

  bool operator==(const Vec2& value) const {
    return data_[0] == value[0] && data_[1] == value[1];
  }

  bool operator!=(const Vec2& value) const {
    return !(this->operator==(value));
  }

  static constexpr int size() { return 2; }

  T& operator[](size_t n) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_9(mht_9_v, 369, "", "./tensorflow/lite/delegates/gpu/common/types.h", "lambda");
 return data_[n]; }
  T operator[](size_t n) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStypesDTh mht_10(mht_10_v, 373, "", "./tensorflow/lite/delegates/gpu/common/types.h", "lambda");
 return data_[n]; }
};

using float2 = Vec2<float>;
using half2 = Vec2<half>;
using byte2 = Vec2<int8_t>;
using ubyte2 = Vec2<uint8_t>;
using short2 = Vec2<int16_t>;
using ushort2 = Vec2<uint16_t>;
using int2 = Vec2<int32_t>;
using uint2 = Vec2<uint32_t>;

using float3 = Vec3<float>;
using half3 = Vec3<half>;
using byte3 = Vec3<int8_t>;
using ubyte3 = Vec3<uint8_t>;
using short3 = Vec3<int16_t>;
using ushort3 = Vec3<uint16_t>;
using int3 = Vec3<int32_t>;
using uint3 = Vec3<uint32_t>;

using float4 = Vec4<float>;
using half4 = Vec4<half>;
using byte4 = Vec4<int8_t>;
using ubyte4 = Vec4<uint8_t>;
using short4 = Vec4<int16_t>;
using ushort4 = Vec4<uint16_t>;
using int4 = Vec4<int32_t>;
using uint4 = Vec4<uint32_t>;

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TYPES_H_
