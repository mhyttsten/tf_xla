/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SAMPLING_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_SAMPLING_KERNELS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh() {
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


#include <cmath>

#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace functor {
// Defines functions for different types of sampling kernels.
enum SamplingKernelType {
  // Lanczos kernel with radius 1.  Aliases but does not ring.
  Lanczos1Kernel,

  // Lanczos kernel with radius 3.  High-quality practical filter but may have
  // some ringing especially on synthetic images.
  Lanczos3Kernel,

  // Lanczos kernel with radius 5.  Very-high-quality filter but may have
  // stronger ringing.
  Lanczos5Kernel,

  // Gaussian kernel with radius 3, sigma = 1.5 / 3.  Less commonly used.
  GaussianKernel,

  // Rectangle function.  Equivalent to "nearest" sampling when upscaling.
  // Has value 1 in interval (-0.5, 0.5), value 0.5 on edge, and 0 elsewhere.
  BoxKernel,

  // Hat/tent function with radius 1.  Equivalent to "bilinear" reconstruction
  // when upsampling.
  // Has value zero at -1.0 and 1.0.
  TriangleKernel,

  // Cubic interpolant of Keys.  Equivalent to Catmull-Rom kernel.  Reasonably
  // good quality and faster than Lanczos3Kernel.
  KeysCubicKernel,

  // Cubic non-interpolating scheme.  For synthetic images (especially those
  // lacking proper prefiltering), less ringing than Keys cubic kernel but less
  // sharp.
  MitchellCubicKernel,

  // Always insert new kernel types before this.
  SamplingKernelTypeEnd
};

// Converts a string into the corresponding kernel type.
// Returns SamplingKernelTypeEnd if the string couldn't be converted.
SamplingKernelType SamplingKernelTypeFromString(const StringPiece str);

// A function object for a Lanczos kernel.
struct LanczosKernelFunc {
  // Pass 1 for Lanczos1 kernel, 3 for Lanczos3 etc.
  explicit LanczosKernelFunc(float _radius) : radius(_radius) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_0(mht_0_v, 239, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "LanczosKernelFunc");
}
  float operator()(float x) const {
    constexpr float kPI = 3.14159265359;
    x = std::abs(x);
    if (x > radius) return 0.0;
    // Need to special case the limit case of sin(x) / x when x is zero.
    if (x <= 1e-3) {
      return 1.0;
    }
    return radius * std::sin(kPI * x) * std::sin(kPI * x / radius) /
           (kPI * kPI * x * x);
  }
  float Radius() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "Radius");
 return radius; }
  const float radius;
};

struct GaussianKernelFunc {
  static constexpr float kRadiusMultiplier = 3.0f;
  // https://en.wikipedia.org/wiki/Gaussian_function
  // We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
  // for Common Resampling Tasks" for kernels with a support of 3 pixels:
  // www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
  // This implies a radius of 1.5,
  explicit GaussianKernelFunc(float _radius = 1.5f)
      : radius(_radius), sigma(_radius / kRadiusMultiplier) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_2(mht_2_v, 269, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "GaussianKernelFunc");
}
  float operator()(float x) const {
    x = std::abs(x);
    if (x >= radius) return 0.0;
    return std::exp(-x * x / (2.0 * sigma * sigma));
  }
  float Radius() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_3(mht_3_v, 278, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "Radius");
 return radius; }
  const float radius;
  const float sigma;  // Gaussian standard deviation
};

struct BoxKernelFunc {
  float operator()(float x) const {
    x = std::abs(x);
    return x < 0.5f ? 1. : x == 0.5f ? 0.5f : 0.0f;
  }
  float Radius() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_4(mht_4_v, 291, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "Radius");
 return 1.f; }
};

struct TriangleKernelFunc {
  // https://en.wikipedia.org/wiki/Triangle_function
  float operator()(float x) const {
    x = std::abs(x);
    return x < 1.0f ? 1.0f - x : 0.0f;
  }
  float Radius() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_5(mht_5_v, 303, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "Radius");
 return 1.f; }
};

struct KeysCubicKernelFunc {
  // http://ieeexplore.ieee.org/document/1163711/
  // R. G. Keys. Cubic convolution interpolation for digital image
  // processing. IEEE Transactions on Acoustics, Speech, and Signal
  // Processing, 29(6):1153–1160, 1981.
  float operator()(float x) const {
    x = std::abs(x);
    if (x >= 2.0f) {
      return 0.0f;
    } else if (x >= 1.0f) {
      return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    } else {
      return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    }
  }
  float Radius() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_6(mht_6_v, 324, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "Radius");
 return 2.f; }
};

struct MitchellCubicKernelFunc {
  // https://doi.org/10.1145/378456.378514
  // D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
  // graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
  // 22(4):221–228, 1988.
  float operator()(float x) const {
    x = std::abs(x);
    if (x >= 2.0f) {
      return 0.0f;
    } else if (x >= 1.0f) {
      return (((-7.0f / 18.0f) * x + 2.0f) * x - 10.0f / 3.0f) * x +
             16.0f / 9.0f;
    } else {
      return (((7.0f / 6.0f) * x - 2.0f) * x) * x + 8.0f / 9.0f;
    }
  }
  float Radius() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_7(mht_7_v, 346, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "Radius");
 return 2.f; }
};

inline LanczosKernelFunc CreateLanczos1Kernel() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_8(mht_8_v, 352, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateLanczos1Kernel");

  return LanczosKernelFunc(1.0);
}

inline LanczosKernelFunc CreateLanczos3Kernel() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_9(mht_9_v, 359, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateLanczos3Kernel");

  return LanczosKernelFunc(3.0);
}

inline LanczosKernelFunc CreateLanczos5Kernel() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_10(mht_10_v, 366, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateLanczos5Kernel");

  return LanczosKernelFunc(5.0);
}

inline GaussianKernelFunc CreateGaussianKernel() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_11(mht_11_v, 373, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateGaussianKernel");

  return GaussianKernelFunc(1.5);
}

inline BoxKernelFunc CreateBoxKernel() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_12(mht_12_v, 380, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateBoxKernel");
 return BoxKernelFunc(); }

inline TriangleKernelFunc CreateTriangleKernel() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_13(mht_13_v, 385, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateTriangleKernel");

  return TriangleKernelFunc();
}

inline KeysCubicKernelFunc CreateKeysCubicKernel() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_14(mht_14_v, 392, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateKeysCubicKernel");

  return KeysCubicKernelFunc();
}

inline MitchellCubicKernelFunc CreateMitchellCubicKernel() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimagePSsampling_kernelsDTh mht_15(mht_15_v, 399, "", "./tensorflow/core/kernels/image/sampling_kernels.h", "CreateMitchellCubicKernel");

  return MitchellCubicKernelFunc();
}

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SAMPLING_KERNELS_H_
