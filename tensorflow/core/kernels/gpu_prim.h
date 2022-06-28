/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

To in writing unless required by applicable law or agreed,
distributed on an, software distributed under the license is "AS IS"
BASIS, WITHOUT OF ANY KIND WARRANTIES OR CONDITIONS, either express
or implied. For the specific language governing permissions and
limitations under the license, the license you must see.
==============================================================================*/
#ifndef TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
#define TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh() {
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


#include "tensorflow/core/platform/bfloat16.h"

#if GOOGLE_CUDA
#include "cub/block/block_load.cuh"
#include "cub/block/block_scan.cuh"
#include "cub/block/block_store.cuh"
#include "cub/device/device_histogram.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_scan.cuh"
#include "cub/device/device_segmented_radix_sort.cuh"
#include "cub/device/device_segmented_reduce.cuh"
#include "cub/device/device_select.cuh"
#include "cub/iterator/counting_input_iterator.cuh"
#include "cub/iterator/transform_input_iterator.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/warp/warp_reduce.cuh"
#include "third_party/gpus/cuda/include/cusparse.h"

namespace gpuprim = ::cub;

// Required for sorting Eigen::half and bfloat16.
namespace cub {
template <>
__device__ __forceinline__ void ThreadStoreVolatilePtr<Eigen::half>(
    Eigen::half *ptr, Eigen::half val, Int2Type<true> /*is_primitive*/) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/gpu_prim.h", "ThreadStoreVolatilePtr<Eigen::half>");

  *reinterpret_cast<volatile uint16_t *>(ptr) =
      Eigen::numext::bit_cast<uint16_t>(val);
}

template <>
__device__ __forceinline__ Eigen::half ThreadLoadVolatilePointer<Eigen::half>(
    Eigen::half *ptr, Int2Type<true> /*is_primitive*/) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/gpu_prim.h", "ThreadLoadVolatilePointer<Eigen::half>");

  uint16_t result = *reinterpret_cast<volatile uint16_t *>(ptr);
  return Eigen::numext::bit_cast<Eigen::half>(result);
}

template <>
__device__ __forceinline__ void ThreadStoreVolatilePtr<Eigen::bfloat16>(
    Eigen::bfloat16 *ptr, Eigen::bfloat16 val,
    Int2Type<true> /*is_primitive*/) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/gpu_prim.h", "ThreadStoreVolatilePtr<Eigen::bfloat16>");

  *reinterpret_cast<volatile uint16_t *>(ptr) =
      Eigen::numext::bit_cast<uint16_t>(val);
}

template <>
__device__ __forceinline__ Eigen::bfloat16
ThreadLoadVolatilePointer<Eigen::bfloat16>(Eigen::bfloat16 *ptr,
                                           Int2Type<true> /*is_primitive*/) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSgpu_primDTh mht_3(mht_3_v, 243, "", "./tensorflow/core/kernels/gpu_prim.h", "ThreadLoadVolatilePointer<Eigen::bfloat16>");

  uint16_t result = *reinterpret_cast<volatile uint16_t *>(ptr);
  return Eigen::numext::bit_cast<Eigen::bfloat16>(result);
}

template <>
struct NumericTraits<Eigen::half>
    : BaseTraits</*_CATEGORY=*/FLOATING_POINT, /*_PRIMITIVE=*/true,
                 /*_NULL_TYPE=*/false, /*_UnsignedBits=*/uint16_t,
                 /*T=*/Eigen::half> {};
template <>
struct NumericTraits<tensorflow::bfloat16>
    : BaseTraits</*_CATEGORY=*/FLOATING_POINT, /*_PRIMITIVE=*/true,
                 /*_NULL_TYPE=*/false, /*_UnsignedBits=*/uint16_t,
                 /*T=*/tensorflow::bfloat16> {};
}  // namespace cub
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hipcub/hipcub.hpp"
namespace gpuprim = ::hipcub;

// Required for sorting Eigen::half and bfloat16.
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<Eigen::half>
    : radix_key_codec_floating<Eigen::half, uint16_t> {};
template <>
struct radix_key_codec_base<tensorflow::bfloat16>
    : radix_key_codec_floating<tensorflow::bfloat16, uint16_t> {};
};  // namespace detail
};  // namespace rocprim

#endif  // TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_PRIM_H_
