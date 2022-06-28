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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"

#include <algorithm>
#include <ostream>
#include <string>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

std::ostream& operator<<(std::ostream& out,
                         const LaunchDimensions& launch_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/xla/service/gpu/launch_dimensions.cc", "operator<<");

  LaunchDimensions::Dim3D block_counts = launch_dims.block_counts();
  LaunchDimensions::Dim3D thread_counts = launch_dims.thread_counts_per_block();
  out << absl::StrFormat("[block: {%d, %d, %d}, thread: {%d, %d, %d}]",
                         block_counts.x, block_counts.y, block_counts.z,
                         thread_counts.x, thread_counts.y, thread_counts.z);
  return out;
}

static int64_t ThreadsPerBlockLimit(GpuDeviceInfo gpu_device_info) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/service/gpu/launch_dimensions.cc", "ThreadsPerBlockLimit");

  int64_t threads_per_block = gpu_device_info.threads_per_block_limit;
  if (threads_per_block <= 0) {
    static std::atomic<int64_t> log_count{0};
    if (log_count.fetch_add(1) < 8) {
      LOG(WARNING) << "Attempting to calculate launch dimensions for GPU "
                      "without full information about its capabilities.  "
                      "StreamExecutor's PopulateDeviceDescription should be "
                      "updated for this device.";
    }
    threads_per_block = gpu_device_info.threads_per_warp;
    if (threads_per_block == 0) {
      // Fall back to *something* if we can't even get num threads per warp.
      threads_per_block = 32;
    }
  }
  return threads_per_block;
}

int64_t ThreadsPerBlockRowVectorized(const Shape& shape,
                                     GpuDeviceInfo gpu_device_info,
                                     LaunchDimensionsConfig dim_config) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/service/gpu/launch_dimensions.cc", "ThreadsPerBlockRowVectorized");

  if (shape.dimensions().empty()) {
    return -1;
  }
  int64_t threads_per_block_row_vectorized =
      shape.dimensions().back() / dim_config.unroll_factor;
  if (dim_config.row_vectorized &&
      shape.dimensions().back() % dim_config.unroll_factor == 0 &&
      // If the row size is a multiple of 256, then use the old code
      // path that use a block size of 256. This give small speed up on V100.
      // Vectorization of the row load was already happening.
      (shape.dimensions().back() % 256) != 0 &&
      // We do not support row that do not fit in one block.
      threads_per_block_row_vectorized <=
          gpu_device_info.threads_per_block_limit) {
    return threads_per_block_row_vectorized;
  }
  return -1;
}

StatusOr<LaunchDimensions> CalculateLaunchDimensions(
    const Shape& shape, GpuDeviceInfo gpu_device_info,
    LaunchDimensionsConfig dim_config) {
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  if (num_elements <= 1) {
    return LaunchDimensions();
  }

  CHECK_EQ(num_elements % dim_config.unroll_factor, 0);
  num_elements = num_elements / dim_config.unroll_factor;

  // Since we don't do any inter-warp communication, we're free to choose any
  // block size we want, subject to hardware constraints.  We choose the largest
  // block size allowed, as empirically, this is a performance win on almost
  // (but not all) benchmarks.
  //
  // My guess is that using a larger block size encourages ptxas to decrease
  // per-thread register usage, thus allowing for higher occupancy, but I
  // haven't verified this.
  //
  // TODO(jlebar): Investigate this further, and tune this heuristic so we can
  // run faster on the few benchmarks where smaller block size helps.
  int64_t threads_per_block_row_vectorized =
      ThreadsPerBlockRowVectorized(shape, gpu_device_info, dim_config);
  // If row vectorized, threads_per_block_x is the vectorized size.
  // Otherwise, we unroll kernels to make use of vectorized
  // loads/stores. This means we need more registers to hold
  // intermediate values. Reduce the number of threads per block to
  // increase the number of registers available to ptxas.  Make sure
  // we still have a multiple of 32.
  int64_t threads_per_block_x = [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSlaunch_dimensionsDTcc mht_3(mht_3_v, 287, "", "./tensorflow/compiler/xla/service/gpu/launch_dimensions.cc", "lambda");

    int64_t max_threads_per_block_x =
        threads_per_block_row_vectorized > 0
            ? threads_per_block_row_vectorized
            : RoundUpTo(ThreadsPerBlockLimit(gpu_device_info) /
                            dim_config.unroll_factor,
                        int64_t{32});
    if (num_elements < max_threads_per_block_x) {
      return num_elements;
    }
    return max_threads_per_block_x;
  }();
  // threads_per_block_y > 1 when we row vectorize and have small row size.
  int64_t threads_per_block_y =
      threads_per_block_row_vectorized > 0 &&
              threads_per_block_row_vectorized < 128 && num_elements > 128
          ? CeilOfRatio(static_cast<int64_t>(128),
                        threads_per_block_row_vectorized)
          : 1;
  VLOG(2) << "Set # of threads per block to (.x=" << threads_per_block_x
          << ", .y=" << threads_per_block_y << ")";

  int64_t block_count =
      CeilOfRatio(num_elements, threads_per_block_x * threads_per_block_y);
  if (dim_config.few_waves && !dim_config.row_vectorized) {
    int64_t capped_threads_per_block_x =
        std::min<int64_t>(threads_per_block_x, 128);
    int64_t capped_block_count =
        gpu_device_info.core_count *
        (gpu_device_info.threads_per_core_limit /
         (capped_threads_per_block_x * threads_per_block_y));
    if (capped_block_count < block_count) {
      threads_per_block_x = capped_threads_per_block_x;
      block_count = capped_block_count;
      VLOG(2) << "Update the # of blocks to " << block_count
              << " and the # of threads per blocks to " << threads_per_block_x
              << " as the few waves mode is enabled.";
    }
  } else if (dim_config.few_waves && dim_config.row_vectorized) {
    int64_t min_block_count = gpu_device_info.core_count *
                              (gpu_device_info.threads_per_core_limit /
                               (threads_per_block_x * threads_per_block_y));
    int64_t capped_block_count = block_count;
    // This multiple of 32 was tuned to not cause regression on multiple
    // benchmarks.  It isn't a value that is optimal for all
    // kernels. Maybe looking at the arithmetic intensity of the
    // kernels can specialize the multiple per kernel.
    while (capped_block_count > (32 * min_block_count)) {
      capped_block_count /= 2;
    }
    // Do not increase the number of blocks. This can happens for
    // small num_elements.
    if (capped_block_count < block_count) {
      VLOG(2) << "Update # of blocks to block_count as few_waves is enabled.";
      block_count = capped_block_count;
    }
  }
  if (gpu_device_info.block_dim_limit_x > 0 &&
      block_count >= gpu_device_info.block_dim_limit_x) {
    return tensorflow::errors::Unimplemented(
        "Kernel launch needs more blocks (", block_count,
        ") than allowed by hardware (", gpu_device_info.block_dim_limit_x,
        ").");
  }

  VLOG(2) << absl::StrFormat(
      "Initialized the block count to %d, the block size .x=%d and .y=%d"
      " for %d elements in the tensor.",
      block_count, threads_per_block_x, threads_per_block_y, num_elements);
  return LaunchDimensions({block_count, 1, 1},
                          {threads_per_block_x, threads_per_block_y, 1});
}

}  // namespace gpu
}  // namespace xla
