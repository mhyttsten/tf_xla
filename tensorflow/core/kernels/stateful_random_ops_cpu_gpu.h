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

#ifndef TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_
#define TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh() {
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


#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/stateful_random_ops.h"

namespace tensorflow {

PHILOX_DEVICE_INLINE PhiloxRandom
GetPhiloxRandomFromMem(StateElementType const* ptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh mht_0(mht_0_v, 194, "", "./tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h", "GetPhiloxRandomFromMem");

  auto ptr_ = reinterpret_cast<uint64 const*>(ptr);
  return GetPhiloxRandomFromCounterKeyMem(ptr_, ptr_ + 2);
}

PHILOX_DEVICE_INLINE void WritePhiloxRandomToMem(PhiloxRandom const& philox,
                                                 StateElementType* ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh mht_1(mht_1_v, 203, "", "./tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h", "WritePhiloxRandomToMem");

  auto ptr_ = reinterpret_cast<uint64*>(ptr);
  WriteCounterToMem(philox.counter(), ptr_);
  WriteKeyToMem(philox.key(), ptr_ + 2);
}

PHILOX_DEVICE_INLINE PhiloxRandom SkipPhiloxRandom(PhiloxRandom const& philox,
                                                   uint64 output_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh mht_2(mht_2_v, 213, "", "./tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h", "SkipPhiloxRandom");

  auto new_philox = philox;
  // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change it
  // just here.
  auto delta = output_size * 256;
  new_philox.Skip(delta);  // do the actual increasing
  return new_philox;
}

PHILOX_DEVICE_INLINE void UpdateMemWithPhiloxRandom(PhiloxRandom const& philox,
                                                    uint64 output_size,
                                                    StateElementType* ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh mht_3(mht_3_v, 227, "", "./tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h", "UpdateMemWithPhiloxRandom");

  auto new_philox = SkipPhiloxRandom(philox, output_size);
  WritePhiloxRandomToMem(new_philox, ptr);
}

PHILOX_DEVICE_INLINE void UpdateCounterMemWithPhiloxRandom(
    PhiloxRandom::ResultType const& counter, uint64 output_size,
    StateElementType* ptr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_cpu_gpuDTh mht_4(mht_4_v, 237, "", "./tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h", "UpdateCounterMemWithPhiloxRandom");

  auto philox = PhiloxRandom(counter, PhiloxRandom::Key() /*dummy*/);
  auto new_philox = SkipPhiloxRandom(philox, output_size);
  WriteCounterToMem(new_philox.counter(), reinterpret_cast<uint64*>(ptr));
}

namespace functor {

// A per-device helper function that does the actual work for
// `UpdateVariableAndFill`.
// Reason to use functor: C++ doesn't allow function-template partial
// specialization.
template <typename Device, typename Distribution>
struct UpdateVariableAndFill_Philox;

template <typename Device>
struct RngSkip_Philox;

}  // end namespace functor

using CPUDevice = Eigen::ThreadPoolDevice;

class ScopedUnlockUnrefVar;

struct UpdateVariableAndFill_Philox_Arg {
  int64_t output_size;
  int64_t alg_tag_skip;
  ScopedUnlockUnrefVar* state_var_guard;
  Tensor* state_tensor;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using GPUDevice = Eigen::GpuDevice;

namespace functor {

// Declares the partially GPU-specialized functor structs.
// must be kept at <=6 arguments because of a gcc/clang ABI incompatibility bug
template <typename Distribution>
struct UpdateVariableAndFill_Philox<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& device,
                  Distribution dist, UpdateVariableAndFill_Philox_Arg* arg,
                  typename Distribution::ResultElementType* output_data);
};

template <>
struct RngSkip_Philox<GPUDevice> {
  void operator()(const GPUDevice& device, const StateElementType* in_data,
                  uint64 delta, StateElementType* out_data);
};

}  // end namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_
