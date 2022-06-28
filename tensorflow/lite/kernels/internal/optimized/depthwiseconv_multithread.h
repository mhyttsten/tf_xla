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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_MULTITHREAD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_MULTITHREAD_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh() {
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


#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"

namespace tflite {
namespace optimized_ops {

// TODO(luwa): add multithread to per-channel depthwise_conv
// DepthwiseConv can run with multi threads on the dim specified by thread_dim.
// Each thread processes output elements on dim, thread_dim, in the range of
// [thread_start, thread_end).
// For example, assume thread_start = 2, thread_end = 6, and thread_dim = 1, it
// means that it will calculate DepthwiseConv for output_data[:, 2:5, :, :].
template <typename T, typename TS>
struct DepthwiseConvWorkerTask : cpu_backend_threadpool::Task {
  DepthwiseConvWorkerTask(const DepthwiseParams& params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& filter_shape,
                          const T* filter_data, const RuntimeShape& bias_shape,
                          const TS* bias_data, const RuntimeShape& output_shape,
                          T* output_data, const CpuFlags& cpu_flags,
                          int thread_start, int thread_end, int thread_dim)
      : params_(params),
        input_shape_(input_shape),
        input_data_(input_data),
        filter_shape_(filter_shape),
        filter_data_(filter_data),
        bias_shape_(bias_shape),
        bias_data_(bias_data),
        output_shape_(output_shape),
        output_data_(output_data),
        cpu_flags_(cpu_flags),
        thread_start_(thread_start),
        thread_end_(thread_end),
        thread_dim_(thread_dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh mht_0(mht_0_v, 223, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h", "DepthwiseConvWorkerTask");
}

  void Run() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh mht_1(mht_1_v, 228, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h", "Run");

    DepthwiseConvImpl(params_, input_shape_, input_data_, filter_shape_,
                      filter_data_, bias_shape_, bias_data_, output_shape_,
                      output_data_, cpu_flags_, thread_start_, thread_end_,
                      thread_dim_);
  }

 private:
  const DepthwiseParams& params_;
  const RuntimeShape& input_shape_;
  const T* input_data_;
  const RuntimeShape& filter_shape_;
  const T* filter_data_;
  const RuntimeShape& bias_shape_;
  const TS* bias_data_;
  const RuntimeShape& output_shape_;
  T* output_data_;
  const CpuFlags& cpu_flags_;
  int thread_start_;
  int thread_end_;
  int thread_dim_;
};

inline int HowManyConvThreads(const RuntimeShape& output_shape,
                              const RuntimeShape& filter_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh mht_2(mht_2_v, 255, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h", "HowManyConvThreads");

  // How many scalar multiplications are needed to make it worth using one
  // more thread
  static constexpr int kMinMulPerThread = 1 << 13;  // 8k
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int num_muls = output_shape.FlatSize() * filter_height * filter_width;
  // Try to avoid real runtime divisions if possible by dividing by a
  // compile-time constant.
  int thread_count = std::max(1, num_muls / kMinMulPerThread);
  return thread_count;
}

inline bool MultithreadAlongBatches(int thread_count, int batches) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSdepthwiseconv_multithreadDTh mht_3(mht_3_v, 271, "", "./tensorflow/lite/kernels/internal/optimized/depthwiseconv_multithread.h", "MultithreadAlongBatches");

  TFLITE_DCHECK_GE(thread_count, 2);
  // If there are fewer batch entries than the number of threads we want to use,
  // then better do intra-batch-entry multithreading.
  if (batches < thread_count) {
    return false;
  }
  // If there are at least 2 batch entries to be handed to each thread, then
  // it's safe to proceed with batch-wise multithreading: each thread will have
  // approximately equal number of batch entries to handle, so the load
  // balancing will be reasonable, and the amount to which the load is not
  // perfectly balanced will be offset by the inherent advantages of
  // batch-wise multithreading (each thread is more efficient thanks to working
  // on larger buffers with less boundary-handling overhead).
  if (batches >= 2 * thread_count) {
    return true;
  }
  // In the limit case were there are at least 1 but not much more than 1
  // batch entries per thread, it may be a good idea to do per-batch
  // multithreading if the number of batch entries is a multiple of the number
  // of threads, so that each thread will have the same number of batch entries
  // to process.
  return ((batches % thread_count) == 0);
}

template <typename T, typename TS>
inline void DepthwiseConv(const DepthwiseParams& params,
                          const RuntimeShape& input_shape, const T* input_data,
                          const RuntimeShape& filter_shape,
                          const T* filter_data, const RuntimeShape& bias_shape,
                          const TS* bias_data, const RuntimeShape& output_shape,
                          T* output_data,
                          CpuBackendContext* cpu_backend_context) {
  ruy::profiler::ScopeLabel label("DepthwiseConv");

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  int thread_count = HowManyConvThreads(output_shape, filter_shape);
  const int max_threads = cpu_backend_context->max_num_threads();
  thread_count = std::max(1, std::min(thread_count, max_threads));
#ifndef TFLITE_WITH_RUY
  // Cap the number of threads to 2 for float path to avoid regression in
  // performance (b/132294857).
  if (std::is_floating_point<T>::value) {
    thread_count = std::min(thread_count, 2);
  }
#endif

  const int output_batches = output_shape.Dims(0);
  const int output_height = output_shape.Dims(1);

  CpuFlags cpu_flags;
  GetCpuFlags(&cpu_flags);

  if (thread_count == 1) {
    DepthwiseConvImpl(params, input_shape, input_data, filter_shape,
                      filter_data, bias_shape, bias_data, output_shape,
                      output_data, cpu_flags, /*thread_start=*/0,
                      /*thread_end=*/output_height, /*thread_dim=*/1);
    return;
  }

  int thread_dim, thread_dim_size;
  if (MultithreadAlongBatches(thread_count, output_batches)) {
    thread_dim = 0;
    thread_dim_size = output_batches;
  } else {
    thread_dim = 1;
    thread_dim_size = output_height;
  }

  std::vector<DepthwiseConvWorkerTask<T, TS>> tasks;
  // TODO(b/131746020) don't create new heap allocations every time.
  // At least we make it a single heap allocation by using reserve().
  tasks.reserve(thread_count);
  int thread_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    int thread_end =
        thread_start + (thread_dim_size - thread_start) / (thread_count - i);
    tasks.emplace_back(params, input_shape, input_data, filter_shape,
                       filter_data, bias_shape, bias_data, output_shape,
                       output_data, cpu_flags, thread_start, thread_end,
                       thread_dim);
    thread_start = thread_end;
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_MULTITHREAD_H_
