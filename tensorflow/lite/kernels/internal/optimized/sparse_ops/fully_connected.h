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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh() {
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


#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {

inline void FullyConnectedSparseWeight(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "FullyConnectedSparseWeight");

  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("Random Sparse");
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  const int output_elements = output_shape.FlatSize();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = FlatSizeSkipDim(output_shape, output_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int accum_depth = weights_shape.Dims(weights_dims_count - 1);
  const int w0_size = sparsity.dim_metadata[0].dense_size;
  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  for (int i = 0; i < output_elements; ++i) {
    output_data[i] = 0.f;
  }

  for (int b = 0; b < batches; ++b) {
    for (int idx_0 = 0; idx_0 < w0_size; ++idx_0) {
      for (int pw1 = w1_segments[idx_0]; pw1 < w1_segments[idx_0 + 1]; ++pw1) {
        int idx_1 = w1_indices[pw1];
        output_data[b * output_depth + idx_0] +=
            weights_data[pw1] * input_data[b * accum_depth + idx_1];
      }
    }
  }

  for (int b = 0; b < batches; ++b) {
    for (int i = 0; i < output_depth; ++i) {
      float total = output_data[b * output_depth + i];
      const float bias_value = bias_data ? bias_data[i] : 0;
      output_data[b * output_depth + i] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

inline void FullyConnectedSparseWeight1x16Impl(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& weights_shape, const int8_t* weights_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data, int thread_start,
    int thread_end, const CpuBackendContext& cpu_backend_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_1(mht_1_v, 255, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "FullyConnectedSparseWeight1x16Impl");

  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("1x16 Block Sparse");

  const int input_dims_count = input_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = thread_end - thread_start;
  const int input_depth = MatchingDim(weights_shape, weights_dims_count - 1,
                                      input_shape, input_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int32_t output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate1x16(
      weights_data, w1_segments, w1_indices, weights_shape.Dims(0),
      weights_shape.Dims(1), input_data + thread_start * input_depth, bias_data,
      batches, input_offset, output_multiplier, output_shift, output_offset,
      output_activation_min, output_activation_max,
      output_data + thread_start * output_depth);
}

inline void FullyConnectedSparseWeight1x4Impl(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data, int thread_start,
    int thread_end, const CpuBackendContext& cpu_backend_context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_2(mht_2_v, 294, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "FullyConnectedSparseWeight1x4Impl");

  ruy::profiler::ScopeLabel label("FullyConnected");
  ruy::profiler::ScopeLabel inner_label("1x4 Block Sparse");
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;

  const int input_dims_count = input_shape.DimensionsCount();
  const int output_dims_count = output_shape.DimensionsCount();
  const int weights_dims_count = weights_shape.DimensionsCount();
  const int batches = thread_end - thread_start;
  const int input_depth = MatchingDim(weights_shape, weights_dims_count - 1,
                                      input_shape, input_dims_count - 1);
  const int output_depth = MatchingDim(weights_shape, weights_dims_count - 2,
                                       output_shape, output_dims_count - 1);
  const int* w1_segments = sparsity.dim_metadata[1].array_segments->data;
  const int* w1_indices = sparsity.dim_metadata[1].array_indices->data;

  tensor_utils::SparseMatrixBatchVectorMultiplyAccumulate1x4(
      weights_data, w1_segments, w1_indices, weights_shape.Dims(0),
      weights_shape.Dims(1), input_data + thread_start * input_depth, batches,
      output_data + thread_start * output_depth);

  ruy::profiler::ScopeLabel activation_label("activation function");
  for (int b = thread_start; b < thread_end; ++b) {
    for (int i = 0; i < output_depth; ++i) {
      float total = output_data[b * output_depth + i];
      const float bias_value = bias_data ? bias_data[i] : 0;
      output_data[b * output_depth + i] = ActivationFunctionWithMinMax(
          total + bias_value, output_activation_min, output_activation_max);
    }
  }
}

struct FullyConnectedSparseWeight1x4Task : cpu_backend_threadpool::Task {
  FullyConnectedSparseWeight1x4Task(
      const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
      const RuntimeShape& input_shape, const float* input_data,
      const RuntimeShape& weights_shape, const float* weights_data,
      const RuntimeShape& bias_shape, const float* bias_data,
      const RuntimeShape& output_shape, float* output_data, int thread_start,
      int thread_end, const CpuBackendContext& cpu_backend_context_x)
      : sparsity(sparsity),
        params(params),
        input_shape(input_shape),
        input_data(input_data),
        weights_shape(weights_shape),
        weights_data(weights_data),
        bias_shape(bias_shape),
        bias_data(bias_data),
        output_shape(output_shape),
        output_data(output_data),
        thread_start(thread_start),
        thread_end(thread_end),
        cpu_backend_context(cpu_backend_context_x) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_3(mht_3_v, 350, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "FullyConnectedSparseWeight1x4Task");
}

  void Run() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_4(mht_4_v, 355, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "Run");

    FullyConnectedSparseWeight1x4Impl(
        sparsity, params, input_shape, input_data, weights_shape, weights_data,
        bias_shape, bias_data, output_shape, output_data, thread_start,
        thread_end, cpu_backend_context);
  }

 private:
  const TfLiteSparsity& sparsity;
  const FullyConnectedParams& params;
  const RuntimeShape& input_shape;
  const float* input_data;
  const RuntimeShape& weights_shape;
  const float* weights_data;
  const RuntimeShape& bias_shape;
  const float* bias_data;
  const RuntimeShape& output_shape;
  float* output_data;
  int thread_start;
  int thread_end;
  const CpuBackendContext& cpu_backend_context;
};

inline void FullyConnectedSparseWeight1x16(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& weights_shape, const int8_t* weights_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data,
    CpuBackendContext* cpu_backend_context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_5(mht_5_v, 387, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "FullyConnectedSparseWeight1x16");

  const int output_elements = output_shape.FlatSize();
  memset(output_data, 0, output_elements * sizeof(int8_t));

  const int batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);

  // TODO(b/220851507): Add multi-thread support for quantized sparse kernel.
  return FullyConnectedSparseWeight1x16Impl(
      sparsity, params, input_shape, input_data, weights_shape, weights_data,
      bias_shape, bias_data, output_shape, output_data, 0, batches,
      *cpu_backend_context);
}

// The multi-threaded kernel slices the workload along the batch dimension. If
// there's not enough batches of data, the number of threads used is equal to
// the batch size. We can improve this later with slicing along the row
// dimension of the weight.
inline void FullyConnectedSparseWeight1x4(
    const TfLiteSparsity& sparsity, const FullyConnectedParams& params,
    const RuntimeShape& input_shape, const float* input_data,
    const RuntimeShape& weights_shape, const float* weights_data,
    const RuntimeShape& bias_shape, const float* bias_data,
    const RuntimeShape& output_shape, float* output_data,
    CpuBackendContext* cpu_backend_context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSsparse_opsPSfully_connectedDTh mht_6(mht_6_v, 414, "", "./tensorflow/lite/kernels/internal/optimized/sparse_ops/fully_connected.h", "FullyConnectedSparseWeight1x4");

  const int output_elements = output_shape.FlatSize();
  memset(output_data, 0, output_elements * sizeof(float));

  const int max_threads = cpu_backend_context->max_num_threads();
  const int batches =
      FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int thread_count = std::max(1, std::min(batches, max_threads));
  if (thread_count == 1) {
    return FullyConnectedSparseWeight1x4Impl(
        sparsity, params, input_shape, input_data, weights_shape, weights_data,
        bias_shape, bias_data, output_shape, output_data, 0, batches,
        *cpu_backend_context);
  }
  std::vector<FullyConnectedSparseWeight1x4Task> tasks;
  tasks.reserve(thread_count);
  int thread_start = 0;
  for (int i = 0; i < thread_count; ++i) {
    // This makes sure the workload is relatively balanced when batches is not a
    // multiple of thread_count. The first mod(batches, thread_count) tasks need
    // to process one more batch than the rest.
    int thread_end = thread_start + batches / thread_count;
    if (i < batches % thread_count) thread_end++;

    tasks.emplace_back(sparsity, params, input_shape, input_data, weights_shape,
                       weights_data, bias_shape, bias_data, output_shape,
                       output_data, thread_start, thread_end,
                       *cpu_backend_context);
    thread_start = thread_end;
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
}

}  // namespace optimized_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_SPARSE_OPS_FULLY_CONNECTED_H_
