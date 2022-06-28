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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh() {
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
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace optimized_integer_ops {

inline void MeanImpl(const tflite::MeanParams& op_params,
                     const RuntimeShape& input_shape, const int8_t* input_data,
                     int32 multiplier, int32 shift, int32 bias,
                     const RuntimeShape& output_shape, int8_t* output_data,
                     int start_depth, int end_depth) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh mht_0(mht_0_v, 199, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h", "MeanImpl");

  ruy::profiler::ScopeLabel label("Mean4D/Int8/MeanImpl");

  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  const int output_batch = output_shape.Dims(0);
  const int output_height = output_shape.Dims(2);
  const int output_width = output_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  constexpr static int32_t kMinValue = std::numeric_limits<int8_t>::min();
  constexpr static int32_t kMaxValue = std::numeric_limits<int8_t>::max();

#ifdef USE_NEON
  const int32x4_t bias_dup = vdupq_n_s32(bias);
  const int32x4_t min_dup = vdupq_n_s32(kMinValue);
  const int32x4_t max_dup = vdupq_n_s32(kMaxValue);
#endif  // USE_NEON
  for (int out_b = 0; out_b < output_batch; ++out_b) {
    int out_d = start_depth;
#ifdef USE_NEON

    for (; out_d <= end_depth - 16; out_d += 16) {
      int32x4x4_t temp_sum;
      temp_sum.val[0] = vdupq_n_s32(0);
      temp_sum.val[1] = vdupq_n_s32(0);
      temp_sum.val[2] = vdupq_n_s32(0);
      temp_sum.val[3] = vdupq_n_s32(0);
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          const int8_t* input_data_ptr =
              input_data + Offset(input_shape, out_b, in_h, in_w, out_d);
          int8x16_t input_data_val = vld1q_s8(input_data_ptr);

          int16x8_t input_data_low_shift =
              vmovl_s8(vget_low_s8(input_data_val));
          int16x8_t input_data_high_shift =
              vmovl_s8(vget_high_s8(input_data_val));

          int32x4_t input_low_low =
              vmovl_s16(vget_low_s16(input_data_low_shift));
          int32x4_t input_high_low =
              vmovl_s16(vget_high_s16(input_data_low_shift));
          int32x4_t input_low_high =
              vmovl_s16(vget_low_s16(input_data_high_shift));
          int32x4_t input_high_high =
              vmovl_s16(vget_high_s16(input_data_high_shift));

          temp_sum.val[0] = vaddq_s32(temp_sum.val[0], input_low_low);
          temp_sum.val[1] = vaddq_s32(temp_sum.val[1], input_high_low);
          temp_sum.val[2] = vaddq_s32(temp_sum.val[2], input_low_high);
          temp_sum.val[3] = vaddq_s32(temp_sum.val[3], input_high_high);
        }
      }

      temp_sum =
          MultiplyByQuantizedMultiplier4Rows(temp_sum, multiplier, shift);

      temp_sum.val[0] = vaddq_s32(temp_sum.val[0], bias_dup);
      temp_sum.val[1] = vaddq_s32(temp_sum.val[1], bias_dup);
      temp_sum.val[2] = vaddq_s32(temp_sum.val[2], bias_dup);
      temp_sum.val[3] = vaddq_s32(temp_sum.val[3], bias_dup);

      temp_sum.val[0] = vminq_s32(vmaxq_s32(temp_sum.val[0], min_dup), max_dup);
      temp_sum.val[1] = vminq_s32(vmaxq_s32(temp_sum.val[1], min_dup), max_dup);
      temp_sum.val[2] = vminq_s32(vmaxq_s32(temp_sum.val[2], min_dup), max_dup);
      temp_sum.val[3] = vminq_s32(vmaxq_s32(temp_sum.val[3], min_dup), max_dup);

      int16x4_t narrowed_low_low = vmovn_s32(temp_sum.val[0]);
      int16x4_t narrowed_high_low = vmovn_s32(temp_sum.val[1]);
      int16x4_t narrowed_low_high = vmovn_s32(temp_sum.val[2]);
      int16x4_t narrowed_high_high = vmovn_s32(temp_sum.val[3]);

      int16x8_t combined_low =
          vcombine_s16(narrowed_low_low, narrowed_high_low);
      int16x8_t combined_high =
          vcombine_s16(narrowed_low_high, narrowed_high_high);

      int8x8_t narrowed_low = vmovn_s16(combined_low);
      int8x8_t narrowed_high = vmovn_s16(combined_high);

      int8x16_t combined_output = vcombine_s8(narrowed_low, narrowed_high);

      int8_t* output_data_ptr =
          output_data + Offset(output_shape, out_b, 0, 0, out_d);
      vst1q_s8(output_data_ptr, combined_output);
    }
#endif  // USE_NEON

    for (; out_d < end_depth; ++out_d) {
      int acc = 0;
      for (int in_h = 0; in_h < input_height; ++in_h) {
        for (int in_w = 0; in_w < input_width; ++in_w) {
          acc += input_data[Offset(input_shape, out_b, in_h, in_w, out_d)];
        }
      }

      acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
      acc += bias;
      acc = std::min(std::max(acc, kMinValue), kMaxValue);
      output_data[Offset(output_shape, out_b, 0, 0, out_d)] =
          static_cast<int8_t>(acc);
    }
  }
}

struct MeanWorkerTask : cpu_backend_threadpool::Task {
  MeanWorkerTask(const tflite::MeanParams& op_params,
                 const RuntimeShape& input_shape, const int8_t* input_data,
                 int32 multiplier, int32 shift, int32 bias,
                 const RuntimeShape& output_shape, int8_t* output_data,
                 int start_height, int end_height)
      : op_params(op_params),
        input_shape(input_shape),
        input_data(input_data),
        multiplier(multiplier),
        shift(shift),
        bias(bias),
        output_shape(output_shape),
        output_data(output_data),
        start_height(start_height),
        end_height(end_height) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh mht_1(mht_1_v, 330, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h", "MeanWorkerTask");
}

  void Run() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh mht_2(mht_2_v, 335, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h", "Run");

    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, start_height, end_height);
  }

 private:
  const tflite::MeanParams& op_params;
  const RuntimeShape& input_shape;
  const int8_t* input_data;
  int32 multiplier;
  int32 shift;
  int32 bias;
  const RuntimeShape& output_shape;
  int8_t* output_data;
  int start_height;
  int end_height;
};

inline void Mean(const tflite::MeanParams& op_params,
                 const RuntimeShape& unextended_input_shape,
                 const int8_t* input_data, int32 input_zero_point,
                 float input_scale, const RuntimeShape& unextended_output_shape,
                 int8_t* output_data, int32 output_zero_point,
                 float output_scale, CpuBackendContext* cpu_backend_context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSoptimizedPSinteger_opsPSmeanDTh mht_3(mht_3_v, 361, "", "./tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h", "Mean");

  ruy::profiler::ScopeLabel label("Mean4D/Int8");
  // Current implementation only supports dimension equals 4 and simultaneous
  // reduction over width and height.
  TFLITE_CHECK_EQ(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_CHECK_LE(unextended_output_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  TFLITE_CHECK_EQ(op_params.axis_count, 2);
  TFLITE_CHECK((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1));
  TFLITE_CHECK_EQ(output_height, 1);
  TFLITE_CHECK_EQ(output_width, 1);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const float num_elements_in_axis = input_width * input_height;

  float temp = input_zero_point * input_scale / output_scale;
  temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
  int32_t bias = output_zero_point - static_cast<int32_t>(temp);
  float real_scale = input_scale / (num_elements_in_axis * output_scale);

  int32 multiplier, shift;
  QuantizeMultiplier(real_scale, &multiplier, &shift);

  constexpr int kMinDepthPerThread = 8;
  int thread_count = output_depth / kMinDepthPerThread;
  thread_count = thread_count > 0 ? thread_count : 1;
  const int capped_thread_count =
      std::min(thread_count, cpu_backend_context->max_num_threads());

  if (capped_thread_count == 1) {
    MeanImpl(op_params, input_shape, input_data, multiplier, shift, bias,
             output_shape, output_data, 0, output_depth);
  } else {
    // Instead parallel for batch, we loop for the output_depth since batch
    // is typical 1.
    std::vector<MeanWorkerTask> tasks;
    // TODO(b/131746020) don't create new heap allocations every time.
    // At least we make it a single heap allocation by using reserve().
    tasks.reserve(capped_thread_count);
    int depth_start = 0;
    for (int i = 0; i < capped_thread_count; ++i) {
      // Try to distribute the tasks as even as possible.
      int depth_end = depth_start +
                      (output_depth - depth_start) / (capped_thread_count - i);
      tasks.emplace_back(op_params, input_shape, input_data, multiplier, shift,
                         bias, output_shape, output_data, depth_start,
                         depth_end);
      depth_start = depth_end;
    }
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                    cpu_backend_context);
  }
}

}  // namespace optimized_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_MEAN_H_
