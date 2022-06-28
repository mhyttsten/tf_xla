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
class MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc() {
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

#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace strided_slice {

enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kEndTensor = 2;
constexpr int kStridesTensor = 3;
constexpr int kOutputTensor = 0;

struct StridedSliceContext {
  StridedSliceContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_0(mht_0_v, 220, "", "./tensorflow/lite/kernels/strided_slice.cc", "StridedSliceContext");

    params = reinterpret_cast<TfLiteStridedSliceParams*>(node->builtin_data);
    input = GetInput(context, node, kInputTensor);
    begin = GetInput(context, node, kBeginTensor);
    end = GetInput(context, node, kEndTensor);
    strides = GetInput(context, node, kStridesTensor);
    output = GetOutput(context, node, kOutputTensor);
    input_dims = NumDimensions(input);
  }
  const TfLiteStridedSliceParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* begin;
  const TfLiteTensor* end;
  const TfLiteTensor* strides;
  TfLiteTensor* output;

  // Equivalent input shape after adding axis according to new_axis_mask.
  RuntimeShape effective_input_shape;
  int input_dims;
};

StridedSliceParams BuildStridedSliceParams(StridedSliceContext* op_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_1(mht_1_v, 244, "", "./tensorflow/lite/kernels/strided_slice.cc", "BuildStridedSliceParams");

  StridedSliceParams op_params;

  // The ellipsis_mask and new_axis_mask in op_params are not used. Those masks
  // are processed here to update begin_mask, end_mask and the index range.
  op_params.begin_mask = 0;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = 0;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = 0;

  // Count indexes where the new_axis_mask is set but the ellipsis_mask is not.
  const int begin_count = GetTensorShape(op_context->begin).Dims(0);
  int num_add_axis = 0;
  for (int i = 0; i < begin_count; ++i) {
    if (!((1 << i) & op_context->params->ellipsis_mask) &&
        ((1 << i) & op_context->params->new_axis_mask)) {
      num_add_axis++;
    }
  }

  // Calculate the dims of input after adding new axises.
  const int effective_dims = op_context->input_dims + num_add_axis;

  // If begin, end and strides are not fully provided, it means Ellipsis should
  // be expanded to multiple dimensions (Ex: for spec [Ellipsis, 2] on a 3D
  // input, the Ellipsis should be applied for the first 2 dimensions). Besides,
  // If the new_axis_mask and the ellipsis_mask are set at the same index, the
  // new_axis_mask will have no effect.
  int effective_ellipsis_mask = 0, effective_new_axis_mask = 0;
  int ellipsis_start_idx = effective_dims, expanded_ellipsis = 0;
  for (int i = 0; i < effective_dims;) {
    if ((1 << i) & op_context->params->ellipsis_mask) {
      ellipsis_start_idx = i;
      int ellipsis_end_idx = std::max(
          i + 1,
          std::min(i + 1 + num_add_axis + op_context->input_dims - begin_count,
                   effective_dims));
      expanded_ellipsis = ellipsis_end_idx - ellipsis_start_idx - 1;

      // Set bit for effective_ellipsis_mask.
      for (; i < ellipsis_end_idx; ++i) {
        effective_ellipsis_mask |= (1 << i);
      }
      continue;
    }

    if ((1 << (i - expanded_ellipsis)) & op_context->params->new_axis_mask) {
      effective_new_axis_mask |= (1 << i);
    }
    ++i;
  }

  // Calculate effective_input_shape and its corresponding begin, end, strides.
  const int32_t* begin_data = GetTensorData<int32_t>(op_context->begin);
  const int32_t* end_data = GetTensorData<int32_t>(op_context->end);
  const int32_t* strides_data = GetTensorData<int32_t>(op_context->strides);
  const RuntimeShape input_shape = GetTensorShape(op_context->input);
  int added_ellipsis = 0, added_axises = 0;
  op_context->effective_input_shape.Resize(effective_dims);

  for (int i = 0; i < effective_dims; ++i) {
    if ((1 << i) & effective_ellipsis_mask) {
      // If ellipsis_mask, set the begin_mask and end_mask at that index.
      added_ellipsis = std::max(0, i - ellipsis_start_idx);
      op_params.begin_mask |= (1 << i);
      op_params.end_mask |= (1 << i);
      op_params.strides[i] = 1;
      op_context->effective_input_shape.SetDim(
          i, input_shape.Dims(i - added_axises));
    } else if ((1 << i) & effective_new_axis_mask) {
      // If new_axis_mask is set, it is equivalent to adding a new dim of 1 to
      // input tensor. Store added shape to effective_input_shape.
      op_params.start_indices[i] = 0;
      op_params.stop_indices[i] = 1;
      op_params.strides[i] = 1;
      op_context->effective_input_shape.SetDim(i, 1);
      added_axises++;
    } else if (i >= begin_count + expanded_ellipsis) {
      op_params.start_indices[i] = 0;
      op_params.stop_indices[i] = 0;
      op_params.strides[i] = 1;
      op_params.begin_mask |= (1 << i);
      op_params.end_mask |= (1 << i);
      op_context->effective_input_shape.SetDim(
          i, input_shape.Dims(i - added_axises));
    } else {
      const int orig_idx = i - added_ellipsis;
      op_params.start_indices[i] = begin_data[orig_idx];
      op_params.stop_indices[i] = end_data[orig_idx];
      op_params.strides[i] = strides_data[orig_idx];
      if (op_context->params->begin_mask & (1 << orig_idx)) {
        op_params.begin_mask |= (1 << i);
      }
      if (op_context->params->end_mask & (1 << orig_idx)) {
        op_params.end_mask |= (1 << i);
      }
      if (op_context->params->shrink_axis_mask & (1 << orig_idx)) {
        op_params.shrink_axis_mask |= (1 << i);
      }
      op_context->effective_input_shape.SetDim(
          i, input_shape.Dims(i - added_axises));
    }
  }
  op_params.start_indices_count = effective_dims;
  op_params.stop_indices_count = effective_dims;
  op_params.strides_count = effective_dims;

  return op_params;
}

// Processes the indexing tensors (begin, end and strides) to resize the
// output tensor. This function is callable from both Prepare() and Eval() as
// long as the caller ensures the indexing tensors are present.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                StridedSliceContext* op_context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_2(mht_2_v, 362, "", "./tensorflow/lite/kernels/strided_slice.cc", "ResizeOutputTensor");

  std::vector<int> output_shape_vector;
  StridedSliceParams op_params = BuildStridedSliceParams(op_context);
  const RuntimeShape effective_input_shape = op_context->effective_input_shape;
  TF_LITE_ENSURE_MSG(
      context, effective_input_shape.DimensionsCount() <= 5,
      "StridedSlice op only supports up to 5D output including added axis.");

  for (int idx = effective_input_shape.DimensionsCount() - 1; idx >= 0; --idx) {
    int32_t stride = op_params.strides[idx];
    TF_LITE_ENSURE_MSG(context, stride != 0, "stride value has to be non-zero");

    int32_t begin = ::tflite::strided_slice::StartForAxis(
        op_params, effective_input_shape, idx);
    int32_t end = ::tflite::strided_slice::StopForAxis(
        op_params, effective_input_shape, idx, begin);

    // When shrinking an axis, the end position does not matter (and can be
    // incorrect when negative indexing is used, see Issue #19260). Always use
    // begin + 1 to generate a length 1 slice, since begin has
    // already been adjusted for negative indices by GetBeginValueAtIndex.
    const bool shrink_axis = op_params.shrink_axis_mask & (1 << idx);
    if (shrink_axis) {
      end = begin + 1;
    }

    // This is valid for both positive and negative strides
    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis) {
      output_shape_vector.push_back(dim_shape);
    }
  }

  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(output_shape_vector.size());

  std::reverse_copy(output_shape_vector.begin(), output_shape_vector.end(),
                    output_shape->data);

  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, op_context->output, output_shape));

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_3(mht_3_v, 411, "", "./tensorflow/lite/kernels/strided_slice.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  StridedSliceContext op_context(context, node);

  // Ensure validity of input tensor and its dimension
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.end), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.strides), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.begin),
                    NumElements(op_context.end));
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // Only INT32 begin/end/strides are supported
  // TODO(b/175642009): add support for INT64
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.begin->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.end->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.strides->type, kTfLiteInt32);
  TF_LITE_ENSURE_MSG(context, op_context.input_dims <= 5,
                     "StridedSlice op only supports 1D-5D input arrays.");

  // Postpone allocation of output if any of the indexing tensors is not
  // constant
  if (!(IsConstantTensor(op_context.begin) &&
        IsConstantTensor(op_context.end) &&
        IsConstantTensor(op_context.strides))) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_4(mht_4_v, 448, "", "./tensorflow/lite/kernels/strided_slice.cc", "Eval");

  StridedSliceContext op_context(context, node);

  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }
  StridedSliceParams op_params = BuildStridedSliceParams(&op_context);

#define TF_LITE_STRIDED_SLICE(data_type)                                 \
  {                                                                      \
    if (kernel_type == kGenericOptimized) {                              \
      optimized_ops::StridedSlice<data_type>(                            \
          op_params, op_context.effective_input_shape, op_context.input, \
          GetTensorShape(op_context.output), op_context.output);         \
    } else {                                                             \
      reference_ops::StridedSlice<data_type>(                            \
          op_params, op_context.effective_input_shape, op_context.input, \
          GetTensorShape(op_context.output), op_context.output);         \
    }                                                                    \
  }

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      TF_LITE_STRIDED_SLICE(float);
      break;
    case kTfLiteInt32:
      TF_LITE_STRIDED_SLICE(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_STRIDED_SLICE(int64_t);
      break;
    case kTfLiteUInt8:
      TF_LITE_STRIDED_SLICE(uint8_t);
      break;
    case kTfLiteInt8:
      TF_LITE_STRIDED_SLICE(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_STRIDED_SLICE(int16_t);
      break;
    case kTfLiteBool:
      TF_LITE_STRIDED_SLICE(bool);
      break;
    case kTfLiteString:
      TF_LITE_STRIDED_SLICE(string);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported "
                         "by StridedSlice.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_STRIDED_SLICE
  return kTfLiteOk;
}

}  // namespace strided_slice

TfLiteRegistration* Register_STRIDED_SLICE_REF() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_5(mht_5_v, 510, "", "./tensorflow/lite/kernels/strided_slice.cc", "Register_STRIDED_SLICE_REF");

  static TfLiteRegistration r = {
      nullptr, nullptr, strided_slice::Prepare,
      strided_slice::Eval<strided_slice::kReference>};
  return &r;
}

TfLiteRegistration* Register_STRIDED_SLICE() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSstrided_sliceDTcc mht_6(mht_6_v, 520, "", "./tensorflow/lite/kernels/strided_slice.cc", "Register_STRIDED_SLICE");

  static TfLiteRegistration r = {
      nullptr, nullptr, strided_slice::Prepare,
      strided_slice::Eval<strided_slice::kGenericOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
