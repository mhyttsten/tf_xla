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
class MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/conv3d_transpose.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv3d_transpose {

enum KernelType {
  kReference,
  kGenericOptimized,
};

const int kTensorNotAllocated = -1;

struct OpData {
  Padding3DValues padding;

  // The id of the temporary col2im tensor.
  int col2im_id = kTensorNotAllocated;

  // The index of col2im tensor in the temporaries list.
  int col2im_index;

  bool need_col2im = false;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_0(mht_0_v, 223, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Init");

  auto* opdata = new OpData;
  return opdata;
}

void Free(TfLiteContext* context, void* buffer) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Free");

  delete static_cast<OpData*>(buffer);
}

static TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context,
                                                       TfLiteNode* node,
                                                       KernelType kernel_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "AllocateTemporaryTensorsIfRequired");

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int temporaries_count = 0;

  // Allocate col2im tensor for the optimized kernel.
  if (kernel_type == kGenericOptimized) {
    if (data->col2im_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->col2im_id);
    }
    data->col2im_index = temporaries_count++;
    data->need_col2im = true;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  return kTfLiteOk;
}

TfLiteStatus ResizeOutputAndTemporaryTensors(
    TfLiteContext* context, OpData* opdata, TfLiteConv3DTransposeParams* params,
    const TfLiteTensor* shape_tensor, const TfLiteTensor* filter,
    const TfLiteTensor* input, TfLiteTensor* col2im, TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_3(mht_3_v, 265, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "ResizeOutputAndTemporaryTensors");

  auto shape_data = GetTensorData<int32_t>(shape_tensor);
  // Output and input tensor must have the same batch size.
  TF_LITE_ENSURE_EQ(context, shape_data[0], SizeOfDimension(input, 0));
  // The number of channels of output must be divisible by that of filter.
  TF_LITE_ENSURE_EQ(context, shape_data[4] % SizeOfDimension(filter, 3), 0);

  // Compute padding.
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const int depth = shape_data[1];
  const int height = shape_data[2];
  const int width = shape_data[3];
  const int filter_depth = filter_shape.Dims(0);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  int unused_out_width, unused_out_height, unused_out_depth;
  opdata->padding = ComputePadding3DValues(
      params->stride_height, params->stride_width, params->stride_depth,
      params->dilation_height_factor, params->dilation_width_factor,
      params->dilation_depth_factor, height, width, depth, filter_height,
      filter_width, filter_depth, params->padding, &unused_out_height,
      &unused_out_width, &unused_out_depth);
  // Computed shape must match the shape of the input tensor.
  TF_LITE_ENSURE_EQ(context, unused_out_depth, SizeOfDimension(input, 1));
  TF_LITE_ENSURE_EQ(context, unused_out_height, SizeOfDimension(input, 2));
  TF_LITE_ENSURE_EQ(context, unused_out_width, SizeOfDimension(input, 3));

  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(NumElements(shape_tensor));
  for (int i = 0; i < output_shape->size; ++i) {
    output_shape->data[i] = GetTensorData<int32_t>(shape_tensor)[i];
  }

  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));

  // Resize col2im tensor.
  if (opdata->need_col2im) {
    TfLiteIntArray* col2im_shape_array = TfLiteIntArrayCreate(2);
    const RuntimeShape& input_shape = GetTensorShape(input);
    col2im_shape_array->data[0] =
        input_shape.Dims(1) * input_shape.Dims(2) * input_shape.Dims(3);
    col2im_shape_array->data[1] =
        filter_depth * filter_height * filter_width * filter_shape.Dims(3);

    col2im->type = kTfLiteFloat32;
    col2im->allocation_type = kTfLiteDynamic;
    return context->ResizeTensor(context, col2im, col2im_shape_array);
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(KernelType kernel_type, TfLiteContext* context,
                     TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_4(mht_4_v, 320, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Prepare");

  auto* params =
      reinterpret_cast<TfLiteConv3DTransposeParams*>(node->builtin_data);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  // Check number of inputs/outputs.
  TF_LITE_ENSURE(context, node->inputs->size == 3 || node->inputs->size == 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &output_shape));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &input));

  // Check dimensionality of inputs/outputs.
  TF_LITE_ENSURE_EQ(context, output_shape->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 5);
  TF_LITE_ENSURE_EQ(context, input->dims->size, 5);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 5);

  // Input and filter must have the same number of channels.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, 4),
                    SizeOfDimension(filter, 4));

  // Check types.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
  TF_LITE_ENSURE_TYPES_EQ(context, output_shape->type, kTfLiteInt32);

  // Check bias.
  const TfLiteTensor* bias = GetInput(context, node, 3);
  if (bias) {
    TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input->type);
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 3));
  }

  // GenericOptimized kernel currently doesn't support dilation.
  if (params->dilation_depth_factor > 1 || params->dilation_height_factor > 1 ||
      params->dilation_width_factor > 1) {
    kernel_type = kReference;
  }

  // Allocate temporary tensors.
  TF_LITE_ENSURE_STATUS(
      AllocateTemporaryTensorsIfRequired(context, node, kernel_type));

  // Check temporary tensors.
  TfLiteTensor* col2im = nullptr;
  if (opdata->need_col2im) {
    node->temporaries->data[opdata->col2im_index] = opdata->col2im_id;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                                opdata->col2im_index, &col2im));
  }

  // Resize the output tensor.
  if (!IsConstantTensor(output_shape)) {
    SetTensorToDynamic(output);
    if (opdata->need_col2im) {
      SetTensorToDynamic(col2im);
    }
  } else {
    TF_LITE_ENSURE_STATUS(ResizeOutputAndTemporaryTensors(
        context, opdata, params, output_shape, filter, input, col2im, output));
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_5(mht_5_v, 394, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Prepare");

  return Prepare(kernel_type, context, node);
}

void EvalFloat(KernelType kernel_type, TfLiteContext* context, TfLiteNode* node,
               TfLiteConv3DTransposeParams* params, OpData* opdata,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* col2im,
               TfLiteTensor* output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_6(mht_6_v, 405, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "EvalFloat");

  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  Conv3DTransposeParams runtime_params;
  runtime_params.padding_values = opdata->padding;
  runtime_params.stride_depth = params->stride_depth;
  runtime_params.stride_height = params->stride_height;
  runtime_params.stride_width = params->stride_width;
  runtime_params.dilation_depth = params->dilation_depth_factor;
  runtime_params.dilation_height = params->dilation_height_factor;
  runtime_params.dilation_width = params->dilation_width_factor;
  runtime_params.float_activation_min = output_activation_min;
  runtime_params.float_activation_max = output_activation_max;

  switch (kernel_type) {
    case kReference: {
      reference_ops::Conv3DTranspose(
          runtime_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    }
    case kGenericOptimized: {
      optimized_ops::Conv3DTranspose(
          runtime_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(col2im), GetTensorData<float>(col2im),
          CpuBackendContext::GetFromContext(context));
    } break;
  }
}

TfLiteStatus Eval(KernelType kernel_type, TfLiteContext* context,
                  TfLiteNode* node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_7(mht_7_v, 446, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Eval");

  auto* params =
      reinterpret_cast<TfLiteConv3DTransposeParams*>(node->builtin_data);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &output_shape));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &input));
  const TfLiteTensor* bias = GetInput(context, node, 3);
  TfLiteTensor* col2im = opdata->need_col2im
                             ? GetTemporary(context, node, opdata->col2im_index)
                             : nullptr;

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputAndTemporaryTensors(
                                   context, opdata, params, output_shape,
                                   filter, input, col2im, output));
  }

  // GenericOptimized kernel currently doesn't support dilation.
  if (params->dilation_depth_factor > 1 || params->dilation_height_factor > 1 ||
      params->dilation_width_factor > 1) {
    kernel_type = kReference;
  }

  switch (input->type) {
    case kTfLiteFloat32:
      EvalFloat(kernel_type, context, node, params, opdata, input, filter, bias,
                col2im, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_8(mht_8_v, 493, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Eval");

  return Eval(kernel_type, context, node);
}

}  // namespace conv3d_transpose

TfLiteRegistration* Register_CONV_3D_TRANSPOSE_REF() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_9(mht_9_v, 502, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Register_CONV_3D_TRANSPOSE_REF");

  static TfLiteRegistration r = {
      conv3d_transpose::Init, conv3d_transpose::Free,
      conv3d_transpose::Prepare<conv3d_transpose::kReference>,
      conv3d_transpose::Eval<conv3d_transpose::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONV_3D_TRANSPOSE_GENERIC_OPT() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_10(mht_10_v, 513, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Register_CONV_3D_TRANSPOSE_GENERIC_OPT");

  static TfLiteRegistration r = {
      conv3d_transpose::Init, conv3d_transpose::Free,
      conv3d_transpose::Prepare<conv3d_transpose::kGenericOptimized>,
      conv3d_transpose::Eval<conv3d_transpose::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONV_3D_TRANSPOSE() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSconv3d_transposeDTcc mht_11(mht_11_v, 524, "", "./tensorflow/lite/kernels/conv3d_transpose.cc", "Register_CONV_3D_TRANSPOSE");

  return Register_CONV_3D_TRANSPOSE_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
