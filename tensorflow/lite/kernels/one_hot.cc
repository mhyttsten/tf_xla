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
class MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace one_hot {

constexpr int kIndicesTensor = 0;
constexpr int kDepthTensor = 1;
constexpr int kOnValueTensor = 2;
constexpr int kOffValueTensor = 3;
constexpr int kOutputTensor = 0;

// Convenience utility for destructuring a node into the appropriate tensors and
// data for the op. Note that this destructuring is quite cheap, so we can avoid
// allocating op-specific, persistent data on the heap.
struct OneHotContext {
  OneHotContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/one_hot.cc", "OneHotContext");

    indices = GetInput(context, node, kIndicesTensor);
    depth = GetInput(context, node, kDepthTensor);
    on_value = GetInput(context, node, kOnValueTensor);
    off_value = GetInput(context, node, kOffValueTensor);
    output = GetOutput(context, node, kOutputTensor);

    const auto* params =
        reinterpret_cast<TfLiteOneHotParams*>(node->builtin_data);
    const int indices_dims = indices->dims->size;
    axis = (params->axis == -1) ? indices_dims : params->axis;
    output_dims = indices_dims + 1;
    dtype = on_value->type;
  }

  const TfLiteTensor* indices;
  const TfLiteTensor* depth;
  const TfLiteTensor* on_value;
  const TfLiteTensor* off_value;
  TfLiteTensor* output;
  int axis;
  int output_dims;
  TfLiteType dtype;
};

template <typename T, typename TI>
void OneHotComputeImpl(const OneHotContext& op_context) {
  // prefix_dim_size == # of elements before the axis
  // depth == # of elements per axis
  // suffix_dim_size == # of elements after the axis
  int prefix_dim_size = 1;
  for (int i = 0; i < op_context.axis; ++i) {
    prefix_dim_size *= op_context.indices->dims->data[i];
  }
  if (prefix_dim_size == 0) {
    // If indices tensor is degenerate, return a degenerate tensor, just like
    // TensorFlow does.
    return;
  }
  const int suffix_dim_size = NumElements(op_context.indices) / prefix_dim_size;
  const int depth = *op_context.depth->data.i32;

  const T on_value = *GetTensorData<T>(op_context.on_value);
  const T off_value = *GetTensorData<T>(op_context.off_value);

  // View the indices as a matrix of size:
  //     prefix_dim_size x suffix_dim_size
  // View the output as a matrix of size:
  //     prefix_dim_size x depth x suffix_dim_size
  // Then the output is:
  //     output(i, j, k) == (indices(i, k) == j) ? on : off
  T* output = GetTensorData<T>(op_context.output);
  const TI* indices = GetTensorData<TI>(op_context.indices);
  for (int i = 0; i < prefix_dim_size; ++i) {
    for (int j = 0; j < depth; ++j) {
      for (int k = 0; k < suffix_dim_size; ++k, ++output) {
        *output = static_cast<int>(indices[i * suffix_dim_size + k]) == j
                      ? on_value
                      : off_value;
      }
    }
  }
}

template <typename T>
void OneHotCompute(const OneHotContext& op_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc mht_1(mht_1_v, 275, "", "./tensorflow/lite/kernels/one_hot.cc", "OneHotCompute");

  if (op_context.indices->type == kTfLiteInt64) {
    OneHotComputeImpl<T, int64_t>(op_context);
  } else {
    OneHotComputeImpl<T, int>(op_context);
  }
}

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const OneHotContext& op_context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc mht_2(mht_2_v, 287, "", "./tensorflow/lite/kernels/one_hot.cc", "ResizeOutputTensor");

  TF_LITE_ENSURE(context, *op_context.depth->data.i32 >= 0);
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(op_context.output_dims);
  for (int i = 0; i < op_context.output_dims; ++i) {
    if (i < op_context.axis) {
      output_size->data[i] = op_context.indices->dims->data[i];
    } else if (i == op_context.axis) {
      output_size->data[i] = *op_context.depth->data.i32;
    } else {
      output_size->data[i] = op_context.indices->dims->data[i - 1];
    }
  }
  return context->ResizeTensor(context, op_context.output, output_size);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc mht_3(mht_3_v, 305, "", "./tensorflow/lite/kernels/one_hot.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OneHotContext op_context{context, node};
  switch (op_context.dtype) {
    // TODO(b/111744875): Support uint8 and quantization.
    case kTfLiteFloat32:
    case kTfLiteInt16:
    case kTfLiteInt32:
    case kTfLiteInt64:
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteBool:
      op_context.output->type = op_context.dtype;
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Unknown output data type: %s",
                         TfLiteTypeGetName(op_context.dtype));
      return kTfLiteError;
  }

  TF_LITE_ENSURE(context, op_context.indices->type == kTfLiteInt32 ||
                              op_context.indices->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, op_context.axis >= 0 &&
                              op_context.axis < op_context.output_dims);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.depth), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.on_value), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.off_value), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.on_value->type, op_context.dtype);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.off_value->type,
                          op_context.dtype);

  if (!IsConstantTensor(op_context.depth)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }

  return ResizeOutputTensor(context, op_context);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc mht_4(mht_4_v, 349, "", "./tensorflow/lite/kernels/one_hot.cc", "Eval");

  OneHotContext op_context{context, node};

  if (IsDynamicTensor(op_context.output)) {
    ResizeOutputTensor(context, op_context);
  }

  switch (op_context.output->type) {
    case kTfLiteFloat32:
      OneHotCompute<float>(op_context);
      break;
    case kTfLiteInt32:
      OneHotCompute<int>(op_context);
      break;
    case kTfLiteInt64:
      OneHotCompute<int64_t>(op_context);
      break;
    case kTfLiteInt8:
      OneHotCompute<int8_t>(op_context);
      break;
    case kTfLiteUInt8:
      OneHotCompute<uint8_t>(op_context);
      break;
    case kTfLiteBool:
      OneHotCompute<bool>(op_context);
      break;
    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace one_hot

TfLiteRegistration* Register_ONE_HOT() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSone_hotDTcc mht_5(mht_5_v, 387, "", "./tensorflow/lite/kernels/one_hot.cc", "Register_ONE_HOT");

  static TfLiteRegistration r = {
      nullptr,
      nullptr,
      one_hot::Prepare,
      one_hot::Eval,
  };
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
