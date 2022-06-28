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
class MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc() {
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
#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"

#include <string.h>

#include <cstdint>
#include <memory>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace broadcastto {

constexpr int kInputTensor = 0;
constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;
constexpr int kMaxDims = 8;

struct BroadcastToContext {
  BroadcastToContext(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/broadcast_to.cc", "BroadcastToContext");

    input = GetInput(context, node, kInputTensor);
    shape = GetInput(context, node, kShapeTensor);
    output = GetOutput(context, node, kOutputTensor);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* shape;
  TfLiteTensor* output;
};

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                BroadcastToContext* op_context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc mht_1(mht_1_v, 220, "", "./tensorflow/lite/kernels/broadcast_to.cc", "ResizeOutputTensor");

  // Ensures the shape is 1D tensor.
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context->shape), 1);

  // Ensure output dims is not less than input dims.
  int input_num_dims = NumDimensions(op_context->input);
  int output_num_dims = SizeOfDimension(op_context->shape, 0);
  TF_LITE_ENSURE_MSG(context, input_num_dims <= output_num_dims,
                     "Output shape must be broadcastable from input shape.");
  TF_LITE_ENSURE_MSG(context, output_num_dims <= kMaxDims,
                     "BroadcastTo only supports 1-8D tensor.");

  // Check if output shape is broadcastable from input shape.
  auto get_shape_data = [op_context](int i) -> int32_t {
    if (op_context->shape->type == kTfLiteInt32) {
      return GetTensorData<int32_t>(op_context->shape)[i];
    } else {
      return GetTensorData<int64_t>(op_context->shape)[i];
    }
  };

  int extending_dims = output_num_dims - input_num_dims;
  for (int idx = 0; idx < input_num_dims; ++idx) {
    TF_LITE_ENSURE_MSG(context,
                       (SizeOfDimension(op_context->input, idx) == 1 ||
                        SizeOfDimension(op_context->input, idx) ==
                            get_shape_data(extending_dims + idx)),
                       "Output shape must be broadcastable from input shape.");
  }
  // Resizing the shape of the output tensor.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_num_dims);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)>
      scoped_output_shape(output_shape, TfLiteIntArrayFree);
  for (int idx = 0; idx < output_num_dims; ++idx) {
    output_shape->data[idx] = get_shape_data(idx);
  }

  return context->ResizeTensor(context, op_context->output,
                               scoped_output_shape.release());
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc mht_2(mht_2_v, 264, "", "./tensorflow/lite/kernels/broadcast_to.cc", "Prepare");

  TF_LITE_ENSURE(context, NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE_MSG(context,
                     (NumDimensions(GetInput(context, node, 0)) <= kMaxDims),
                     "BroadcastTo only supports 1-8D tensor.");

  BroadcastToContext op_context(context, node);
  TF_LITE_ENSURE(context, op_context.shape->type == kTfLiteInt32 ||
                              op_context.shape->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // Not yet support string type due to the use of memcopy with fixed size.
  TF_LITE_ENSURE(context, op_context.input->type != kTfLiteString);

  if (IsConstantTensor(op_context.shape)) {
    return ResizeOutputTensor(context, &op_context);
  }

  SetTensorToDynamic(op_context.output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc mht_3(mht_3_v, 290, "", "./tensorflow/lite/kernels/broadcast_to.cc", "Eval");

  BroadcastToContext op_context(context, node);
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

  // BroadcastTo op support upto 8 dims, matching the support of Tensorflow.
  reference_ops::BroadcastTo<kMaxDims>(
      GetTensorShape(op_context.input), op_context.input->data.raw,
      GetTensorShape(op_context.output), op_context.output->data.raw,
      op_context.input->type);
  return kTfLiteOk;
}

}  // namespace broadcastto

TfLiteRegistration* Register_BROADCAST_TO() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSbroadcast_toDTcc mht_4(mht_4_v, 309, "", "./tensorflow/lite/kernels/broadcast_to.cc", "Register_BROADCAST_TO");

  static TfLiteRegistration r = {nullptr, nullptr, broadcastto::Prepare,
                                 broadcastto::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
