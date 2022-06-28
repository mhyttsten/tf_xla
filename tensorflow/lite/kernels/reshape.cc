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
class MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc() {
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

#include <cstdint>
#include <cstring>
#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reshape {

constexpr int kInputTensor = 0;
constexpr int kShapeTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteIntArray* GetOutputShape(TfLiteContext*, TfLiteNode*);

TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_0(mht_0_v, 205, "", "./tensorflow/lite/kernels/reshape.cc", "ResizeOutput");

  TfLiteIntArray* output_shape = GetOutputShape(context, node);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)>
      scoped_output_shape(output_shape, TfLiteIntArrayFree);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements is the same as the number of input elements.
  int64_t non_zero_num_input_elements = 1, num_input_elements = 1;
  const RuntimeShape& input_shape = GetTensorShape(input);
  for (int i = 0; i < input_shape.DimensionsCount(); ++i) {
    const int value = input_shape.Dims(i);
    num_input_elements *= value;
    if (value != 0) {
      non_zero_num_input_elements *= value;
    }
  }

  int64_t non_zero_num_output_elements = 1, num_output_elements = 1;
  int stretch_dim = -1;
  for (int i = 0; i < output_shape->size; ++i) {
    const int value = output_shape->data[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
      continue;
    } else if (value != 0) {
      non_zero_num_output_elements *= value;
    }
    num_output_elements *= value;
  }

  if (stretch_dim != -1) {
    if (num_input_elements == 0 && num_output_elements != 0) {
      output_shape->data[stretch_dim] = 0;
    } else {
      output_shape->data[stretch_dim] =
          non_zero_num_input_elements / non_zero_num_output_elements;
    }
    num_output_elements *= output_shape->data[stretch_dim];
  }

  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);
  return context->ResizeTensor(context, output, scoped_output_shape.release());
}

inline TfLiteIntArray* GetOutputShapeFromTensor(TfLiteContext* context,
                                                TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_1(mht_1_v, 262, "", "./tensorflow/lite/kernels/reshape.cc", "GetOutputShapeFromTensor");

  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  if (shape == nullptr) return nullptr;

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(shape->dims->data[0]);
  for (int i = 0; i < output_shape->size; ++i) {
    output_shape->data[i] = shape->data.i32[i];
  }

  return output_shape;
}

inline TfLiteIntArray* GetOutputShapeFromParam(TfLiteContext* context,
                                               TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_2(mht_2_v, 278, "", "./tensorflow/lite/kernels/reshape.cc", "GetOutputShapeFromParam");

  auto* params = reinterpret_cast<TfLiteReshapeParams*>(node->builtin_data);

  // The function is returned above this line if the shape tensor is usable.
  // Now fallback to the shape parameter in `TfLiteReshapeParams`.
  int num_dimensions = params->num_dimensions;
  if (num_dimensions == 1 && params->shape[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    num_dimensions = 0;
  }
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->data[i] = params->shape[i];
  }

  return output_shape;
}

// Check if the shape tensor is valid. Shapes should be int32 vectors.
inline bool ShapeIsVector(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_3(mht_3_v, 302, "", "./tensorflow/lite/kernels/reshape.cc", "ShapeIsVector");

  const TfLiteTensor* shape = GetInput(context, node, kShapeTensor);
  return (shape != nullptr && shape->dims->size == 1 &&
          shape->type == kTfLiteInt32);
}

TfLiteIntArray* GetOutputShape(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_4(mht_4_v, 311, "", "./tensorflow/lite/kernels/reshape.cc", "GetOutputShape");

  if (NumInputs(node) == 2 && ShapeIsVector(context, node)) {
    return GetOutputShapeFromTensor(context, node);
  } else {
    return GetOutputShapeFromParam(context, node);
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_5(mht_5_v, 322, "", "./tensorflow/lite/kernels/reshape.cc", "Prepare");

  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Always postpone sizing string tensors, even if we could in principle
  // calculate their shapes now. String tensors don't benefit from having their
  // shapes precalculated because the actual memory can only be allocated after
  // we know all the content.
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  if (output->type != kTfLiteString) {
    if (NumInputs(node) == 1 ||
        IsConstantTensor(GetInput(context, node, kShapeTensor))) {
      TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
    } else {
      SetTensorToDynamic(output);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_6(mht_6_v, 347, "", "./tensorflow/lite/kernels/reshape.cc", "Eval");

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // There are two ways in which the 'output' can be made dynamic: it could be
  // a string tensor, or its shape cannot be calculated during Prepare(). In
  // either case, we now have all the information to calculate its shape.
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }

  // Note that string tensors are always "dynamic" in the sense that their size
  // is not known until we have all the content. This applies even when their
  // shape is known ahead of time. As a result, a string tensor is never given
  // any memory by ResizeOutput(), and we need to do it manually here. Since
  // reshape doesn't change the data, the output tensor needs exactly as many
  // bytes as the input tensor.
  if (output->type == kTfLiteString) {
    auto bytes_required = input->bytes;
    TfLiteTensorRealloc(bytes_required, output);
    output->bytes = bytes_required;
  }

  memcpy(output->data.raw, input->data.raw, input->bytes);

  return kTfLiteOk;
}

}  // namespace reshape

TfLiteRegistration* Register_RESHAPE() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSreshapeDTcc mht_7(mht_7_v, 383, "", "./tensorflow/lite/kernels/reshape.cc", "Register_RESHAPE");

  static TfLiteRegistration r = {nullptr, nullptr, reshape::Prepare,
                                 reshape::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
