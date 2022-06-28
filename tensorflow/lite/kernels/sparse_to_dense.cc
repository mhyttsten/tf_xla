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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc() {
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
#include <stdint.h>

#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace sparse_to_dense {

constexpr int kIndicesTensor = 0;
constexpr int kOutputShapeTensor = 1;
constexpr int kValueInputTensor = 2;
constexpr int kDefaultValueTensor = 3;
constexpr int kOutputTensor = 0;

constexpr int kMaxDimensions = 4;

template <typename T>
TfLiteStatus Resize(TfLiteContext* context, const TfLiteTensor* output_shape,
                    TfLiteTensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "Resize");

  const int output_dimensions = NumElements(output_shape);
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(output_dimensions);
  for (int i = 0; i < output_dimensions; ++i) {
    output_shape_array->data[i] = GetTensorData<T>(output_shape)[i];
  }

  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus CheckDimensionsMatch(TfLiteContext* context,
                                  const TfLiteTensor* indices,
                                  const TfLiteTensor* output_shape,
                                  const TfLiteTensor* values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "CheckDimensionsMatch");

  switch (NumDimensions(indices)) {
    case 0:
    case 1: {
      if (NumDimensions(values) == 0) {
        TF_LITE_ENSURE_EQ(context, NumElements(indices), NumElements(values));
      }
      TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 1);
      break;
    }
    case 2: {
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(indices, 1),
                        NumElements(output_shape));
      if (NumDimensions(values) == 0)
        TF_LITE_ENSURE_EQ(context, SizeOfDimension(indices, 0),
                          NumElements(values));
      break;
    }
    default:
      context->ReportError(
          context, "Wrong indices dimensions %d, should be less than 3.",
          NumDimensions(indices));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// Convert indices into a vector of 4-d vectors.
// TODO(renjieliu): Revisit here to improve the performance, since multiple
// allocations of std::vectors will be quite slow on phones.
template <typename T>
TfLiteStatus GetIndicesVector(TfLiteContext* context,
                              const TfLiteTensor* indices,
                              const int num_indices,
                              std::vector<std::vector<T>>* indices_vector) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_2(mht_2_v, 263, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "GetIndicesVector");

  // Note because TfLite will reverse the dimensions, so pad zeros upfront.
  switch (NumDimensions(indices)) {
    case 0:
    case 1: {
      const auto indices_data = GetTensorData<T>(indices);
      for (int i = 0; i < num_indices; ++i) {
        std::vector<T> index({0, 0, 0, indices_data[i]});
        indices_vector->push_back(index);
      }
      break;
    }
    case 2: {
      const int true_dimensions = SizeOfDimension(indices, 1);
      TF_LITE_ENSURE(context, true_dimensions <= kMaxDimensions);
      for (int i = 0; i < num_indices; ++i) {
        std::vector<T> index;
        index.reserve(kMaxDimensions);
        // Fill the index with 1 up to kMaxDimensions - true_dimensions to
        // satisfy the needs for 4-dimension index.
        for (int j = 0; j < kMaxDimensions - true_dimensions; ++j) {
          index.push_back(0);
        }
        for (int j = 0; j < true_dimensions; ++j) {
          index.push_back(GetTensorData<T>(indices)[i * true_dimensions + j]);
        }

        indices_vector->push_back(index);
      }
      break;
    }
    default:
      context->ReportError(context,
                           "Indices dimensions problem, got %d dimensions",
                           NumDimensions(indices));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus ResizeOutputShape(TfLiteContext* context,
                               const TfLiteTensor* output_shape,
                               TfLiteTensor* output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_3(mht_3_v, 308, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "ResizeOutputShape");

  if (output_shape->type == kTfLiteInt32) {
    return Resize<int32_t>(context, output_shape, output);
  } else if (output_shape->type == kTfLiteInt64) {
    return Resize<int64_t>(context, output_shape, output);
  } else {
    context->ReportError(context, "Dense shape type %d not supported.",
                         output_shape->type);
    return kTfLiteError;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_4(mht_4_v, 323, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "Prepare");

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndicesTensor, &indices));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kOutputShapeTensor, &output_shape));
  const TfLiteTensor* values;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueInputTensor, &values));
  const TfLiteTensor* default_value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDefaultValueTensor,
                                          &default_value));

  // TODO(renjieliu): Handle validate_indices.

  // Indices can be 0-D, 1-D or 2-D.
  TF_LITE_ASSERT(NumDimensions(indices) >= 0);
  TF_LITE_ENSURE(context, NumDimensions(indices) < 3);
  TF_LITE_ASSERT(NumDimensions(output_shape) >= 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);
  // Values can be 0-D or 1-D.
  TF_LITE_ASSERT(NumDimensions(values) >= 0);
  TF_LITE_ENSURE(context, NumDimensions(values) < 2);

  TF_LITE_ENSURE_EQ(context, NumElements(default_value), 1);

  TF_LITE_ENSURE(
      context, indices->type == kTfLiteInt32 || indices->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, output_shape->type == kTfLiteInt32 ||
                              output_shape->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, values->type == kTfLiteInt32 ||
                              values->type == kTfLiteInt64 ||
                              values->type == kTfLiteInt8 ||
                              values->type == kTfLiteUInt8 ||
                              values->type == kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, values->type, default_value->type);

  // Ensure dimensions match.
  TF_LITE_ENSURE_OK(
      context, CheckDimensionsMatch(context, indices, output_shape, values));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = values->type;
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);

  if (!IsConstantOrPersistentTensor(output_shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputShape(context, output_shape, output);
}

template <typename T, typename TI>
TfLiteStatus SparseToDenseImpl(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndicesTensor, &indices));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kOutputShapeTensor, &output_shape));
  const TfLiteTensor* values;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueInputTensor, &values));
  const TfLiteTensor* default_value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDefaultValueTensor,
                                          &default_value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputShape(context, output_shape, output));
  }

  const int num_indices = SizeOfDimension(indices, 0);
  const bool value_is_scalar = NumDimensions(values) == 0;
  std::vector<std::vector<TI>> indices_vector;
  indices_vector.reserve(num_indices);
  TF_LITE_ENSURE_OK(context, GetIndicesVector<TI>(context, indices, num_indices,
                                                  &indices_vector));
  reference_ops::SparseToDense(indices_vector, GetTensorData<T>(values),
                               *GetTensorData<T>(default_value),
                               value_is_scalar, GetTensorShape(output),
                               GetTensorData<T>(output));

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalForIndexType(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteTensor* indices) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_5(mht_5_v, 423, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "EvalForIndexType");

  switch (indices->type) {
    case kTfLiteInt32: {
      return SparseToDenseImpl<T, int32_t>(context, node);
    }
    case kTfLiteInt64: {
      return SparseToDenseImpl<T, int64_t>(context, node);
    }
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Indice type %s is currently not supported by sparse to dense.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_6(mht_6_v, 443, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "Eval");

  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndicesTensor, &indices));
  const TfLiteTensor* values;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueInputTensor, &values));

  switch (values->type) {
    case kTfLiteFloat32:
      return EvalForIndexType<float>(context, node, indices);
    case kTfLiteInt32:
      return EvalForIndexType<int32_t>(context, node, indices);
    case kTfLiteInt64:
      return EvalForIndexType<int64_t>(context, node, indices);
    case kTfLiteInt8:
      return EvalForIndexType<int8_t>(context, node, indices);
    case kTfLiteUInt8:
      return EvalForIndexType<uint8_t>(context, node, indices);
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Value type %s is currently not supported by sparse to dense.",
          TfLiteTypeGetName(values->type));
      return kTfLiteError;
  }
}

}  // namespace sparse_to_dense

TfLiteRegistration* Register_SPARSE_TO_DENSE() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsparse_to_denseDTcc mht_7(mht_7_v, 476, "", "./tensorflow/lite/kernels/sparse_to_dense.cc", "Register_SPARSE_TO_DENSE");

  static TfLiteRegistration r = {nullptr, nullptr, sparse_to_dense::Prepare,
                                 sparse_to_dense::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
