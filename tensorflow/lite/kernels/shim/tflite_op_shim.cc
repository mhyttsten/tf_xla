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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc() {
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
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"

namespace tflite {
namespace shim {

namespace internal {
absl::StatusOr<AttrValue> GetAttr(const flexbuffers::Map* attr_map,
                                  const std::string& attr_name) {
  const auto value = (*attr_map)[attr_name.data()];
  if (value.IsNull())
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent attribute: ", attr_name));
  AttrValue ret;
  switch (value.GetType()) {
    case ::flexbuffers::FBT_BOOL: {
      ret = value.AsBool();
      break;
    }
    case ::flexbuffers::FBT_INT: {
      ret = static_cast<int64_t>(value.AsInt64());
      break;
    }
    case ::flexbuffers::FBT_FLOAT: {
      ret = value.AsFloat();
      break;
    }
    case ::flexbuffers::FBT_STRING: {
      const auto str_val = value.AsString();
      ret = absl::string_view(str_val.c_str(), str_val.length());
      break;
    }
    default: {
      return absl::FailedPreconditionError(
          absl::StrCat("Unsupported type for attribute: ", attr_name,
                       " with value: ", value.ToString()));
    }
  }
  return ret;
}
}  // namespace internal

TfLiteInitContext::TfLiteInitContext(const TfLiteContext* context,
                                     const flexbuffers::Map* attr_map)
    : attr_map_(attr_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_0(mht_0_v, 237, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInitContext::TfLiteInitContext");
}

absl::StatusOr<AttrValue> TfLiteInitContext::GetAttr(
    const std::string& attr_name) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_1(mht_1_v, 244, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInitContext::GetAttr");

  return internal::GetAttr(attr_map_, attr_name);
}

TfLiteInvokeContext::TfLiteInvokeContext(TfLiteContext* context,
                                         TfLiteNode* node)
    : context_(context), node_(node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_2(mht_2_v, 253, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInvokeContext::TfLiteInvokeContext");
}

ConstTensorViewOr TfLiteInvokeContext::GetInput(const int idx) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_3(mht_3_v, 258, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInvokeContext::GetInput");

  // Scope is used to ensure tensor_view string contents are flushed
  const auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(
        absl::StrCat("input tensor is null during invocation. idx: ", idx));
  SH_ASSIGN_OR_RETURN(const TfLiteTensorView& tensor_view,
                      TensorView::New(tflite_tensor));
  return absl::make_unique<const TfLiteTensorView>(tensor_view);
}

TensorViewOr TfLiteInvokeContext::GetOutput(const int idx,
                                            const Shape& output_shape) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_4(mht_4_v, 273, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInvokeContext::GetOutput");

  if (!output_shape.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("output_shape value should be populated. idx: ", idx));
  }
  auto* tflite_tensor = ::tflite::GetOutput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(
        absl::StrCat("output tensor is null during invocation. idx: ", idx));
  if (tflite_tensor->data.raw == nullptr) {
    TfLiteIntArray* output_shape_array =
        ShapeToTfLiteShape(output_shape.value());
    context_->ResizeTensor(context_, tflite_tensor, output_shape_array);
  } else {
    DCHECK(TfLiteShapeToShape(tflite_tensor->dims) == output_shape);
  }
  SH_ASSIGN_OR_RETURN(TfLiteTensorView tensor_view,
                      TensorView::New(tflite_tensor));
  return absl::make_unique<TfLiteTensorView>(std::move(tensor_view));
}

int TfLiteInvokeContext::NumInputs() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_5(mht_5_v, 297, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInvokeContext::NumInputs");

  return ::tflite::NumInputs(node_);
}

int TfLiteInvokeContext::NumOutputs() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_6(mht_6_v, 304, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteInvokeContext::NumOutputs");

  return ::tflite::NumOutputs(node_);
}

TfLiteShapeInferenceContext::TfLiteShapeInferenceContext(
    TfLiteContext* context, TfLiteNode* node, const flexbuffers::Map* attr_map,
    std::vector<Shape>* inferred_shapes)
    : context_(context),
      node_(node),
      attr_map_(attr_map),
      inferred_shapes_(inferred_shapes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_7(mht_7_v, 317, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::TfLiteShapeInferenceContext");
}

ShapeOr TfLiteShapeInferenceContext::GetInputShape(const int idx) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_8(mht_8_v, 322, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::GetInputShape");

  auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(absl::StrCat(
        "input tensor is null during shape inference. idx: ", idx));
  return TfLiteShapeToShape(tflite_tensor->dims);
}

// A function object to set output shape information from a Shape
// object
absl::Status TfLiteShapeInferenceContext::SetOutputShape(const int idx,
                                                         const Shape& shape) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_9(mht_9_v, 336, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::SetOutputShape");

  if (idx >= inferred_shapes_->size()) {
    return absl::InternalError(absl::StrCat("output idx out of bounds: ", idx,
                                            " >= ", inferred_shapes_->size()));
  }
  (*inferred_shapes_)[idx] = shape;
  return absl::OkStatus();
}

// A function object to read input tensor during shape inference as a TensorView
ConstTensorViewOr TfLiteShapeInferenceContext::GetInputTensor(
    const int idx) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_10(mht_10_v, 350, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::GetInputTensor");

  const auto* tflite_tensor = ::tflite::GetInput(context_, node_, idx);
  if (tflite_tensor == nullptr)
    return absl::InternalError(absl::StrCat(
        "input tensor is null during shape inference. idx: ", idx));
  if (::tflite::IsConstantTensor(tflite_tensor)) {
    SH_ASSIGN_OR_RETURN(const TfLiteTensorView& tensor_view,
                        TensorView::New(tflite_tensor));
    return absl::make_unique<const TfLiteTensorView>(tensor_view);
  } else {
    return absl::FailedPreconditionError(absl::StrCat(
        "input tensor is unavailable during shape inference. idx: ", idx));
  }
}

absl::StatusOr<AttrValue> TfLiteShapeInferenceContext::GetAttr(
    const std::string& attr_name) const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_11(mht_11_v, 370, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::GetAttr");

  return internal::GetAttr(attr_map_, attr_name);
}

int TfLiteShapeInferenceContext::NumInputs() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_12(mht_12_v, 377, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::NumInputs");

  return ::tflite::NumInputs(node_);
}

int TfLiteShapeInferenceContext::NumOutputs() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_13(mht_13_v, 384, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeInferenceContext::NumOutputs");

  return ::tflite::NumOutputs(node_);
}

TfLiteStatus StatusToTfLiteStatus(TfLiteContext* context,
                                  const absl::Status& status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_14(mht_14_v, 392, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "StatusToTfLiteStatus");

  if (status.ok()) return kTfLiteOk;
  const auto error_string = std::string(status.message());
  TF_LITE_KERNEL_LOG(context, "error: %s", error_string.c_str());
  return kTfLiteError;
}

TfLiteIntArray* ShapeToTfLiteShape(const std::vector<int>& shape) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_15(mht_15_v, 402, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "ShapeToTfLiteShape");

  TfLiteIntArray* tflite_shape = TfLiteIntArrayCreate(shape.size());
  // TfLiteIntArray has a data array inside which is not a pointer so there's a
  // need for copy
  std::memcpy(tflite_shape->data, shape.data(), sizeof(int) * shape.size());
  return tflite_shape;
}

// Converts an int array representing shape in TFLite to Shape.
Shape TfLiteShapeToShape(const TfLiteIntArray* tflite_shape) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStflite_op_shimDTcc mht_16(mht_16_v, 414, "", "./tensorflow/lite/kernels/shim/tflite_op_shim.cc", "TfLiteShapeToShape");

  Shape ret(tflite_shape->size);
  std::memcpy(ret->data(), tflite_shape->data,
              sizeof(int) * tflite_shape->size);
  return ret;
}

}  // namespace shim
}  // namespace tflite
