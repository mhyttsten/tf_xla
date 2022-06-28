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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc() {
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
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tflite {
namespace shim {

namespace {
// Converts a TF AttrValue into a TF Shim AttrValue
absl::StatusOr<AttrValue> TfAttrValueToShimAttrValue(
    const ::tensorflow::AttrValue& attr_value) {
  AttrValue ret;
  switch (attr_value.value_case()) {
    case ::tensorflow::AttrValue::kB: {
      ret = attr_value.b();
      break;
    }
    case ::tensorflow::AttrValue::kI: {
      ret = attr_value.i();
      break;
    }
    case ::tensorflow::AttrValue::kF: {
      ret = attr_value.f();
      break;
    }
    case ::tensorflow::AttrValue::kS: {
      ret = attr_value.s();
      break;
    }
    default: {
      return absl::FailedPreconditionError(absl::StrCat(
          "Unsupported attribute type: ", attr_value.DebugString()));
    }
  }
  return ret;
}
}  // namespace

TfInitContext::TfInitContext(const ::tensorflow::OpKernelConstruction* context)
    : context_(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_0(mht_0_v, 232, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInitContext::TfInitContext");
}

absl::StatusOr<AttrValue> TfInitContext::GetAttr(
    const std::string& attr_name) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInitContext::GetAttr");

  if (!context_->HasAttr(attr_name))
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent attribute: ", attr_name, "\nop def:\n",
                     context_->def().DebugString()));
  const auto& attr_value = context_->def().attr().at(attr_name);
  return TfAttrValueToShimAttrValue(attr_value);
}

TfInvokeContext::TfInvokeContext(::tensorflow::OpKernelContext* context)
    : context_(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_2(mht_2_v, 252, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInvokeContext::TfInvokeContext");
}

ConstTensorViewOr TfInvokeContext::GetInput(const int idx) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_3(mht_3_v, 257, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInvokeContext::GetInput");

  if (idx >= context_->num_inputs()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected idx < num_inputs. idx: ", idx,
                     " num_inputs: ", context_->num_inputs()));
  }
  const auto tf_tensor = context_->input(idx);
  SH_ASSIGN_OR_RETURN(const TfTensorView& tensor_view,
                      TensorView::New(&tf_tensor));
  return absl::make_unique<const TfTensorView>(tensor_view);
}

TensorViewOr TfInvokeContext::GetOutput(const int idx,
                                        const Shape& shape) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_4(mht_4_v, 273, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInvokeContext::GetOutput");

  tensorflow::Tensor* output_t = nullptr;
  if (!shape.has_value())
    return absl::InvalidArgumentError("Output shape needs to be specified.");
  std::vector<int64_t> shape_64(shape->size());
  for (int i = 0; i < shape->size(); ++i) shape_64[i] = (*shape)[i];
  auto status = context_->allocate_output(
      idx, ::tensorflow::TensorShape(shape_64), &output_t);
  if (!status.ok()) return ToAbslStatus(status);
  SH_ASSIGN_OR_RETURN(const TfTensorView& tensor_view,
                      TensorView::New(output_t));
  return absl::make_unique<TfTensorView>(std::move(tensor_view));
}

int TfInvokeContext::NumInputs() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_5(mht_5_v, 290, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInvokeContext::NumInputs");
 return context_->num_inputs(); }

int TfInvokeContext::NumOutputs() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_6(mht_6_v, 295, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfInvokeContext::NumOutputs");
 return context_->num_outputs(); }

TfShapeInferenceContext::TfShapeInferenceContext(
    ::tensorflow::shape_inference::InferenceContext* context)
    : context_(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_7(mht_7_v, 302, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::TfShapeInferenceContext");
}

ShapeOr TfShapeInferenceContext::GetInputShape(const int idx) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_8(mht_8_v, 307, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::GetInputShape");

  std::vector<int> ret;
  const auto& shape = context_->input(idx);
  if (!context_->RankKnown(shape)) return Shape();
  ret.resize(context_->Rank(shape));
  for (int i = 0; i < ret.size(); ++i)
    ret[i] = context_->Value(context_->Dim(shape, i));
  return Shape(ret);
}

absl::Status TfShapeInferenceContext::SetOutputShape(const int idx,
                                                     const Shape& shape) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_9(mht_9_v, 321, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::SetOutputShape");

  tensorflow::shape_inference::ShapeHandle output_shape;
  if (shape.has_value()) {
    std::vector<::tensorflow::shape_inference::DimensionHandle> tf_shape;
    tf_shape.reserve(shape.value().size());
    for (const auto dim : shape.value())
      tf_shape.emplace_back(context_->MakeDim(dim));
    output_shape = context_->MakeShape(tf_shape);
  } else {
    output_shape = context_->UnknownShape();
  }
  context_->set_output(idx, output_shape);
  return absl::OkStatus();
}

ConstTensorViewOr TfShapeInferenceContext::GetInputTensor(const int idx) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_10(mht_10_v, 339, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::GetInputTensor");

  const auto* tf_tensor = context_->input_tensor(idx);
  if (tf_tensor == nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Tensor is not available. idx: ", idx));
  }
  SH_ASSIGN_OR_RETURN(const TfTensorView& tensor_view,
                      TensorView::New(tf_tensor));
  return absl::make_unique<const TfTensorView>(tensor_view);
}

absl::StatusOr<AttrValue> TfShapeInferenceContext::GetAttr(
    const std::string& attr_name) const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_11(mht_11_v, 355, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::GetAttr");

  const auto* tf_attr_value = context_->GetAttr(attr_name);
  if (tf_attr_value == nullptr)
    return absl::InvalidArgumentError(
        absl::StrCat("Non-existent attribute: ", attr_name));
  return TfAttrValueToShimAttrValue(*tf_attr_value);
}

int TfShapeInferenceContext::NumInputs() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_12(mht_12_v, 366, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::NumInputs");

  return context_->num_inputs();
}

int TfShapeInferenceContext::NumOutputs() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_13(mht_13_v, 373, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "TfShapeInferenceContext::NumOutputs");

  return context_->num_outputs();
}

::tensorflow::Status FromAbslStatus(const absl::Status& s) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_14(mht_14_v, 380, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "FromAbslStatus");

  if (s.ok()) return ::tensorflow::Status();
  return ::tensorflow::Status(static_cast<::tensorflow::error::Code>(s.code()),
                              s.message());
}

absl::Status ToAbslStatus(const ::tensorflow::Status& s) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTcc mht_15(mht_15_v, 389, "", "./tensorflow/lite/kernels/shim/tf_op_shim.cc", "ToAbslStatus");

  return s.ok() ? absl::OkStatus()
                : absl::Status(static_cast<absl::StatusCode>(s.code()),
                               s.error_message());
}

}  // namespace shim
}  // namespace tflite
