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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

#include <numeric>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

XlaOpKernelContext::XlaOpKernelContext(OpKernelContext* context)
    : context_(context),
      dynamic_dimension_is_minus_one_(false),
      value_inference_(xla_context()->builder()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::XlaOpKernelContext");
}

bool XlaOpKernelContext::ValidateInputsAreSameShape(OpKernel* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ValidateInputsAreSameShape");

  return context_->ValidateInputsAreSameShape(op);
}

XlaContext* XlaOpKernelContext::xla_context() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_2(mht_2_v, 220, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::xla_context");

  return &XlaContext::Get(context_);
}

xla::XlaBuilder* XlaOpKernelContext::builder() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_3(mht_3_v, 227, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::builder");

  return xla_context()->builder();
}

xla::ValueInference& XlaOpKernelContext::value_inference() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_4(mht_4_v, 234, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::value_inference");

  return value_inference_;
}

XlaCompiler* XlaOpKernelContext::compiler() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_5(mht_5_v, 241, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::compiler");

  return xla_context()->compiler();
}

const XlaExpression& XlaOpKernelContext::InputExpression(int index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_6(mht_6_v, 248, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputExpression");

  return *XlaExpression::CastExpressionFromTensor(context_->input(index));
}

const XlaExpression& XlaOpKernelContext::InputExpression(
    absl::string_view name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_7(mht_7_v, 257, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputExpression");

  return *XlaExpression::CastExpressionFromTensor(GetInputTensorByName(name));
}

xla::XlaOp XlaOpKernelContext::Input(int index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_8(mht_8_v, 264, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::Input");

  return InputExpression(index).AsXlaOp(builder());
}

xla::XlaOp XlaOpKernelContext::Input(absl::string_view name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_9(mht_9_v, 272, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::Input");

  return InputExpression(name).AsXlaOp(builder());
}

TensorShape XlaOpKernelContext::InputShape(int index) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_10(mht_10_v, 279, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputShape");

  return context_->input(index).shape();
}

TensorShape XlaOpKernelContext::InputShape(absl::string_view name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_11(mht_11_v, 287, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputShape");

  return GetInputTensorByName(name).shape();
}

StatusOr<xla::Shape> XlaOpKernelContext::InputXlaShape(int index) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_12(mht_12_v, 294, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputXlaShape");

  return builder()->GetShape(Input(index));
}

StatusOr<xla::Shape> XlaOpKernelContext::InputXlaShape(absl::string_view name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_13(mht_13_v, 302, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputXlaShape");

  return builder()->GetShape(Input(name));
}

DataType XlaOpKernelContext::input_type(int index) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_14(mht_14_v, 309, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::input_type");

  DataType type = context_->input_dtype(index);
  if (type == DT_UINT8) {
    // Masqueraded XlaExpression could have different type. See
    // XlaOpKernelContext::SetOutputExpression for details.
    auto expression =
        XlaExpression::CastExpressionFromTensor(context_->input(index));
    type = expression->dtype();
  }
  return type;
}

DataType XlaOpKernelContext::InputType(absl::string_view name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_15(mht_15_v, 325, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputType");

  const Tensor& tensor = GetInputTensorByName(name);
  DataType type = tensor.dtype();
  if (type == DT_UINT8) {
    // Masqueraded XlaExpression could have different type. See
    // XlaOpKernelContext::SetOutputExpression for details.
    auto expression = XlaExpression::CastExpressionFromTensor(tensor);
    type = expression->dtype();
  }
  return type;
}

xla::PrimitiveType XlaOpKernelContext::input_xla_type(int index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_16(mht_16_v, 340, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::input_xla_type");

  xla::PrimitiveType type;
  Status status = DataTypeToPrimitiveType(input_type(index), &type);
  if (!status.ok()) {
    SetStatus(status);
    return xla::PRIMITIVE_TYPE_INVALID;
  }
  return type;
}

xla::PrimitiveType XlaOpKernelContext::InputXlaType(absl::string_view name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_17(mht_17_v, 354, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputXlaType");

  xla::PrimitiveType type;
  Status status = DataTypeToPrimitiveType(InputType(name), &type);
  if (!status.ok()) {
    SetStatus(status);
    return xla::PRIMITIVE_TYPE_INVALID;
  }
  return type;
}

Status XlaOpKernelContext::ConstantInput(int index,
                                         xla::Literal* constant_literal,
                                         xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_18(mht_18_v, 369, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInput");

  if (this->InputXlaShape(index)->is_dynamic()) {
    return errors::InvalidArgument(
        "Reading input as constant from a dynamic tensor is not yet supported. "
        "Xla shape: ",
        this->InputXlaShape(index)->ToString());
  }
  return ConstantInputReshaped(index,
                               context_->input(index).shape().dim_sizes(),
                               constant_literal, mode);
}

static StatusOr<int> InputIndex(XlaOpKernelContext* context,
                                absl::string_view name) {
  int start, stop;
  TF_RETURN_IF_ERROR(context->op_kernel().InputRange(name, &start, &stop));
  if (stop != start + 1) {
    return errors::InvalidArgument("OpKernel used list-valued input name '",
                                   name,
                                   "' when single-valued input was "
                                   "expected");
  }
  return start;
}

Status XlaOpKernelContext::ResolveInputDynamism(
    int index, xla::Literal* dynamism_literal) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_19(mht_19_v, 398, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamism");

  return ResolveInputDynamismReshaped(
      index, context_->input(index).shape().dim_sizes(), dynamism_literal);
}

Status XlaOpKernelContext::ResolveInputDynamism(
    absl::string_view name, xla::Literal* dynamism_literal) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_20(mht_20_v, 408, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamism");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ResolveInputDynamism(index, dynamism_literal);
}

Status XlaOpKernelContext::ConstantInput(absl::string_view name,
                                         xla::Literal* constant_literal,
                                         xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_21(mht_21_v, 419, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInput");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInput(index, constant_literal, mode);
}

Status XlaOpKernelContext::ConstantInputReshaped(
    int index, absl::Span<const int64_t> new_dims,
    xla::Literal* constant_literal, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_22(mht_22_v, 429, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputReshaped");

  TF_ASSIGN_OR_RETURN(Tensor constant, ConstantInputTensor(index, mode));
  Tensor temp(constant.dtype());
  if (!temp.CopyFrom(constant, TensorShape(new_dims))) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        constant.shape().DebugString(),
        " but was asked to be reshaped to incompatible shape ",
        TensorShape(new_dims).DebugString());
  }

  TF_ASSIGN_OR_RETURN(*constant_literal, HostTensorToLiteral(temp));
  return Status::OK();
}

// Converts an int32 or int64 scalar literal to an int64.
static Status LiteralToInt64Scalar(const xla::LiteralSlice& literal,
                                   int64_t* out) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_23(mht_23_v, 449, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "LiteralToInt64Scalar");

  if (literal.shape().rank() != 0) {
    return errors::InvalidArgument("value is not a scalar");
  }
  if (literal.shape().element_type() == xla::S32) {
    *out = literal.Get<int32>({});
  } else if (literal.shape().element_type() == xla::S64) {
    *out = literal.Get<int64_t>({});
  } else {
    return errors::InvalidArgument("value must be either int32 or int64");
  }
  return Status::OK();
}

// Converts an float32 or float64 scalar literal to a float64.
static Status LiteralToFloat64Scalar(const xla::LiteralSlice& literal,
                                     double* out) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_24(mht_24_v, 468, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "LiteralToFloat64Scalar");

  if (literal.shape().rank() != 0) {
    return errors::InvalidArgument("value is not a scalar");
  }
  if (literal.shape().element_type() == xla::F32) {
    *out = literal.Get<float>({});
  } else if (literal.shape().element_type() == xla::F64) {
    *out = literal.Get<double>({});
  } else {
    return errors::InvalidArgument("value must be either float32 or float64");
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputAsIntScalar(
    int index, int64_t* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_25(mht_25_v, 486, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsIntScalar");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  return LiteralToInt64Scalar(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsIntScalar(
    absl::string_view name, int64_t* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_26(mht_26_v, 497, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsIntScalar");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsIntScalar(index, out, mode);
}

Status XlaOpKernelContext::ConstantInputAsFloatScalar(
    int index, double* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_27(mht_27_v, 506, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsFloatScalar");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  return LiteralToFloat64Scalar(literal, out);
}

static Status LiteralToPredVector(const xla::LiteralSlice& literal,
                                  std::vector<bool>* out) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_28(mht_28_v, 516, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "LiteralToPredVector");

  if (literal.shape().rank() != 1) {
    return errors::InvalidArgument("output_shape must be rank 1, got shape ",
                                   literal.shape().DebugString());
  }
  int64_t size = xla::ShapeUtil::ElementsIn(literal.shape());
  if (literal.shape().element_type() != xla::PRED) {
    return errors::InvalidArgument("value is not PRED");
  }
  for (int64_t i = 0; i < size; ++i) {
    out->push_back(literal.Get<bool>({i}));
  }
  return Status::OK();
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPred(int index, bool* out) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_29(mht_29_v, 534, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamismIntoPred");

  xla::Literal literal;
  XlaExpression e = InputExpression(index);
  auto* client = compiler() ? compiler()->client() : nullptr;
  StatusOr<Tensor> dynamism_or_status = e.ResolveDynamism(client);
  if (!dynamism_or_status.ok()) {
    // When failed to resolve dynamism, conservatively consider the value
    // dynamic. This could happen if the input depends on some ops like
    // custom-call that is not supported generally for dynamism computation.
    //
    // TODO(b/176993339): Support resolving dynamism across computations so
    // resolving dynamism will not fail in those cases.
    *out = true;
    return Status::OK();
  }
  Tensor dynamism = dynamism_or_status.ValueOrDie();

  Tensor temp(dynamism.dtype());
  TensorShape tensor_shape({});
  if (!temp.CopyFrom(dynamism, tensor_shape)) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        dynamism.shape().DebugString(), " which is not a R0 ", tensor_shape);
  }

  TF_ASSIGN_OR_RETURN(literal, HostTensorToLiteral(temp));
  *out = literal.Get<bool>({});
  return Status::OK();
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPredVector(
    absl::string_view name, std::vector<bool>* out) {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_30(mht_30_v, 569, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamismIntoPredVector");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ResolveInputDynamismIntoPredVector(index, out);
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPred(absl::string_view name,
                                                        bool* out) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_31(mht_31_v, 579, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamismIntoPred");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ResolveInputDynamismIntoPred(index, out);
}

Status XlaOpKernelContext::ResolveInputDynamismReshaped(
    int index, absl::Span<const int64_t> new_dims,
    xla::Literal* dynamism_literal) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_32(mht_32_v, 589, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamismReshaped");

  XlaExpression e = InputExpression(index);
  auto* client = compiler() ? compiler()->client() : nullptr;
  StatusOr<Tensor> dynamism_or_status = e.ResolveDynamism(client);
  if (!dynamism_or_status.ok()) {
    xla::Literal true_literal = xla::LiteralUtil::CreateR0<bool>(true);
    // When failed to resolve dynamism, conservatively consider the value
    // dynamic. This could happen if the input depends on some ops like
    // custom-call that is not supported generally for dynamism computation.
    *dynamism_literal =
        true_literal
            .Broadcast(xla::ShapeUtil::MakeShape(xla::PRED, new_dims), {})
            .ValueOrDie();

    return Status::OK();
  }
  Tensor dynamism = dynamism_or_status.ValueOrDie();

  Tensor temp(dynamism.dtype());
  if (!temp.CopyFrom(dynamism, TensorShape(new_dims))) {
    return errors::InvalidArgument(
        context_->op_kernel().name(), " input ", index, " has shape ",
        dynamism.shape().DebugString(),
        " but was asked to be reshaped to incompatible shape ",
        TensorShape(new_dims).DebugString());
  }

  TF_ASSIGN_OR_RETURN(*dynamism_literal, HostTensorToLiteral(temp));
  return Status::OK();
}

Status XlaOpKernelContext::ResolveInputDynamismIntoPredVector(
    int index, std::vector<bool>* out) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_33(mht_33_v, 624, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ResolveInputDynamismIntoPredVector");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ResolveInputDynamismReshaped(
      index, {InputShape(index).num_elements()}, &literal));

  return LiteralToPredVector(literal, out);
}

// Converts an int32 or int64 1D literal to an int64 vector.
static Status LiteralToInt64Vector(const xla::LiteralSlice& literal,
                                   std::vector<int64_t>* out) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_34(mht_34_v, 637, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "LiteralToInt64Vector");

  if (literal.shape().rank() != 1) {
    return errors::InvalidArgument("output_shape must be rank 1, got shape ",
                                   literal.shape().DebugString());
  }
  int64_t size = xla::ShapeUtil::ElementsIn(literal.shape());
  if (literal.shape().element_type() == xla::S32) {
    for (int64_t i = 0; i < size; ++i) {
      out->push_back(literal.Get<int32>({i}));
    }
  } else if (literal.shape().element_type() == xla::S64) {
    for (int64_t i = 0; i < size; ++i) {
      out->push_back(literal.Get<int64_t>({i}));
    }
  } else {
    return errors::InvalidArgument("value must be either int32 or int64");
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputAsIntVector(
    int index, std::vector<int64_t>* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_35(mht_35_v, 661, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsIntVector");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsIntVector(
    absl::string_view name, std::vector<int64_t>* out,
    xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_36(mht_36_v, 673, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsIntVector");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsIntVector(index, out, mode);
}

Status XlaOpKernelContext::ConstantInputReshapedToIntVector(
    int index, std::vector<int64_t>* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_37(mht_37_v, 682, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputReshapedToIntVector");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInputReshaped(
      index, {InputShape(index).num_elements()}, &literal, mode));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputReshapedToIntVector(
    absl::string_view name, std::vector<int64_t>* out,
    xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_38(mht_38_v, 695, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputReshapedToIntVector");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInputReshaped(
      index, {InputShape(index).num_elements()}, &literal, mode));
  return LiteralToInt64Vector(literal, out);
}

Status XlaOpKernelContext::ConstantInputAsInt64Literal(
    int index, xla::Literal* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_39(mht_39_v, 707, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsInt64Literal");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  switch (literal.shape().element_type()) {
    case xla::S32: {
      *out = xla::Literal(
          xla::ShapeUtil::ChangeElementType(literal.shape(), xla::S64));
      auto src_data = literal.data<int32>();
      for (int64_t i = 0; i < src_data.size(); ++i) {
        out->data<int64_t>()[i] = src_data[i];
      }
      return Status::OK();
    }
    case xla::S64:
      *out = std::move(literal);
      return Status::OK();

    default:
      return errors::InvalidArgument(
          "Invalid argument to ConstantInputAsInt64Literal: ",
          xla::ShapeUtil::HumanString(literal.shape()));
  }
}

Status XlaOpKernelContext::ConstantInputAsInt64Literal(
    absl::string_view name, xla::Literal* out, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_40(mht_40_v, 736, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsInt64Literal");

  TF_ASSIGN_OR_RETURN(int index, InputIndex(this, name));
  return ConstantInputAsInt64Literal(index, out, mode);
}

// TODO(phawkins): validate that the dimensions form a valid shape, fail
// gracefully if they do not.
Status XlaOpKernelContext::ConstantInputAsShape(int index, TensorShape* shape,
                                                xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_41(mht_41_v, 747, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsShape");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal, mode));
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(LiteralToInt64Vector(literal, &dims));
  *shape = TensorShape(dims);
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputAsPartialShape(
    int index, PartialTensorShape* shape) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_42(mht_42_v, 760, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputAsPartialShape");

  xla::Literal literal;
  TF_RETURN_IF_ERROR(ConstantInput(index, &literal));
  // If `literal` is a scalar it's value must be -1.
  if (literal.shape().rank() == 0) {
    int64_t shape_val;
    TF_RETURN_IF_ERROR(LiteralToInt64Scalar(literal, &shape_val));
    if (shape_val != -1) {
      return errors::InvalidArgument(
          "Cannot convert value to PartialTensorShape: ", shape_val);
    }
    *shape = PartialTensorShape();  // Shape with unknown rank.
    return Status::OK();
  }
  std::vector<int64_t> dims;
  TF_RETURN_IF_ERROR(LiteralToInt64Vector(literal, &dims));
  *shape = PartialTensorShape(dims);
  return Status::OK();
}

Status XlaOpKernelContext::InputList(absl::string_view name,
                                     std::vector<xla::XlaOp>* handles,
                                     std::vector<TensorShape>* shapes) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_43(mht_43_v, 786, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::InputList");

  OpInputList inputs;
  TF_RETURN_IF_ERROR(context_->input_list(name, &inputs));
  handles->clear();
  shapes->clear();
  for (const Tensor& input : inputs) {
    handles->push_back(
        XlaExpression::CastExpressionFromTensor(input)->AsXlaOp(builder()));
    shapes->push_back(input.shape());
  }
  return Status::OK();
}

Status XlaOpKernelContext::ConstantInputList(absl::string_view name,
                                             std::vector<xla::Literal>* outputs,
                                             xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_44(mht_44_v, 805, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputList");

  int start, stop;
  TF_RETURN_IF_ERROR(op_kernel().InputRange(name, &start, &stop));
  outputs->resize(stop - start);
  for (int i = start; i < stop; ++i) {
    TF_RETURN_IF_ERROR(ConstantInput(i, &(*outputs)[i], mode));
  }
  return Status::OK();
}

StatusOr<Tensor> XlaOpKernelContext::ConstantInputTensor(
    int index, xla::ValueInferenceMode mode) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_45(mht_45_v, 819, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ConstantInputTensor");

  XlaExpression e = InputExpression(index);
  auto* client = compiler() ? compiler()->client() : nullptr;
  StatusOr<absl::optional<Tensor>> constant_or_status =
      e.ResolveConstant(client, dynamic_dimension_is_minus_one_, mode);
  if (!constant_or_status.ok()) {
    Status status = constant_or_status.status();
    errors::AppendToMessage(&status, "while evaluating input ", index, " of ",
                            context_->op_kernel().type_string(),
                            " operator as a compile-time constant.");
    return status;
  }
  absl::optional<Tensor> constant = constant_or_status.ValueOrDie();
  if (!constant.has_value()) {
    return errors::InvalidArgument(
        "Input ", index, " to node `", context_->op_kernel().name(),
        "` with op ", context_->op_kernel().type_string(),
        " must be a compile-time constant.\n\n"
        "XLA compilation requires that operator arguments that represent "
        "shapes or dimensions be evaluated to concrete values at compile time. "
        "This error means that a shape or dimension argument could not be "
        "evaluated at compile time, usually because the value of the argument "
        "depends on a parameter to the computation, on a variable, or on a "
        "stateful operation such as a random number generator.");
  }
  return *constant;
}

namespace {

Status ReadVariableInputTensor(const Tensor& tensor, DataType type,
                               const XlaOpKernelContext* ctx,
                               TensorShape* shape, xla::XlaOp* value) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_46(mht_46_v, 854, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "ReadVariableInputTensor");

  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);
  if (!variable->initialized()) {
    return errors::FailedPrecondition(
        "Read variable failure ", variable->name(),
        ". It could mean the variable is uninitialized or the variable is on "
        "another device ");
  }
  if (variable->type() != type) {
    return errors::InvalidArgument(
        "Trying to read variable with wrong dtype. Expected ",
        DataTypeString(type), " got ", DataTypeString(variable->type()));
  }
  if (shape) {
    *shape = variable->shape();
  }

  if (!variable->IsOverwritten() && expression->constant_value()) {
    TF_ASSIGN_OR_RETURN(xla::Literal literal,
                        HostTensorToLiteral(*expression->constant_value()));
    *value = xla::ConstantLiteral(ctx->builder(), literal);
    return Status::OK();
  }
  auto shape_determination_fns =
      ctx->compiler()->options().shape_determination_fns;
  XlaLayoutPreference layout_preference =
      shape_determination_fns.layout_preference_fn(
          variable->shape(), variable->type(), absl::nullopt);
  TF_ASSIGN_OR_RETURN(xla::Shape representation_shape,
                      shape_determination_fns.shape_representation_fn(
                          variable->shape(), variable->type(),
                          /*use_fast_memory=*/false, layout_preference));
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(
      TensorShapeToXLAShape(variable->type(), variable->shape(), &xla_shape));
  if (xla::ShapeUtil::Compatible(xla_shape, representation_shape)) {
    *value = variable->value();
  } else {
    *value = xla::Reshape(variable->value(), variable->shape().dim_sizes());
  }
  return Status::OK();
}

}  // namespace

Status XlaOpKernelContext::ReadVariableInput(int index, DataType type,
                                             TensorShape* shape,
                                             xla::XlaOp* value) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_47(mht_47_v, 908, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ReadVariableInput");

  return ReadVariableInputTensor(context_->input(index), type, this, shape,
                                 value);
}

Status XlaOpKernelContext::ReadVariableInput(absl::string_view name,
                                             DataType type, TensorShape* shape,
                                             xla::XlaOp* value) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_48(mht_48_v, 919, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::ReadVariableInput");

  return ReadVariableInputTensor(GetInputTensorByName(name), type, this, shape,
                                 value);
}

Status XlaOpKernelContext::GetVariableTypeAndShape(int index, DataType* type,
                                                   TensorShape* shape) const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_49(mht_49_v, 928, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetVariableTypeAndShape");

  const Tensor& tensor = context_->input(index);
  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);
  if (!variable->initialized()) {
    return errors::InvalidArgument(
        "Read variable failure ", variable->name(),
        ". It could mean the variable is uninitialized or the variable is on "
        "another device ");
  }
  *type = variable->type();
  *shape = variable->shape();
  return Status::OK();
}

void XlaOpKernelContext::SetOutputExpression(int index,
                                             const XlaExpression& expression) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_50(mht_50_v, 950, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::SetOutputExpression");

  Status status = [&] {
    // The step's default allocator is the dummy XlaCompilationAllocator which
    // simply allocates a metadata buffer to hold the expression to which it
    // corresponds.
    // Provides a special behavior for DT_VARIANT and other types that are not
    // trivially copyable. In those cases, allocate a tensor of type DT_UINT8.
    if (!DataTypeCanUseMemcpy(expression.dtype())) {
      // tensor_data() is not supported for tensors that cannot be copied via
      // memcpy, as the copy logic might try to inspect the stored data (e.g.
      // a std::string). This is likely to fail, as the data is invalid given
      // that it actually encodes an XlaExpression. Using a uint8 tensor is
      // always safe, so simply do that.
      // TODO(jpienaar): This should be refactored to stop masquerading
      // XlaExpressions as Tensors.
      Tensor output;
      TensorShape tensor_shape;
      TF_RETURN_IF_ERROR(
          context_->allocate_temp(DT_UINT8, tensor_shape, &output));
      context_->set_output(index, output);
    } else {
      Tensor* output = nullptr;
      TF_ASSIGN_OR_RETURN(TensorShape shape, expression.GetShape());
      TF_RETURN_IF_ERROR(context_->allocate_output(index, shape, &output));
    }
    XlaExpression::AssignExpressionToTensor(expression,
                                            context_->mutable_output(index));
    return Status::OK();
  }();
  if (!status.ok()) {
    SetStatus(status);
  }
}

xla::PrimitiveType XlaOpKernelContext::output_xla_type(int index) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_51(mht_51_v, 987, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::output_xla_type");

  xla::PrimitiveType type;
  Status status = DataTypeToPrimitiveType(expected_output_dtype(index), &type);
  if (!status.ok()) {
    SetStatus(status);
    return xla::PRIMITIVE_TYPE_INVALID;
  }
  return type;
}

void XlaOpKernelContext::SetOutput(int index, const xla::XlaOp& handle) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_52(mht_52_v, 1000, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::SetOutput");

  SetOutputExpression(
      index,
      XlaExpression::XlaOp(handle, context_->expected_output_dtype(index)));
}

void XlaOpKernelContext::SetConstantOutput(int index, const Tensor& constant) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_53(mht_53_v, 1009, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::SetConstantOutput");

  SetOutputExpression(index, XlaExpression::Constant(constant));
}

void XlaOpKernelContext::SetTensorListOutput(int index,
                                             const xla::XlaOp& handle) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_54(mht_54_v, 1017, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::SetTensorListOutput");

  SetOutputExpression(index, XlaExpression::TensorList(handle));
}

void XlaOpKernelContext::SetResourceOutput(int index, XlaResource* resource) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_55(mht_55_v, 1024, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::SetResourceOutput");

  SetOutputExpression(index, XlaExpression::Resource(resource));
}

Status XlaOpKernelContext::GetResourceInput(int index, XlaResource** resource) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_56(mht_56_v, 1031, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetResourceInput");

  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(context_->input(index));
  TF_RET_CHECK(expression->resource() != nullptr);
  *resource = expression->resource();
  return Status::OK();
}

namespace {

Status AssignVariableTensor(const Tensor& tensor, DataType type,
                            const XlaOpKernelContext* ctx, xla::XlaOp handle,
                            xla::XlaBuilder* builder) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_57(mht_57_v, 1046, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "AssignVariableTensor");

  const XlaExpression* expression =
      XlaExpression::CastExpressionFromTensor(tensor);
  XlaResource* variable = expression->resource();
  TF_RET_CHECK(variable != nullptr);
  TF_RET_CHECK(variable->kind() == XlaResource::kVariable);

  auto shape_or_status = builder->GetShape(handle);
  if (!shape_or_status.ok()) {
    return shape_or_status.status();
  }
  TensorShape shape;
  TF_RETURN_IF_ERROR(
      XLAShapeToTensorShape(shape_or_status.ValueOrDie(), &shape));

  TF_RETURN_IF_ERROR(variable->SetTypeAndShape(type, shape));

  auto shape_determination_fns =
      ctx->compiler()->options().shape_determination_fns;
  XlaLayoutPreference layout_preference =
      shape_determination_fns.layout_preference_fn(shape, type, absl::nullopt);
  TF_ASSIGN_OR_RETURN(xla::Shape representation_shape,
                      shape_determination_fns.shape_representation_fn(
                          shape, type,
                          /*use_fast_memory=*/false, layout_preference));
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
  if (!xla::ShapeUtil::Compatible(xla_shape, representation_shape)) {
    handle = xla::Reshape(handle, representation_shape.dimensions());
  }
  variable->SetRepresentationShape(representation_shape);
  return variable->SetValue(handle);
}

}  // namespace

Status XlaOpKernelContext::AssignVariable(int input_index, DataType type,
                                          xla::XlaOp handle) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_58(mht_58_v, 1086, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::AssignVariable");

  TF_RET_CHECK(handle.valid());
  return AssignVariableTensor(context_->input(input_index), type, this, handle,
                              builder());
}

Status XlaOpKernelContext::AssignVariable(absl::string_view name, DataType type,
                                          xla::XlaOp handle) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_59(mht_59_v, 1097, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::AssignVariable");

  TF_RET_CHECK(handle.valid());
  return AssignVariableTensor(GetInputTensorByName(name), type, this, handle,
                              builder());
}

static Status GetStatusWithStackTrace(const Status& s,
                                      const XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_60(mht_60_v, 1107, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "GetStatusWithStackTrace");

  if (s.code() == error::INVALID_ARGUMENT) {
    return Status{s.code(),
                  absl::StrCat(s.error_message(), "\n", ctx->StackTrace())};
  }
  return s;
}

void XlaOpKernelContext::CtxFailure(const Status& s) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_61(mht_61_v, 1118, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::CtxFailure");

  context_->CtxFailure(GetStatusWithStackTrace(s, this));
}
void XlaOpKernelContext::CtxFailureWithWarning(const Status& s) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_62(mht_62_v, 1124, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::CtxFailureWithWarning");

  context_->CtxFailureWithWarning(GetStatusWithStackTrace(s, this));
}

void XlaOpKernelContext::CtxFailure(const char* file, int line,
                                    const Status& s) {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_63(mht_63_v, 1133, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::CtxFailure");

  context_->CtxFailure(file, line, GetStatusWithStackTrace(s, this));
}
void XlaOpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                               const Status& s) {
   std::vector<std::string> mht_64_v;
   mht_64_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_64(mht_64_v, 1141, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::CtxFailureWithWarning");

  context_->CtxFailureWithWarning(file, line, GetStatusWithStackTrace(s, this));
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMax(
    const DataType type) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_65(mht_65_v, 1149, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetOrCreateMax");

  return xla_context()->GetOrCreateMax(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMin(
    const DataType type) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_66(mht_66_v, 1157, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetOrCreateMin");

  return xla_context()->GetOrCreateMin(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateAdd(
    const DataType type) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_67(mht_67_v, 1165, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetOrCreateAdd");

  return xla_context()->GetOrCreateAdd(type);
}

const xla::XlaComputation* XlaOpKernelContext::GetOrCreateMul(
    const DataType type) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_68(mht_68_v, 1173, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetOrCreateMul");

  return xla_context()->GetOrCreateMul(type);
}

const Tensor& XlaOpKernelContext::GetInputTensorByName(absl::string_view name) {
   std::vector<std::string> mht_69_v;
   mht_69_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_69(mht_69_v, 1181, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::GetInputTensorByName");

  const Tensor* tensor;
  CHECK(context_->input(name, &tensor).ok());
  return *tensor;
}

XlaOpKernel::XlaOpKernel(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_70(mht_70_v, 1190, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernel::XlaOpKernel");
}

void XlaOpKernel::Compute(OpKernelContext* context) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_71(mht_71_v, 1195, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernel::Compute");

  XlaOpKernelContext xla_context(context);
  Compile(&xla_context);
}

std::string XlaOpKernelContext::StackTrace() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_op_kernelDTcc mht_72(mht_72_v, 1203, "", "./tensorflow/compiler/tf2xla/xla_op_kernel.cc", "XlaOpKernelContext::StackTrace");

  if (const AbstractStackTrace* stack_trace =
          xla_context()->StackTraceForNodeName(op_kernel().name())) {
    AbstractStackTrace::TracePrintingOptions opts;
    opts.show_line_contents = true;
    opts.filter_common_prefix = true;
    opts.drop_internal_frames = true;
    return absl::StrCat("\nStack trace for op definition: \n",
                        stack_trace->ToString(opts), "\n");
  } else {
    return "";
  }
}

}  // namespace tensorflow
