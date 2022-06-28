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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_expression.h"

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

XlaExpression::XlaExpression() = default;

XlaExpression XlaExpression::Invalid() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::Invalid");

  XlaExpression e;
  e.kind_ = Kind::kInvalid;
  return e;
}

XlaExpression XlaExpression::Constant(Tensor value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::Constant");

  XlaExpression e;
  e.kind_ = Kind::kConstant;
  e.dtype_ = value.dtype();
  e.constant_value_ = value;
  return e;
}

XlaExpression XlaExpression::ConstantResource(Tensor value,
                                              XlaResource* resource) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_2(mht_2_v, 218, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::ConstantResource");

  XlaExpression e;
  e.kind_ = Kind::kResource;
  e.dtype_ = DT_RESOURCE;
  e.resource_ = resource;
  e.constant_value_ = value;
  return e;
}

XlaExpression XlaExpression::XlaOp(xla::XlaOp value, DataType dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::XlaOp");

  XlaExpression e;
  e.kind_ = Kind::kXlaOp;
  e.dtype_ = dtype;
  e.handle_ = value;
  return e;
}

XlaExpression XlaExpression::TensorList(xla::XlaOp tensor_list) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_4(mht_4_v, 241, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::TensorList");

  XlaExpression e;
  e.kind_ = Kind::kTensorList;
  e.dtype_ = DT_VARIANT;
  e.handle_ = tensor_list;
  return e;
}

XlaExpression XlaExpression::Resource(XlaResource* resource) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_5(mht_5_v, 252, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::Resource");

  XlaExpression e;
  e.kind_ = Kind::kResource;
  e.dtype_ = DT_RESOURCE;
  e.resource_ = resource;
  return e;
}

string XlaExpression::HumanString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_6(mht_6_v, 263, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::HumanString");

  switch (kind_) {
    case Kind::kInvalid:
      return "invalid";
    case Kind::kConstant:
      return "constant";
    case Kind::kXlaOp:
      return "xla_op";
    case Kind::kResource:
      return "resource";
    case Kind::kTensorList:
      return "tensor_list";
  }
}

xla::XlaOp XlaExpression::AsXlaOp(xla::XlaBuilder* builder) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_7(mht_7_v, 281, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::AsXlaOp");

  return builder->ReportErrorOrReturn([&]() -> StatusOr<xla::XlaOp> {
    switch (kind_) {
      case Kind::kConstant: {
        xla::BorrowingLiteral literal;
        TF_RETURN_IF_ERROR(
            HostTensorToBorrowingLiteral(*constant_value_, &literal));
        return xla::ConstantLiteral(builder, literal);
      }
      case Kind::kTensorList:
        TF_FALLTHROUGH_INTENDED;
      case Kind::kXlaOp:
        if (builder != handle_.builder()) {
          return errors::InvalidArgument(
              "Mismatched builders in XlaExpression::AsXlaOp");
        }
        return handle_;
      default:
        return errors::InvalidArgument("AsXlaOp called on XlaExpression: ",
                                       HumanString());
    }
  });
}

StatusOr<Tensor> XlaExpression::ResolveDynamism(xla::Client* client) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_8(mht_8_v, 308, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::ResolveDynamism");

  switch (kind()) {
    case Kind::kConstant: {
      // Constant values are considered static.
      Tensor constant_false(DT_BOOL, constant_value()->shape());
      auto flat = constant_false.flat<bool>();
      for (int64_t i = 0; i < flat.size(); ++i) flat(i) = false;
      return constant_false;
    }
    case Kind::kXlaOp:
      break;
    case Kind::kTensorList:
      TF_FALLTHROUGH_INTENDED;
    case Kind::kResource:
      TF_FALLTHROUGH_INTENDED;
    case Kind::kInvalid:
      return errors::InvalidArgument(
          "ResolveDynamism called on unsupported XlaExpression: ",
          HumanString());
  }

  if (!client)
    return errors::InvalidArgument("client is required to resolve constant");

  TF_ASSIGN_OR_RETURN(TensorShape shape, GetShape());

  // The XLA layout is specified minor to major, and TensorFlow uses a major to
  // minor order.
  std::vector<int64_t> layout_indices(shape.dims());
  std::iota(layout_indices.rbegin(), layout_indices.rend(), 0);
  xla::ValueInference value_inference(handle().builder());
  TF_ASSIGN_OR_RETURN(xla::LiteralSlice literal,
                      value_inference.AnalyzeIsDynamic(handle()));
  Tensor tensor(DT_BOOL);
  TF_RETURN_IF_ERROR(LiteralToHostTensor(literal, DT_BOOL, &tensor));
  return tensor;
}

StatusOr<absl::optional<Tensor>> XlaExpression::ResolveConstant(
    xla::Client* client, bool dynamic_dimension_is_minus_one,
    xla::ValueInferenceMode mode) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_9(mht_9_v, 351, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::ResolveConstant");

  switch (kind()) {
    case Kind::kConstant:
    case Kind::kResource:
      return constant_value();
    case Kind::kXlaOp:
      break;
    case Kind::kTensorList:
      TF_FALLTHROUGH_INTENDED;
    case Kind::kInvalid:
      return errors::InvalidArgument(
          "ResolveConstant called on XlaExpression: ", HumanString());
  }
  TF_ASSIGN_OR_RETURN(TensorShape shape, GetShape());
  // The XLA layout is specified minor to major, and TensorFlow uses a major to
  // minor order.
  std::vector<int64_t> layout_indices(shape.dims());
  std::iota(layout_indices.rbegin(), layout_indices.rend(), 0);
  xla::Layout layout = xla::LayoutUtil::MakeLayout(layout_indices);
  if (mode == xla::ValueInferenceMode::kLowerBound ||
      mode == xla::ValueInferenceMode::kUpperBound ||
      mode == xla::ValueInferenceMode::kValue) {
    std::vector<int64_t> layout_indices(shape.dims());
    std::iota(layout_indices.rbegin(), layout_indices.rend(), 0);
    xla::ValueInference value_inference(handle().builder());
    TF_ASSIGN_OR_RETURN(xla::OptionalLiteral literal,
                        value_inference.AnalyzeConstant(handle(), mode));
    if (!literal.GetValue().has_value()) {
      return {absl::nullopt};
    }
    Tensor tensor;
    TF_RETURN_IF_ERROR(LiteralToHostTensor(
        literal.GetValue().value().Relayout(layout), dtype(), &tensor));
    return {tensor};
  }

  TF_ASSIGN_OR_RETURN(bool is_constant,
                      handle().builder()->IsConstant(handle()));
  if (!is_constant) {
    return {absl::nullopt};
  }

  if (!client)
    return errors::InvalidArgument("client is required to resolve constant");

  TF_ASSIGN_OR_RETURN(xla::XlaComputation constant_graph,
                      handle().builder()->BuildConstantSubGraph(
                          handle(), dynamic_dimension_is_minus_one));

  TF_ASSIGN_OR_RETURN(xla::Literal literal,
                      client->ComputeConstant(constant_graph, &layout));
  Tensor tensor;
  TF_RETURN_IF_ERROR(LiteralToHostTensor(literal, dtype(), &tensor));
  return {tensor};
}

StatusOr<TensorShape> XlaExpression::GetShape() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_10(mht_10_v, 410, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::GetShape");

  switch (kind_) {
    case Kind::kConstant:
      return constant_value()->shape();
    case Kind::kResource:
      if (constant_value()) {
        return constant_value()->shape();
      }
      return TensorShape({});
    case Kind::kXlaOp: {
      TF_ASSIGN_OR_RETURN(xla::Shape xla_shape,
                          handle().builder()->GetShape(handle()));
      TensorShape shape;
      TF_RETURN_IF_ERROR(XLAShapeToTensorShape(xla_shape, &shape));
      return shape;
    }
    case Kind::kTensorList:
      return TensorShape({});
    case Kind::kInvalid:
      return errors::InvalidArgument(
          "GetShape() called on invalid XlaExpression");
  }
}

const XlaExpression* XlaExpression::CastExpressionFromTensor(
    const Tensor& tensor) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_11(mht_11_v, 438, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::CastExpressionFromTensor");

  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor.tensor_data().data());
  CHECK(expression->kind() != XlaExpression::Kind::kInvalid)
      << expression->HumanString();
  return expression;
}

// Assigns an XlaExpression to a tensor on an XLA compilation device.
void XlaExpression::AssignExpressionToTensor(const XlaExpression& value,
                                             Tensor* tensor) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTcc mht_12(mht_12_v, 451, "", "./tensorflow/compiler/tf2xla/xla_expression.cc", "XlaExpression::AssignExpressionToTensor");

  const XlaExpression* expression =
      reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
  CHECK(expression->kind() == XlaExpression::Kind::kInvalid)
      << expression->HumanString();
  *const_cast<XlaExpression*>(expression) = value;
}

}  // namespace tensorflow
