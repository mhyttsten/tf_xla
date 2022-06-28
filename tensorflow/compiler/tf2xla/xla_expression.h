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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_EXPRESSION_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_EXPRESSION_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh() {
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


#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// A XlaExpression represents a symbolic TensorFlow value in a TF->XLA
// compilation.
// An expression is one of:
// * a constant tensor.
// * an xla::XlaOp, representing a symbolic XLA value.
// * a resource, e.g., a variable, represented as an XlaResource pointer.
// * a tensor list, represented by a tuple of tensors and the list length.
//
// Constant tensors are mostly an optimization to avoid passing large constants
// to XLA, but are also sometimes used to represent tensors that have no XLA
// representation, for example, DT_STRING tensors. A canonical use case might be
// an error message string.
//
// Tensor lists are very similar to xla::XlaOp, however they require some
// specific logic around shape management since the tuples are not supported by
// TensorFlow.
class XlaExpression {
 public:
  enum class Kind {
    kInvalid,
    kConstant,
    kXlaOp,
    kResource,
    kTensorList,
  };

  XlaExpression();
  XlaExpression(const XlaExpression&) = default;
  XlaExpression& operator=(const XlaExpression&) = default;

  // Builds an invalid expression. (Same as the default constructor, but makes
  // the intent clearer.)
  static XlaExpression Invalid();

  // Builds a constant XLA expression.
  static XlaExpression Constant(Tensor value);

  // Builds a XlaOp expression. Since the mapping from TF data types to XLA
  // types is not 1-1, the TF type must also be provided; in general it cannot
  // be derived from the XLA type.
  static XlaExpression XlaOp(xla::XlaOp value, DataType dtype);

  // Builds a tensor list expression.
  static XlaExpression TensorList(xla::XlaOp tensor_list);

  // Builds a resource expression.
  static XlaExpression Resource(XlaResource* resource);

  // Builds a resource whose value is known at a compile time.
  static XlaExpression ConstantResource(Tensor value, XlaResource* resource);

  Kind kind() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh mht_0(mht_0_v, 250, "", "./tensorflow/compiler/tf2xla/xla_expression.h", "kind");
 return kind_; }

  DataType dtype() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh mht_1(mht_1_v, 255, "", "./tensorflow/compiler/tf2xla/xla_expression.h", "dtype");
 return dtype_; }

  // handle() returns the XlaOp that backs a kXlaOp expression.
  const xla::XlaOp& handle() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh mht_2(mht_2_v, 261, "", "./tensorflow/compiler/tf2xla/xla_expression.h", "handle");
 return handle_; }

  // Return a constant value associated with this expression. Always set for
  // constants, might be set for resources.
  absl::optional<Tensor> constant_value() const {
    if (kind_ == Kind::kResource && resource_->IsOverwritten()) {
      // The constant is no longer available if the value was overwritten.
      return absl::nullopt;
    }
    return constant_value_;
  }

  // Set the bound of the expression.
  void set_value_bound(Tensor tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh mht_3(mht_3_v, 277, "", "./tensorflow/compiler/tf2xla/xla_expression.h", "set_value_bound");

    value_bound_.emplace(std::move(tensor));
  }

  // Return the bound of the expression, if available.
  absl::optional<Tensor> value_bound() const { return value_bound_; }

  // Set the dynamism of the expression, indicating whether or not each value in
  // this expression is dynamic.
  void set_value_dynamism(Tensor tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh mht_4(mht_4_v, 289, "", "./tensorflow/compiler/tf2xla/xla_expression.h", "set_value_dynamism");

    value_dynamism_.emplace(std::move(tensor));
  }

  // Return the dynamism of the expression, if available.
  absl::optional<Tensor> value_dynamism() const { return value_dynamism_; }

  XlaResource* resource() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_expressionDTh mht_5(mht_5_v, 299, "", "./tensorflow/compiler/tf2xla/xla_expression.h", "resource");
 return resource_; }

  // Returns a human-readable summary of the expression.
  string HumanString() const;

  // Returns the value of a kValue or kXlaOp as an xla::XlaOp. Returns
  // an erroneous XlaOp if the expression is not a constant or an expression.
  xla::XlaOp AsXlaOp(xla::XlaBuilder* builder) const;

  // If a kXlaOp or kValue expression can be resolved to a compile-time
  // constant, returns the value as a host-memory Tensor. Returns an empty
  // optional if it cannot be resolved. Returns an error if passed a resource
  // expression.
  StatusOr<absl::optional<Tensor>> ResolveConstant(
      xla::Client* client, bool dynamic_dimension_is_minus_one = false,
      xla::ValueInferenceMode mode = xla::ValueInferenceMode::kValue) const;

  // ResolveDynamism computes where a value inside this op is dynamic or can be
  // inferred at compile time.
  StatusOr<Tensor> ResolveDynamism(xla::Client* client) const;

  // Returns the shape of the tensor.
  // The shape of a resource is the shape of a resource handle (i.e., a scalar),
  // not the shape of the resource's value.
  StatusOr<TensorShape> GetShape() const;

  // Retrieves an XlaExpression that was allocated by a previous Op.
  static const XlaExpression* CastExpressionFromTensor(const Tensor& tensor);

  // Assigns an XlaExpression to a tensor on an XLA compilation device.
  static void AssignExpressionToTensor(const XlaExpression& value,
                                       Tensor* tensor);

 private:
  Kind kind_ = Kind::kInvalid;

  DataType dtype_ = DT_INVALID;

  // The XLA handle of the expression's computation, if kind_ == kXlaOp or
  // a tuple expression if kind_ == kTensorList.
  xla::XlaOp handle_;

  // The value of the constant, if available.
  absl::optional<Tensor> constant_value_;

  // The bound of the expression, if available.
  absl::optional<Tensor> value_bound_;

  // Indicate whether each value inside a tensor is dynamic or not.
  absl::optional<Tensor> value_dynamism_;

  // The resource, if kind_ == kResource. Not owned.
  XlaResource* resource_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_EXPRESSION_H_
