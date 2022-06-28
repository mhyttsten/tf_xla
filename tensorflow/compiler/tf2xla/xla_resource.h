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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_RESOURCE_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_RESOURCE_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh() {
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


#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/managed_stack_trace.h"

namespace tensorflow {

// Represents a resource, such as a Variable or TensorArray.
class XlaResource {
 public:
  enum Kind {
    kInvalid,
    kVariable,
    kTensorArray,
    kStack,
  };
  static absl::string_view KindToString(Kind kind);

  // Creates a new Stack resource.
  static std::unique_ptr<XlaResource> CreateStack(string name, DataType type,
                                                  int64_t max_size);

  // Creates a new TensorArray resource.
  static std::unique_ptr<XlaResource> CreateTensorArray(
      string name, DataType type, TensorShape shape, xla::XlaOp initial_value,
      int64_t max_array_size);

  XlaResource(Kind kind, int arg_num, string name, DataType type,
              TensorShape shape, xla::XlaOp initial_value,
              int64_t max_array_size,
              const std::set<string>& tensor_array_gradients,
              bool tensor_array_multiple_writes_aggregate,
              const absl::optional<ManagedStackTrace>& definition_stack_trace =
                  absl::nullopt);

  XlaResource(const XlaResource&) = delete;
  XlaResource(XlaResource&&) = delete;
  XlaResource& operator=(const XlaResource&) = delete;
  XlaResource& operator=(XlaResource&&) = delete;

  Kind kind() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_0(mht_0_v, 233, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "kind");
 return kind_; }

  // If this resource is visible externally to the computation, what was its
  // argument number?
  // < 0 means "not visible externally".
  int arg_num() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_1(mht_1_v, 241, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "arg_num");
 return arg_num_; }

  // A descriptive name for the resource, used in error messages.
  const string& name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_2(mht_2_v, 247, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "name");
 return name_; }

  // Current type and value of the resource. Uninitialized resources are
  // represented by a default (zero) handle and type DT_INVALID.
  // While the type of a resource is notionally fixed during execution, when
  // a resource is first initialized we do not yet know its type, so we keep
  // track of its type dynamically.
  DataType type() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_3(mht_3_v, 257, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "type");
 return type_; }

  // Shape of the resource. For an uninitialized resource, this is ignored.
  // For a Variable, this is the shape of the value. For a TensorArray or Stack
  // this is the shape of each entry in the TensorArray/Stack.
  const TensorShape& shape() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_4(mht_4_v, 265, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "shape");
 return shape_; }

  const xla::XlaOp& value() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_5(mht_5_v, 270, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "value");
 return value_; }

  // Value of the resource at computation entry. Used to detect which
  // variables have new values that need to be written back.
  const xla::XlaOp& initial_value() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_6(mht_6_v, 277, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "initial_value");
 return initial_value_; }

  // An xla shape that indicates how this resource variable is represented on
  // device.
  const absl::optional<xla::Shape>& representation_shape() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_7(mht_7_v, 284, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "representation_shape");

    return representation_shape_;
  }

  // A variable is initialized if it has a value.
  bool initialized() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_8(mht_8_v, 292, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "initialized");
 return value_.valid(); }

  // Sets the type and shape of the resource. The type and shape of a resource
  // must not change once the variable has been initialized.
  Status SetTypeAndShape(DataType type, const TensorShape& shape);

  // Sets the current value of the resource. Returns an error if the type is not
  // set to a valid value.
  Status SetValue(const xla::XlaOp& value);

  // Sets the current value of the resource to an all-zero value.
  Status SetZeroValue(xla::XlaBuilder* builder);

  // Sets the representational shape of the resource on device.
  void SetRepresentationShape(const xla::Shape& shape) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_9(mht_9_v, 309, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "SetRepresentationShape");

    representation_shape_ = absl::make_optional(shape);
  }

  // Looks up the gradient for `source`, or creates it if it does not already
  // exist. The call target must be an initialized TensorArray resource. A
  // TensorArray can have multiple named gradients; see the operator
  // documentation for TensorArrayGradV3 for details.
  Status GetOrCreateTensorArrayGradient(const string& source,
                                        xla::XlaBuilder* builder,
                                        XlaResource** gradient_out);

  // Packs a resource into a single XLA value `pack`, suitable for use as
  // an XlaCompiler::Argument. For non-TensorArrays or TensorArrays without
  // gradients, sets `*pack` to `value`.
  // For TensorArrays with gradients, packs the value and its gradient values in
  // a tuple; the gradients values are packed in order by source name.
  Status Pack(xla::XlaOp* pack, xla::XlaBuilder* builder) const;

  // Updates the resource with values from `pack`. If `gradient_sources` is
  // non-empty, treats `pack` as a tuple that represents a TensorArray and
  // its gradients, and unpacks and updates the gradient resources.
  // If `reset_initial_values` is true, sets the initial_values as well as the
  // values.
  // Opposite of Pack().
  Status SetFromPack(const std::set<string>& gradient_sources,
                     const xla::XlaOp& pack, xla::XlaBuilder* builder);

  bool IsOverwritten() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_10(mht_10_v, 340, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "IsOverwritten");
 return is_overwritten_; }

  // TensorArray and Stack specific fields
  // TODO(phawkins): refactor this code to use subclasses, rather than putting
  // kind-specific fields in XlaResource.

  // 'max_array_size' stores the expected size of the TensorArray or Stack.
  // We need to store this since sometimes TensorArrays must be initialized
  // lazily since we do not know the element shape at construction time.
  // Used by both TensorArrays and Stacks.
  int64_t max_array_size() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_11(mht_11_v, 353, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "max_array_size");
 return max_array_size_; }
  void set_max_array_size(int64_t size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_12(mht_12_v, 357, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "set_max_array_size");
 max_array_size_ = size; }

  bool tensor_array_multiple_writes_aggregate() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_13(mht_13_v, 362, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "tensor_array_multiple_writes_aggregate");

    return tensor_array_multiple_writes_aggregate_;
  }

  // 'tensor_array_gradient' is a map from TensorArrayGradV3 'source' attributes
  // to an XlaResource containing the gradient TensorArrays. We store a pointer
  // here since there should only be one gradient TensorArray per 'source'
  // string, irrespective of the number of calls to TensorArrayGrad. The map
  // is ordered since values are packed into tuples by Pack() sorted by name
  // order.
  const std::map<string, std::unique_ptr<XlaResource>>& tensor_array_gradients()
      const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSxla_resourceDTh mht_14(mht_14_v, 376, "", "./tensorflow/compiler/tf2xla/xla_resource.h", "tensor_array_gradients");

    return tensor_array_gradients_;
  }

 private:
  const Kind kind_;
  const int arg_num_;
  const string name_;

  DataType type_;
  TensorShape shape_;
  xla::XlaOp value_;
  xla::XlaOp initial_value_;

  // An xla shape that indicates how this resource variable is represented on
  // device.
  absl::optional<xla::Shape> representation_shape_;

  int64_t max_array_size_ = -1;
  bool tensor_array_multiple_writes_aggregate_ = false;

  std::map<string, std::unique_ptr<XlaResource>> tensor_array_gradients_;
  bool is_overwritten_ = false;

  absl::optional<ManagedStackTrace> definition_stack_trace_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_RESOURCE_H_
