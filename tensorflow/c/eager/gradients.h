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

#ifndef TENSORFLOW_C_EAGER_GRADIENTS_H_
#define TENSORFLOW_C_EAGER_GRADIENTS_H_
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
class MHTracer_DTPStensorflowPScPSeagerPSgradientsDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSgradientsDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tape.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"

namespace tensorflow {
namespace gradients {

// =============== Experimental C++ API for computing gradients ===============

// Sample gradient function:
//
// class AddGradientFunction : public GradientFunction {
//  public:
//   Status Compute(Context* ctx,
//                  absl::Span<AbstractTensorHandle* const> grad_inputs,
//                  absl::Span<AbstractTensorHandle*> grad_outputs) override {
//     grad_outputs[0] = grad_inputs[0];
//     grad_outputs[1] = grad_inputs[0];
//     grad_outputs[0]->Ref();
//     grad_outputs[1]->Ref();
//     return Status::OK();
//   }
//   ~AddGradientFunction() override {}
// };
//
// GradientFunction* AddRegisterer(const ForwardOperation& op) {
//   // More complex gradient functions can use inputs/attrs etc. from the
//   // forward `op`.
//   return new AddGradientFunction;
// }
//
// Status RegisterGradients(GradientRegistry* registry) {
//   return registry->Register("Add", AddRegisterer);
// }
class GradientFunction {
 public:
  virtual Status Compute(AbstractContext* ctx,
                         absl::Span<AbstractTensorHandle* const> grad_outputs,
                         absl::Span<AbstractTensorHandle*> grad_inputs) = 0;
  virtual ~GradientFunction() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSgradientsDTh mht_0(mht_0_v, 229, "", "./tensorflow/c/eager/gradients.h", "~GradientFunction");
}
};

// Metadata from the forward operation that is made available to the
// gradient registerer to instantiate a GradientFunction.
struct ForwardOperation {
 public:
  string op_name;
  std::vector<AbstractTensorHandle*> inputs;
  std::vector<AbstractTensorHandle*> outputs;
  std::vector<int64_t> skip_input_indices;
  AttrBuilder attrs;
};

using GradientFunctionFactory =
    std::function<GradientFunction*(const ForwardOperation& op)>;

// Map from op name to a `GradientFunctionFactory`.
class GradientRegistry {
 public:
  Status Register(const string& op,
                  GradientFunctionFactory gradient_function_factory);
  Status Lookup(const ForwardOperation& op,
                std::unique_ptr<GradientFunction>* gradient_function) const;

 private:
  absl::flat_hash_map<string, GradientFunctionFactory> registry_;
};

// TODO(srbs): Figure out if we can avoid declaring this in the public header.
// Wrapper for a tensor output of an operation executing under a tape.
//
// `GetID` returns a unique id for the wrapped tensor which is used to maintain
// a map (`tensorflow::eager::TensorTape`) from the wrapped tensor to the id of
// the op that produced it (or -1 if this tensor was watched using
// `GradientTape::Watch`.) The op_id is simply a unique index assigned to each
// op executed under the tape. A separate map (`tensorflow::eager::OpTape`)
// maintains the map from `op_id` to a `OpTapeEntry` which stores the `op_type`,
// inputs and outputs and the gradient function These data structures combined
// allow us to trace the data dependencies between operations and hence compute
// gradients.
//
// `ZerosLike` is not expected to be called and returns a nullptr. The creation
// of default zeros grads is handled by the `DefaultGradientFunction` registered
// for each op.
// TODO(srbs): We need to define `ZerosLike` here to keep the compiler happy.
// Figure out a way to avoid this.
// TODO(srbs): Should ZerosLike check-fail instead of returning nullptr?
class TapeTensor {
 public:
  explicit TapeTensor(AbstractTensorHandle* handle);
  TapeTensor(const TapeTensor& other);
  ~TapeTensor();

  int64_t GetID() const;
  tensorflow::DataType GetDType() const;

  AbstractTensorHandle* ZerosLike() const;

  AbstractTensorHandle* GetHandle() const;

 private:
  AbstractTensorHandle* handle_;
};

// A tracing/immediate-execution agnostic tape.
//
// Gradient functions defined for this tape must support handling null incoming
// gradients.
class Tape : protected eager::GradientTape<AbstractTensorHandle,
                                           GradientFunction, TapeTensor> {
 public:
  using GradientTape<AbstractTensorHandle, GradientFunction,
                     TapeTensor>::GradientTape;
  // Returns whether the tape is persistent, i.e., whether the tape will hold
  // onto its internal state after a call to `ComputeGradient`.
  using GradientTape<AbstractTensorHandle, GradientFunction,
                     TapeTensor>::IsPersistent;

  // Adds this tensor to the list of watched tensors.
  //
  // This is a no-op if the tensor is already being watched either from an
  // earlier call to `GradientTape::Watch` or being an output of an op with
  // watched inputs.
  void Watch(const AbstractTensorHandle*);
  // Records an operation with given inputs and outputs
  // on the tape and marks all its outputs as watched if at
  // least one input of the op is watched and has a trainable dtype.
  // op_name is optional and is used for debugging only.
  void RecordOperation(absl::Span<AbstractTensorHandle* const> inputs,
                       absl::Span<AbstractTensorHandle* const> outputs,
                       GradientFunction* gradient_function,
                       const string& op_name = "");
  // Returns whether any tensor in a list of tensors is being watched and has
  // a trainable dtype.
  bool ShouldRecord(
      absl::Span<const AbstractTensorHandle* const> tensors) const;
  // Unwatches this tensor on the tape. Mainly used for cleanup when deleting
  // eager tensors.
  void DeleteTrace(const AbstractTensorHandle*);

  // Consumes the internal state of the tape (so cannot be called more than
  // once unless the tape is persistent) and produces the gradient of the target
  // tensors with respect to the source tensors. The output gradients are used
  // if not empty and not null. The result is populated with one tensor per
  // target element.
  Status ComputeGradient(
      AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> targets,
      absl::Span<AbstractTensorHandle* const> sources,
      absl::Span<AbstractTensorHandle* const> output_gradients,
      absl::Span<AbstractTensorHandle*> result);
};

}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_GRADIENTS_H_
