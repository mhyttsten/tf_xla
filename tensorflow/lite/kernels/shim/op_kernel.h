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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_OP_KERNEL_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_OP_KERNEL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh() {
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


// This file defines a shim layer on top of TF and TFLite custom op APIs.
// The goal is for a custom op to be written once and used for both runtimes
//
// It consists of two pieces:
//   * A set of *context* interfaces:
//     ** InvokeContext, InitContext, ShapeInferenceContext
//     These are passed on to the custom op implementation to read/write
//     tensors, etc.
//
//   * An OpKernelShim interface:
//     This is what a custom op needs to implement. By using that interface the
//     custom op can then be easily converted to both a TF op kernel and a
//     TFLite op kernel.

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tflite {
namespace shim {

// List of the TF custom op APIs this shim library is abstracting away.
//
// This enum is used as the template parameter in various places in
// order to pick the correct set of types (eg. TfInvokeContext vs.
// TfLiteInvokeContext) in the op implementation.
enum class Runtime { kTf, kTfLite };

// TensorView or error
using TensorViewOr = absl::StatusOr<std::unique_ptr<TensorView>>;
using ConstTensorViewOr = absl::StatusOr<std::unique_ptr<const TensorView>>;

// Below are the interfaces for various "Context" objects to abstract away the
// TF and TFLite differences.
//
// The interfaces are static and use the CRTP pattern instead of virtual
// methods.

// The attribute dictionary passed to the op
using AttrValue = absl::variant<bool, int64_t, float, absl::string_view>;

// The interface for available methods during an op kernel initialization
template <typename SubType>
class InitContext {
 public:
  // Read the given attribute and populate the given value.
  template <typename AttrType>
  absl::Status GetAttr(const std::string& attr_name, AttrType* value) const;

 protected:
  // Read a given attribute or return error
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const {
    return static_cast<const SubType&>(*this).GetAttr(attr_name);
  }
};

// The interface for available methods during an op kernel invocation
template <typename SubType>
class InvokeContext {
 public:
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_0(mht_0_v, 255, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "GetInput");

    return static_cast<const SubType&>(*this).GetInput(idx);
  }
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape& shape) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_1(mht_1_v, 262, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "GetOutput");

    return static_cast<const SubType&>(*this).GetOutput(idx, shape);
  }
  // Number of input tensors
  int NumInputs() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_2(mht_2_v, 269, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "NumInputs");

    return static_cast<const SubType&>(*this).NumInputs();
  }
  // Number of output tensors
  int NumOutputs() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_3(mht_3_v, 276, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "NumOutputs");

    return static_cast<const SubType&>(*this).NumOutputs();
  }
};

// The interface for available methods during shape inference
template <typename SubType>
class ShapeInferenceContext {
 public:
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_4(mht_4_v, 289, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "GetInputShape");

    return static_cast<const SubType&>(*this).GetInputShape(idx);
  }
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_5(mht_5_v, 296, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "SetOutputShape");

    return static_cast<SubType&>(*this).SetOutputShape(idx, shape);
  }
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_6(mht_6_v, 303, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "GetInputTensor");

    return static_cast<const SubType&>(*this).GetInputTensor(idx);
  }
  // Number of input tensors
  int NumInputs() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_7(mht_7_v, 310, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "NumInputs");

    return static_cast<const SubType&>(*this).NumInputs();
  }
  // Number of output tensors
  int NumOutputs() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_8(mht_8_v, 317, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "NumOutputs");

    return static_cast<const SubType&>(*this).NumOutputs();
  }
  // Read the given attribute and populate the given value.
  template <typename AttrType>
  absl::Status GetAttr(const std::string& attr_name, AttrType* value) const;

 protected:
  // Read a given attribute or return error
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const {
    return static_cast<const SubType&>(*this).GetAttr(attr_name);
  }
};

// Maps the Runtime to the correct context types.
// eg. ContextTypeForRuntime<Runtime::Tf>  -->
//       { TfInitContext, TfInvokeContext, TfShapreInferenceContext }
template <Runtime Rt>
struct ContextTypeForRuntime {
  // * Init
  // * Invoke
  // * ShapeInference
};

// A Tensorflow operation interface which is then adapted to both TF and TFLite
// runtimes.
//
// Example usage:
//
//   template<Runtime R>
//   class MyOp : public OpKernelShim<MyOp, R> {
//
//     // Attributes declaration
//     // (syntax: https://www.tensorflow.org/guide/create_op)
//     static std::vector<std::string> Attrs();
//
//     // Input tensors declaration
//     // (syntax: https://www.tensorflow.org/guide/create_op)
//     static std::vector<std::string> Inputs();
//
//     // Output tensors declaration
//     // (syntax: https://www.tensorflow.org/guide/create_op)
//     static std::vector<std::string> Outputs();
//
//     // Initializes the op
//     absl::Status Init(InitContext* ctx);
//
//     // Runs the operation
//     absl::Status Invoke(InvokeContext* ctx);
//
//     // Shape inference
//     static absl::Status ShapeInference(ShapeInferenceContext* ctx);
//
//   };
//
// WARNING: Experimental interface, subject to change
template <template <Runtime> typename SubType, Runtime Rt>
class OpKernelShim {
 public:
  // Some typedefs for convenience
  using Shape = ::tflite::shim::Shape;
  using InitContext =
      ::tflite::shim::InitContext<typename ContextTypeForRuntime<Rt>::Init>;
  using InvokeContext =
      ::tflite::shim::InvokeContext<typename ContextTypeForRuntime<Rt>::Invoke>;
  using ShapeInferenceContext = ::tflite::shim::ShapeInferenceContext<
      typename ContextTypeForRuntime<Rt>::ShapeInference>;

  // Needed because the pointer to this class is stored
  virtual ~OpKernelShim() = default;

  // If the operation has any attributes they are passed here.
  absl::Status Init(InitContext* ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_9(mht_9_v, 392, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "Init");

    return static_cast<SubType<Rt>&>(*this).Init(ctx);
  }

  // The actual computations of the operation
  absl::Status Invoke(InvokeContext* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_10(mht_10_v, 400, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "Invoke");

    return static_cast<SubType<Rt>&>(*this).Invoke(ctx);
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_11(mht_11_v, 408, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "ShapeInference");

    return SubType<Rt>::ShapeInference(ctx);
  }

 protected:
  OpKernelShim() = default;
};

/////////////////////// Implementations

namespace internal {
// Extract the given AttrType from the AttrValue variant or returns error.
template <typename AttrType>
absl::Status GetAttr(const std::string& attr_name,
                     const absl::StatusOr<AttrValue> attr_value_or,
                     AttrType* value) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_12(mht_12_v, 427, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "GetAttr");

  if (!attr_value_or.ok()) return attr_value_or.status();
  const AttrValue& attr_value = attr_value_or.value();
  if (!absl::holds_alternative<AttrType>(attr_value)) {
    return absl::InternalError(
        absl::StrCat("The attribute type does not match the provided "
                     "type: attr_name: ",
                     attr_name));
  }
  *value = absl::get<AttrType>(attr_value);
  return absl::OkStatus();
}
}  // namespace internal

template <typename SubType>
template <typename AttrType>
absl::Status InitContext<SubType>::GetAttr(const std::string& attr_name,
                                           AttrType* value) const {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_13(mht_13_v, 448, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "InitContext<SubType>::GetAttr");

  const auto attr_value_or = GetAttr(attr_name);
  return internal::GetAttr<AttrType>(attr_name, attr_value_or, value);
}

template <typename SubType>
template <typename AttrType>
absl::Status ShapeInferenceContext<SubType>::GetAttr(
    const std::string& attr_name, AttrType* value) const {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("attr_name: \"" + attr_name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPSop_kernelDTh mht_14(mht_14_v, 460, "", "./tensorflow/lite/kernels/shim/op_kernel.h", "ShapeInferenceContext<SubType>::GetAttr");

  const auto attr_value_or = GetAttr(attr_name);
  return internal::GetAttr<AttrType>(attr_name, attr_value_or, value);
}

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_ABSTRACT_OP_H_
