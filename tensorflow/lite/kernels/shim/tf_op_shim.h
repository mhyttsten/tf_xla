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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TF_OP_SHIM_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TF_OP_SHIM_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh() {
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
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/tf_tensor_view.h"

// This file contains the TF adapter. That is, it takes a `OpKernelShim`
// class and provides a TF kernel out of it.

namespace tflite {
namespace shim {

// TF implementation of the methods during an op kernel initialization
class TfInitContext : public InitContext<TfInitContext> {
 public:
  explicit TfInitContext(const ::tensorflow::OpKernelConstruction* context);
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;

 private:
  const ::tensorflow::OpKernelConstruction* context_;
};

// TF implementation of the methods during an op kernel invocation
class TfInvokeContext : public InvokeContext<TfInvokeContext> {
 public:
  explicit TfInvokeContext(::tensorflow::OpKernelContext* context);
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const;
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape& shape) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  ::tensorflow::OpKernelContext* context_;
};

// TF implementation of the methods during shape inference
class TfShapeInferenceContext
    : public ShapeInferenceContext<TfShapeInferenceContext> {
 public:
  explicit TfShapeInferenceContext(
      ::tensorflow::shape_inference::InferenceContext* context);
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const;
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  ::tensorflow::shape_inference::InferenceContext* context_;
};

// Converts absl::Status to tensorflow::Status
::tensorflow::Status FromAbslStatus(const ::absl::Status& s);
// Converts to tensorflow::Status to absl::Status
::absl::Status ToAbslStatus(const ::tensorflow::Status& s);

// The adaptor between an op implementation (OpKernelShim subclass) and TF
// runtime
template <template <Runtime> typename Impl>
class TfOpKernel : public ::tensorflow::OpKernel {
 public:
  using ImplType = Impl<Runtime::kTf>;

  explicit TfOpKernel(::tensorflow::OpKernelConstruction* c)
      : OpKernel(c), impl_(absl::make_unique<ImplType>()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh mht_0(mht_0_v, 272, "", "./tensorflow/lite/kernels/shim/tf_op_shim.h", "TfOpKernel");

    TfInitContext ctx(c);
    c->SetStatus(FromAbslStatus(impl_->Init(&ctx)));
  }

  // The main computation of the op
  void Compute(::tensorflow::OpKernelContext* c) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh mht_1(mht_1_v, 281, "", "./tensorflow/lite/kernels/shim/tf_op_shim.h", "Compute");

    TfInvokeContext ctx(c);
    OP_REQUIRES_OK(c, FromAbslStatus(impl_->Invoke(&ctx)));
  }

  // Shape inference for the op.
  static tensorflow::Status ShapeInference(
      ::tensorflow::shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh mht_2(mht_2_v, 291, "", "./tensorflow/lite/kernels/shim/tf_op_shim.h", "ShapeInference");

    TfShapeInferenceContext ctx(c);
    return FromAbslStatus(ImplType::ShapeInference(&ctx));
  }

  // The operation name
  static const char* OpName() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh mht_3(mht_3_v, 300, "", "./tensorflow/lite/kernels/shim/tf_op_shim.h", "OpName");
 return ImplType::kOpName; }

 protected:
  std::unique_ptr<OpKernelShim<Impl, Runtime::kTf>> impl_;
};

static_assert(::tensorflow::shape_inference::InferenceContext::kUnknownDim ==
                  Shape::kUnknownDim,
              "The values must match.");
static_assert(::tensorflow::shape_inference::InferenceContext::kUnknownRank ==
                  Shape::kUnknownRank,
              "The values must match.");

// Builds the OpDef to register theop with the TF runtime
template <typename Kernel>
::tensorflow::register_op::OpDefBuilderWrapper CreateOpDefBuilderWrapper() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSshimPStf_op_shimDTh mht_4(mht_4_v, 318, "", "./tensorflow/lite/kernels/shim/tf_op_shim.h", "CreateOpDefBuilderWrapper");

  auto ret =
      ::tensorflow::register_op::OpDefBuilderWrapper(Kernel::ImplType::kOpName);
  for (const auto& input : Kernel::ImplType::Inputs()) ret = ret.Input(input);
  for (const auto& output : Kernel::ImplType::Outputs())
    ret = ret.Output(output);
  for (const auto& attr : Kernel::ImplType::Attrs()) ret = ret.Attr(attr);
  ret.SetShapeFn(Kernel::ShapeInference).Doc(Kernel::ImplType::kDoc);
  return ret;
}

template <>
struct ContextTypeForRuntime<Runtime::kTf> {
  using Init = TfInitContext;
  using Invoke = TfInvokeContext;
  using ShapeInference = TfShapeInferenceContext;
};

// Macros for defining an op. These are taken from op.h because they need to be
// slightly modified here.
#define REGISTER_OP_SHIM_IMPL(ctr, op_kernel_cls)                            \
  static ::tensorflow::InitOnStartupMarker const register_op##ctr            \
      TF_ATTRIBUTE_UNUSED =                                                  \
          TF_INIT_ON_STARTUP_IF(SHOULD_REGISTER_OP(op_kernel_cls::OpName())) \
          << ::tflite::shim::CreateOpDefBuilderWrapper<op_kernel_cls>()

#define REGISTER_TF_OP_SHIM(op_kernel_cls) \
  TF_ATTRIBUTE_ANNOTATE("tf:op")           \
  TF_NEW_ID_FOR_INIT(REGISTER_OP_SHIM_IMPL, op_kernel_cls)

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TF_OP_SHIM_H_
