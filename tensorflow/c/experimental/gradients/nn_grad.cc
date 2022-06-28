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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc() {
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
#include "tensorflow/c/experimental/gradients/nn_grad.h"

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/experimental/ops/array_ops.h"
#include "tensorflow/c/experimental/ops/math_ops.h"
#include "tensorflow/c/experimental/ops/nn_ops.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"

using std::vector;
using tensorflow::ops::BiasAddGrad;
using tensorflow::ops::Mul;
using tensorflow::ops::ReluGrad;

namespace tensorflow {
namespace gradients {
namespace {

class ReluGradientFunction : public GradientFunction {
 public:
  explicit ReluGradientFunction(vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs_(f_outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_0(mht_0_v, 208, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "ReluGradientFunction");

    for (auto output : forward_outputs_) {
      if (output) {
        output->Ref();
      }
    }
  }

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_1(mht_1_v, 221, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "Compute");

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    AbstractTensorHandle* activations = forward_outputs_[0];

    // Calculate Grad
    std::string name = "relu_grad";
    TF_RETURN_IF_ERROR(ReluGrad(ctx, upstream_grad, activations,
                                &grad_inputs[0], name.c_str()));
    return Status::OK();
  }
  ~ReluGradientFunction() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_2(mht_2_v, 234, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "~ReluGradientFunction");

    for (auto output : forward_outputs_) {
      if (output) {
        output->Unref();
      }
    }
  }

 private:
  // TODO(b/174778737): Only hold needed outputs.
  vector<AbstractTensorHandle*> forward_outputs_;
};

Status BroadcastMul(AbstractContext* ctx, AbstractTensorHandle* vec,
                    AbstractTensorHandle* mat,
                    absl::Span<AbstractTensorHandle*> outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_3(mht_3_v, 252, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "BroadcastMul");

  if (!isa<ImmediateExecutionContext>(ctx)) {
    // TODO(b/168850692): Fix this.
    return errors::Unimplemented(
        "BroadcastMul is not supported in tracing mode yet.");
  }
  auto imm_ctx = dyn_cast<ImmediateExecutionContext>(ctx);
  AbstractTensorPtr minus_1(imm_ctx->CreateInt32Scalar(-1));
  ImmediateTensorHandlePtr dim(imm_ctx->CreateLocalHandle(minus_1.get()));
  AbstractTensorHandle* expand_dims_outputs;
  TF_RETURN_IF_ERROR(
      ops::ExpandDims(ctx, vec, dim.get(), &expand_dims_outputs, "ExpandDims"));
  TF_RETURN_IF_ERROR(
      ops::Mul(ctx, expand_dims_outputs, mat, &outputs[0], "Mul"));
  expand_dims_outputs->Unref();
  return Status::OK();
}

class SparseSoftmaxCrossEntropyWithLogitsGradientFunction
    : public GradientFunction {
 public:
  explicit SparseSoftmaxCrossEntropyWithLogitsGradientFunction(
      vector<AbstractTensorHandle*> f_outputs)
      : forward_outputs_(f_outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_4(mht_4_v, 278, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "SparseSoftmaxCrossEntropyWithLogitsGradientFunction");
}

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_5(mht_5_v, 285, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "Compute");

    // Grad for Softmax Input
    TF_RETURN_IF_ERROR(BroadcastMul(
        ctx, grad_outputs[0], forward_outputs_[1],
        grad_inputs.subspan(0, 1)));  // upstream_grad * local softmax grad

    // Grad for labels is null
    grad_inputs[1] = nullptr;
    return Status::OK();
  }
  ~SparseSoftmaxCrossEntropyWithLogitsGradientFunction() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_6(mht_6_v, 298, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "~SparseSoftmaxCrossEntropyWithLogitsGradientFunction");
}

 private:
  vector<AbstractTensorHandle*> forward_outputs_;
};

// TODO(vnvo2409): Add python test
class BiasAddGradientFunction : public GradientFunction {
 public:
  explicit BiasAddGradientFunction(AttrBuilder f_attrs)
      : forward_attrs_(f_attrs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_7(mht_7_v, 311, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "BiasAddGradientFunction");
}

  Status Compute(AbstractContext* ctx,
                 absl::Span<AbstractTensorHandle* const> grad_outputs,
                 absl::Span<AbstractTensorHandle*> grad_inputs) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_8(mht_8_v, 318, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "Compute");

    /* Given upstream grad U and a BiasAdd: A + bias, the gradients are:
     *
     *    dA = U
     *    dbias = reduceSum(U, dims = channel_dim)
     */

    AbstractTensorHandle* upstream_grad = grad_outputs[0];
    DCHECK(upstream_grad);

    // Recover data format from forward pass for gradient.
    std::string data_format;
    TF_RETURN_IF_ERROR(forward_attrs_.Get("data_format", &data_format));

    // Grad for A
    grad_inputs[0] = upstream_grad;
    grad_inputs[0]->Ref();

    // Grad for bias
    std::string name = "bias_add_grad";
    TF_RETURN_IF_ERROR(BiasAddGrad(ctx, upstream_grad, &grad_inputs[1],
                                   data_format.c_str(), name.c_str()));

    return Status::OK();
  }
  ~BiasAddGradientFunction() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_9(mht_9_v, 346, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "~BiasAddGradientFunction");
}

 private:
  AttrBuilder forward_attrs_;
};

}  // namespace

GradientFunction* ReluRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_10(mht_10_v, 357, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "ReluRegisterer");

  return new ReluGradientFunction(op.outputs);
}

GradientFunction* SparseSoftmaxCrossEntropyWithLogitsRegisterer(
    const ForwardOperation& op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_11(mht_11_v, 365, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "SparseSoftmaxCrossEntropyWithLogitsRegisterer");

  return new SparseSoftmaxCrossEntropyWithLogitsGradientFunction(op.outputs);
}

GradientFunction* BiasAddRegisterer(const ForwardOperation& op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPSnn_gradDTcc mht_12(mht_12_v, 372, "", "./tensorflow/c/experimental/gradients/nn_grad.cc", "BiasAddRegisterer");

  return new BiasAddGradientFunction(op.attrs);
}

}  // namespace gradients
}  // namespace tensorflow
