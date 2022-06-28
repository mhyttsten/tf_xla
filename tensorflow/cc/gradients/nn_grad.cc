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
class MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc {
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
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"

namespace tensorflow {
namespace ops {
namespace {

Status SoftmaxGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_0(mht_0_v, 198, "", "./tensorflow/cc/gradients/nn_grad.cc", "SoftmaxGrad");

  // Softmax gradient function.
  // p = softmax(x) maps from [batch, n] to [batch, m]
  // dp/dx = [dp0/dx0   ... dp0/dxn-1  ]
  //         [  ...           ...      ]
  //         [dpm-1/dx0 ... dpm-1/dxn-1]
  // dL/dx = dp/dx * dL/dy
  //
  // Using alternative formula:
  // dL/dx = dL/dy * y - sum(dL/dy * y) * y
  //    = (dL/dy - sum(dL/dy * y)) * y
  auto y = op.output(0);
  auto dyy = Mul(scope, grad_inputs[0], y);
  auto sum = Sum(scope, dyy, /*axis=*/-1, Sum::KeepDims(true));
  auto sub = Sub(scope, grad_inputs[0], sum);
  auto dx = Mul(scope, sub, y);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softmax", SoftmaxGrad);

bool IsZero(const Scope& scope, const Output& grad) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_1(mht_1_v, 222, "", "./tensorflow/cc/gradients/nn_grad.cc", "IsZero");

  string op_type_name = grad.op().node()->type_string();
  if (op_type_name == "ZerosLike" || op_type_name == "Zeros") {
    return true;
  }
  // The Operation we were provided is not named something obvious so
  // we need to actually look at its contents.
  // The original python code did this by calling a utility function called
  // tensor_util.constant_value.
  // There is no C++ equivalent to tensor_util.constant_value so we do nothing
  // for the moment.
  return false;
}

// Multiply after broadcasting vec to match dimensions of mat.
//   Args:
//     vec: A 1-D tensor of dimension [D0]
//     mat: A 2-D tensor of dimension [D0, D1]
//
//   Returns:
//     A tensor of dimension [D0, D1], the result for vec * mat.
Output BroadcastMul(const Scope& scope, const Output& vec, const Output& mat) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_2(mht_2_v, 246, "", "./tensorflow/cc/gradients/nn_grad.cc", "BroadcastMul");

  auto reshaped = ExpandDims(scope, vec, -1);
  return Multiply(scope, reshaped, mat);
}

Status SoftmaxCrossEntropyWithLogitsGrad(const Scope& scope,
                                         const Operation& op,
                                         const std::vector<Output>& grad_inputs,
                                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_3(mht_3_v, 257, "", "./tensorflow/cc/gradients/nn_grad.cc", "SoftmaxCrossEntropyWithLogitsGrad");

  // Softmax gradient with cross entropy logits function.
  // We multiply the backprop for cost with the gradients - op.output[1].
  // There is no gradient for labels.

  // The outputs of the network are at input index 0.
  auto logits = op.input(0);
  // The "truth" labels are at index 1.
  auto softmax_grad = op.output(1);

  // The loss is the output at index 0, and backprop is the output at index 1.
  auto grad_loss = grad_inputs[0];
  auto grad_grad = grad_inputs[1];

  auto grad = BroadcastMul(scope, grad_loss, softmax_grad);
  if (!IsZero(scope, grad_grad)) {
    std::vector<int> axis;
    auto logits_softmax = Softmax(scope, logits);

    auto grad_grad_expand = ExpandDims(scope, grad_grad, 1);
    auto logits_softmax_expand = ExpandDims(scope, logits_softmax, 2);
    auto matmul_result =
        BatchMatMul(scope, grad_grad_expand, logits_softmax_expand);
    axis.push_back(1);
    auto squeeze_result = Squeeze(scope, matmul_result, Squeeze::Axis(axis));
    auto subtraction_result = Subtract(scope, grad_grad, squeeze_result);
    auto multiply_result = Multiply(scope, subtraction_result, logits_softmax);
    grad = Add(scope, grad, multiply_result);
  }
  auto minus_log_softmax = Multiply(scope, LogSoftmax(scope, logits), -1.0f);
  grad_outputs->push_back(grad);
  grad_outputs->push_back(BroadcastMul(scope, grad_loss, minus_log_softmax));
  return scope.status();
}
REGISTER_GRADIENT_OP("SoftmaxCrossEntropyWithLogits",
                     SoftmaxCrossEntropyWithLogitsGrad);

Status LogSoftmaxGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_4(mht_4_v, 299, "", "./tensorflow/cc/gradients/nn_grad.cc", "LogSoftmaxGrad");

  auto softmax = Exp(scope, op.output(0));
  auto sum = Sum(scope, grad_inputs[0], {1}, Sum::KeepDims(true));
  auto mul = Mul(scope, sum, softmax);
  auto dx = Sub(scope, grad_inputs[0], mul);
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("LogSoftmax", LogSoftmaxGrad);

Status ReluGradHelper(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_5(mht_5_v, 314, "", "./tensorflow/cc/gradients/nn_grad.cc", "ReluGradHelper");

  auto dx = internal::ReluGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Relu", ReluGradHelper);

Status Relu6GradHelper(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_6(mht_6_v, 326, "", "./tensorflow/cc/gradients/nn_grad.cc", "Relu6GradHelper");

  auto dx = internal::Relu6Grad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Relu6", Relu6GradHelper);

Status LeakyReluGradHelper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_7(mht_7_v, 338, "", "./tensorflow/cc/gradients/nn_grad.cc", "LeakyReluGradHelper");

  float alpha;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "alpha", &alpha));
  internal::LeakyReluGrad::Attrs attrs;
  auto dx = internal::LeakyReluGrad(scope, grad_inputs[0], op.input(0),
                                    attrs.Alpha(alpha));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("LeakyRelu", LeakyReluGradHelper);

Status LeakyReluGradGradHelper(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_8(mht_8_v, 354, "", "./tensorflow/cc/gradients/nn_grad.cc", "LeakyReluGradGradHelper");

  float alpha;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "alpha", &alpha));
  internal::LeakyReluGrad::Attrs attrs;
  auto dx = internal::LeakyReluGrad(scope, grad_inputs[0], op.input(1),
                                    attrs.Alpha(alpha));
  grad_outputs->push_back(dx);
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("LeakyReluGrad", LeakyReluGradGradHelper);

Status EluGradHelper(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_9(mht_9_v, 371, "", "./tensorflow/cc/gradients/nn_grad.cc", "EluGradHelper");

  auto dx = internal::EluGrad(scope, grad_inputs[0], op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Elu", EluGradHelper);

Status SeluGradHelper(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_10(mht_10_v, 383, "", "./tensorflow/cc/gradients/nn_grad.cc", "SeluGradHelper");

  auto dx = internal::SeluGrad(scope, grad_inputs[0], op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Selu", SeluGradHelper);

Status L2LossGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_11(mht_11_v, 395, "", "./tensorflow/cc/gradients/nn_grad.cc", "L2LossGrad");

  grad_outputs->push_back(Mul(scope, op.input(0), grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("L2Loss", L2LossGrad);

Status BiasAddGradHelper(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_12(mht_12_v, 406, "", "./tensorflow/cc/gradients/nn_grad.cc", "BiasAddGradHelper");

  string data_format;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "data_format", &data_format));
  auto dx_1 =
      BiasAddGrad(scope, grad_inputs[0], BiasAddGrad::DataFormat(data_format));
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(dx_1);
  return scope.status();
}
REGISTER_GRADIENT_OP("BiasAdd", BiasAddGradHelper);

Status Conv2DGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_13(mht_13_v, 423, "", "./tensorflow/cc/gradients/nn_grad.cc", "Conv2DGrad");

  string data_format;
  string padding;
  std::vector<int32> strides;
  bool use_cudnn_on_gpu;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  auto dx_1 = Conv2DBackpropInput(scope, Shape(scope, op.input(0)), op.input(1),
                                  grad_inputs[0], strides, padding,
                                  Conv2DBackpropInput::DataFormat(data_format)
                                      .UseCudnnOnGpu(use_cudnn_on_gpu));
  grad_outputs->push_back(dx_1);
  auto dx_2 =
      Conv2DBackpropFilter(scope, op.input(0), Shape(scope, op.input(1)),
                           grad_inputs[0], strides, padding,
                           Conv2DBackpropFilter::DataFormat(data_format)
                               .UseCudnnOnGpu(use_cudnn_on_gpu));
  grad_outputs->push_back(dx_2);
  return scope.status();
}
REGISTER_GRADIENT_OP("Conv2D", Conv2DGrad);

Status MaxPoolGradHelper(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_14(mht_14_v, 453, "", "./tensorflow/cc/gradients/nn_grad.cc", "MaxPoolGradHelper");

  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> ksize;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  auto dx = internal::MaxPoolGrad(
      scope, op.input(0), op.output(0), grad_inputs[0], ksize, strides, padding,
      internal::MaxPoolGrad::DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPool", MaxPoolGradHelper);

Status MaxPoolGradV2Helper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_15(mht_15_v, 476, "", "./tensorflow/cc/gradients/nn_grad.cc", "MaxPoolGradV2Helper");

  string data_format;
  string padding;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  auto dx = MaxPoolGradV2(scope, op.input(0), op.output(0), grad_inputs[0],
                          op.input(1), op.input(2), padding,
                          MaxPoolGradV2::DataFormat(data_format));
  grad_outputs->push_back(dx);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPoolV2", MaxPoolGradV2Helper);

Status MaxPool3DGradHelper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_16(mht_16_v, 497, "", "./tensorflow/cc/gradients/nn_grad.cc", "MaxPool3DGradHelper");

  std::vector<int32> ksize;
  std::vector<int32> strides;
  string padding;
  string data_format;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  MaxPool3DGrad::Attrs grad_attrs;
  auto dx =
      MaxPool3DGrad(scope, op.input(0), op.output(0), grad_inputs[0], ksize,
                    strides, padding, grad_attrs.DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("MaxPool3D", MaxPool3DGradHelper);

Status AvgPoolGradHelper(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_17(mht_17_v, 521, "", "./tensorflow/cc/gradients/nn_grad.cc", "AvgPoolGradHelper");

  std::vector<int32> ksize;
  std::vector<int32> strides;
  string padding;
  string data_format;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  internal::AvgPoolGrad::Attrs grad_attrs;
  auto dx = internal::AvgPoolGrad(scope, Shape(scope, op.input(0)),
                                  grad_inputs[0], ksize, strides, padding,
                                  grad_attrs.DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("AvgPool", AvgPoolGradHelper);

Status AvgPool3DGradHelper(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_18(mht_18_v, 545, "", "./tensorflow/cc/gradients/nn_grad.cc", "AvgPool3DGradHelper");

  std::vector<int32> ksize;
  std::vector<int32> strides;
  string padding;
  string data_format;
  auto attrs = op.output(0).node()->attrs();
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "ksize", &ksize));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "strides", &strides));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "padding", &padding));
  TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "data_format", &data_format));
  AvgPool3DGrad::Attrs grad_attrs;
  auto dx =
      AvgPool3DGrad(scope, Shape(scope, op.input(0)), grad_inputs[0], ksize,
                    strides, padding, grad_attrs.DataFormat(data_format));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("AvgPool3D", AvgPool3DGradHelper);

Status LRNGradHelper(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_19(mht_19_v, 569, "", "./tensorflow/cc/gradients/nn_grad.cc", "LRNGradHelper");

  auto dx = internal::LRNGrad(scope, grad_inputs[0], op.input(0), op.output(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("LRN", LRNGradHelper);

Status SoftplusGradHelper(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_20(mht_20_v, 581, "", "./tensorflow/cc/gradients/nn_grad.cc", "SoftplusGradHelper");

  auto dx = internal::SoftplusGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softplus", SoftplusGradHelper);

Status SoftsignGradHelper(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_21(mht_21_v, 593, "", "./tensorflow/cc/gradients/nn_grad.cc", "SoftsignGradHelper");

  auto dx = internal::SoftsignGrad(scope, grad_inputs[0], op.input(0));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("Softsign", SoftsignGradHelper);

Status FractionalAvgPoolGradHelper(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_22(mht_22_v, 605, "", "./tensorflow/cc/gradients/nn_grad.cc", "FractionalAvgPoolGradHelper");

  bool overlapping;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "overlapping", &overlapping));
  auto dx = internal::FractionalAvgPoolGrad(
      scope, Shape(scope, op.input(0), Shape::OutType(DT_INT64)),
      grad_inputs[0], op.output(1), op.output(2),
      internal::FractionalAvgPoolGrad::Overlapping(overlapping));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("FractionalAvgPool", FractionalAvgPoolGradHelper);

Status FractionalMaxPoolGradHelper(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_23(mht_23_v, 623, "", "./tensorflow/cc/gradients/nn_grad.cc", "FractionalMaxPoolGradHelper");

  bool overlapping;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.output(0).node()->attrs(), "overlapping", &overlapping));
  auto dx = internal::FractionalMaxPoolGrad(
      scope, op.input(0), op.output(0), grad_inputs[0], op.output(1),
      op.output(2), internal::FractionalMaxPoolGrad::Overlapping(overlapping));
  grad_outputs->push_back(dx);
  return scope.status();
}
REGISTER_GRADIENT_OP("FractionalMaxPool", FractionalMaxPoolGradHelper);

// Templated constructor for FusedBatchNormGrad[..]::Attrs.
template <typename T>
T FusedBatchNormGradAttrs(float epsilon, std::string data_format,
                          bool is_training) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("data_format: \"" + data_format + "\"");
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_24(mht_24_v, 642, "", "./tensorflow/cc/gradients/nn_grad.cc", "FusedBatchNormGradAttrs");

  T result;
  result.epsilon_ = epsilon;
  result.data_format_ = data_format;
  result.is_training_ = is_training;
  return result;
}

using BatchNormGradFn =
    std::function<Status(const Scope&, Output x, Output grad_y, Output scale,
                         const std::vector<Output>& reserve_spaces,
                         float epsilon, std::string data_format,
                         bool is_training, std::vector<Output>* grad_outputs)>;

Status BaseFusedBatchNormGrad(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              BatchNormGradFn grad_fn,
                              std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_25(mht_25_v, 662, "", "./tensorflow/cc/gradients/nn_grad.cc", "BaseFusedBatchNormGrad");

  if (op.num_outputs() < 5) {
    return errors::InvalidArgument(
        "FusedBatchNorm requires at least 5 outputs");
  }
  if (grad_inputs.empty()) {
    return errors::InvalidArgument("FusedBatchNorm grad requires 1 grad input");
  }
  if (op.num_inputs() < 3) {
    return errors::InvalidArgument("FusedBatchNorm has too few inputs");
  }

  Output x = op.input(0);
  Output grad_y = grad_inputs[0];
  Output scale = op.input(1);
  float epsilon;
  std::string data_format;
  bool is_training;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "epsilon", &epsilon));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "is_training", &is_training));

  std::vector<Output> reserve_spaces;
  reserve_spaces.push_back(op.output(3));
  reserve_spaces.push_back(op.output(4));
  if (op.num_outputs() > 5) {
    reserve_spaces.push_back(op.output(5));
  }

  if (is_training) {
    return grad_fn(scope, x, grad_y, scale, reserve_spaces, epsilon,
                   data_format, is_training, grad_outputs);
  } else {
    if (op.num_inputs() < 5) {
      return errors::InvalidArgument(
          "FusedBatchNorm requires 5 inputs in eval mode");
    }

    reserve_spaces[0] = op.input(3);  // pop_mean
    reserve_spaces[1] = op.input(4);  // pop_var
    if (data_format == "NCHW") {
      x = Transpose(scope, x, {0, 2, 3, 1});
      grad_y = Transpose(scope, grad_y, {0, 2, 3, 1});
    } else if (data_format == "NCDHW") {
      x = Transpose(scope, x, {0, 2, 3, 4, 1});
      grad_y = Transpose(scope, grad_y, {0, 2, 3, 4, 1});
    }

    std::string target_data_format;
    if (data_format == "NCHW" || data_format == "NHWC") {
      target_data_format = "NHWC";
    } else {
      target_data_format = "NDHWC";
    }

    TF_RETURN_IF_ERROR(grad_fn(scope, x, grad_y, scale, reserve_spaces, epsilon,
                               target_data_format, is_training, grad_outputs));
    if (data_format == "NCHW") {
      (*grad_outputs)[0] = Transpose(scope, (*grad_outputs)[0], {0, 3, 1, 2});
    } else if (data_format == "NCDHW") {
      (*grad_outputs)[0] =
          Transpose(scope, (*grad_outputs)[0], {0, 4, 1, 2, 3});
    }
    return scope.status();
  }
}

Status FusedBatchNormV3Grad(const Scope& scope, const Operation& op,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_26(mht_26_v, 736, "", "./tensorflow/cc/gradients/nn_grad.cc", "FusedBatchNormV3Grad");

  return BaseFusedBatchNormGrad(
      scope, op, grad_inputs,
      [](const Scope& scope, Output x, Output grad_y, Output scale,
         const std::vector<Output>& reserve_spaces, float epsilon,
         std::string data_format, bool is_training,
         std::vector<Output>* grad_outputs) {
        FusedBatchNormGradV3 grad(
            scope, grad_y, x, scale, reserve_spaces[0], reserve_spaces[1],
            reserve_spaces[2],
            FusedBatchNormGradAttrs<FusedBatchNormGradV3::Attrs>(
                epsilon, data_format, is_training));
        grad_outputs->push_back(grad.x_backprop);
        grad_outputs->push_back(grad.scale_backprop);
        grad_outputs->push_back(grad.offset_backprop);
        grad_outputs->push_back(NoGradient());
        grad_outputs->push_back(NoGradient());
        return scope.status();
      },
      grad_outputs);
}

REGISTER_GRADIENT_OP("FusedBatchNormV3", FusedBatchNormV3Grad);

Status Conv2DBackpropInputGrad(const Scope& scope, const Operation& op,
                               const std::vector<Output>& grad_inputs,
                               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_27(mht_27_v, 765, "", "./tensorflow/cc/gradients/nn_grad.cc", "Conv2DBackpropInputGrad");

  if (op.num_inputs() != 3) {
    return errors::InvalidArgument("Conv2DBackpropInput requires 3 inputs.");
  }
  if (grad_inputs.empty()) {
    return errors::InvalidArgument(
        "Conv2DBackpropInput grad requires 1 grad input");
  }

  std::vector<int> dilations, strides, explicit_paddings;
  bool use_cudnn_on_gpu;
  std::string data_format, padding;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "dilations", &dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "strides", &strides));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "explicit_paddings", &explicit_paddings));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "padding", &padding));

  grad_outputs->push_back(NoGradient());

  Conv2DBackpropFilter::Attrs filter_attrs;
  filter_attrs.use_cudnn_on_gpu_ = use_cudnn_on_gpu;
  filter_attrs.explicit_paddings_ = explicit_paddings;
  filter_attrs.data_format_ = data_format;
  filter_attrs.dilations_ = dilations;
  grad_outputs->push_back(
      Conv2DBackpropFilter(scope, grad_inputs[0], Shape(scope, op.input(1)),
                           op.input(2), strides, padding, filter_attrs));

  Conv2D::Attrs conv_attrs;
  conv_attrs.use_cudnn_on_gpu_ = use_cudnn_on_gpu;
  conv_attrs.explicit_paddings_ = explicit_paddings;
  conv_attrs.data_format_ = data_format;
  conv_attrs.dilations_ = dilations;
  grad_outputs->push_back(
      Conv2D(scope, grad_inputs[0], op.input(1), strides, padding, conv_attrs));
  return scope.status();
}
REGISTER_GRADIENT_OP("Conv2DBackpropInput", Conv2DBackpropInputGrad);

Status DepthwiseConv2dNativeGrad(const Scope& scope, const Operation& op,
                                 const std::vector<Output>& grad_inputs,
                                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSnn_gradDTcc mht_28(mht_28_v, 814, "", "./tensorflow/cc/gradients/nn_grad.cc", "DepthwiseConv2dNativeGrad");

  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("DepthwiseConv2dNative requires 2 inputs.");
  }
  if (grad_inputs.empty()) {
    return errors::InvalidArgument(
        "DepthwiseConv2dNative grad requires 1 grad input");
  }

  std::vector<int> dilations, strides, explicit_paddings;
  std::string data_format, padding;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "dilations", &dilations));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "strides", &strides));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "explicit_paddings", &explicit_paddings));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "data_format", &data_format));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "padding", &padding));

  DepthwiseConv2dNativeBackpropInput::Attrs input_attrs;
  input_attrs.explicit_paddings_ = explicit_paddings;
  input_attrs.data_format_ = data_format;
  input_attrs.dilations_ = dilations;
  grad_outputs->push_back(DepthwiseConv2dNativeBackpropInput(
      scope, Shape(scope, op.input(0)), op.input(1), grad_inputs[0], strides,
      padding, input_attrs));

  DepthwiseConv2dNativeBackpropFilter::Attrs filter_attrs;
  filter_attrs.explicit_paddings_ = explicit_paddings;
  filter_attrs.data_format_ = data_format;
  filter_attrs.dilations_ = dilations;
  grad_outputs->push_back(DepthwiseConv2dNativeBackpropFilter(
      scope, op.input(0), Shape(scope, op.input(1)), grad_inputs[0], strides,
      padding, filter_attrs));
  return scope.status();
}
REGISTER_GRADIENT_OP("DepthwiseConv2dNative", DepthwiseConv2dNativeGrad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
