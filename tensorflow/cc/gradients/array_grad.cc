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
class MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc {
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
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc() {
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

#include <vector>

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace ops {
namespace {

REGISTER_NO_GRADIENT_OP("Const");
REGISTER_NO_GRADIENT_OP("StopGradient");
REGISTER_NO_GRADIENT_OP("ConcatOffset");
REGISTER_NO_GRADIENT_OP("EditDistance");
REGISTER_NO_GRADIENT_OP("ZerosLike");
REGISTER_NO_GRADIENT_OP("InvertPermutation");
REGISTER_NO_GRADIENT_OP("Shape");
REGISTER_NO_GRADIENT_OP("ShapeN");
REGISTER_NO_GRADIENT_OP("Rank");
REGISTER_NO_GRADIENT_OP("Size");
REGISTER_NO_GRADIENT_OP("BroadcastGradientArgs");
REGISTER_NO_GRADIENT_OP("OneHot");

Status PackGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_0(mht_0_v, 212, "", "./tensorflow/cc/gradients/array_grad.cc", "PackGrad");

  int N;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "N", &N));
  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));

  grad_outputs->reserve(N);
  auto grad_op = Unstack(scope, grad_inputs[0], N, Unstack::Axis(axis));
  for (const Output& o : grad_op.output) {
    grad_outputs->emplace_back(o);
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("Pack", PackGrad);

Status UnpackGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_1(mht_1_v, 232, "", "./tensorflow/cc/gradients/array_grad.cc", "UnpackGrad");

  int axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));
  grad_outputs->push_back(Stack(scope, grad_inputs, Stack::Axis(axis)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Unpack", UnpackGrad);

Status IdentityGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_2(mht_2_v, 245, "", "./tensorflow/cc/gradients/array_grad.cc", "IdentityGrad");

  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Identity", IdentityGrad);

Status RefIdentityGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_3(mht_3_v, 256, "", "./tensorflow/cc/gradients/array_grad.cc", "RefIdentityGrad");

  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("RefIdentity", RefIdentityGrad);

Status QuantizeAndDequantizeGrad(const Scope& scope, const Operation& op,
                                 const std::vector<Output>& grad_inputs,
                                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_4(mht_4_v, 267, "", "./tensorflow/cc/gradients/array_grad.cc", "QuantizeAndDequantizeGrad");

  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantize", QuantizeAndDequantizeGrad);

Status QuantizeAndDequantizeV4GradHelper(const Scope& scope,
                                         const Operation& op,
                                         const std::vector<Output>& grad_inputs,
                                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_5(mht_5_v, 279, "", "./tensorflow/cc/gradients/array_grad.cc", "QuantizeAndDequantizeV4GradHelper");

  Input input = Shape(scope, op.input(0));
  Input input_min = op.input(1);
  Input input_max = op.input(2);
  int64_t axis;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "axis", &axis));
  auto qdq_v4_grad = QuantizeAndDequantizeV4Grad(
      scope, grad_inputs[0], input, input_min, input_max,
      QuantizeAndDequantizeV4Grad::Axis(axis));
  grad_outputs->push_back(qdq_v4_grad.input_backprop);
  grad_outputs->push_back(qdq_v4_grad.input_min_backprop);
  grad_outputs->push_back(qdq_v4_grad.input_max_backprop);
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV4",
                     QuantizeAndDequantizeV4GradHelper);

Status QuantizeAndDequantizeV3Grad(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_6(mht_6_v, 301, "", "./tensorflow/cc/gradients/array_grad.cc", "QuantizeAndDequantizeV3Grad");

  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("QuantizeAndDequantizeV3", QuantizeAndDequantizeV3Grad);

Status SplitGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_7(mht_7_v, 315, "", "./tensorflow/cc/gradients/array_grad.cc", "SplitGrad");

  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(Concat(scope, grad_inputs, op.input(0)));
  return scope.status();
}
REGISTER_GRADIENT_OP("Split", SplitGrad);

Status SplitVGrad(const Scope& scope, const Operation& op,
                  const std::vector<Output>& grad_inputs,
                  std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_8(mht_8_v, 327, "", "./tensorflow/cc/gradients/array_grad.cc", "SplitVGrad");

  if (op.num_inputs() < 3) {
    return errors::InvalidArgument("SplitV requires 3 arguments");
  }
  grad_outputs->push_back(Concat(scope, grad_inputs, op.input(2)));
  for (int i = 0; i < op.num_inputs() - 1; ++i) {
    grad_outputs->push_back(NoGradient());
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("SplitV", SplitVGrad);

Status FillGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_9(mht_9_v, 344, "", "./tensorflow/cc/gradients/array_grad.cc", "FillGrad");

  // y = fill(fill_shape, x)
  // No gradient returned for the fill_shape argument.
  grad_outputs->push_back(NoGradient());
  // The gradient for x (which must be a scalar) is just the sum of
  // all the gradients from the shape it fills.
  // We use ReduceSum to implement this, which needs an argument providing
  // the indices of all the dimensions of the incoming gradient.
  // grad(x) = reduce_sum(grad(y), [0..rank(grad(y))])
  auto all_dims = Range(scope, Const(scope, 0), Rank(scope, grad_inputs[0]),
                        Const(scope, 1));
  grad_outputs->push_back(ReduceSum(scope, grad_inputs[0], all_dims));
  return scope.status();
}
REGISTER_GRADIENT_OP("Fill", FillGrad);

Status DiagGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_10(mht_10_v, 365, "", "./tensorflow/cc/gradients/array_grad.cc", "DiagGrad");

  grad_outputs->push_back(DiagPart(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("Diag", DiagGrad);

Status DiagPartGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_11(mht_11_v, 376, "", "./tensorflow/cc/gradients/array_grad.cc", "DiagPartGrad");

  grad_outputs->push_back(Diag(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("DiagPart", DiagPartGrad);

Status MatrixDiagGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_12(mht_12_v, 387, "", "./tensorflow/cc/gradients/array_grad.cc", "MatrixDiagGrad");

  grad_outputs->push_back(MatrixDiagPart(scope, grad_inputs[0]));
  return scope.status();
}
REGISTER_GRADIENT_OP("MatrixDiag", MatrixDiagGrad);

Status MatrixBandPartGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_13(mht_13_v, 398, "", "./tensorflow/cc/gradients/array_grad.cc", "MatrixBandPartGrad");

  auto num_lower = op.input(1);
  auto num_upper = op.input(2);
  grad_outputs->push_back(
      MatrixBandPart(scope, grad_inputs[0], num_lower, num_upper));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MatrixBandPart", MatrixBandPartGrad);

Status GatherNdGrad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_14(mht_14_v, 414, "", "./tensorflow/cc/gradients/array_grad.cc", "GatherNdGrad");

  auto ref = op.input(0);
  auto ref_shape = Shape(scope, ref);
  auto indices = op.input(1);
  grad_outputs->push_back(ScatterNd(scope, indices, grad_inputs[0], ref_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("GatherNd", GatherNdGrad);

Status CheckNumericsGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_15(mht_15_v, 429, "", "./tensorflow/cc/gradients/array_grad.cc", "CheckNumericsGrad");

  string message;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "message", &message));
  string err_msg = strings::StrCat(
      "Not a number (NaN) or infinity (Inf) values detected in gradient. ",
      message);
  grad_outputs->push_back(CheckNumerics(scope, grad_inputs[0], err_msg));
  return scope.status();
}
REGISTER_GRADIENT_OP("CheckNumerics", CheckNumericsGrad);

Status ReshapeGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_16(mht_16_v, 445, "", "./tensorflow/cc/gradients/array_grad.cc", "ReshapeGrad");

  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Reshape", ReshapeGrad);

Status ExpandDimsGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_17(mht_17_v, 458, "", "./tensorflow/cc/gradients/array_grad.cc", "ExpandDimsGrad");

  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ExpandDims", ExpandDimsGrad);

Status SqueezeGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_18(mht_18_v, 471, "", "./tensorflow/cc/gradients/array_grad.cc", "SqueezeGrad");

  auto input_shape = Shape(scope, op.input(0));
  grad_outputs->push_back(Reshape(scope, grad_inputs[0], input_shape));
  return scope.status();
}
REGISTER_GRADIENT_OP("Squeeze", SqueezeGrad);

Status TransposeGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_19(mht_19_v, 483, "", "./tensorflow/cc/gradients/array_grad.cc", "TransposeGrad");

  auto inverted_perm = InvertPermutation(scope, op.input(1));
  grad_outputs->push_back(Transpose(scope, grad_inputs[0], inverted_perm));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Transpose", TransposeGrad);

Status ReverseSequenceGrad(const Scope& scope, const Operation& op,
                           const std::vector<Output>& grad_inputs,
                           std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_20(mht_20_v, 496, "", "./tensorflow/cc/gradients/array_grad.cc", "ReverseSequenceGrad");

  auto seq_lengths = op.input(1);
  int batch_dim;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "batch_dim", &batch_dim));
  int seq_dim;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "seq_dim", &seq_dim));
  grad_outputs->push_back(
      ReverseSequence(scope, grad_inputs[0], seq_lengths, seq_dim,
                      ReverseSequence::BatchDim(batch_dim)));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ReverseSequence", ReverseSequenceGrad);

Status ReverseGrad(const Scope& scope, const Operation& op,
                   const std::vector<Output>& grad_inputs,
                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_21(mht_21_v, 515, "", "./tensorflow/cc/gradients/array_grad.cc", "ReverseGrad");

  auto reverse_dims = op.input(1);
  grad_outputs->push_back(Reverse(scope, grad_inputs[0], reverse_dims));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ReverseV2", ReverseGrad);

Status ScatterNdGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_22(mht_22_v, 528, "", "./tensorflow/cc/gradients/array_grad.cc", "ScatterNdGrad");

  auto indices = op.input(0);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(GatherNd(scope, grad_inputs[0], indices));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("ScatterNd", ScatterNdGrad);

Status ScatterNdNonAliasingAddGrad(const Scope& scope, const Operation& op,
                                   const std::vector<Output>& grad_inputs,
                                   std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_23(mht_23_v, 542, "", "./tensorflow/cc/gradients/array_grad.cc", "ScatterNdNonAliasingAddGrad");

  auto indices = op.input(1);
  grad_outputs->push_back(Identity(scope, grad_inputs[0]));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(GatherNd(scope, grad_inputs[0], indices));
  return scope.status();
}
REGISTER_GRADIENT_OP("ScatterNdNonAliasingAdd", ScatterNdNonAliasingAddGrad);

template <bool IsPadV2>
Status PadGrad(const Scope& scope, const Operation& op,
               const std::vector<Output>& grad_inputs,
               std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_24(mht_24_v, 557, "", "./tensorflow/cc/gradients/array_grad.cc", "PadGrad");

  auto x = op.input(0);
  auto a = op.input(1);  // [Rank(x), 2]
  // Takes a slice of a. The 1st column. [Rank(x), 1].
  auto size = Stack(scope, {Rank(scope, x), 1});
  auto pad_before = Slice(scope, a, {0, 0}, size);
  // Make it a 1-D tensor.
  auto begin = Reshape(scope, pad_before, {-1});
  grad_outputs->push_back(Slice(scope, grad_inputs[0], begin, Shape(scope, x)));
  grad_outputs->push_back(NoGradient());
  // PadV2 adds a "constant_values" input.
  if (IsPadV2) {
    grad_outputs->push_back(NoGradient());
  }
  return scope.status();
}
REGISTER_GRADIENT_OP("Pad", PadGrad<false>);
REGISTER_GRADIENT_OP("PadV2", PadGrad<true>);

Status SpaceToBatchGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_25(mht_25_v, 581, "", "./tensorflow/cc/gradients/array_grad.cc", "SpaceToBatchGrad");

  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(
      BatchToSpace(scope, grad_inputs[0], op.input(1), block_size));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToBatch", SpaceToBatchGrad);

Status SpaceToBatchNDGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_26(mht_26_v, 597, "", "./tensorflow/cc/gradients/array_grad.cc", "SpaceToBatchNDGrad");

  grad_outputs->push_back(
      BatchToSpaceND(scope, grad_inputs[0], op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToBatchND", SpaceToBatchNDGrad);

Status BatchToSpaceGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_27(mht_27_v, 611, "", "./tensorflow/cc/gradients/array_grad.cc", "BatchToSpaceGrad");

  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(
      SpaceToBatch(scope, grad_inputs[0], op.input(1), block_size));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("BatchToSpace", BatchToSpaceGrad);

Status BatchToSpaceNDGrad(const Scope& scope, const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_28(mht_28_v, 627, "", "./tensorflow/cc/gradients/array_grad.cc", "BatchToSpaceNDGrad");

  grad_outputs->push_back(
      SpaceToBatchND(scope, grad_inputs[0], op.input(1), op.input(2)));
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("BatchToSpaceND", BatchToSpaceNDGrad);

Status SpaceToDepthGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_29(mht_29_v, 641, "", "./tensorflow/cc/gradients/array_grad.cc", "SpaceToDepthGrad");

  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(DepthToSpace(scope, grad_inputs[0], block_size));
  return scope.status();
}
REGISTER_GRADIENT_OP("SpaceToDepth", SpaceToDepthGrad);

Status DepthToSpaceGrad(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_30(mht_30_v, 655, "", "./tensorflow/cc/gradients/array_grad.cc", "DepthToSpaceGrad");

  int block_size;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "block_size", &block_size));
  grad_outputs->push_back(SpaceToDepth(scope, grad_inputs[0], block_size));
  return scope.status();
}
REGISTER_GRADIENT_OP("DepthToSpace", DepthToSpaceGrad);

Status MirrorPadGrad(const Scope& scope, const Operation& op,
                     const std::vector<Output>& grad_inputs,
                     std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_31(mht_31_v, 669, "", "./tensorflow/cc/gradients/array_grad.cc", "MirrorPadGrad");

  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "mode", &mode));
  grad_outputs->push_back(tensorflow::ops::internal::MirrorPadGrad(
      scope, grad_inputs[0], op.input(1), mode));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MirrorPad", MirrorPadGrad);

// TODO(suharshs): b/34770860. This gradient was within 1e-3 but not 1e-4.
Status MirrorPadGradGrad(const Scope& scope, const Operation& op,
                         const std::vector<Output>& grad_inputs,
                         std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_32(mht_32_v, 685, "", "./tensorflow/cc/gradients/array_grad.cc", "MirrorPadGradGrad");

  string mode;
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "mode", &mode));
  grad_outputs->push_back(MirrorPad(scope, grad_inputs[0], op.input(1), mode));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("MirrorPadGrad", MirrorPadGradGrad);

Status StridedSliceGradHelper(const Scope& scope, const Operation& op,
                              const std::vector<Output>& grad_inputs,
                              std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_33(mht_33_v, 699, "", "./tensorflow/cc/gradients/array_grad.cc", "StridedSliceGradHelper");

  Input x = Shape(scope, op.input(0));
  Input begin = op.input(1);
  Input end = op.input(2);
  Input strides = op.input(3);
  int64_t begin_mask;
  int64_t end_mask;
  int64_t ellipsis_mask;
  int64_t new_axis_mask;
  int64_t shrink_axis_mask;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "begin_mask", &begin_mask));
  TF_RETURN_IF_ERROR(GetNodeAttr(op.node()->attrs(), "end_mask", &end_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "ellipsis_mask", &ellipsis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "new_axis_mask", &new_axis_mask));
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "shrink_axis_mask", &shrink_axis_mask));
  grad_outputs->push_back(
      StridedSliceGrad(scope, x, begin, end, strides, grad_inputs[0],
                       StridedSliceGrad::BeginMask(begin_mask)
                           .EndMask(end_mask)
                           .EllipsisMask(ellipsis_mask)
                           .NewAxisMask(new_axis_mask)
                           .ShrinkAxisMask(shrink_axis_mask)));
  // No gradients returned for begin, end and strides
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("StridedSlice", StridedSliceGradHelper);

Status SliceGrad(const Scope& scope, const Operation& op,
                 const std::vector<Output>& grad_inputs,
                 std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_34(mht_34_v, 738, "", "./tensorflow/cc/gradients/array_grad.cc", "SliceGrad");

  // Propagate the incoming gradient along all the selected values,
  // and zero everywhere else. Use the Pad operator for this.
  //
  // First create an Nx2 padding where N is the number of input
  // dimensions. The first column is the number of prepended zeros
  // for each dimension, and the second column is the number of
  // appended zeros.
  //
  // The first column is just the begin vector.
  // The second column is the shape of the input element-wise
  // subtracted by begin+size

  // Running example:
  // input.shape = [3, 5, 3]
  // begin = [1, 2, 1], size = [1, 3, 2]
  Input input = op.input(0);
  Input begin = op.input(1);
  // input_rank = 3
  auto input_rank = Rank(scope, input);
  // slice_size = [1, 3, 2]
  auto slice_size = Shape(scope, op.output(0));
  // padding_shape = [3, 1]
  auto padding_shape = Stack(scope, {input_rank, 1});
  // before_padding = [[1]
  //                   [2]
  //                   [1]]
  Input before_padding = Reshape(scope, begin, padding_shape);
  // after_padding_sizes = shape(input) - slice_size - begin
  //                     = [3, 5, 3] - [1, 3, 2] - [1, 2, 1]
  //                     = [1, 0, 0]
  auto after_padding_sizes =
      Sub(scope, Sub(scope, Shape(scope, input), slice_size), begin);
  // after_padding = [[1]
  //                  [0]
  //                  [0]]
  Input after_padding = Reshape(scope, after_padding_sizes, padding_shape);
  // paddings = [[1 1]
  //             [2 0]
  //             [1 0]]
  auto paddings =
      Concat(scope, {before_padding, after_padding}, Const(scope, 1));
  grad_outputs->push_back(Pad(scope, grad_inputs[0], paddings));
  // Nothing propagated for "begin" and "size" inputs
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Slice", SliceGrad);

Status ConcatGradHelper(const Scope& scope, const Operation& op,
                        const std::vector<Output>& grad_inputs,
                        std::vector<Output>* grad_outputs,
                        int start_value_index, int end_value_index,
                        int dim_index) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_35(mht_35_v, 795, "", "./tensorflow/cc/gradients/array_grad.cc", "ConcatGradHelper");

  if (end_value_index >= op.num_inputs()) {
    return errors::Internal("Invalid input index");
  }
  std::vector<Output> inputs;
  inputs.reserve(end_value_index - start_value_index);
  for (int i = start_value_index; i < end_value_index; ++i) {
    inputs.push_back(op.input(i));
  }

  auto shapes = ShapeN(scope, inputs);
  const auto unique_name = scope.GetUniqueNameForOp("ConcatOffset");
  auto builder =
      ::tensorflow::NodeBuilder(unique_name, "ConcatOffset")
          .Input(::tensorflow::ops::AsNodeOut(scope, op.input(dim_index)))
          .Input(::tensorflow::ops::AsNodeOutList(scope, shapes.output));
  scope.UpdateBuilder(&builder);
  ::tensorflow::Node* concat_offset_node;
  scope.UpdateStatus(builder.Finalize(scope.graph(), &concat_offset_node));
  scope.UpdateStatus(scope.DoShapeInference(concat_offset_node));
  if (concat_offset_node->num_outputs() != inputs.size()) {
    return errors::Internal("ConcatOffset has invalid output count");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Concat grad should have 1 input");
  }

  // For each dx[i], we take a slice of dy. The offset and size of the
  // slice is given by offset[i] and shape[i].
  const Output& dy = grad_inputs[0];
  for (int i = 0; i < inputs.size(); ++i) {
    grad_outputs->push_back(
        Slice(scope, dy, Output(concat_offset_node, i), shapes.output[i]));
  }

  // Insert a NoGradient for the axis.
  grad_outputs->insert(grad_outputs->begin() + dim_index, NoGradient());
  return scope.status();
}

Status ConcatV2Grad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_36(mht_36_v, 840, "", "./tensorflow/cc/gradients/array_grad.cc", "ConcatV2Grad");

  return ConcatGradHelper(scope, op, grad_inputs, grad_outputs,
                          /*start_value_index=*/0,
                          /*end_value_index=*/op.num_inputs() - 1,
                          /*dim+index=*/op.num_inputs() - 1);
}

REGISTER_GRADIENT_OP("ConcatV2", ConcatV2Grad);

Status BroadcastToGrad(const Scope& scope, const Operation& op,
                       const std::vector<Output>& grad_inputs,
                       std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_37(mht_37_v, 854, "", "./tensorflow/cc/gradients/array_grad.cc", "BroadcastToGrad");

  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("BroadcastTo grad should have 1 grad input");
  }
  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("BroadcastTo requires 2 inputs");
  }

  auto x_shape = Shape(scope, op.input(0));
  auto args = internal::BroadcastGradientArgs(scope, x_shape, op.input(1));
  auto sum_gx = Sum(scope, grad_inputs[0], args.r0);
  grad_outputs->push_back(Reshape(scope, sum_gx, x_shape));
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("BroadcastTo", BroadcastToGrad);

Status TileGrad(const Scope& scope, const Operation& op,
                const std::vector<Output>& grad_inputs,
                std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_38(mht_38_v, 877, "", "./tensorflow/cc/gradients/array_grad.cc", "TileGrad");

  if (op.num_inputs() != 2) {
    return errors::InvalidArgument("Tile requires 2 inputs");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Tile grad requires 1 grad input");
  }

  Shape::Attrs shape_attrs;
  shape_attrs.out_type_ = op.input_type(1);
  auto input_shape = Shape(scope, op.input(0), shape_attrs);
  // We interleave multiples and input_shape to get split_shape,
  // reshape grad to split_shape, and reduce along all even
  // dimensions (the tiled dimensions) to get the result
  // with shape input_shape.  For example
  //   input_shape = [20, 30, 40]
  //   multiples = [2, 3, 4]
  //   split_shape = [2, 20, 3, 30, 4, 40]
  //   axes = [0, 2, 4]
  auto stack = Stack(scope, {op.input(1), input_shape.output});
  auto perm = Range(scope, Sub(scope, Rank(scope, stack), 1), -1, -1);
  auto split_shape = Reshape(scope, Transpose(scope, stack, perm), {-1});
  auto axes = Range(scope, Const(scope, 0), Size(scope, split_shape.output), 2);
  auto input_grad = ReduceSum(
      scope, Reshape(scope, grad_inputs[0], split_shape.output), axes.output);
  grad_outputs->push_back(input_grad.output);
  grad_outputs->push_back(NoGradient());
  return scope.status();
}
REGISTER_GRADIENT_OP("Tile", TileGrad);

// Create a constant of the provided d_type;
Output ConstHelper(const Scope& scope, int value, DataType d_type) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_39(mht_39_v, 912, "", "./tensorflow/cc/gradients/array_grad.cc", "ConstHelper");

  return Cast(scope, Const(scope, value), d_type);
}

// Adds the batch offsets to the given indices and returns the results.
Output GetBatchIndices(const Scope& scope, const Output& params_shape,
                       const Output& indices, int batch_dims) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_40(mht_40_v, 921, "", "./tensorflow/cc/gradients/array_grad.cc", "GetBatchIndices");

  Output batch_indices = indices;
  auto indices_ndims = Rank(scope, indices);
  auto casted_params_shape = Cast(scope, params_shape, indices.type());
  Output accum_dim_value = ConstHelper(scope, 1, indices.type());
  for (int dim = batch_dims; dim > 0; dim--) {
    Output dim_value = Slice(scope, casted_params_shape, {dim - 1}, {1});
    accum_dim_value = Multiply(scope, accum_dim_value,
                               Slice(scope, casted_params_shape, {dim}, {1}));
    auto start = ConstHelper(scope, 0, indices.type());
    auto step = ConstHelper(scope, 1, indices.type());
    Output dim_indices = Range(scope, start, Squeeze(scope, dim_value), step);
    dim_indices = Multiply(scope, dim_indices, accum_dim_value);
    auto one = Cast(scope, Const(scope, {1}), indices.type());
    auto dim_shape = Concat(
        scope,
        {Output(Tile(scope, one, Const(scope, {dim - 1}))), dim_value,
         Output(Tile(scope, one,
                     ExpandDims(scope, Sub(scope, indices_ndims, dim), 0)))},
        /*axis=*/0);
    batch_indices =
        Add(scope, batch_indices, Reshape(scope, dim_indices, dim_shape));
  }

  return batch_indices;
}

Output BatchGatherGrad(const Scope& scope, Output params_shape, Output values,
                       Output indices, int batch_dims, Output gather_dim_size) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_41(mht_41_v, 952, "", "./tensorflow/cc/gradients/array_grad.cc", "BatchGatherGrad");

  // Axis is the first non-batch dimension.
  auto indices_size = ExpandDims(scope, Size(scope, indices), 0);
  Output outer_shape, flat_values_shape;
  if (batch_dims != 0) {
    auto values_shape = Shape(scope, values);
    // Add the batch offsets to indices and flatten the batch dimensions.
    outer_shape = Slice(scope, values_shape, {0}, {batch_dims});
    auto inner_shape =
        Slice(scope, Slice(scope, values_shape, {batch_dims}, {-1}), {1}, {-1});
    auto batch_size = Prod(scope, outer_shape, /*axis=*/0);
    flat_values_shape = Concat(scope, {{-1}, inner_shape}, /*axis=*/0);
    gather_dim_size = Multiply(scope, gather_dim_size, batch_size);
    indices = GetBatchIndices(scope, params_shape, indices, batch_dims);
    values = Reshape(scope, values, flat_values_shape);
  }

  indices = Reshape(scope, indices, indices_size);
  Output params_grad =
      UnsortedSegmentSum(scope, values, indices, gather_dim_size);

  if (batch_dims != 0) {
    // Put back the batch dimensions.
    params_grad = Reshape(scope, params_grad, params_shape);
  }
  return params_grad;
}

Status GatherV2Grad(const Scope& scope, const Operation& op,
                    const std::vector<Output>& grad_inputs,
                    std::vector<Output>* grad_outputs) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSccPSgradientsPSarray_gradDTcc mht_42(mht_42_v, 985, "", "./tensorflow/cc/gradients/array_grad.cc", "GatherV2Grad");

  if (op.num_inputs() != 3) {
    return errors::InvalidArgument("Gather requires 3 inputs");
  }
  if (grad_inputs.size() != 1) {
    return errors::InvalidArgument("Gather grad requires 1 grad input");
  }

  // params can be large, so colocate the shape calculation with it.
  // params can be very large for sparse model, array_ops.shape raises
  // exception on the Windows platform when any dimension is larger than
  // int32. params_shape is not used in optimizer apply_sparse gradients,
  // so it's fine to convert it back to int32 regardless of truncation.
  auto params = op.input(0);
  auto colocate_scope = scope.ColocateWith(params);
  Shape::Attrs shape_attrs;
  shape_attrs.out_type_ = DT_INT64;
  auto params_shape64 = Shape(colocate_scope, params, shape_attrs);
  Output params_shape = Cast(colocate_scope, params_shape64, DT_INT32);

  auto indices = op.input(1);
  auto indices_size = ExpandDims(scope, Size(scope, indices), 0);
  auto axis = op.input(2);
  auto axis_expand = ExpandDims(scope, axis, 0);

  int batch_dims;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(op.node()->attrs(), "batch_dims", &batch_dims));
  if (batch_dims < 0) {
    // TODO(bdodson): Figure out if we can find the param rank here, like the
    // python implementation does.
    return errors::InvalidArgument(
        "C++ GatherV2 gradient does not support negative batch_dims.");
  }

  // Handle axis by transposing the axis dimension to be the first non-batch
  // dimension, compute the gradient and transpose the result back.
  auto outer_shape = Slice(scope, params_shape, {0}, axis_expand);
  auto inner_shape =
      Slice(scope, Slice(scope, params_shape, axis_expand, {-1}), {1}, {-1});
  auto values_shape = Concat(scope, {outer_shape, {-1}, inner_shape}, 0);
  auto values_dims = Size(scope, values_shape);
  auto axis_dims = Size(scope, outer_shape);

  Output outer_batches_indices = Range(scope, 0, batch_dims, /*delta=*/1);
  Output batch_axis_indices = Range(scope, batch_dims, axis_dims, /*delta=*/1);
  Output inner_axes_indices =
      Range(scope, Add(scope, axis_dims, 1), values_dims, /*delta=*/1);
  Output axis_dims_expand = ExpandDims(scope, axis_dims, 0);

  auto values = Reshape(scope, grad_inputs[0], values_shape);

  // Move values[axis] up to values[batch_dims]
  Output transpose_dims = Concat(scope,
                                 {outer_batches_indices, axis_dims_expand,
                                  batch_axis_indices, inner_axes_indices},
                                 0);
  auto values_transpose = Transpose(scope, values, transpose_dims);
  Output gather_dim_size =
      Squeeze(scope, Slice(scope, params_shape, axis_expand, {1}));
  params_shape = Gather(scope, params_shape, transpose_dims);

  auto params_grad = BatchGatherGrad(scope, params_shape, values_transpose,
                                     indices, batch_dims, gather_dim_size);

  // Inverts the above transpose by moving dimension batch_dims back to its
  // original position.
  Output invert_transpose_dims = Concat(scope,
                                        {outer_batches_indices,
                                         Add(scope, batch_axis_indices, 1),
                                         {batch_dims},
                                         inner_axes_indices},
                                        0);

  params_grad = Transpose(scope, params_grad, invert_transpose_dims);

  grad_outputs->push_back(params_grad);
  grad_outputs->push_back(NoGradient());
  grad_outputs->push_back(NoGradient());
  return scope.status();
}

REGISTER_GRADIENT_OP("GatherV2", GatherV2Grad);

}  // anonymous namespace
}  // namespace ops
}  // namespace tensorflow
