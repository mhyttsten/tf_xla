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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_binary_into_following_affineDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_binary_into_following_affineDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_binary_into_following_affineDTcc() {
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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

void FuseAddOrSubParamsIntoFollowingAffine(Model* model, Operator* following_op,
                                           const Operator* add_or_sub_op,
                                           int index_of_constant_input) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_binary_into_following_affineDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/graph_transformations/fuse_binary_into_following_affine.cc", "FuseAddOrSubParamsIntoFollowingAffine");

  CHECK(add_or_sub_op->type == OperatorType::kAdd ||
        add_or_sub_op->type == OperatorType::kSub);
  CHECK(index_of_constant_input == 0 || index_of_constant_input == 1);
  // If the op is a subtraction, the constant input should be the right hand
  // side.
  // This should have been checked before this point.
  CHECK(add_or_sub_op->type != OperatorType::kSub ||
        index_of_constant_input == 1);
  if (following_op->inputs.size() < 3) {
    LOG(FATAL) << "Missing bias parameter";
  }
  const auto& weights = model->GetArray(following_op->inputs[1]);
  auto& bias = model->GetArray(following_op->inputs[2]);
  bias.minmax = nullptr;
  const auto& operand =
      model->GetArray(add_or_sub_op->inputs[index_of_constant_input]);
  // We're only supporting the case of a scalar operand. Should have
  // been checked earlier.
  CHECK_EQ(RequiredBufferSizeForShape(operand.shape()), 1);

  const float scalar_operand =
      operand.GetBuffer<ArrayDataType::kFloat>().data[0];
  // At this point we reduce the case of subtraction to that of addition
  // by negating the operand.
  float add_scalar_operand = 0.f;
  if (add_or_sub_op->type == OperatorType::kAdd) {
    add_scalar_operand = scalar_operand;
  } else if (add_or_sub_op->type == OperatorType::kSub &&
             index_of_constant_input == 1) {
    add_scalar_operand = -scalar_operand;
  } else {
    LOG(FATAL) << "Should not get here";
  }
  // From here on we are fusing an addition. add_or_sub_op->type does not
  // matter anymore.

  const Shape& weights_shape = weights.shape();
  const Shape& bias_shape = bias.shape();
  const auto& weights_buffer = weights.GetBuffer<ArrayDataType::kFloat>();
  const float* const weights_data = weights_buffer.data.data();
  auto& bias_buffer = bias.GetMutableBuffer<ArrayDataType::kFloat>();
  float* const bias_data = bias_buffer.data.data();

  if (following_op->type == OperatorType::kConv ||
      following_op->type == OperatorType::kFullyConnected) {
    const int output_depth = weights_shape.dims(0);
    // TODO(b/62904716): Bias array should become 1-D when padding removed.
    CHECK_EQ(output_depth, bias_shape.dims(bias_shape.dimensions_count() - 1));
    const int weights_size = RequiredBufferSizeForShape(weights_shape);
    const int weights_per_depth = weights_size / output_depth;
    CHECK_EQ(weights_size, weights_per_depth * output_depth);

    for (int d = 0; d < output_depth; d++) {
      float accumulation = 0;
      for (int i = 0; i < weights_per_depth; i++) {
        accumulation +=
            add_scalar_operand * weights_data[d * weights_per_depth + i];
      }
      bias_data[d] += accumulation;
    }
  } else if (following_op->type == OperatorType::kDepthwiseConv) {
    const int output_depth =
        weights_shape.dims(weights_shape.dimensions_count() - 1);
    const int weights_size = RequiredBufferSizeForShape(weights_shape);
    const int weights_per_depth = weights_size / output_depth;
    CHECK_EQ(weights_size, weights_per_depth * output_depth);

    for (int c = 0; c < output_depth; c++) {
      float accumulation = 0;
      for (int k = 0; k < weights_per_depth; k++) {
        accumulation += add_scalar_operand * weights_data[k * output_depth + c];
      }
      bias_data[c] += accumulation;
    }
  } else {
    LOG(FATAL) << "Should not get here.";
  }
}

void FuseMulOrDivParamsIntoFollowingAffine(Model* model, Operator* following_op,
                                           const Operator* mul_or_div_op,
                                           int index_of_constant_input) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_binary_into_following_affineDTcc mht_1(mht_1_v, 287, "", "./tensorflow/lite/toco/graph_transformations/fuse_binary_into_following_affine.cc", "FuseMulOrDivParamsIntoFollowingAffine");

  CHECK(mul_or_div_op->type == OperatorType::kMul ||
        mul_or_div_op->type == OperatorType::kDiv);
  CHECK(index_of_constant_input == 0 || index_of_constant_input == 1);
  // If the op is a division, the constant input should be the right hand side.
  // This should have been checked before this point.
  CHECK(mul_or_div_op->type != OperatorType::kDiv ||
        index_of_constant_input == 1);
  const auto& weights_name = following_op->inputs[1];
  const auto& bias_name = following_op->inputs[2];
  auto& weights = model->GetArray(weights_name);
  DropMinMax(model, weights_name);
  DropMinMax(model, bias_name);
  const auto& operand =
      model->GetArray(mul_or_div_op->inputs[index_of_constant_input]);
  // We're only supporting the case of a scalar operand. Should have
  // been checked earlier.
  CHECK_EQ(RequiredBufferSizeForShape(operand.shape()), 1);

  const float scalar_operand =
      operand.GetBuffer<ArrayDataType::kFloat>().data[0];

  float* weights_data =
      weights.GetMutableBuffer<ArrayDataType::kFloat>().data.data();
  const int weights_size = RequiredBufferSizeForShape(weights.shape());
  for (int i = 0; i < weights_size; i++) {
    if (mul_or_div_op->type == OperatorType::kMul) {
      weights_data[i] *= scalar_operand;
    } else if (mul_or_div_op->type == OperatorType::kDiv) {
      weights_data[i] /= scalar_operand;
    } else {
      LOG(FATAL) << "Should not get here";
    }
  }
}

}  // namespace

::tensorflow::Status FuseBinaryIntoFollowingAffine::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSfuse_binary_into_following_affineDTcc mht_2(mht_2_v, 330, "", "./tensorflow/lite/toco/graph_transformations/fuse_binary_into_following_affine.cc", "FuseBinaryIntoFollowingAffine::Run");

  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(binary_op->inputs.size(), 2);

  // We only can fuse an binary when the two operands break down as follows:
  //   1. One operand is the (variable) output of a typical affine (linear plus
  //   bias)
  //      op of a finite list of possible types: at the moment Conv,
  //      DepthwiseConv and
  //      FullyConnected are supported.
  //   2. The other operand is a constant param array.
  const bool is_input_constant[2] = {
      IsConstantParameterArray(*model, binary_op->inputs[0]),
      IsConstantParameterArray(*model, binary_op->inputs[1]),
  };
  if (!is_input_constant[0] && !is_input_constant[1]) {
    // Neither input is constant, so nothing we can fuse into a constant.
    return ::tensorflow::Status::OK();
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants
    // propagation, not for us to handle here.
    return ::tensorflow::Status::OK();
  }
  const int index_of_constant_input = is_input_constant[0] ? 0 : 1;
  const int index_of_variable_input = is_input_constant[0] ? 1 : 0;
  CHECK(is_input_constant[index_of_constant_input]);
  CHECK(!is_input_constant[index_of_variable_input]);

  // For division, we can only fuse if the denominator is constant.
  if (binary_op->type == OperatorType::kDiv) {
    if (index_of_constant_input != 1) {
      AddMessageF("Not fusing %s because the denominator is not constant",
                  LogName(*binary_op));
      return ::tensorflow::Status::OK();
    }
  }

  const auto& operand_shape =
      model->GetArray(binary_op->inputs[index_of_constant_input]).shape();
  for (const auto& dim : operand_shape.dims()) {
    if (dim > 1) {
      AddMessageF(
          "Not fusing %s into the following affine op, because we only know "
          "how to do so when the constant operand is a scalar",
          LogName(*binary_op));
      return ::tensorflow::Status::OK();
    }
  }

  if (binary_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF("Not fusing %s because it has a fused activation function",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  if (CountOpsWithInput(*model, binary_op->outputs[0]) != 1) {
    AddMessageF("Not fusing %s because it's consumed by multiple ops",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  Operator* following_op = GetOpWithInput(*model, binary_op->outputs[0]);

  if (!following_op) {
    AddMessageF("Not fusing %s because it is not consumed by any op",
                LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  if (following_op->type != OperatorType::kConv &&
      following_op->type != OperatorType::kFullyConnected &&
      following_op->type != OperatorType::kDepthwiseConv) {
    AddMessageF(
        "Not fusing %s because the following %s is not of one of the supported "
        "types",
        LogName(*binary_op), LogName(*following_op));
    return ::tensorflow::Status::OK();
  }

  if (following_op->inputs.size() < 3) {
    AddMessageF(
        "Not fusing %s because the following %s does not have a bias vector",
        LogName(*following_op), LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  const auto& weights = model->GetArray(following_op->inputs[1]);
  const auto& bias = model->GetArray(following_op->inputs[2]);
  if (!weights.buffer || !bias.buffer) {
    AddMessageF(
        "Not fusing %s because the following %s has non-constant weights or "
        "bias arrays",
        LogName(*binary_op), LogName(*following_op));
    return ::tensorflow::Status::OK();
  }

  // Try to fuse the binary params into the following op's params
  if (binary_op->type == OperatorType::kAdd ||
      binary_op->type == OperatorType::kSub) {
    if (following_op->type == OperatorType::kConv) {
      if (static_cast<ConvOperator*>(following_op)->padding.type !=
          PaddingType::kValid) {
        AddMessageF(
            "Not fusing %s because the following %s does not use VALID padding",
            LogName(*binary_op), LogName(*following_op));
        return ::tensorflow::Status::OK();
      }
    }
    if (following_op->type == OperatorType::kDepthwiseConv) {
      if (static_cast<DepthwiseConvOperator*>(following_op)->padding.type !=
          PaddingType::kValid) {
        AddMessageF(
            "Not fusing %s because the following %s does not use VALID padding",
            LogName(*binary_op), LogName(*following_op));
        return ::tensorflow::Status::OK();
      }
    }
    FuseAddOrSubParamsIntoFollowingAffine(model, following_op, binary_op,
                                          index_of_constant_input);
  } else if (binary_op->type == OperatorType::kMul ||
             binary_op->type == OperatorType::kDiv) {
    FuseMulOrDivParamsIntoFollowingAffine(model, following_op, binary_op,
                                          index_of_constant_input);
  } else {
    LOG(FATAL) << "should not get here";
  }

  AddMessageF("Fusing %s into the following %s", LogName(*binary_op),
              LogName(*following_op));

  model->EraseArray(binary_op->outputs[0]);

  following_op->inputs[0] = binary_op->inputs[index_of_variable_input];
  DeleteOpAndArrays(model, binary_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
