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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_unaryDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_unaryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_unaryDTcc() {
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
#include <string.h>
#include <algorithm>
#include <cmath>
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

// Using the function reducer, reduce input along all axes in axes.
// Put the reduced data in output, which should already be appropriately sized.
// check_output_shape is set to what this code computes the final shape
// to be, so it can be cross checked with the shape computation logic.
void ReduceGeneric(bool keep_dims, const std::vector<int>& axes,
                   const Shape& input_shape, const std::vector<float>& input,
                   Shape* check_output_shape, std::vector<float>* output,
                   const std::function<float(float, float)>& reducer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_unaryDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_unary.cc", "ReduceGeneric");

  if (!IsNonEmpty(input_shape)) {
    // Zero-dimensions will break the NextIndices() logic, so just early out if
    // we have an empty shape.
    return;
  }

  // Set up output_shape to be the same length as input_shape, with
  // appropriate dimensions squashed to 1.  If keep_dims is false, we'll strip
  // out the one dimensions at the end, but it's convenient to leave them for
  // now.  We recompute the shape because we need the output shape to have
  // 1-dims in all the squashed dimensions; the shape from shape computation may
  // remove those squashed dimensions, depending on the options used.
  Shape output_shape = input_shape;

  // Reduction mask will be elementwise multiplied against the input
  // indices to figure out the output index for the element.
  std::vector<int> reduction_mask(input_shape.dimensions_count(), 1);
  for (const auto& axis : axes) {
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_shape.dimensions_count());
    reduction_mask[axis] = 0;
    output_shape.mutable_dims()->at(axis) = 1;
  }

  std::vector<int> output_indices(input_shape.dimensions_count());
  for (size_t input_offset = 0; input_offset < input.size(); ++input_offset) {
    std::vector<int> input_indices = ReverseOffset(input_shape, input_offset);
    // Calculate the output location by squashing input indices to 0
    // in reduced axes.
    for (int i = 0; i < input_shape.dimensions_count(); ++i) {
      output_indices[i] = input_indices[i] * reduction_mask[i];
    }
    int output_offset = Offset(output_shape, output_indices);
    if (input_indices == output_indices) {
      // Base element for the reduced axes
      output->at(output_offset) = input.at(input_offset);
    } else {
      // Reduce with existing element.
      output->at(output_offset) =
          reducer(output->at(output_offset), input.at(input_offset));
    }
  }

  if (!keep_dims) {
    // Strip out the dims from output_shape.
    std::vector<int> new_dims;
    for (int i = 0; i < output_shape.dimensions_count(); ++i) {
      if (reduction_mask[i]) {
        new_dims.push_back(output_shape.dims(i));
      }
    }
    output_shape.mutable_dims()->swap(new_dims);
  }
  *check_output_shape = output_shape;
}

}  // namespace

bool CopyMinMaxFromFirstInput(const Operator& op, Model* model) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_unaryDTcc mht_1(mht_1_v, 270, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_unary.cc", "CopyMinMaxFromFirstInput");

  auto& output_array = model->GetArray(op.outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op.inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax.min;
  output_minmax.max = input_minmax.max;
  return true;
}

::tensorflow::Status ResolveConstantUnaryOperator::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_unaryDTcc mht_2(mht_2_v, 292, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_unary.cc", "ResolveConstantUnaryOperator::Run");

  *modified = false;
  const auto unary_it = model->operators.begin() + op_index;
  const auto* unary_op = unary_it->get();
  // Test for unary ops of types that we know how to resolve.
  switch (unary_op->type) {
    case OperatorType::kCast:
    case OperatorType::kExp:
    case OperatorType::kLog:
    case OperatorType::kNeg:
    case OperatorType::kRsqrt:
    case OperatorType::kSqrt:
    case OperatorType::kSquare:
    case OperatorType::kSum:
    case OperatorType::kReduceMin:  //  Reduction Min
    case OperatorType::kReduceMax:  //  Reduction Max
    case OperatorType::kReshape:
    case OperatorType::kRelu6:
    case OperatorType::kRelu1:
    case OperatorType::kRelu:
      break;
    default:
      return ::tensorflow::Status::OK();
  }

  // Check if the input is a constant parameter.
  if (!IsConstantParameterArray(*model, unary_op->inputs[0])) {
    return ::tensorflow::Status::OK();
  }

  // if the unary op involves a tensor required by a rnn state, ignore it
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (unary_op->inputs[0] == rnn_state.back_edge_source_array()) {
      return ::tensorflow::Status::OK();
    }
    if (unary_op->inputs[0] == rnn_state.state_array()) {
      return ::tensorflow::Status::OK();
    }
  }

  auto& output_array = model->GetArray(unary_op->outputs[0]);
  if (!output_array.has_shape()) {
    // Yield until the output array dims have been resolved.
    return ::tensorflow::Status::OK();
  }

  // At the moment we don't want to care about fused activation functions.
  // The idea is that we should do the present constants-propagation before
  // activation functions get fused.
  if (unary_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF(
        "Not resolving constant %s "
        " because it has a fused activation function",
        LogName(*unary_op));
    return ::tensorflow::Status::OK();
  }

  // The min-max is only copied for ops that copy data without arithmetic.
  // In future trivial transpose, etc, can be handled here.
  if (unary_op->type == OperatorType::kReshape) {
    CopyMinMaxFromFirstInput(*unary_op, model);
  }

  const auto& input_array = model->GetArray(unary_op->inputs[0]);
  // We have already tested above for existence of buffers (synonymous to being
  // a constant param).
  CHECK(input_array.buffer);
  std::vector<DataType<ArrayDataType::kFloat>> const* input_float_data =
      nullptr;
  if (unary_op->type == OperatorType::kCast) {
    CastOperator const* cast_op = static_cast<CastOperator const*>(unary_op);
    if (cast_op->dst_data_type != ArrayDataType::kFloat) {
      AddMessageF(
          "Not resolving constant %s because we currently only support casting "
          "to float",
          LogName(*unary_op));
      return ::tensorflow::Status::OK();
    }
    if (cast_op->src_data_type != input_array.buffer->type) {
      AddMessageF(
          "Not resolving constant %s because cast op source type does not "
          "match input type",
          LogName(*unary_op));
    }
  } else {
    if (input_array.buffer->type != ArrayDataType::kFloat) {
      return ::tensorflow::Status::OK();
    }
    input_float_data = &(input_array.GetBuffer<ArrayDataType::kFloat>().data);
  }

  // Create a float buffer on the output array, which are always constant.
  const Shape& output_shape = output_array.shape();
  const int output_dims_count = output_shape.dimensions_count();
  const int output_buffer_size = RequiredBufferSizeForShape(output_shape);
  auto& output_float_data =
      output_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  output_float_data.resize(output_buffer_size);

  const Shape& input_shape = input_array.shape();
  const int input_buffer_size = RequiredBufferSizeForShape(input_shape);
  if (unary_op->type == OperatorType::kCast) {
    for (int i = 0; i < output_buffer_size; i++) {
      float outval = 0.0f;
      if (input_array.buffer->type == ArrayDataType::kFloat) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kFloat>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kUint8) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kUint8>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kInt32) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kInt32>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kInt64) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kInt64>().data[i]);
      } else if (input_array.buffer->type == ArrayDataType::kBool) {
        outval = static_cast<float>(
            input_array.GetBuffer<ArrayDataType::kBool>().data[i]);
      } else {
        LOG(FATAL) << "Unsupported cast op input type";
      }
      output_float_data[i] = outval;
    }
  } else if (unary_op->type == OperatorType::kReshape) {
    CHECK(input_buffer_size == output_buffer_size);
    output_float_data = *input_float_data;
  } else if (unary_op->type == OperatorType::kSum) {
    CHECK_EQ(unary_op->inputs.size(), 2) << "Sum needs 2 inputs";
    if (!IsConstantParameterArray(*model, unary_op->inputs[1])) {
      AddMessageF("Axis input is non-constant");
      return ::tensorflow::Status::OK();
    }
    auto& axis_array = model->GetArray(unary_op->inputs[1]);
    CHECK(axis_array.data_type == ArrayDataType::kInt32);

    // We only support keep_dims=true; shape prop will need to change otherwise.
    auto sum_op = static_cast<const TensorFlowSumOperator*>(unary_op);
    Shape check_output_shape;

    ReduceGeneric(
        sum_op->keep_dims, axis_array.GetBuffer<ArrayDataType::kInt32>().data,
        input_shape, *input_float_data, &check_output_shape, &output_float_data,
        [](float existing, float current) -> float {
          return existing + current;
        });
    CHECK(check_output_shape == output_shape)
        << "Shape propagation output shape doesn't match output shape from op";
  } else if (unary_op->type == OperatorType::kReduceMin) {
    // At the moment only full reduction across all dimensions is supported.
    // TODO(starka): Output should not be padded.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), 1);
    }
    float min = (*input_float_data)[0];
    for (int i = 0; i < input_buffer_size; i++) {
      min = std::min(min, (*input_float_data)[i]);
    }
    output_float_data[0] = min;
  } else if (unary_op->type == OperatorType::kReduceMax) {
    // At the moment only full reduction across all dimensions is supported.
    // TODO(starka): Output should not be padded.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), 1);
    }
    float max = (*input_float_data)[0];
    for (int i = 0; i < input_buffer_size; i++) {
      max = std::max(max, (*input_float_data)[i]);
    }
    output_float_data[0] = max;
  } else if (unary_op->type == OperatorType::kExp ||
             unary_op->type == OperatorType::kNeg ||
             unary_op->type == OperatorType::kLog ||
             unary_op->type == OperatorType::kRsqrt ||
             unary_op->type == OperatorType::kSqrt ||
             unary_op->type == OperatorType::kSquare) {
    // Element-wise ops. Should have perfectly matching sizes here.
    for (int i = 0; i < output_dims_count; i++) {
      CHECK_EQ(output_shape.dims(i), input_shape.dims(i));
    }

    for (int i = 0; i < output_buffer_size; i++) {
      const float val = (*input_float_data)[i];
      float outval = 0.f;
      if (unary_op->type == OperatorType::kExp) {
        outval = std::exp(val);
      } else if (unary_op->type == OperatorType::kNeg) {
        outval = -val;
      } else if (unary_op->type == OperatorType::kLog) {
        outval = std::log(val);
      } else if (unary_op->type == OperatorType::kRsqrt) {
        outval = 1.0f / std::sqrt(val);
      } else if (unary_op->type == OperatorType::kSqrt) {
        outval = std::sqrt(val);
      } else if (unary_op->type == OperatorType::kSquare) {
        outval = val * val;
      } else {
        LOG(FATAL) << "should not get here.";
      }
      output_float_data[i] = outval;
    }
  } else if (unary_op->type == OperatorType::kRelu6 ||
             unary_op->type == OperatorType::kRelu1 ||
             unary_op->type == OperatorType::kRelu) {
    for (int i = 0; i < output_buffer_size; ++i) {
      const float value = (*input_float_data)[i];
      float new_value = 0.0f;
      switch (unary_op->type) {
        case OperatorType::kRelu: {
          static constexpr float kLower = 0;
          new_value = value < kLower ? kLower : value;
          break;
        }
        case OperatorType::kRelu1: {
          static constexpr float kUpper = 1;
          static constexpr float kLower = -1;
          new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
          break;
        }
        case OperatorType::kRelu6: {
          static constexpr float kUpper = 6;
          static constexpr float kLower = 0;
          new_value = value > kUpper ? kUpper : value < kLower ? kLower : value;
          break;
        }
        default:
          LOG(FATAL) << "Unsupported activation function "
                     << LogName(*unary_op);
          return ::tensorflow::Status::OK();
      }
      output_float_data[i] = new_value;
    }
  } else {
    LOG(FATAL) << "should not get here.";
  }

  DeleteOpAndArrays(model, unary_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
