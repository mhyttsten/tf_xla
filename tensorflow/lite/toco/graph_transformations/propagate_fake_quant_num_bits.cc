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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

bool ChangeArrayDataType(GraphTransformation* transformation, Array* array,
                         ArrayDataType new_data_type,
                         const MinMax* new_minmax) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_0(mht_0_v, 201, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "ChangeArrayDataType");

  // Ensure the array ends up in the new type (if it hasn't yet been quantized).
  bool data_type_changed = array->final_data_type != new_data_type;
  array->final_data_type = new_data_type;

  if (array->minmax && array->quantization_params && data_type_changed) {
    // The array is already quantized and has min/max info.
    // As we are changing the data type we need to fix up the existing min/max
    // to the new data type range.

    double old_quantized_min, old_quantized_max;
    CHECK(GetQuantizedDataTypeNumericalRange(
        array->data_type, &old_quantized_min, &old_quantized_max))
        << "Existing data type is not quantized: "
        << ArrayDataTypeName(array->data_type);
    double new_quantized_min, new_quantized_max;
    CHECK(GetQuantizedDataTypeNumericalRange(new_data_type, &new_quantized_min,
                                             &new_quantized_max))
        << "New data type is not quantized: "
        << ArrayDataTypeName(new_data_type);

    // Compute new minmax values.
    double min = (old_quantized_min - array->quantization_params->zero_point) *
                 array->quantization_params->scale;
    double max =
        (old_quantized_max + 1 - array->quantization_params->zero_point) *
        array->quantization_params->scale;
    max = max - 1.0 / (new_quantized_max + 1);

    auto& array_minmax = array->GetOrCreateMinMax();
    transformation->AddMessageF(
        "Rescaling min/max from %g,%g (%s) to %g,%g (%s)", array_minmax.min,
        array_minmax.max, ArrayDataTypeName(array->data_type), min, max,
        ArrayDataTypeName(new_data_type));
    array_minmax.min = min;
    array_minmax.max = max;
    ChooseQuantizationParamsForArrayAndQuantizedDataType(
        *array, new_data_type, array->quantization_params.get());
    // Directly change the type as the array was already quantized.
    array->data_type = new_data_type;
    return true;
  }

  // Array has not yet been quantized so we can just set the final data type
  // and assign the new min/max value (if provided).
  if (!array->quantization_params && !array->minmax && new_minmax) {
    transformation->AddMessageF("Forcing new minmax to %g,%g (%s)",
                                new_minmax->min, new_minmax->max,
                                ArrayDataTypeName(new_data_type));
    auto& array_minmax = array->GetOrCreateMinMax();
    array_minmax.min = new_minmax->min;
    array_minmax.max = new_minmax->max;
    return true;
  }

  return data_type_changed;
}

// Returns true if the op blocks our backward recursive data type propagation.
bool DoesOpBlockBackwardPropagation(const Operator& op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_1(mht_1_v, 263, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "DoesOpBlockBackwardPropagation");

  switch (op.type) {
    case OperatorType::kConcatenation:
    case OperatorType::kConcat:
    case OperatorType::kConcatV2:
      // Concat shouldn't block propagation, but we do expect that all inputs
      // have the same range.
      return false;
    case OperatorType::kDequantize:
      // Dequantize ops are inserted between the value we care about and the
      // FakeQuant so make sure we move across them.
    case OperatorType::kGather:
      // Gathers need their parameters changed to the appropriate data type.
    case OperatorType::kReshape:
    case OperatorType::kTranspose:
    case OperatorType::kSelect:
    case OperatorType::kTile:
      // Reshapes and transposes don't change values.
    case OperatorType::kRelu:
    case OperatorType::kRelu1:
    case OperatorType::kRelu6:
      // Relus only clamp the output. If min/max of parent is unknown, just
      // prop the range backward. This only happens for cases where activations
      // are not fused to avoid a default being set on the RELU input and
      // propagating forward to the RELU output.
      return false;
    default:
      return true;
  }
}

// Returns true if the input of an op blocks our backward recursive data type
// propagation.
bool DoesOpInputBlockBackwardPropagation(const Operator& op, int input_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_2(mht_2_v, 299, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "DoesOpInputBlockBackwardPropagation");

  switch (op.type) {
    case OperatorType::kSelect:
      return input_index == 0;
    case OperatorType::kGather:
      // Ignore gather indices.
      return input_index != 0;
      break;
    case OperatorType::kReshape:
    case OperatorType::kTranspose:
      // Ignore reshape/transpose shapes/dimensions.
      return input_index != 0;
    case OperatorType::kTile:
      // Ignore tile multiples.
      return input_index != 0;
    default:
      return false;
  }
}

// Propagates the data type up into the input arrays if they are model inputs
// that may need their type changed. May act recursively if the inputs are
// produced by ops that we can move over (such as Dequantize).
bool RecursivelyBackwardPropagateDataType(GraphTransformation* transformation,
                                          Model* model, Operator* op,
                                          ArrayDataType new_data_type,
                                          const MinMax& new_minmax) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_3(mht_3_v, 328, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "RecursivelyBackwardPropagateDataType");

  bool did_change = false;
  for (size_t input_index = 0; input_index < op->inputs.size(); ++input_index) {
    const auto& input = op->inputs[input_index];
    auto& input_array = model->GetArray(input);

    // Prevent moving into constant param args that we don't want to modify.
    if (DoesOpInputBlockBackwardPropagation(*op, input_index)) {
      continue;
    }

    bool array_did_change = ChangeArrayDataType(transformation, &input_array,
                                                new_data_type, &new_minmax);
    if (array_did_change) {
      transformation->AddMessageF(
          "Adjusting input final data type of array %s from %s to %s", input,
          ArrayDataTypeName(input_array.final_data_type),
          ArrayDataTypeName(new_data_type));
    }
    did_change |= array_did_change;

    // Walk up into all ops producing the inputs to this op.
    for (auto& producing_op : model->operators) {
      if (!DoesOpBlockBackwardPropagation(*producing_op)) {
        for (const auto& output : producing_op->outputs) {
          if (input == output) {
            did_change |= RecursivelyBackwardPropagateDataType(
                transformation, model, producing_op.get(), new_data_type,
                new_minmax);
          }
        }
      }
    }
  }
  return did_change;
}

// Returns true if the op blocks our forward recursive data type propagation.
bool DoesOpBlockForwardPropagation(const Operator& op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_4(mht_4_v, 369, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "DoesOpBlockForwardPropagation");

  switch (op.type) {
    case OperatorType::kFakeQuant:
      // Always stop at another FakeQuant, as it will likely have different
      // parameters.
      return true;
    default:
      return false;
  }
}

// Recurses down the graph setting the data type of all arrays until an operator
// that blocks propagation (like another FakeQuant) or a final_data_type is
// already specified.
bool RecursivelyForwardPropagateDataType(GraphTransformation* transformation,
                                         Model* model, Operator* op,
                                         ArrayDataType new_data_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_5(mht_5_v, 388, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "RecursivelyForwardPropagateDataType");

  bool did_change = false;
  for (const auto& output : op->outputs) {
    auto& output_array = model->GetArray(output);
    if (output_array.final_data_type == new_data_type) {
      // Final data type is already - skip.
      continue;
    }

    if (output_array.final_data_type == ArrayDataType::kNone ||
        output_array.final_data_type != new_data_type) {
      transformation->AddMessageF(
          "Adjusting output final data type of array %s from %s to %s", output,
          ArrayDataTypeName(output_array.final_data_type),
          ArrayDataTypeName(new_data_type));
      did_change |= ChangeArrayDataType(transformation, &output_array,
                                        new_data_type, nullptr);

      // Walk down into all ops consuming the output of this op.
      for (auto& consuming_op : model->operators) {
        if (!DoesOpBlockForwardPropagation(*consuming_op)) {
          for (const auto& input : consuming_op->inputs) {
            if (input == output) {
              did_change |= RecursivelyForwardPropagateDataType(
                  transformation, model, consuming_op.get(), new_data_type);
            }
          }
        }
      }
    }
  }
  return did_change;
}

}  // namespace

// Propagates the num_bits on a FakeQuant operator into the final data types
// of inputs and outputs. For example, if FakeQuant.num_bits==16 then we know
// the output must be int16 and assume all inputs up until the preceding op are
// also 16.
//
// This can be thought of as a bidirectional flood-fill of the num_bits implied
// final_data_type that terminates at other FakeQuant ops (and a few others as
// determined by DoesOpBlockBackwardPropagation/DoesOpBlockForwardPropagation).
// Once all FakeQuant ops have been visited the arrays should all have
// appropriate final_data_types if the source graph was annotated with the
// proper FakeQuant ops.
//
// Annotating a graph requires following a few hard rules:
// - every input MUST have a FakeQuant immediately following it
// - every output MUST have a FakeQuant immediately preceding it
// - important arithmetic ops (such as FullyConnected) SHOULD have a FakeQuant
//   immediately following it
// - all trained weights (RHS of FullyConnected ops, params on Gather ops, etc)
//   MUST have FakeQuants between them and the consuming op
// Additional FakeQuants may be used if desired, especially in areas that may
// suffer from large precision changes - such as between a Softmax and a
// FullyConnected. Only by validating accuracy differences between float
// inference with the FakeQuant ops simulating quantization and the actually
// quantized graph can you be sure the appropriate FakeQuant ops are present.
//
// You can tell if you're missing some FakeQuants by looking for warnings from
// quantize.cc about minmax ranges being determined by the contents of constant
// arrays. This will almost never produce functional models during inference.
//
// As this op may change the data types and ranges of input and output arrays
// downstream tools must also be sure to parse the output model flags to get the
// post-Transform values that may have changed due to this transformation.
//
// This isn't a GraphTransformation in the traditional respect as it affects ops
// outside of the one under transformation. This is primarily so that we can
// utilize the graph traversal and repeated pass system underlying the
// transformation system to exhaustively find all FakeQuant ops. It also gets us
// nice logging and integration with the graphviz video dumping mode.
// In general you should not copy this style of transformation and stick to
// local-only changes as seen in the other transformations.
::tensorflow::Status PropagateFakeQuantNumBits::Run(Model* model,
                                                    std::size_t op_index,
                                                    bool* modified) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_fake_quant_num_bitsDTcc mht_6(mht_6_v, 469, "", "./tensorflow/lite/toco/graph_transformations/propagate_fake_quant_num_bits.cc", "PropagateFakeQuantNumBits::Run");

  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if (op->type != OperatorType::kFakeQuant) {
    return ::tensorflow::Status::OK();
  }
  auto* fakequant_op = static_cast<FakeQuantOperator*>(op);

  ArrayDataType quantized_data_type = ArrayDataType::kNone;
  if (!InferQuantizedDataTypeFromFakeQuant(*fakequant_op,
                                           &quantized_data_type)) {
    AddMessageF("FakeQuant op %s num_bits=%d is out of range, ignoring",
                LogName(*op), fakequant_op->num_bits);
    return ::tensorflow::Status::OK();
  }
  const auto& final_minmax = *fakequant_op->minmax;

  AddMessageF(
      "Beginning propagation of fake quant %s num_bits=%d min=%g max=%g to %s",
      LogName(*op), fakequant_op->num_bits, final_minmax.min, final_minmax.max,
      ArrayDataTypeName(quantized_data_type));

  bool did_change = false;

  // Propagate the FakeQuant information backward up the graph.
  // This will possibly adjust input arrays or constant types (like Gather).
  did_change |= RecursivelyBackwardPropagateDataType(
      this, model, op, quantized_data_type, final_minmax);

  // Propagate the FakeQuant information forward down the graph.
  // This will possibly adjust output arrays.
  did_change |=
      RecursivelyForwardPropagateDataType(this, model, op, quantized_data_type);

  *modified = did_change;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
