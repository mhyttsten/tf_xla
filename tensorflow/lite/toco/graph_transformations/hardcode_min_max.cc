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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc() {
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
#include <vector>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool HardcodeMinMaxForIm2colArray(Model* model, Operator* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForIm2colArray");

  if (op->outputs.size() != 2) {
    return false;
  }
  auto& im2col_array = model->GetArray(op->outputs[1]);
  if (im2col_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!im2col_array.minmax);
  auto& im2col_minmax = im2col_array.GetOrCreateMinMax();
  im2col_minmax.min = input_minmax.min;
  im2col_minmax.max = input_minmax.max;
  return true;
}

bool HardcodeMinMaxForL2Normalization(Model* model, Operator* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForL2Normalization");

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax.min >= 0. ? 0. : -1.;
  output_minmax.max = input_minmax.max <= 0. ? 0. : 1.;
  return true;
}

bool HardcodeInputMinMaxFromOutput(Model* model, Operator* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_2(mht_2_v, 241, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeInputMinMaxFromOutput");

  auto& input = model->GetArray(op->inputs[0]);
  if (input.minmax) {
    const auto* minmax = input.minmax.get();
    if (minmax) {
      return false;
    }
  }
  auto& output = model->GetArray(op->outputs[0]);
  if (output.minmax) {
    const auto* minmax = model->GetArray(op->outputs[0]).minmax.get();
    if (minmax) {
      input.GetOrCreateMinMax() = *minmax;
      return true;
    }
  }
  return false;
}

bool HardcodeMinMaxForConcatenation(Model* model, Operator* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_3(mht_3_v, 263, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForConcatenation");

  // Do not early return if the output already has min/max:
  // we may still need to adjust the inputs min/max.
  bool has_minmax = false;
  double overall_min = std::numeric_limits<double>::infinity();
  double overall_max = -std::numeric_limits<double>::infinity();
  for (const auto& input : op->inputs) {
    if (model->GetArray(input).minmax) {
      has_minmax = true;
      const auto* minmax = model->GetArray(input).minmax.get();
      if (minmax) {
        overall_min = std::min(overall_min, minmax->min);
        overall_max = std::max(overall_max, minmax->max);
      }
    }
  }
  auto& output = model->GetArray(op->outputs[0]);
  if (output.minmax) {
    has_minmax = true;
    const auto* minmax = model->GetArray(op->outputs[0]).minmax.get();
    if (minmax) {
      overall_min = std::min(overall_min, minmax->min);
      overall_max = std::max(overall_max, minmax->max);
    }
  }
  if (!has_minmax) {
    return false;
  }
  MinMax overall_minmax;
  overall_minmax.min = overall_min;
  overall_minmax.max = overall_max;
  bool changed = false;
  if (model->flags.change_concat_input_ranges()) {
    for (const auto& input : op->inputs) {
      auto& array = model->GetArray(input);
      if (!array.minmax) {
        changed = true;
      } else if (!(overall_minmax == array.GetMinMax())) {
        changed = true;
        LOG(WARNING)
            << "Tweaking the MinMax of array " << input << ", which is "
            << "an input to " << LogName(*op) << ", because we want all inputs "
            << "and outputs of a Concatenation operator to have the same "
            << "MinMax so that it can be implemented as a pure byte-copy, no "
               "arithmetic.";
      }
      array.GetOrCreateMinMax() = overall_minmax;
    }
  }
  if (!output.minmax) {
    changed = true;
  } else if (!(overall_minmax == output.GetMinMax())) {
    if (model->flags.change_concat_input_ranges()) {
      changed = true;
      LOG(WARNING)
          << "Tweaking the MinMax of the output array of " << LogName(*op)
          << ", because we want all inputs "
          << "and outputs of a Concatenation operator to have the same MinMax "
          << "so that it can be implemented as a pure byte-copy, no "
          << "arithmetic.";
    } else {
      return false;
    }
  }
  output.GetOrCreateMinMax() = overall_minmax;

  return changed;
}

bool HardcodeMinMaxForSplit(Model* model, Operator* op) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_4(mht_4_v, 335, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForSplit");

  // Data is in second input.
  auto& input_array = model->GetArray(op->inputs[1]);
  if (!input_array.minmax) {
    return false;
  }
  bool changed = false;
  for (const auto& output : op->outputs) {
    auto& array = model->GetArray(output);
    if (!array.minmax || !(array.GetMinMax() == input_array.GetMinMax())) {
      changed = true;
      array.GetOrCreateMinMax() = *input_array.minmax;
    }
  }
  return changed;
}

// The output of average or max pooling is within the same range as its input.
bool HardcodeMinMaxForAverageOrMaxPool(Model* model, Operator* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_5(mht_5_v, 356, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForAverageOrMaxPool");

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  const auto& input_minmax = input_array.GetMinMax();
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = std::min(input_minmax.min, 0.);
  output_minmax.max = std::max(input_minmax.max, 0.);
  return true;
}

bool HardcodeMinMaxFromFirstInput(Model* model, Operator* op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_6(mht_6_v, 376, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxFromFirstInput");

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
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

bool HardcodeMinMaxForSelect(Model* model, Operator* op) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_7(mht_7_v, 396, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForSelect");

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }

  auto& input_array_1 = model->GetArray(op->inputs[1]);
  auto& input_array_2 = model->GetArray(op->inputs[2]);

  if (!input_array_1.minmax && !input_array_2.minmax) {
    return false;
  }

  // Propagate up if one input is quantized and the other is constant.
  if (!input_array_1.minmax &&
      IsConstantParameterArray(*model, op->inputs[1])) {
    auto& minmax_1 = input_array_1.GetOrCreateMinMax();
    const auto& minmax_2 = input_array_2.GetMinMax();
    minmax_1.min = minmax_2.min;
    minmax_1.max = minmax_2.max;
  }

  if (!input_array_2.minmax &&
      IsConstantParameterArray(*model, op->inputs[2])) {
    auto& minmax_2 = input_array_2.GetOrCreateMinMax();
    const auto& minmax_1 = input_array_1.GetMinMax();
    minmax_2.min = minmax_1.min;
    minmax_2.max = minmax_1.max;
  }

  if (!input_array_1.minmax || !input_array_2.minmax) {
    return false;
  }

  const auto& input_minmax_1 = input_array_1.GetMinMax();
  const auto& input_minmax_2 = input_array_2.GetMinMax();

  CHECK_EQ(input_minmax_1.min, input_minmax_2.min);
  CHECK_EQ(input_minmax_1.max, input_minmax_2.max);
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = input_minmax_1.min;
  output_minmax.max = input_minmax_1.max;
  return true;
}

bool HardcodeMinMaxForOutput(Model* model, Operator* op, double min,
                             double max) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_8(mht_8_v, 446, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForOutput");

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.minmax) {
    return false;
  }
  CHECK(!output_array.minmax);
  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = min;
  output_minmax.max = max;
  return true;
}

bool MinMaxApproximatelyEqual(const MinMax& minmax1, const MinMax& minmax2) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_9(mht_9_v, 466, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "MinMaxApproximatelyEqual");

  const double magnitude =
      std::min(minmax1.max - minmax1.min, minmax2.max - minmax2.min);
  const double tolerated = 1e-6 * magnitude;
  return std::abs(minmax1.min - minmax2.min) <= tolerated &&
         std::abs(minmax1.max - minmax2.max) <= tolerated;
}

// Propagates MinMax from any of the listed arrays, to all others.
// If multiple of these arrays have MinMax, then these are required
// to agree with each other.
bool PropagateMinMaxAmongArrays(Model* model,
                                const std::vector<std::string>& array_names) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_10(mht_10_v, 481, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "PropagateMinMaxAmongArrays");

  std::string reference_array_name;
  MinMax* reference_minmax = nullptr;
  for (const std::string& array_name : array_names) {
    if (model->GetArray(array_name).minmax) {
      reference_array_name = array_name;
      reference_minmax = model->GetArray(array_name).minmax.get();
      break;
    }
  }
  // No MinMax info is available to propagate.
  if (!reference_minmax) {
    return false;
  }
  bool changed = false;
  for (const std::string& array_name : array_names) {
    auto& array = model->GetArray(array_name);
    if (array.minmax) {
      CHECK(MinMaxApproximatelyEqual(*array.minmax, *reference_minmax))
          << "Both the following arrays have minmax, and they disagree: "
          << reference_array_name << " (" << reference_minmax->min << ","
          << reference_minmax->max << ") and " << array_name << " ("
          << array.minmax->min << "," << array.minmax->max
          << "). Expected that either only one of them would have minmax, or "
             "at "
             "least that they would agree.";
    } else {
      array.GetOrCreateMinMax() = *reference_minmax;
      changed = true;
    }
  }
  return changed;
}

bool HardcodeMinMaxForReshape(Model* model, Operator* op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_11(mht_11_v, 518, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForReshape");

  Array& input = model->GetArray(op->inputs[0]);
  Array& output = model->GetArray(op->outputs[0]);

  // If input and output both exist or do not exist, do nothing.
  if ((!input.minmax && !output.minmax) || (input.minmax && output.minmax)) {
    return false;
  }

  // Otherwise propagate info amongst the input and output array.
  return PropagateMinMaxAmongArrays(model, {op->inputs[0], op->outputs[0]});
}

bool HardcodeMinMaxForLstmCell(Model* model, Operator* op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_12(mht_12_v, 534, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForLstmCell");

  CHECK_EQ(op->inputs.size(), LstmCellOperator::NUM_INPUTS);
  CHECK_EQ(op->outputs.size(), LstmCellOperator::NUM_OUTPUTS);

  bool changed = false;
  changed |= PropagateMinMaxAmongArrays(
      model, {op->inputs[LstmCellOperator::PREV_STATE_INPUT],
              op->outputs[LstmCellOperator::STATE_OUTPUT]});

  auto& input_activations =
      model->GetArray(op->inputs[LstmCellOperator::DATA_INPUT]);
  if (!input_activations.minmax) {
    auto& minmax = input_activations.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  auto& prev_output_activations =
      model->GetArray(op->inputs[LstmCellOperator::PREV_ACTIV_INPUT]);
  if (!prev_output_activations.minmax) {
    auto& minmax = prev_output_activations.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  auto& output_concat_temp =
      model->GetArray(op->outputs[LstmCellOperator::CONCAT_TEMP]);
  if (!output_concat_temp.minmax) {
    auto& minmax = output_concat_temp.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  auto& output_activations =
      model->GetArray(op->outputs[LstmCellOperator::ACTIV_OUTPUT]);
  if (!output_activations.minmax) {
    auto& minmax = output_activations.GetOrCreateMinMax();
    minmax.min = -1;
    minmax.max = 127. / 128.;
    changed = true;
  }

  // (This comment should morph into proper documentation for
  // quantization of LSTM models. It isn't just a local implementation detail,
  // the training code for LSTM models needs to be adjusted to that.)
  //
  // Finally, output_activations_temp holds the output of the fully-connected
  // node inside the LSTM cell. For it, we hardcode a minmax of [-8, 8].
  // The rationale for that is given in a lengthy comment on the LstmCell
  // quantized runtime implementation in reference_ops.h.
  auto& output_activations_temp =
      model->GetArray(op->outputs[LstmCellOperator::ACTIV_TEMP]);
  if (!output_activations_temp.minmax) {
    auto& minmax = output_activations_temp.GetOrCreateMinMax();
    minmax.min = -8;
    minmax.max = 8 * 32767. / 32768.;
    changed = true;
  }

  return changed;
}

bool HardcodeMinMaxForPack(Model* model, Operator* op) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_13(mht_13_v, 602, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMaxForPack");

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.minmax) {
    return false;
  }

  // If all tensors being packed have the same min/max range, hardcode min/max
  // for the output.
  const auto& first_input_array = model->GetArray(op->inputs[0]);
  if (!first_input_array.minmax) {
    return false;
  }
  const auto& first_input_minmax = first_input_array.GetMinMax();

  for (size_t i = 1; i < op->inputs.size(); i++) {
    const auto& input_array = model->GetArray(op->inputs[i]);
    if (!input_array.minmax) {
      return false;
    }
    if (first_input_minmax != input_array.GetMinMax()) {
      return false;
    }
  }

  auto& output_minmax = output_array.GetOrCreateMinMax();
  output_minmax.min = first_input_minmax.min;
  output_minmax.max = first_input_minmax.max;
  return true;
}

}  // namespace

::tensorflow::Status HardcodeMinMax::Run(Model* model, std::size_t op_index,
                                         bool* modified) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPShardcode_min_maxDTcc mht_14(mht_14_v, 638, "", "./tensorflow/lite/toco/graph_transformations/hardcode_min_max.cc", "HardcodeMinMax::Run");

  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  bool changed = false;
  switch (op->type) {
    case OperatorType::kConv:
      changed = HardcodeMinMaxForIm2colArray(model, op);
      break;

    case OperatorType::kL2Normalization:
      changed = HardcodeMinMaxForL2Normalization(model, op);
      break;

    case OperatorType::kRelu:
      // For any normalization other than batch norm, the quantizations ranges
      // before and after relu are expected to be known. Having a quantization
      // op before relu would reduce the number of bits of precision for the
      // activation in half. So we deduce the range before relu from that after
      // the relu. This would eliminate the need for two fake quantization nodes
      // and would not reduce the bits of precision available for activation.
      changed = HardcodeInputMinMaxFromOutput(model, op);
      break;

    case OperatorType::kConcatenation:
      changed = HardcodeMinMaxForConcatenation(model, op);
      break;

    case OperatorType::kSplit:
      changed = HardcodeMinMaxForSplit(model, op);
      break;

    case OperatorType::kAveragePool:
    case OperatorType::kMaxPool:
      changed = HardcodeMinMaxForAverageOrMaxPool(model, op);
      break;

    case OperatorType::kResizeBilinear:
    case OperatorType::kResizeNearestNeighbor:
    case OperatorType::kSlice:
    case OperatorType::kStridedSlice:
    case OperatorType::kSqueeze:
    case OperatorType::kExpandDims:
    case OperatorType::kPad:
    case OperatorType::kGather:
    case OperatorType::kTranspose:
    case OperatorType::kMean:
    case OperatorType::kReduceMax:
    case OperatorType::kReduceMin:
      changed = HardcodeMinMaxFromFirstInput(model, op);
      break;
    case OperatorType::kPack:
      changed = HardcodeMinMaxForPack(model, op);
      break;
    case OperatorType::kSum:
      // reduce_sum is expected to change the output range. Hence
      // a fake_quant op is necessary in the output to minimize error. However
      // in special circumstances like when computing expected value using
      // reduce_sum the input range and the output range matches. Hence the
      // below code would act as a fallback. If a fake_quant node is observed in
      // the output that takes precedence over the hard coding logic below.
      changed = HardcodeMinMaxFromFirstInput(model, op);
      if (changed) {
        LOG(WARNING) << "Using the input range for output in reduce_sum op."
                     << "This could have an impact on your model accuracy.";
      }
      break;
    case OperatorType::kSelect:
      changed = HardcodeMinMaxForSelect(model, op);
      break;
    case OperatorType::kLogistic:
      // We hardcode quantization_params to: zero_point=0, scale=1/256.
      // This choice of minmax is the one that is equivalent to that.
      changed = HardcodeMinMaxForOutput(model, op, 0, 255. / 256.);
      break;

    case OperatorType::kSoftmax:
      // We hardcode quantization_params to: zero_point=0, scale=1/256.
      // This choice of minmax is the one that is equivalent to that.
      changed = HardcodeMinMaxForOutput(model, op, 0, 255. / 256.);
      break;

    case OperatorType::kTanh:
      // We hardcode quantization_params to: zero_point=127, scale=1/128.
      // This choice of minmax is the one that is equivalent to that.
      changed = HardcodeMinMaxForOutput(model, op, -127. / 128., 1.0);
      break;

    case OperatorType::kLstmCell:
      changed = HardcodeMinMaxForLstmCell(model, op);
      break;

    case OperatorType::kReshape:
      changed = HardcodeMinMaxForReshape(model, op);
      break;

    default:
      break;
  }
  if (changed) {
    AddMessageF("Hardcoded min-max through %s", LogName(*op));
  }
  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
