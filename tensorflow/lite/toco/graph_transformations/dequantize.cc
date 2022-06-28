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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType A>
void DequantizeBuffer(Array* array) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/toco/graph_transformations/dequantize.cc", "DequantizeBuffer");

  const auto old_data = array->GetBuffer<A>().data;
  array->buffer = nullptr;
  array->data_type = ArrayDataType::kFloat;
  auto& new_data = array->GetMutableBuffer<ArrayDataType::kFloat>().data;
  new_data.resize(old_data.size());
  const auto& qparams = array->GetQuantizationParams();
  for (int i = 0, end = old_data.size(); i < end; i++) {
    new_data[i] = qparams.scale * (old_data[i] - qparams.zero_point);
  }
}

std::vector<std::unique_ptr<Operator>>::iterator FindFirstOpWithInput(
    Model* model, const std::string& array_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("array_name: \"" + array_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc mht_1(mht_1_v, 217, "", "./tensorflow/lite/toco/graph_transformations/dequantize.cc", "FindFirstOpWithInput");

  for (auto it = model->operators.begin(); it != model->operators.end(); ++it) {
    for (const auto& input : it->get()->inputs) {
      if (input == array_name) {
        return it;
      }
    }
  }
  return model->operators.end();
}

void ClearArrayQuantizationParams(const std::string& array_name, Model* model) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("array_name: \"" + array_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc mht_2(mht_2_v, 232, "", "./tensorflow/lite/toco/graph_transformations/dequantize.cc", "ClearArrayQuantizationParams");

  auto* array = &model->GetArray(array_name);
  CHECK(array->quantization_params);
  for (auto& input_array : *model->flags.mutable_input_arrays()) {
    if (input_array.name() == array_name) {
      auto& qparams = *array->quantization_params;
      const double new_std_value = 1. / qparams.scale;
      const double new_mean_value = qparams.zero_point;
      if (input_array.has_std_value()) {
        CHECK_LE(std::abs(new_std_value - input_array.std_value()), 0.001);
      } else {
        input_array.set_std_value(new_std_value);
      }
      if (input_array.has_mean_value()) {
        CHECK_LE(std::abs(new_mean_value - input_array.mean_value()), 0.001);
      } else {
        input_array.set_mean_value(new_mean_value);
      }
    }
  }
  array->quantization_params = nullptr;
}

bool DequantizeArray(const std::string& array_name,
                     GraphTransformation* transformation, Model* model) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("array_name: \"" + array_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc mht_3(mht_3_v, 260, "", "./tensorflow/lite/toco/graph_transformations/dequantize.cc", "DequantizeArray");

  auto* array = &model->GetArray(array_name);
  if (!array->quantization_params) {
    return false;
  }
  transformation->AddMessageF("Dequantizing array: %s", array_name);

  // Dequantize any buffer
  if (array->buffer) {
    if (array->data_type == ArrayDataType::kUint8) {
      DequantizeBuffer<ArrayDataType::kUint8>(array);
    } else if (array->data_type == ArrayDataType::kInt32) {
      DequantizeBuffer<ArrayDataType::kInt32>(array);
    } else {
      LOG(FATAL) << "Unhandled data type";
    }
    CHECK(array->data_type == ArrayDataType::kFloat);
    CHECK(array->buffer->type == ArrayDataType::kFloat);

    // Clear quantization params, officially makes this a non-quantized array.
    ClearArrayQuantizationParams(array_name, model);
    return true;
  } else {
    array->data_type = ArrayDataType::kFloat;
  }

  // Clear quantization params, officially makes this a non-quantized array.
  ClearArrayQuantizationParams(array_name, model);

  if (array->buffer) {
    return true;
  }

  auto* op_outputting_array = GetOpWithOutput(*model, array_name);
  if (op_outputting_array) {
    if (op_outputting_array->type == OperatorType::kReshape) {
      return true;
    }
  }

  // If there was no minmax info, we can return now. Indeed,
  // the below only serves to create a FakeQuant node, but some arrays are
  // quantized without MinMax (see the CHECK above) and that corresponds to
  // places where a FakeQuant node is actually not wanted, because the
  // quantization params are meant to be inferred in another way (e.g. bias
  // vector for a Conv op, see their special-casing in quantize.cc).
  if (!array->minmax) {
    return true;
  }

  // Determine whether to insert a FakeQuant before or after
  // this array.
  bool must_insert_fakequant_before = false;
  bool must_insert_fakequant_after = false;
  if (IsInputArray(*model, array_name)) {
    must_insert_fakequant_after = true;
  }
  for (const std::string& output_array : model->flags.output_arrays()) {
    if (array_name == output_array) {
      must_insert_fakequant_before = true;
    }
  }
  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (array_name == rnn_state.state_array()) {
      must_insert_fakequant_after = true;
    }
    if (array_name == rnn_state.back_edge_source_array()) {
      must_insert_fakequant_before = true;
    }
  }
  CHECK(!(must_insert_fakequant_before && must_insert_fakequant_after));

  // Create and insert the FakeQuant node
  auto* fakequant_op = new FakeQuantOperator;
  model->operators.emplace(FindFirstOpWithInput(model, array_name),
                           fakequant_op);
  const std::string& new_array_name = AvailableArrayName(*model, array_name);
  auto& new_array = model->GetOrCreateArray(new_array_name);
  new_array.data_type = ArrayDataType::kFloat;
  new_array.copy_shape(array->shape());
  new_array.GetOrCreateMinMax() = array->GetMinMax();
  fakequant_op->minmax.reset(new MinMax);
  *fakequant_op->minmax = array->GetMinMax();
  fakequant_op->narrow_range = array->narrow_range;
  if (must_insert_fakequant_before) {
    for (const auto& op : model->operators) {
      for (std::string& output : op->outputs) {
        if (output == array_name) {
          output = new_array_name;
        }
      }
    }
    fakequant_op->inputs = {new_array_name};
    fakequant_op->outputs = {array_name};
  } else {
    for (const auto& op : model->operators) {
      for (std::string& input : op->inputs) {
        if (input == array_name) {
          input = new_array_name;
        }
      }
    }
    fakequant_op->inputs = {array_name};
    fakequant_op->outputs = {new_array_name};
  }
  return true;
}

}  // namespace

::tensorflow::Status Dequantize::Run(Model* model, std::size_t op_index,
                                     bool* modified) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSdequantizeDTcc mht_4(mht_4_v, 374, "", "./tensorflow/lite/toco/graph_transformations/dequantize.cc", "Dequantize::Run");

  *modified = false;
  const auto op_it = model->operators.begin() + op_index;
  auto* op = op_it->get();

  if (op->type == OperatorType::kDequantize) {
    auto& input_array = model->GetArray(op->inputs[0]);
    if (input_array.data_type == ArrayDataType::kFloat) {
      return ::tensorflow::Status::OK();
    }
    if (input_array.final_data_type != ArrayDataType::kFloat) {
      return ::tensorflow::Status::OK();
    }
    input_array.data_type = ArrayDataType::kFloat;
    input_array.quantization_params = nullptr;
    auto& output_array = model->GetArray(op->outputs[0]);
    output_array.data_type = ArrayDataType::kFloat;
    output_array.quantization_params = nullptr;
    *modified = RemoveTrivialPassthroughOp(this, model, op_index);
    return ::tensorflow::Status::OK();
  }

  std::vector<std::string> arrays;
  for (const std::string& input : op->inputs) {
    arrays.push_back(input);
  }
  for (const std::string& output : op->outputs) {
    arrays.push_back(output);
  }
  bool changed = false;
  for (const std::string& array : arrays) {
    if (!model->IsOptionalArray(array)) {
      changed |= DequantizeArray(array, this, model);
    }
  }

  *modified = changed;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
