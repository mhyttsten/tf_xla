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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_batch_normalizationDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_batch_normalizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_batch_normalizationDTcc() {
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
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ResolveBatchNormalization::Run(Model* model,
                                                    std::size_t op_index,
                                                    bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_batch_normalizationDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/resolve_batch_normalization.cc", "ResolveBatchNormalization::Run");

  *modified = false;
  auto bn_it = model->operators.begin() + op_index;
  if (bn_it->get()->type != OperatorType::kBatchNormalization) {
    return ::tensorflow::Status::OK();
  }
  const auto* bn_op =
      static_cast<const BatchNormalizationOperator*>(bn_it->get());

  auto& mean_array = model->GetArray(bn_op->inputs[1]);
  const auto& multiplier_array = model->GetArray(bn_op->inputs[2]);
  const auto& offset_array = model->GetArray(bn_op->inputs[3]);

  // This graph transformation needs to address constant buffers below, so
  // we need to exit early if these buffers don't exist yet (i.e. if the params
  // haven't yet been resolved as constants) and will process it once they have.
  if (!mean_array.buffer || !multiplier_array.buffer || !offset_array.buffer) {
    return ::tensorflow::Status::OK();
  }

  CHECK(IsConstantParameterArray(*model, bn_op->inputs[1]) &&
        IsConstantParameterArray(*model, bn_op->inputs[2]) &&
        IsConstantParameterArray(*model, bn_op->inputs[3]))
      << "Batch normalization resolution requires that mean, multiplier and "
         "offset arrays be constant.";

  // We should only have *float* BatchNormalizations... let's guard this
  // assumption by CHECK's.
  CHECK(mean_array.data_type == ArrayDataType::kFloat);
  CHECK(multiplier_array.data_type == ArrayDataType::kFloat);
  CHECK(offset_array.data_type == ArrayDataType::kFloat);

  // Create the new Mul, Add operators
  auto* mul_op = new MulOperator;
  auto* add_op = new AddOperator;
  const std::string mul_name =
      AvailableArrayName(*model, bn_op->outputs[0] + "_mul");
  const std::string add_name =
      AvailableArrayName(*model, bn_op->outputs[0] + "_add");
  const std::string mul_param_name =
      AvailableArrayName(*model, mul_name + "_param");
  const std::string add_param_name =
      AvailableArrayName(*model, add_name + "_param");
  mul_op->inputs = {bn_op->inputs[0], mul_param_name};
  mul_op->outputs = {mul_name};
  add_op->inputs = {mul_name, add_param_name};
  add_op->outputs = {bn_op->outputs[0]};
  AddMessageF("Splitting %s into %s and %s", LogName(*bn_op), LogName(*mul_op),
              LogName(*add_op));

  // Create the intermediate activation array (output of mul, input of add)
  auto& intermediate_array = model->GetOrCreateArray(mul_op->outputs[0]);
  intermediate_array.data_type = model->GetArray(bn_op->inputs[0]).data_type;

  // Insert the new operators in the graph
  auto add_it = model->operators.emplace(bn_it, add_op);
  auto mul_it = model->operators.emplace(add_it, mul_op);
  // update invalidated iterators.
  DCHECK_EQ(mul_it->get(), mul_op);
  add_it = mul_it + 1;
  DCHECK_EQ(add_it->get(), add_op);
  bn_it = add_it + 1;
  DCHECK_EQ(bn_it->get(), bn_op);

  // Create the new param arrays
  auto& mean_shape = *mean_array.mutable_shape();
  const auto& multiplier_shape = multiplier_array.shape();
  const auto& offset_shape = offset_array.shape();
  if (mean_shape.dims().empty()) {
    *mean_shape.mutable_dims() = multiplier_shape.dims();
    auto& data = mean_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
    CHECK_EQ(data.size(), 1);
    data.resize(RequiredBufferSizeForShape(mean_shape), data[0]);
  }
  CHECK(mean_shape.dims() == multiplier_shape.dims());
  CHECK(mean_shape.dims() == offset_shape.dims());
  const auto& param_shape = mean_shape;
  const int buffer_size = RequiredBufferSizeForShape(param_shape);
  auto& mul_param_array = model->GetOrCreateArray(mul_param_name);
  auto& add_param_array = model->GetOrCreateArray(add_param_name);
  DropMinMax(model, mul_param_name);
  DropMinMax(model, add_param_name);
  mul_param_array.copy_shape(param_shape);
  add_param_array.copy_shape(param_shape);
  mul_param_array.data_type = ArrayDataType::kFloat;
  add_param_array.data_type = ArrayDataType::kFloat;
  auto& mul_float_data =
      mul_param_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  auto& add_float_data =
      add_param_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  mul_float_data.resize(buffer_size);
  add_float_data.resize(buffer_size);
  const auto& mean_float_data =
      mean_array.GetBuffer<ArrayDataType::kFloat>().data;
  const auto& multiplier_float_data =
      multiplier_array.GetBuffer<ArrayDataType::kFloat>().data;
  const auto& offset_float_data =
      offset_array.GetBuffer<ArrayDataType::kFloat>().data;
  size_t buffer_size_for_compare = buffer_size;
  CHECK(mul_float_data.size() == buffer_size_for_compare);
  CHECK(add_float_data.size() == buffer_size_for_compare);
  CHECK(mean_float_data.size() == buffer_size_for_compare);
  CHECK(multiplier_float_data.size() == buffer_size_for_compare);
  CHECK(offset_float_data.size() == buffer_size_for_compare);

  for (int i = 0; i < buffer_size; i++) {
    mul_float_data[i] = multiplier_float_data[i];
    add_float_data[i] =
        offset_float_data[i] - mean_float_data[i] * multiplier_float_data[i];
  }

  DeleteOpAndArrays(model, bn_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
