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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc() {
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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

int GetOutputDepthFromWeights(const Model& model, const Operator& op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/toco/graph_transformations/ensure_bias_vectors.cc", "GetOutputDepthFromWeights");

  const std::string& weights_name = op.inputs[1];
  const auto& weights_shape = model.GetArray(weights_name).shape();
  if (op.type == OperatorType::kConv ||
      op.type == OperatorType::kFullyConnected ||
      op.type == OperatorType::kTransposeConv) {
    return weights_shape.dims(0);
  }
  if (op.type == OperatorType::kDepthwiseConv) {
    return weights_shape.dims(3);
  }
  LOG(FATAL) << "Unhandled operator type";
  return 0;
}

bool CheckOpInputSize(const Operator& op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/toco/graph_transformations/ensure_bias_vectors.cc", "CheckOpInputSize");

  if (op.type == OperatorType::kConv ||
      op.type == OperatorType::kFullyConnected ||
      op.type == OperatorType::kDepthwiseConv) {
    return (op.inputs.size() >= 3);
  } else if (op.type == OperatorType::kTransposeConv) {
    return (op.inputs.size() >= 4);
  }
  return true;
}

bool ProcessLinearOperator(Model* model, Operator* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/toco/graph_transformations/ensure_bias_vectors.cc", "ProcessLinearOperator");

  if (CheckOpInputSize(*op)) {
    return false;
  }
  const std::string& output_name = op->outputs[0];
  const std::string& weights_name = op->inputs[1];
  if (!model->GetArray(weights_name).has_shape()) {
    return false;
  }
  const int depth = GetOutputDepthFromWeights(*model, *op);
  const std::string& bias_name =
      AvailableArrayName(*model, output_name + "_bias");
  op->inputs.push_back(bias_name);
  auto& bias_array = model->GetOrCreateArray(bias_name);
  bias_array.data_type = ArrayDataType::kFloat;
  bias_array.mutable_shape()->mutable_dims()->push_back(depth);
  auto& bias_buffer = bias_array.GetMutableBuffer<ArrayDataType::kFloat>();
  bias_buffer.data.resize(depth, 0.f);
  return true;
}
}  // namespace

::tensorflow::Status EnsureBiasVectors::Run(Model* model, std::size_t op_index,
                                            bool* modified) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSensure_bias_vectorsDTcc mht_3(mht_3_v, 256, "", "./tensorflow/lite/toco/graph_transformations/ensure_bias_vectors.cc", "EnsureBiasVectors::Run");

  *modified = false;
  auto* op = model->operators[op_index].get();
  if (op->type == OperatorType::kConv ||
      op->type == OperatorType::kDepthwiseConv ||
      op->type == OperatorType::kFullyConnected ||
      op->type == OperatorType::kTransposeConv) {
    if (ProcessLinearOperator(model, op)) {
      AddMessageF("Added bias vector to %s as %s", LogName(*op), op->inputs[2]);
      *modified = true;
      return ::tensorflow::Status::OK();
    }
  }
  return ::tensorflow::Status::OK();
}

}  // namespace toco
