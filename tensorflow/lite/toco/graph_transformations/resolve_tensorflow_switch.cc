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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_switchDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_switchDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_switchDTcc() {
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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status ResolveTensorFlowSwitch::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_tensorflow_switchDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/resolve_tensorflow_switch.cc", "ResolveTensorFlowSwitch::Run");

  *modified = false;
  const auto switch_it = model->operators.begin() + op_index;
  const auto* switch_op = switch_it->get();
  if (switch_op->type != OperatorType::kSwitch) {
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(switch_op->inputs.size(), 2);
  CHECK_EQ(switch_op->outputs.size(), 2);
  const std::string& predicate_name = switch_op->inputs[1];
  // If the predicate array hasn't been resolved to a constant yet,
  // we need to yield.
  if (!IsConstantParameterArray(*model, predicate_name)) {
    AddMessageF(
        "Waiting for the boolean predicate of %s to be resolved to a constant",
        LogName(*switch_op));
    return ::tensorflow::Status::OK();
  }

  // The predicate should be boolean, and should consist of a single value.
  const auto& predicate_array = model->GetArray(predicate_name);
  CHECK(predicate_array.data_type == ArrayDataType::kBool);
  for (const auto& dim : predicate_array.shape().dims()) {
    CHECK_EQ(dim, 1);
  }

  // Obtain the predicate boolean value.
  const auto& predicate_data =
      predicate_array.GetBuffer<ArrayDataType::kBool>().data;
  CHECK_EQ(predicate_data.size(), 1);
  const bool predicate_value = predicate_data[0];

  // From the TensorFlow docs on .switch() in
  // third_party/tensorflow/python/ops/control_flow_ops.py
  //
  //    If `pred` is false, the `data` input is forwarded to the first output.
  //    Otherwise, the data goes to the second output.
  //
  // Note that this comment used to say the opposite and was recently fixed:
  // https://github.com/tensorflow/tensorflow/commit/bc456e361d49d1d89a74b80060c70efb51fd7d87#diff-76ab9dafbe12c20ddc3769c6b108986c
  const int selected_output_index = predicate_value ? 1 : 0;
  const int nonselected_output_index = predicate_value ? 0 : 1;

  // Update the edges of the graph ahead of removing the node:
  // edges that were pointing to the selected output, should instead
  // point to the input of the Switch node.
  for (const auto& other_op : model->operators) {
    for (auto& input : other_op->inputs) {
      if (input == switch_op->outputs[selected_output_index]) {
        input = switch_op->inputs[0];
      }
    }
  }

  // There remains to handle the edges that were pointing to the nonselected
  // output. We will just discard those edges. Concretely, at the moment,
  // our only examples of graphs with Switch nodes have them feeding into Merge
  // nodes, so what we're saying here is that we'll make the convention,
  // in our toco internal representation, that Merge nodes with only 1 input
  // are Merge nodes that have been resolved already and should be have as
  // Identity nodes, simply forwarding their input.
  //
  for (const auto& other_op : model->operators) {
    auto input_it = other_op->inputs.begin();
    while (input_it != other_op->inputs.end()) {
      if (*input_it == switch_op->outputs[nonselected_output_index]) {
        // Let us guard our assumption that only Merge nodes consume the outputs
        // of Switch nodes:
        if (other_op->type != OperatorType::kMerge) {
          return ::tensorflow::Status(
              ::tensorflow::error::FAILED_PRECONDITION,
              ::absl::StrCat(
                  "Found ", HelpfulOperatorTypeName(*other_op),
                  " as non-selected output from Switch, but only "
                  "Merge supported. Control flow ops like Switch and Merge are "
                  "not generally supported. We are working on fixing this, "
                  "please see the Github issue at "
                  "https://github.com/tensorflow/tensorflow/issues/28485."));
        }
        input_it = other_op->inputs.erase(input_it);
      } else {
        ++input_it;
      }
    }
  }

  // Remove the output arrays if they are now unused.
  for (int i = 0; i < 2; i++) {
    if (!GetOpWithInput(*model, switch_op->outputs[i])) {
      model->EraseArray(switch_op->outputs[i]);
    }
  }
  // Remove input arrays if they are only used by the switch itself and aren't
  // the output of another op (will get handled by RemoveUnusedOp in that case).
  for (const auto& input : switch_op->inputs) {
    if (CountOpsWithInput(*model, input) == 1 &&
        !GetOpWithOutput(*model, input)) {
      model->EraseArray(input);
    }
  }
  // Remove the switch node itself.
  AddMessageF("Removing already-resolved %s", LogName(*switch_op));
  DeleteOpAndArrays(model, switch_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
