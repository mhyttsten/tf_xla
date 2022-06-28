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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_passthroughDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_passthroughDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_passthroughDTcc() {
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
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {

// Reroute all edges involving a given discardable array to another
// array instead. from_array is assumed to be discardable, and consequently
// this only updates operator edges (since discardable arrays only
// appear there, and not e.g. in model flags).
void Reroute(const std::string& from, const std::string& to, Model* model) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("from: \"" + from + "\"");
   mht_0_v.push_back("to: \"" + to + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_passthroughDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.cc", "Reroute");

  for (const auto& op : model->operators) {
    for (auto& output : op->outputs) {
      if (output == from) {
        output = to;
      }
    }
    for (auto& input : op->inputs) {
      if (input == from) {
        input = to;
      }
    }
  }
  const Array& from_array = model->GetArray(from);
  Array& to_array = model->GetOrCreateArray(to);
  // Preserve minmax information if to_array didn't already have any.
  if (from_array.minmax && !to_array.minmax) {
    to_array.GetOrCreateMinMax() = from_array.GetMinMax();
    // If we're copying minmax info, then we should also be copying
    // narrow_range, which affects how minmax info is to be interpreted.
    to_array.narrow_range = from_array.narrow_range;
  }
  // Separately, also preserve final_data_type if to_array didn't already
  // have any.
  if (from_array.final_data_type != ArrayDataType::kNone &&
      to_array.final_data_type == ArrayDataType::kNone) {
    to_array.final_data_type = from_array.final_data_type;
  }
  // The 'from' array may now be unused. We delete it here immediately
  // so that this function doesn't violate graph invariants (no unused arrays)
  // and as it's not trivial to get this right for the caller since
  // DeleteOpAndArrays will no longer delete this array, since it's no longer
  // referenced by this op.
  DeleteArrayIfUnused(from, model);
}

}  // namespace

bool RemoveTrivialPassthroughOp(GraphTransformation* transformation,
                                Model* model, std::size_t op_index,
                                int input_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSremove_trivial_passthroughDTcc mht_1(mht_1_v, 246, "", "./tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.cc", "RemoveTrivialPassthroughOp");

  auto passthru_it = model->operators.begin() + op_index;
  auto* passthru_op = passthru_it->get();
  CHECK_EQ(passthru_op->outputs.size(), 1);
  CHECK_GE(passthru_op->inputs.size(), 1);

  int main_input_array_index = 0;
  if (input_index != -1) {
    main_input_array_index = input_index;
  } else {
    // We call 'main input' the unique nonconstant input array if there is one,
    // or else the 0-th input.
    int count_nonconstant_input_arrays = 0;
    for (size_t i = 0; i < passthru_op->inputs.size(); i++) {
      if (!model->GetArray(passthru_op->inputs[i]).buffer) {
        count_nonconstant_input_arrays++;
        if (count_nonconstant_input_arrays == 1) {
          main_input_array_index = i;
        }
      }
    }
  }

  const std::string main_input_name =
      passthru_op->inputs[main_input_array_index];
  const std::string output_name = passthru_op->outputs[0];

  if (IsDiscardableArray(*model, output_name)) {
    transformation->AddMessageF(
        "Removing %s, keeping its non-constant input array %s and removing %s",
        LogName(*passthru_op), main_input_name, output_name);
    Reroute(output_name, main_input_name, model);
  } else if (IsDiscardableArray(*model, main_input_name) &&
             !IsConstantParameterArray(*model, main_input_name)) {
    transformation->AddMessageF(
        "Removing %s, keeping its output array %s and removing non-constant "
        "input %s",
        LogName(*passthru_op), output_name, main_input_name);
    Reroute(main_input_name, output_name, model);
  } else {
    transformation->AddMessageF(
        "Cannot remove %s, neither its main input nor its output may be "
        "discarded",
        LogName(*passthru_op));
    if (passthru_op->type != OperatorType::kReshape &&
        model->GetArray(main_input_name).has_shape()) {
      // We can't remove either array but we can remove the op. Converting it to
      // a reshape gives us some hope of later on fixing that (either in the
      // final runtime or as an additional fixup step).
      //
      // Note that we don't try to insert copies in place of reshapes as the
      // copy itself is a trivial reshape and we'd go into an infinite loop!
      transformation->AddMessageF("Replacing with a copy (reshape) instead");
      InsertCopyOperator(model, main_input_name, output_name);
      // To avoid using invalidated iterator, evaluate passthru_it again.
      passthru_it = model->operators.begin() + op_index;
    } else {
      return false;
    }
  }

  // Remove the pass-through node.
  DeleteOpAndArrays(model, passthru_op);

  return true;
}

}  // namespace toco
