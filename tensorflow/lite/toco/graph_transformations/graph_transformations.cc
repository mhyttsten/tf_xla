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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc() {
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
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

void PrintModelStats(const std::string& label, const Model& model) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.cc", "PrintModelStats");

  int quantized_arrays = 0;
  for (const auto& array : model.GetArrayMap()) {
    if (array.second->quantization_params) {
      quantized_arrays++;
    }
  }
  LOG(INFO) << label << ": " << model.operators.size() << " operators, "
            << model.GetArrayMap().size() << " arrays (" << quantized_arrays
            << " quantized)";
}

// Some graphs have RNN back-edges that are discardable, having been
// created typically by TensorFlow import rather than specified by the user.
// Such graphs might have cycles (closed by RNN back-edges) that may be pruned.
// Local graph transformations can't identify such global features,
// so this function performs this global transformation.
//
// The other (and related) thing that is peculiar about RNN back-edges
// is that they do not prevent the arrays that they touch, from being
// pruned. Thus, they may refer to array names which no longer exist.
// The intent is for that to result in the eventual pruning of such
// 'dangling' RNN back-edges. We perform this pruning at the end of this
// function, as the pruning of connected components done here may leave
// more RNN back-edges dangling.
void DiscardUselessConnectedComponentsAndRNNBackEdges(Model* model) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc mht_1(mht_1_v, 230, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.cc", "DiscardUselessConnectedComponentsAndRNNBackEdges");

  // Identify the set of arrays that are in 'useful' connected components
  // of the graph, which means connected to output arrays.
  std::unordered_set<std::string> useful_arrays;
  for (const std::string& output_array : model->flags.output_arrays()) {
    useful_arrays.insert(output_array);
  }
  bool found_new_useful_arrays;
  do {
    found_new_useful_arrays = false;
    for (const auto& op : model->operators) {
      bool op_touches_useful_arrays = false;
      for (const std::string& output : op->outputs) {
        op_touches_useful_arrays |= useful_arrays.count(output);
      }
      if (op_touches_useful_arrays) {
        for (const std::string& input : op->inputs) {
          found_new_useful_arrays |= !useful_arrays.count(input);
          useful_arrays.insert(input);
        }
        for (const std::string& output : op->outputs) {
          found_new_useful_arrays |= !useful_arrays.count(output);
          useful_arrays.insert(output);
        }
      }
    }
    for (const auto& rnn_state : model->flags.rnn_states()) {
      bool rnn_back_edge_touches_useful_arrays =
          useful_arrays.count(rnn_state.state_array());
      if (rnn_back_edge_touches_useful_arrays) {
        found_new_useful_arrays |=
            !useful_arrays.count(rnn_state.back_edge_source_array());
        useful_arrays.insert(rnn_state.back_edge_source_array());
      }
    }
  } while (found_new_useful_arrays);
  // Erase arrays that aren't useful, and that are discardable.
  model->EraseArrays([&](const std::string& name) {
    return (!useful_arrays.count(name) && IsDiscardableArray(*model, name));
  });
  // Erase operators that do not produce a useful output array.
  for (auto it = model->operators.begin(); it != model->operators.end();) {
    // Only need to test the first output, as we simultaneously added all of
    // an operator's outputs to the list of output arrays.
    if (useful_arrays.count((*it)->outputs[0])) {
      ++it;
    } else {
      for (const std::string& output : (*it)->outputs) {
        CHECK(!useful_arrays.count(output));
      }
      it = model->operators.erase(it);
    }
  }
  // Erase RNN back-edges that are 'dangling' i.e. that touch an array
  // that no longer exists. This should only happen for discardable RNN
  // back-edges.
  std::vector<RnnState> rnn_states_to_keep;
  for (const auto& rnn_state : model->flags.rnn_states()) {
    const bool dangling =
        !model->HasArray(rnn_state.back_edge_source_array()) ||
        !model->HasArray(rnn_state.state_array());
    if (dangling) {
      CHECK(rnn_state.discardable());
    } else {
      rnn_states_to_keep.push_back(rnn_state);
    }
  }
  model->flags.clear_rnn_states();
  for (const auto& rnn_state : rnn_states_to_keep) {
    *model->flags.add_rnn_states() = rnn_state;
  }
}

bool GraphTransformationsPass(int increment, Model* model,
                              const GraphTransformationsSet& transformations,
                              tensorflow::Status* status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc mht_2(mht_2_v, 308, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.cc", "GraphTransformationsPass");

  CHECK(increment == 1 || increment == -1);
  bool changed = false;
  if (model->operators.empty()) {
    LOG(INFO) << "Model is empty!!!";
    return false;
  }
  int op_index = increment == 1 ? 0 : model->operators.size() - 1;
  while (true) {
    bool changed_now = false;
    // Loop over all transformations at the current position in the graph.
    for (const auto& transformation : transformations) {
      CHECK(!changed_now);
      CHECK(transformation->Messages().empty());
      *status = transformation->Run(model, op_index, &changed_now);
      if (!status->ok()) {
        return false;
      }
      const char* made_a_change_msg =
          changed_now ? "made a change" : "did NOT make a change";
      const int log_level =
          changed_now ? kLogLevelModelChanged : kLogLevelModelUnchanged;
      if (transformation->Messages().empty()) {
        VLOG(log_level) << transformation->Name() << " " << made_a_change_msg
                        << " at op_index=" << op_index << "/"
                        << model->operators.size() - 1;
      }
      for (const std::string& message : transformation->Messages()) {
        VLOG(log_level) << transformation->Name() << " " << made_a_change_msg
                        << " at op_index=" << op_index << "/"
                        << model->operators.size() - 1 << ": " << message;
      }
      transformation->ClearMessages();
      if (changed_now) {
        DumpGraphvizVideoFrame(*model);
        if (model->operators.empty()) return true;
        op_index = std::min<int>(op_index, model->operators.size() - 1);
        // Uncomment for debugging
        // CheckInvariants(*model);
      }
      if (changed_now) {
        break;
      }
    }
    if (changed_now) {
      changed = true;
    } else {
      const int op_index_last =
          increment == 1 ? model->operators.size() - 1 : 0;
      if (op_index == op_index_last) {
        break;
      }
      op_index += increment;
    }
  }
  DiscardUselessConnectedComponentsAndRNNBackEdges(model);
  return changed;
}

}  // namespace

tensorflow::Status RunGraphTransformationsWithStatus(
    Model* model, const std::string& msg,
    const GraphTransformationsSet& transformations) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("msg: \"" + msg + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSgraph_transformationsDTcc mht_3(mht_3_v, 375, "", "./tensorflow/lite/toco/graph_transformations/graph_transformations.cc", "RunGraphTransformationsWithStatus");

  PrintModelStats(toco::port::StringF("Before %s", msg), *model);
  int pass_index = 0;
  tensorflow::Status status;
  while (GraphTransformationsPass((pass_index % 2) ? -1 : 1, model,
                                  transformations, &status)) {
    pass_index++;
    const auto& label =
        toco::port::StringF("After %s pass %d", msg, pass_index);
    PrintModelStats(label, *model);
    CheckInvariants(*model);
  }
  return status;
}

}  // namespace toco
