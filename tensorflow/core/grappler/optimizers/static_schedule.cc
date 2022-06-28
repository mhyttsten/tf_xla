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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSstatic_scheduleDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSstatic_scheduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSstatic_scheduleDTcc() {
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

#include "tensorflow/core/grappler/optimizers/static_schedule.h"

#include <deque>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {

static Costs::NanoSeconds PredictExecutionTime(
    const GraphProperties& properties, const OpLevelCostEstimator& estimator,
    const VirtualPlacer& placer, const NodeDef& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSstatic_scheduleDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/optimizers/static_schedule.cc", "PredictExecutionTime");

  OpContext op_context;
  op_context.op_info.set_op(node.op());
  *op_context.op_info.mutable_attr() = node.attr();

  std::vector<OpInfo::TensorProperties> inputs =
      properties.GetInputProperties(node.name());
  for (auto& input : inputs) {
    op_context.op_info.add_inputs()->Swap(&input);
  }

  std::vector<OpInfo::TensorProperties> outputs =
      properties.GetOutputProperties(node.name());
  for (auto& output : outputs) {
    op_context.op_info.add_outputs()->Swap(&output);
  }

  DeviceProperties device = placer.get_device(node);
  op_context.op_info.mutable_device()->Swap(&device);

  Costs::NanoSeconds estimate =
      estimator.PredictCosts(op_context).execution_time;

  // Make sure our estimates are at least one nanosecond per node.
  return std::max(estimate, Costs::NanoSeconds(1));
}

Status EstimateEarliestExecutionTimes(
    const GrapplerItem& item, const Cluster* cluster,
    std::unordered_map<const NodeDef*, Costs::NanoSeconds>* completion_times) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSstatic_scheduleDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/grappler/optimizers/static_schedule.cc", "EstimateEarliestExecutionTimes");

  std::unordered_map<string, const NodeDef*> name_map;
  std::unordered_map<const NodeDef*, int> pending_inputs;
  std::deque<const NodeDef*> ready_nodes;
  for (const NodeDef& node : item.graph.node()) {
    name_map[node.name()] = &node;
    if (node.input_size() == 0) {
      ready_nodes.push_back(&node);
      (*completion_times)[&node] = 0;
    } else if (IsMerge(node)) {
      // Merge nodes are processed as soon as one of the input becomes
      // available.
      pending_inputs[&node] = 1;
    } else {
      pending_inputs[&node] = node.input_size();
    }
  }

  std::unordered_map<const NodeDef*, std::vector<const NodeDef*>> fanouts;
  for (const NodeDef& node : item.graph.node()) {
    for (const string& input : node.input()) {
      string node_name = NodeName(input);
      auto it = name_map.find(node_name);
      if (it == name_map.end()) {
        return errors::InvalidArgument(
            strings::StrCat("Unknown input node ", input));
      }
      const NodeDef* fanin = it->second;
      fanouts[fanin].push_back(&node);
    }
  }
  name_map.clear();

  GraphProperties properties(item);
  TF_RETURN_IF_ERROR(
      properties.InferStatically(/*assume_valid_feeds=*/true,
                                 /*aggressive_shape_inference=*/false,
                                 /*include_tensor_values=*/false));
  OpLevelCostEstimator estimator;
  VirtualPlacer placer(cluster->GetDevices());

  while (!ready_nodes.empty()) {
    const NodeDef* node = ready_nodes.front();
    ready_nodes.pop_front();

    Costs::NanoSeconds execution_time =
        PredictExecutionTime(properties, estimator, placer, *node);
    Costs::NanoSeconds completion_time =
        execution_time + (*completion_times)[node];
    (*completion_times)[node] = completion_time;

    for (const NodeDef* fanout : fanouts[node]) {
      int pending = pending_inputs[fanout];
      if (pending == 0) {
        // Already processed. Avoid going through loops more than once.
        continue;
      } else if (pending == 1) {
        ready_nodes.push_back(fanout);
      }
      pending_inputs[fanout]--;

      Costs::NanoSeconds ready_time =
          std::max(completion_time, (*completion_times)[fanout]);
      (*completion_times)[fanout] = ready_time;
    }
  }

  return Status::OK();
}

Status EstimateRequiredTimes(
    const GrapplerItem& item, const Cluster* cluster,
    const std::unordered_map<const NodeDef*, Costs::NanoSeconds>&
        execution_times,
    std::unordered_map<const NodeDef*, Costs::NanoSeconds>* required_times) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSstatic_scheduleDTcc mht_2(mht_2_v, 312, "", "./tensorflow/core/grappler/optimizers/static_schedule.cc", "EstimateRequiredTimes");

  std::unordered_map<string, const NodeDef*> name_map;
  for (const NodeDef& node : item.graph.node()) {
    name_map[node.name()] = &node;
    (*required_times)[&node] = Costs::NanoSeconds::max();
  }

  std::unordered_map<const NodeDef*, int> pending_fanouts;
  for (const NodeDef& node : item.graph.node()) {
    for (const string& input : node.input()) {
      string node_name = NodeName(input);
      auto it = name_map.find(node_name);
      if (it == name_map.end()) {
        return errors::InvalidArgument(
            strings::StrCat("Unknown input node ", input));
      }
      const NodeDef* fanin = it->second;
      pending_fanouts[fanin] += 1;
    }
  }
  std::deque<const NodeDef*> ready_nodes;
  for (const NodeDef& node : item.graph.node()) {
    if (pending_fanouts[&node] == 0) {
      auto it = execution_times.find(&node);
      if (it != execution_times.end()) {
        (*required_times)[&node] = it->second;
      }
      ready_nodes.push_back(&node);
    }
  }
  GraphProperties properties(item);
  TF_RETURN_IF_ERROR(
      properties.InferStatically(/*assume_valid_feeds=*/true,
                                 /*aggressive_shape_inference=*/false,
                                 /*include_tensor_values=*/false));
  OpLevelCostEstimator estimator;
  VirtualPlacer placer(cluster->GetDevices());

  while (!ready_nodes.empty()) {
    const NodeDef* node = ready_nodes.front();
    ready_nodes.pop_front();

    Costs::NanoSeconds execution_time =
        PredictExecutionTime(properties, estimator, placer, *node);
    Costs::NanoSeconds required_time = (*required_times)[node] - execution_time;

    for (const string& fanin_name : node->input()) {
      const NodeDef* fanin = name_map[NodeName(fanin_name)];
      (*required_times)[fanin] =
          std::min((*required_times)[fanin], required_time);

      int pending = pending_fanouts[fanin];
      if (pending == 0) {
        // Already processed. Avoid going through loops more than once.
        continue;
      } else if (pending == 1) {
        ready_nodes.push_back(fanin);
      }
      pending_fanouts[fanin]--;
    }
  }

  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
