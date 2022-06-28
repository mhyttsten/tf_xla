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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/inspecting_placer.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

string IOColocationGroups::DebugString() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/inspecting_placer.cc", "IOColocationGroups::DebugString");

  std::unordered_map<int, std::vector<string>> group_members;
  for (int arg_index = 0; arg_index < input_groups.size(); ++arg_index) {
    int group_id = input_groups[arg_index];
    group_members[group_id].push_back(strings::StrCat("i:", arg_index));
  }
  for (int ret_index = 0; ret_index < output_groups.size(); ++ret_index) {
    int group_id = output_groups[ret_index];
    group_members[group_id].push_back(strings::StrCat("o:", ret_index));
  }

  std::vector<string> group_strings;
  for (const auto& it : group_members) {
    int group_id = it.first;
    const std::vector<string>& members = it.second;
    const PossibleDevices& devices = group_devices[group_id];
    group_strings.push_back(strings::StrCat(
        "Group(", group_id, " members = [", absl::StrJoin(members, ", "),
        "] requested_device_name = \"",
        DeviceNameUtils::ParsedNameToString(devices.requested_device_name),
        "\" resource_device_name = \"",
        DeviceNameUtils::ParsedNameToString(devices.resource_device_name),
        "\" device_types = [",
        absl::StrJoin(
            devices.device_types, ", ",
            [](string* out, const std::pair<DeviceType, int32>& type_and_pref) {
              out->append(DeviceTypeString(type_and_pref.first));
            }),
        "])"));
  }

  return absl::StrJoin(group_strings, "\n\t");
}

// Utility class for constructing IOColocationGroups from a ColocationGraph.
class ColocationGraphToIOColocationGroups {
 public:
  // colocation_graph is mutable because finding root nodes can update
  // parent pointers. It is not modified otherwise.
  explicit ColocationGraphToIOColocationGroups(
      ColocationGraph* colocation_graph)
      : colocation_graph_(colocation_graph), next_group_id_(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc mht_1(mht_1_v, 248, "", "./tensorflow/core/common_runtime/inspecting_placer.cc", "ColocationGraphToIOColocationGroups");
}

  void AssignGroups(const gtl::InlinedVector<Node*, 4>& nodes,
                    std::vector<int>* groups) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/common_runtime/inspecting_placer.cc", "AssignGroups");

    for (int i = 0; i < nodes.size(); ++i) {
      int root_id = colocation_graph_->FindAndUpdateRoot(nodes[i]->id());
      const auto& it = group_ids_.find(root_id);
      int assigned_group_id;
      if (it == group_ids_.end()) {
        group_ids_[root_id] = next_group_id_;
        assigned_group_id = next_group_id_;
        ++next_group_id_;
      } else {
        assigned_group_id = it->second;
      }
      groups->push_back(assigned_group_id);
    }
  }

  Status FillGroups(std::vector<PossibleDevices>* group_devices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc mht_3(mht_3_v, 273, "", "./tensorflow/core/common_runtime/inspecting_placer.cc", "FillGroups");

    group_devices->resize(group_ids_.size());
    for (const auto& it : group_ids_) {
      int assigned_group_id = it.second;
      PossibleDevices& possible_devices = (*group_devices)[assigned_group_id];
      const Member& member = colocation_graph_->members()[it.first];
      TF_RETURN_IF_ERROR(member.FillPossibleDevices(&possible_devices));
    }
    return Status::OK();
  }

 private:
  ColocationGraph* colocation_graph_;
  // Allocated group ids: collocation_graph root id -> allocated group id.
  std::unordered_map<int, int> group_ids_;
  int next_group_id_;
};

InspectingPlacer::InspectingPlacer(const FunctionStack& stack,
                                   const FunctionLibraryDefinition* flib_def,
                                   const DeviceSet* device_set,
                                   const Device* default_device,
                                   bool allow_soft_placement,
                                   bool log_device_placement)
    : stack_(stack),
      flib_def_(*flib_def),
      device_set_(*device_set),
      default_device_(default_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc mht_4(mht_4_v, 305, "", "./tensorflow/core/common_runtime/inspecting_placer.cc", "InspectingPlacer::InspectingPlacer");
}

Status InspectingPlacer::ComputeIOColocationGroups(const Node& node,
                                                   IOColocationGroups* groups) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSinspecting_placerDTcc mht_5(mht_5_v, 311, "", "./tensorflow/core/common_runtime/inspecting_placer.cc", "InspectingPlacer::ComputeIOColocationGroups");

  const FunctionDef* fdef;
  NameAttrList func;
  TF_RETURN_IF_ERROR(GetFunctionDefAndAttrs(flib_def_, node, &fdef, &func));
  std::unique_ptr<FunctionBody> fbody;

  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*fdef, AttrSlice(&func.attr()),
                                             &flib_def_, &fbody));

  TF_RETURN_IF_ERROR(
      IsolatePlacerInspectionRequiredOps(flib_def_, fbody->graph));
  if (stack_.HasFunction(func.name())) {
    return errors::Unimplemented(
        "Recursive function calls are not supported. Node ",
        FormatNodeForError(node), " inside the body of ",
        errors::FormatFunctionForError(stack_.current_function_name()),
        " calls function ", errors::FormatFunctionForError(func.name()),
        " which is already present in the call stack:\n  ",
        stack_.FormatForError());
  }

  ColocationGraph colocation_graph(
      fbody->graph, stack_.Push(&node, func.name()), &flib_def_, &device_set_,
      default_device_, allow_soft_placement_, log_device_placement_);
  TF_RETURN_IF_ERROR(colocation_graph.Initialize());

  ColocationGraphToIOColocationGroups converter(&colocation_graph);
  converter.AssignGroups(fbody->arg_nodes, &groups->input_groups);
  converter.AssignGroups(fbody->ret_nodes, &groups->output_groups);
  TF_RETURN_IF_ERROR(converter.FillGroups(&groups->group_devices));
  return Status::OK();
}

}  // namespace tensorflow
