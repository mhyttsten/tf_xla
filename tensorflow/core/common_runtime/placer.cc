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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/placer.h"

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/colocation_graph.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

namespace {

struct NameCounts {
  mutex counts_mutex;
  std::unordered_map<string, int> counts;
};

string MakeUniqueFilename(string name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/common_runtime/placer.cc", "MakeUniqueFilename");

  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0; i < name.size(); ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?') {
      name[i] = '_';
    }
  }

  int count;
  {
    mutex_lock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  string filename = name;
  if (count > 0) {
    absl::StrAppend(&filename, "_", count);
  }
  absl::StrAppend(&filename, ".txt");
  return filename;
}

Status GetFileName(string base_name, string* fname) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("base_name: \"" + base_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/common_runtime/placer.cc", "GetFileName");

  const char* dir = nullptr;
  dir = getenv("TF_DUMP_GRAPH_PREFIX");
  if (!dir) {
    return errors::Internal("Failed to get the directory for ", base_name,
                            " because dump location is not specified through "
                            "TF_DUMP_GRAPH_PREFIX environment variable");
  }
  base_name = MakeUniqueFilename(base_name);
  *fname = absl::StrCat(dir, "/", base_name);
  return Status::OK();
}

void DumpColocationGraph(const string& base_name,
                         const ColocationGraph& colocation_graph) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("base_name: \"" + base_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_2(mht_2_v, 260, "", "./tensorflow/core/common_runtime/placer.cc", "DumpColocationGraph");

  string fname;
  Status status = GetFileName(base_name, &fname);
  if (status.ok()) {
    status = WriteStringToFile(Env::Default(), fname,
                               colocation_graph.DebugString());
    if (status.ok()) {
      LOG(INFO) << "Wrote ColocationGraph to " << fname;
    }
  }
  if (!status.ok()) {
    LOG(ERROR) << "Failed to write final colocation graph to file " << fname
               << " with " << status.ToString();
  }
}

// Returns true if the node has no inputs and produces outputs
// that are consumed by a single node.
//
// TODO(vrv): Currently this handles only nodes with one output, but
// this could be extended to handle the case where a node has many
// outputs that are connected to nodes in the same colocation group.
bool IsGeneratorNode(const Node* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_3(mht_3_v, 285, "", "./tensorflow/core/common_runtime/placer.cc", "IsGeneratorNode");

  return node->num_inputs() == 0 && node->num_outputs() == 1 &&
         !IsRefType(node->output_type(0));
}

void LogDeviceAssignment(const Node* node, bool log_device_placement) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_4(mht_4_v, 293, "", "./tensorflow/core/common_runtime/placer.cc", "LogDeviceAssignment");

  // Log placement if log_device_placement is set.
  if (log_device_placement) {
    printf("%s: (%s): %s\n", node->name().c_str(), node->type_string().c_str(),
           node->assigned_device_name().c_str());
    LOG(INFO) << node->name() << ": "
              << "(" << node->type_string()
              << "): " << node->assigned_device_name();
  }
  if (VLOG_IS_ON(1)) {
    if (VLOG_IS_ON(4)) {
      VLOG(4) << "\nNode:\n"
              << node->def().DebugString()
              << "placed on: " << node->assigned_device_name();
    } else {
      VLOG(1) << node->name() << "(" << node->type_string()
              << ") placed on: " << node->assigned_device_name();
    }
  }
}

Status AssignAndLog(int assigned_device, Node* node,
                    ColocationGraph* colocation_graph,
                    bool log_device_placement) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/common_runtime/placer.cc", "AssignAndLog");

  node->set_assigned_device_name_index(assigned_device);

  // Constraint the group of node to the assigned device.
  TF_RETURN_IF_ERROR(colocation_graph->LimitToAssignedDevice(*node));

  LogDeviceAssignment(node, log_device_placement);
  return Status::OK();
}

}  // namespace

Placer::Placer(Graph* graph, const string& function_name,
               const FunctionLibraryDefinition* flib_def,
               const DeviceSet* devices, const Device* default_local_device,
               bool allow_soft_placement, bool log_device_placement)
    : graph_(graph),
      function_name_(function_name),
      flib_def_(flib_def),
      devices_(devices),
      default_local_device_(default_local_device),
      allow_soft_placement_(allow_soft_placement),
      log_device_placement_(log_device_placement) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_6(mht_6_v, 345, "", "./tensorflow/core/common_runtime/placer.cc", "Placer::Placer");
}

Placer::Placer(Graph* graph, const string& function_name,
               const FunctionLibraryDefinition* flib_def,
               const DeviceSet* devices, const Device* default_local_device)
    : Placer(graph, function_name, flib_def, devices, default_local_device,
             true, false) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_7(mht_7_v, 355, "", "./tensorflow/core/common_runtime/placer.cc", "Placer::Placer");
}
Placer::Placer(Graph* graph, const string& function_name,
               const FunctionLibraryDefinition* flib_def,
               const DeviceSet* devices)
    : Placer(graph, function_name, flib_def, devices, nullptr, true, false) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_8(mht_8_v, 363, "", "./tensorflow/core/common_runtime/placer.cc", "Placer::Placer");
}

Placer::~Placer() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_9(mht_9_v, 368, "", "./tensorflow/core/common_runtime/placer.cc", "Placer::~Placer");
}

Status Placer::Run() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_10(mht_10_v, 373, "", "./tensorflow/core/common_runtime/placer.cc", "Placer::Run");

  if (devices_->devices().empty()) {
    return errors::FailedPrecondition("No devices are registered");
  }

  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("placer_input", *graph_, nullptr);
  }
  if (VLOG_IS_ON(5)) {
    for (const Node* node : graph_->op_nodes()) {
      VLOG(5) << "    " << node->name() << ": requested: '"
              << node->requested_device() << "' assigned: '"
              << node->assigned_device_name() << "'";
    }
  }

  FunctionStack stack(function_name_);
  ColocationGraph colocation_graph(graph_, stack, flib_def_, devices_,
                                   default_local_device_, allow_soft_placement_,
                                   log_device_placement_);

  TF_RETURN_IF_ERROR(colocation_graph.Initialize());

  // For each node, assign a device based on the constraints in the disjoint
  // node set.
  std::vector<Node*> second_pass;
  for (Node* node : graph_->op_nodes()) {
    // The graph may have come pre-populated by the framework with assigned
    // devices (e.g., for stateful placements), so the placer should not try to
    // place nodes that are already placed.
    if (node->has_assigned_device_name()) {
      TF_RETURN_IF_ERROR(colocation_graph.LimitToAssignedDevice(*node));
      LogDeviceAssignment(node, log_device_placement_);
      continue;
    }

    // Heuristic A: prefer to place "generators" with their only
    // consumers.
    //
    // If this is a node with no inputs and one output, we save
    // this for a second pass, so that the consumer's placement
    // is chosen.
    if (IsGeneratorNode(node)) {
      second_pass.push_back(node);
      continue;
    }

    const std::vector<Device*>* devices;
    Status status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device for operation ",
                                  node->name(), ": ", status.error_message()),
          *node);
    }

    // TODO(mdan): This is a constrained optimization solver. Write it like one.

    // Returns the first device in sorted devices list so we will always
    // choose the same device.
    //
    // TODO(vrv): Factor this assignment out into a pluggable
    // algorithm, so that Placer is responsible for enforcing
    // preconditions and we can experiment with other algorithms when
    // given a choice of devices. Once we have a better idea of the
    // types of heuristics we want to use and the information needed
    // to perform good placement we can add an interface for this.
    int assigned_device = -1;

    // Heuristic B: If the node only operates on metadata, not data,
    // then it is desirable to place that metadata node with its
    // input.
    if (IsMetadata(node)) {
      // Make sure that the input device type is in the list of supported
      // device types for this node.
      const Node* input = (*node->in_edges().begin())->src();
      // TODO(vrv): if the input is empty, consider postponing this
      // node's assignment to the second pass, so that we handle the
      // case where a metadata node's input comes from a backedge
      // of a loop.
      if (CanAssignToDevice(input->assigned_device_name(), *devices)) {
        assigned_device = input->assigned_device_name_index();
      }
    }

    // Provide the default, if necessary.
    if (assigned_device == -1) {
      assigned_device = graph_->InternDeviceName((*devices)[0]->name());
    }

    TF_RETURN_IF_ERROR(AssignAndLog(assigned_device, node, &colocation_graph,
                                    log_device_placement_));
  }

  // Perform a second pass assignment for those nodes explicitly
  // skipped during the first pass.
  for (Node* node : second_pass) {
    const std::vector<Device*>* devices;
    Status status = colocation_graph.GetDevicesForNode(node, &devices);
    if (!status.ok()) {
      return AttachDef(
          errors::InvalidArgument("Cannot assign a device for operation ",
                                  node->name(), ": ", status.error_message()),
          *node);
    }

    int assigned_device = -1;

    // Heuristic A application.
    if (IsGeneratorNode(node) && !node->out_edges().empty()) {
      const Node* output = (*node->out_edges().begin())->dst();
      int output_device_name = output->assigned_device_name_index();

      const bool consumers_on_same_device = std::all_of(
          node->out_edges().begin(), node->out_edges().end(),
          [output_device_name](const Edge* e) {
            return e->dst()->assigned_device_name_index() == output_device_name;
          });

      if (consumers_on_same_device &&
          CanAssignToDevice(output->assigned_device_name(), *devices)) {
        assigned_device = output_device_name;
      }
    }

    // Provide the default, if necessary.
    if (assigned_device == -1) {
      assigned_device = graph_->InternDeviceName((*devices)[0]->name());
    }

    TF_RETURN_IF_ERROR(AssignAndLog(assigned_device, node, &colocation_graph,
                                    log_device_placement_));
  }

  if (VLOG_IS_ON(3)) {
    DumpGraphToFile("placer_output", *graph_, nullptr);
    DumpColocationGraph("colocation_graph", colocation_graph);
  }
  return Status::OK();
}

bool Placer::CanAssignToDevice(const string& candidate_device_name,
                               const std::vector<Device*>& devices) const {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("candidate_device_name: \"" + candidate_device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSplacerDTcc mht_11(mht_11_v, 519, "", "./tensorflow/core/common_runtime/placer.cc", "Placer::CanAssignToDevice");

  if (!candidate_device_name.empty()) {
    // 'devices' lists the set of devices that the placer or the user has
    // constrained the operation to.  "candidate_device_name" must
    // refer to a concrete Device that is in the list of 'devices'.
    const Device* other_device =
        devices_->FindDeviceByName(candidate_device_name);
    if (std::find(devices.begin(), devices.end(), other_device) !=
        devices.end()) {
      return true;
    }
  }

  return false;
}

}  // namespace tensorflow
