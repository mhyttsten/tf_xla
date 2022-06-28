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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc() {
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

#include "tensorflow/core/grappler/grappler_item.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/transitive_fanin.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

GrapplerItem::OptimizationOptions CreateOptOptionsForEager() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/grappler_item.cc", "CreateOptOptionsForEager");

  GrapplerItem::OptimizationOptions optimization_options;
  // Tensorflow 2.0 in eager mode with automatic control dependencies will
  // prune all nodes that are not in the transitive fanin of the fetch nodes.
  // However because the function will be executed via FunctionLibraryRuntime,
  // and current function implementation does not prune stateful and dataset
  // ops, we rely on Grappler to do the correct graph pruning.
  optimization_options.allow_pruning_stateful_and_dataset_ops = true;

  optimization_options.is_eager_mode = true;

  // All the nested function calls will be executed and optimized via
  // PartitionedCallOp, there is no need to optimize functions now.
  optimization_options.optimize_function_library = false;

  return optimization_options;
}

GrapplerItem GrapplerItem::WithGraph(GraphDef&& graph_def) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::WithGraph");

  GrapplerItem item;
  item.id = id;
  item.feed = feed;
  item.fetch = fetch;
  item.init_ops = init_ops;
  item.keep_ops = keep_ops;
  item.expected_init_time = expected_init_time;
  item.save_op = save_op;
  item.restore_op = restore_op;
  item.save_restore_loc_tensor = save_restore_loc_tensor;
  item.queue_runners = queue_runners;
  item.devices_ = devices_;
  item.optimization_options_ = optimization_options_;
  item.graph.Swap(&graph_def);
  return item;
}

std::vector<const NodeDef*> GrapplerItem::MainOpsFanin() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::MainOpsFanin");

  std::vector<const NodeDef*> fanin_nodes;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, fetch, &fanin_nodes));
  return fanin_nodes;
}

std::vector<const NodeDef*> GrapplerItem::EnqueueOpsFanin() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::EnqueueOpsFanin");

  std::vector<string> enqueue_ops;
  for (const auto& queue_runner : queue_runners) {
    for (const string& enqueue_op : queue_runner.enqueue_op_name()) {
      enqueue_ops.push_back(enqueue_op);
    }
  }
  std::vector<const NodeDef*> fanin_nodes;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, fetch, &fanin_nodes));
  return fanin_nodes;
}

std::vector<const NodeDef*> GrapplerItem::InitOpsFanin() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_4(mht_4_v, 269, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::InitOpsFanin");

  std::vector<const NodeDef*> fanin_nodes;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, init_ops, &fanin_nodes));
  return fanin_nodes;
}

std::vector<const NodeDef*> GrapplerItem::MainVariables() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::MainVariables");

  std::vector<const NodeDef*> fanin;
  TF_CHECK_OK(ComputeTransitiveFanin(graph, init_ops, &fanin));
  std::vector<const NodeDef*> vars;
  for (const NodeDef* node : fanin) {
    if (IsVariable(*node)) {
      vars.push_back(node);
    }
  }
  return vars;
}

std::unordered_set<string> GrapplerItem::NodesToPreserve() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::NodesToPreserve");

  std::unordered_set<string> result;
  for (const string& f : fetch) {
    VLOG(1) << "Add fetch " << f;
    result.insert(NodeName(f));
  }
  for (const auto& f : feed) {
    VLOG(1) << "Add feed " << f.first;
    result.insert(NodeName(f.first));
  }
  for (const auto& node : init_ops) {
    result.insert(NodeName(node));
  }
  for (const auto& node : keep_ops) {
    result.insert(NodeName(node));
  }
  if (!save_op.empty()) {
    result.insert(NodeName(save_op));
  }
  if (!restore_op.empty()) {
    result.insert(NodeName(restore_op));
  }
  if (!save_restore_loc_tensor.empty()) {
    result.insert(NodeName(save_restore_loc_tensor));
  }

  for (const auto& queue_runner : queue_runners) {
    for (const string& enqueue_op : queue_runner.enqueue_op_name()) {
      result.insert(NodeName(enqueue_op));
    }
    if (!queue_runner.close_op_name().empty()) {
      result.insert(NodeName(queue_runner.close_op_name()));
    }
    if (!queue_runner.cancel_op_name().empty()) {
      result.insert(NodeName(queue_runner.cancel_op_name()));
    }
  }

  absl::optional<FunctionLibraryDefinition> fn_library;
  if (!optimization_options_.allow_pruning_stateful_and_dataset_ops) {
    fn_library.emplace(OpRegistry::Global(), graph.library());
  }
  for (const NodeDef& node : graph.node()) {
    const auto attrs = AttrSlice(&node.attr());

    // Tensorflow functions do not prune stateful or dataset-output ops from
    // the function body (see PruneFunctionBody in common_runtime/function.cc).
    if (!optimization_options_.allow_pruning_stateful_and_dataset_ops &&
        (IsStateful(node, &*fn_library) || IsDataset(node))) {
      result.insert(node.name());
    }

    // Do not remove ops with attribute _grappler_do_not_remove. This is useful
    // for debugging.
    bool do_not_remove;
    if (TryGetNodeAttr(attrs, "_grappler_do_not_remove", &do_not_remove) &&
        do_not_remove) {
      result.insert(node.name());
    }
  }

  return result;
}

const std::unordered_set<string>& GrapplerItem::devices() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_7(mht_7_v, 360, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::devices");

  return devices_;
}

Status GrapplerItem::AddDevice(const string& device) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_8(mht_8_v, 368, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::AddDevice");

  DeviceNameUtils::ParsedName name;

  if (!DeviceNameUtils::ParseFullName(device, &name)) {
    return errors::InvalidArgument("Invalid device name: device=", device);

  } else if (!name.has_job || !name.has_replica || !name.has_task ||
             !name.has_type || !name.has_id) {
    return errors::InvalidArgument("Not a fully defined device name: device=",
                                   device);
  }

  devices_.insert(DeviceNameUtils::ParsedNameToString(name));
  return Status::OK();
}

Status GrapplerItem::AddDevices(const GrapplerItem& other) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_9(mht_9_v, 387, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::AddDevices");

  std::vector<absl::string_view> invalid_devices;
  for (const string& device : other.devices()) {
    Status added = AddDevice(device);
    if (!added.ok()) invalid_devices.emplace_back(device);
  }
  return invalid_devices.empty()
             ? Status::OK()
             : errors::InvalidArgument("Skipped invalid devices: [",
                                       absl::StrJoin(invalid_devices, ", "),
                                       "]");
}

Status GrapplerItem::InferDevicesFromGraph() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_10(mht_10_v, 403, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::InferDevicesFromGraph");

  absl::flat_hash_set<absl::string_view> invalid_devices;
  for (const NodeDef& node : graph.node()) {
    Status added = AddDevice(node.device());
    if (!added.ok()) invalid_devices.insert(node.device());
  }
  VLOG(2) << "Inferred device set: [" << absl::StrJoin(devices_, ", ") << "]";
  return invalid_devices.empty()
             ? Status::OK()
             : errors::InvalidArgument("Skipped invalid devices: [",
                                       absl::StrJoin(invalid_devices, ", "),
                                       "]");
}

void GrapplerItem::ClearDevices() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_11(mht_11_v, 420, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::ClearDevices");
 devices_.clear(); }

const GrapplerItem::OptimizationOptions& GrapplerItem::optimization_options()
    const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_12(mht_12_v, 426, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::optimization_options");

  return optimization_options_;
}

GrapplerItem::OptimizationOptions& GrapplerItem::optimization_options() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgrappler_itemDTcc mht_13(mht_13_v, 433, "", "./tensorflow/core/grappler/grappler_item.cc", "GrapplerItem::optimization_options");

  return optimization_options_;
}

}  // end namespace grappler
}  // end namespace tensorflow
