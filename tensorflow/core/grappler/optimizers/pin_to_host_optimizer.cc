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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/pin_to_host_optimizer.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/grappler/utils/tpu.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace grappler {
namespace internal {

// TODO(williamchan): Change this constant to be something smarter, maybe
// dynamically determined.
constexpr int64_t kTensorMaxSize = 64;

// All the nodes that should be denylisted and not swapped.
bool IsDenylisted(const NodeDef& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "IsDenylisted");

  return
      // Collective ops should not be swapped.
      IsCollective(node) ||
      // ControlFlow ops should not be swapped.
      IsControlFlow(node) ||
      // NoOp ops should not be swapped (due to group dependencies).
      IsNoOp(node);
}

// Check if Tensor is either a string or is integer and small size
bool IsTensorSmall(const OpInfo::TensorProperties& prop) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "IsTensorSmall");

  if (prop.dtype() == DataType::DT_STRING) {
    return true;
  }

  // Check type to be int32 or int64.
  if (prop.dtype() != DataType::DT_INT32 &&
      prop.dtype() != DataType::DT_INT64 &&
      prop.dtype() != DataType::DT_FLOAT) {
    return false;
  }

  // Check size known and small.
  const int64_t size = NumCoefficients(prop.shape());
  if (size < 0 || size > kTensorMaxSize) {
    return false;
  }

  return true;
}

// Find KernelDef for `node`, greedily return first found from `devices`.
Status TryFindKernelDef(const std::vector<DeviceType>& devices,
                        const NodeDef& node, const KernelDef** kdef) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "TryFindKernelDef");

  for (const DeviceType& device : devices) {
    const KernelDef* kernel = nullptr;
    Status s = FindKernelDef(device, node, &kernel, nullptr);
    if (s.ok()) {
      if (kdef) {
        *kdef = kernel;
      }
      return Status::OK();
    }
  }

  return errors::NotFound("Could not find KernelDef for op: ", node.op());
}

// Checks if a node's output port is host friendly.
// Roughly this means checking if the output port is on Host memory.
Status IsNodeOutputPortHostFriendly(const GraphView& graph,
                                    GraphProperties* properties,
                                    const NodeDef& node, int port_id,
                                    bool* is_candidate) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_3(mht_3_v, 272, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "IsNodeOutputPortHostFriendly");

  *is_candidate = false;

  // Make sure we are not a denylisted op.
  if (IsDenylisted(node)) {
    return Status::OK();
  }

  // Check to make sure we have the right properties (i.e., statically shaped).
  if (!properties->has_properties()) {
    // This is an expensive call, call it lazily.
    TF_RETURN_IF_ERROR(properties->InferStatically(
        /*assume_valid_feeds=*/false, /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/false));
  }
  const auto& output_properties = properties->GetOutputProperties(node.name());
  int output_properties_size = output_properties.size();
  if (port_id >= output_properties_size) {
    LOG(WARNING) << "port_id=" << port_id
                 << " but output_properties.size()=" << output_properties.size()
                 << "\n"
                 << node.DebugString();
    return Status::OK();
  }
  if (!IsTensorSmall(output_properties[port_id])) {
    return Status::OK();
  }

  // These nodes may be optimized away downstream (even if pinned to Host), we
  // should (recursively) check their source.
  if (IsIdentity(node) || IsIdentityNSingleInput(node)) {
    for (const auto& fanin : graph.GetFanins(node, false)) {
      bool fanin_candidate = false;
      TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
          graph, properties, *fanin.node, fanin.port_id, &fanin_candidate));
      if (!fanin_candidate) {
        return Status::OK();
      }
    }
    *is_candidate = true;
    return Status::OK();
  }

  // Check if op's device is on CPU.
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return Status::OK();
  }

  // Check if op's output port is pinned to HostMemory.
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    return Status::OK();
  }

  // Map the port_id to output_arg_id.
  const int output_arg_id = OpOutputPortIdToArgId(node, *op, port_id);
  if (output_arg_id < 0) {
    LOG(WARNING) << "Invalid port: " << port_id << "!\n"
                 << node.DebugString() << "\n"
                 << op->DebugString();
    return Status::OK();
  }

  // Find the kernel.
  const KernelDef* kernel = nullptr;
  s = TryFindKernelDef({node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node,
                       &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    return Status::OK();
  }

  // Check if the output_arg is pinned to Host.
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->output_arg(output_arg_id).name() == host_memory_arg) {
      *is_candidate = true;
      break;
    }
  }

  return Status::OK();
}

// Checks if a node's input port is Host friendly.
// Roughly this means checking if the input port is on Host memory.
bool IsNodeInputPortHostFriendly(const NodeDef& node, int port_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_4(mht_4_v, 363, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "IsNodeInputPortHostFriendly");

  // If node is on Host, assume its inputs are Host friendly.
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    return true;
  }

  // Check if op's input port is pinned to HostMemory.
  const OpDef* op = nullptr;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op);
  if (!s.ok()) {
    LOG(WARNING) << "Could not find OpDef for : " << node.op();
    return false;
  }
  const int input_arg_id = OpInputPortIdToArgId(node, *op, port_id);

  // Find the kernel.
  const KernelDef* kernel = nullptr;
  s = internal::TryFindKernelDef(
      {node.device().c_str(), DEVICE_GPU, DEVICE_CPU}, node, &kernel);
  if (!s.ok()) {
    LOG(INFO) << "Could not find KernelDef for: " << node.op();
    return false;
  }

  // Check if the input_arg is pinned to Host.
  for (const string& host_memory_arg : kernel->host_memory_arg()) {
    if (op->input_arg(input_arg_id).name() == host_memory_arg) {
      return true;
    }
  }

  return false;
}

// Checks if a node is a candidate to pin to Host.
// The rough algorithm is as follows:
// 1] Check if node is denylisted.
// 2] Check if node can run on Host.
// 3] Check all input/outputs are Host "friendly" (atm, friendly means small,
//    ints, and pinned to Host).
Status IsNodeHostCandidate(const GraphView& graph, GraphProperties* properties,
                           const NodeDef& node, bool* is_candidate) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_5(mht_5_v, 407, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "IsNodeHostCandidate");

  *is_candidate = false;

  // Check if node already on CPU.
  if (absl::StrContains(node.device(), DEVICE_CPU)) {
    *is_candidate = true;
    return Status::OK();
  }

  // Skip these node types.
  if (IsDenylisted(node)) {
    return Status::OK();
  }

  // Check the node can be run on CPU.
  Status s = TryFindKernelDef({DEVICE_CPU}, node, nullptr);
  if (!s.ok()) {
    return Status::OK();
  }

  // Check all inputs are Host friendly.
  for (const GraphView::OutputPort& fanin :
       graph.GetFanins(node, /*include_controlling_nodes=*/false)) {
    bool fanin_candidate = false;
    TF_RETURN_IF_ERROR(IsNodeOutputPortHostFriendly(
        graph, properties, *fanin.node, fanin.port_id, &fanin_candidate));
    if (!fanin_candidate) {
      return Status::OK();
    }
  }

  // Check all outputs are Host friendly.
  if (!properties->has_properties()) {
    // This is an expensive call, call it lazily.
    TF_RETURN_IF_ERROR(properties->InferStatically(
        /*assume_valid_feeds=*/false, /*aggressive_shape_inference=*/false,
        /*include_tensor_values=*/false));
  }
  for (const auto& prop : properties->GetOutputProperties(node.name())) {
    if (!IsTensorSmall(prop)) {
      return Status::OK();
    }
  }

  *is_candidate = true;
  return Status::OK();
}

// Tries to find a Host device from `devices`. Returns empty string if no
// matching Host device is found.
string TryFindHostDevice(const gtl::FlatSet<string>& devices,
                         bool has_device_cpu, const string& device) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_6(mht_6_v, 462, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "TryFindHostDevice");

  // Force this node onto the CPU.
  if (device.empty() && has_device_cpu) {
    return "/device:CPU:0";
  } else if (absl::StrContains(device, DEVICE_GPU)) {
    // Sometimes the cluster can have:
    //   devices = {"/device:CPU:0", "/device:XLA_GPU:0"}
    // and we need to handle them properly.
    for (const auto& device_match :
         {std::pair<string, string>("GPU", "CPU:0"),
          std::pair<string, string>("/device", "/device:CPU:0")}) {
      const string device_host =
          strings::StrCat(device.substr(0, device.rfind(device_match.first)),
                          device_match.second);
      if (devices.find(device_host) != devices.end()) {
        return device_host;
      }
    }
  }

  // We couldn't find an appropriate Host device, return no device.
  return "";
}
}  // end namespace internal

Status PinToHostOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                    GraphDef* optimized_graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSpin_to_host_optimizerDTcc mht_7(mht_7_v, 491, "", "./tensorflow/core/grappler/optimizers/pin_to_host_optimizer.cc", "PinToHostOptimizer::Optimize");

  *optimized_graph = item.graph;

  // Skip all TPU graphs.
  if (IsTPUGraphDef(*optimized_graph)) {
    return Status::OK();
  }

  GraphProperties properties(item);
  GraphView graph(optimized_graph);

  gtl::FlatSet<string> devices;
  if (cluster) {
    const std::vector<string> device_names = cluster->GetDeviceNames();
    devices.insert(device_names.begin(), device_names.end());
  } else {
    devices = {"/device:CPU:0"};
  }

  const bool has_device_cpu = devices.find("/device:CPU:0") != devices.end();

  // Topologically sort the graph, so that we traverse the nodes in order. This
  // will help us discover producer->consumer chains of Host ops.
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph));

  // All the Const nodes, and their original devices in topological order.
  std::vector<std::pair<NodeDef*, string>> const_nodes;

  for (auto& node : *optimized_graph->mutable_node()) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    bool is_candidate = false;
    TF_RETURN_IF_ERROR(
        internal::IsNodeHostCandidate(graph, &properties, node, &is_candidate));
    if (!is_candidate) {
      continue;
    }

    string device =
        internal::TryFindHostDevice(devices, has_device_cpu, node.device());
    if (!device.empty()) {
      // Keep track of all Const nodes that we swapped.
      if (IsConstant(node)) {
        const_nodes.emplace_back(&node, node.device());
      }
      VLOG(2) << "Moving node " << node.name() << " to device " << device;
      *node.mutable_device() = std::move(device);
    }
  }

  // Traverse all `const_nodes`, and map them back to GPU greedily.
  for (auto& it : const_nodes) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    NodeDef* node = it.first;
    const string& device = it.second;

    // Check all the consumers of this node, if any of them are not on CPU, swap
    // this node back onto the original device.
    for (const GraphView::InputPort& fanout : graph.GetFanouts(*node, false)) {
      // The consumer is not Host friendly, swap it back to the original device.
      if (!internal::IsNodeInputPortHostFriendly(*fanout.node,
                                                 fanout.port_id)) {
        VLOG(2) << "Swapping node " << node->name() << " back to device "
                << device;
        node->set_device(device);
        break;
      }
    }
  }
  return Status::OK();
}

}  // end namespace grappler
}  // end namespace tensorflow
