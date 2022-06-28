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
class MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Helper functions for TPU rewrite passes.

#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"

#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// LINT.IfChange
Status DistributedTPURewriteHelpers::GetSystemDevice(
    const string& system_spec_string, const DeviceSet& device_set,
    DeviceNameUtils::ParsedName* system_spec, Device** system_device) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("system_spec_string: \"" + system_spec_string + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.cc", "DistributedTPURewriteHelpers::GetSystemDevice");

  if (!DeviceNameUtils::ParseFullName(system_spec_string, system_spec)) {
    system_spec->Clear();
  }

  // Callers may have relied on an Op only being registered on TPU_SYSTEM
  // devices to ensure the Op is placed there. Augment the device spec to make
  // the device type explicit.
  if (!system_spec->has_type || system_spec->type != DEVICE_TPU_SYSTEM) {
    system_spec->type = DEVICE_TPU_SYSTEM;
    system_spec->has_type = true;
    system_spec->id = 0;
    system_spec->has_id = true;
  }

  std::vector<Device*> system_devices;
  device_set.FindMatchingDevices(*system_spec, &system_devices);
  if (system_devices.empty()) {
    if (system_spec_string.empty()) {
      return errors::InvalidArgument(
          "No TPU_SYSTEM device found. Please ensure that you're connected to "
          "a host with a TPU_SYSTEM device.");
    }
    return errors::InvalidArgument("No matching devices found for '",
                                   system_spec_string, "'");
  } else if (system_devices.size() > 1) {
    // Validate that all system devices are part of the same job.
    std::unordered_set<string> job_names;
    for (auto device : system_devices) {
      const auto& parsed_name = device->parsed_name();
      TF_RET_CHECK(parsed_name.has_job);
      job_names.insert(parsed_name.job);
    }
    if (job_names.size() > 1) {
      return errors::InvalidArgument(
          "System devices cannot be part "
          "of multiple different jobs.  Found: ",
          str_util::Join(job_names, ","));
    }

    // Identify the lexicographically first device from the list of
    // valid TPU SYSTEM devices, so that every process in the same
    // 'cluster' definition uses the same system device.
    std::sort(system_devices.begin(), system_devices.end(),
              [](Device* i, Device* j) {
                auto i_name = i->parsed_name();
                auto j_name = j->parsed_name();
                if (i_name.replica != j_name.replica) {
                  return i_name.replica < j_name.replica;
                }
                return i_name.task < j_name.task;
              });
  }

  *system_device = system_devices[0];
  if (!DeviceNameUtils::ParseFullName((*system_device)->name(), system_spec)) {
    return errors::InvalidArgument("Unable to re-parse system device name ",
                                   (*system_device)->name(),
                                   " as a device spec.");
  }
  return Status::OK();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

// LINT.IfChange
Status DistributedTPURewriteHelpers::GetHostSystemDevices(
    const DeviceNameUtils::ParsedName& system_spec, const DeviceSet& device_set,
    std::vector<Device*>* host_system_devices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc mht_1(mht_1_v, 271, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.cc", "DistributedTPURewriteHelpers::GetHostSystemDevices");

  DeviceNameUtils::ParsedName host_spec;
  if (system_spec.has_job) {
    // The system Op has been explicitly assigned to a job, so we want
    // all the hosts in that job.
    CHECK(DeviceNameUtils::ParseFullName(
        strings::StrCat("/job:", system_spec.job, "/device:", DEVICE_TPU_SYSTEM,
                        ":0"),
        &host_spec));
  } else {
    // The system Op has not been explicitly assigned to a
    // job, so take all hosts in the system. There will be a runtime
    // error if some of those hosts don't contain TPU devices.
    CHECK(DeviceNameUtils::ParseFullName(
        strings::StrCat("/device:", DEVICE_TPU_SYSTEM, ":0"), &host_spec));
  }
  device_set.FindMatchingDevices(host_spec, host_system_devices);

  TF_RET_CHECK(!host_system_devices->empty())
      << "No hosts found matching device spec "
      << DeviceNameUtils::ParsedNameToString(host_spec);

  // Check that all the devices belong to the same job.
  TF_RET_CHECK((*host_system_devices)[0]->parsed_name().has_job);
  const string& job_name = (*host_system_devices)[0]->parsed_name().job;
  int replica = (*host_system_devices)[0]->parsed_name().replica;
  for (const auto host_device : *host_system_devices) {
    const auto& parsed_name = host_device->parsed_name();
    TF_RET_CHECK(parsed_name.has_job);
    if (parsed_name.job != job_name) {
      return errors::InvalidArgument(
          "All TPU host devices must be in the same job");
    }
    TF_RET_CHECK(parsed_name.has_replica);
    if (parsed_name.replica != replica) {
      return errors::InvalidArgument(
          "All TPU host devices must be in the same replica");
    }
  }

  // Sort the devices by replica and then task.
  std::sort(host_system_devices->begin(), host_system_devices->end(),
            [](Device* i, Device* j) {
              auto i_name = i->parsed_name();
              auto j_name = j->parsed_name();
              return i_name.task < j_name.task;
            });
  return Status::OK();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

// LINT.IfChange
Status DistributedTPURewriteHelpers::GetTPUDevices(
    const DeviceNameUtils::ParsedName& system_spec, const DeviceSet& device_set,
    int* num_tpus_per_host, std::vector<std::vector<Device*>>* tpu_devices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc mht_2(mht_2_v, 328, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.cc", "DistributedTPURewriteHelpers::GetTPUDevices");

  // GetHostSystemDevices returns the CPU device on each host that is
  // going to be used for executing TPU code.
  std::vector<Device*> host_system_devices;
  TF_RETURN_IF_ERROR(DistributedTPURewriteHelpers::GetHostSystemDevices(
      system_spec, device_set, &host_system_devices));

  // Enumerate all the physical devices. Enumerate devices on task 0,
  // then task 1, etc.
  std::sort(host_system_devices.begin(), host_system_devices.end(),
            [](Device* i, Device* j) {
              return i->parsed_name().task < j->parsed_name().task;
            });

  *num_tpus_per_host = 0;
  tpu_devices->clear();
  tpu_devices->reserve(host_system_devices.size());
  for (const auto device : host_system_devices) {
    // Make a copy of the parsed name because we are going to change it.
    DeviceNameUtils::ParsedName device_spec = device->parsed_name();
    device_spec.has_type = true;
    device_spec.type = "TPU";
    // Enumerate all the available TPUs.
    device_spec.has_id = false;
    std::vector<Device*> host_tpu_devices;
    device_set.FindMatchingDevices(device_spec, &host_tpu_devices);
    // Sort the devices by device id.
    std::sort(host_tpu_devices.begin(), host_tpu_devices.end(),
              [](Device* i, Device* j) {
                return i->parsed_name().id < j->parsed_name().id;
              });
    if (tpu_devices->empty()) {
      // First iteration: set *num_tpus_per_host to the number of TPUs on the
      // first host.
      *num_tpus_per_host = host_tpu_devices.size();
    } else if (*num_tpus_per_host != host_tpu_devices.size()) {
      // Subsequent iterations: check the number of TPUs match the number on
      // the first host.
      return errors::InvalidArgument(
          "Mismatched number of TPU devices in cluster ", *num_tpus_per_host,
          " vs. ", host_tpu_devices.size());
    }
    tpu_devices->push_back(std::move(host_tpu_devices));
  }
  return Status::OK();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

Status DistributedTPURewriteHelpers::ForConfigurationNodeMatchingType(
    const string& node_type, Graph* graph, const DeviceSet& device_set,
    const std::function<
        Status(const NodeDef& configuration_node_def,
               const string& configuration_device_name,
               const std::vector<Device*>& host_devices,
               const std::vector<Node*>& input_dependencies,
               const std::vector<OutputDependency>& output_dependencies,
               Graph* graph)>& action) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("node_type: \"" + node_type + "\"");
   mht_3_v.push_back("configuration_device_name: \"" + configuration_device_name + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSgraph_rewritePSdistributed_tpu_rewrite_helpersDTcc mht_3(mht_3_v, 389, "", "./tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.cc", "DistributedTPURewriteHelpers::ForConfigurationNodeMatchingType");

  // Find all the matching nodes before mutating the graph.
  std::vector<Node*> nodes;
  for (Node* node : graph->nodes()) {
    if (node->type_string() == node_type) {
      nodes.push_back(node);
    }
  }

  for (Node* node : nodes) {
    string spec_string = node->requested_device();
    DeviceNameUtils::ParsedName spec;
    Device* device;
    TF_RETURN_IF_ERROR(
        GetSystemDevice(spec_string, device_set, &spec, &device));
    const string& device_name = device->name();

    std::vector<Device*> host_devices;
    TF_RETURN_IF_ERROR(GetHostSystemDevices(spec, device_set, &host_devices));

    std::vector<Node*> input_dependencies;
    for (const Edge* edge : node->in_edges()) {
      // Config ops have no inputs, so all edges must be control edges.
      CHECK(edge->IsControlEdge());
      input_dependencies.push_back(edge->src());
    }
    std::vector<OutputDependency> output_dependencies;
    for (const Edge* edge : node->out_edges()) {
      OutputDependency dep;
      dep.src_output = edge->src_output();
      dep.dst = edge->dst();
      dep.dst_input = edge->dst_input();
      output_dependencies.push_back(dep);
    }
    NodeDef node_def = node->def();

    // Remove the node now so we can insert a new node with the same
    // name inside the action.
    graph->RemoveNode(node);

    TF_RETURN_IF_ERROR(action(node_def, device_name, host_devices,
                              input_dependencies, output_dependencies, graph));
  }

  return Status::OK();
}

}  // namespace tensorflow
