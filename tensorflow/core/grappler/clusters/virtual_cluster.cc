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
class MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc() {
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

#include "tensorflow/core/grappler/clusters/virtual_cluster.h"

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"

namespace tensorflow {
namespace grappler {

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices)
    : VirtualCluster(devices, absl::make_unique<OpLevelCostEstimator>(),
                     ReadyNodeManagerFactory("FirstReady")) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::VirtualCluster");
}

VirtualCluster::VirtualCluster(
    const std::unordered_map<string, DeviceProperties>& devices,
    std::unique_ptr<OpLevelCostEstimator> node_estimator,
    std::unique_ptr<ReadyNodeManager> node_manager)
    : Cluster(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::VirtualCluster");

  devices_ = devices;

  // Note that we do not use aggressive shape inference to preserve unknown
  // shapes from the input graph.
  estimator_ = absl::make_unique<AnalyticalCostEstimator>(
      this, std::move(node_estimator), std::move(node_manager),
      /*use_static_shapes=*/true, /*use_aggressive_shape_inference=*/false);
}

VirtualCluster::VirtualCluster(const DeviceSet* device_set)
    : VirtualCluster(std::unordered_map<string, DeviceProperties>()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::VirtualCluster");

  device_set_ = device_set;
  for (const auto& device : device_set_->devices()) {
    DeviceProperties props = GetDeviceInfo(device->parsed_name());
    if (props.type() == "UNKNOWN") continue;
    auto attrs = device->attributes();
    props.set_memory_size(attrs.memory_limit());
    devices_[device->name()] = props;
  }
}

VirtualCluster::~VirtualCluster() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::~VirtualCluster");
}

Status VirtualCluster::Provision() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_4(mht_4_v, 241, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::Provision");
 return Status::OK(); }

Status VirtualCluster::Initialize(const GrapplerItem& item) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_5(mht_5_v, 246, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::Initialize");

  return Status::OK();
}

Status VirtualCluster::Run(const GraphDef& graph,
                           const std::vector<std::pair<string, Tensor>>& feed,
                           const std::vector<string>& fetch,
                           RunMetadata* metadata) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_6(mht_6_v, 256, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::Run");

  GrapplerItem item;
  item.graph = graph;
  item.feed = feed;
  item.fetch = fetch;
  return Run(item, metadata);
}

Status VirtualCluster::Run(const GrapplerItem& item, RunMetadata* metadata) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSvirtual_clusterDTcc mht_7(mht_7_v, 267, "", "./tensorflow/core/grappler/clusters/virtual_cluster.cc", "VirtualCluster::Run");

  // Initializes an analytical cost estimator to estimate the graph cost. Makes
  // sure to use static shape inference to prevent the virtual scheduler from
  // calling the Run method on the cluster and creating an infinite loop.
  if (metadata) {
    metadata->clear_step_stats();
    metadata->clear_cost_graph();
    metadata->clear_partition_graphs();
  }

  TF_RETURN_IF_ERROR(estimator_->Initialize(item));
  TF_RETURN_IF_ERROR(
      estimator_->PredictCosts(item.graph, metadata, /*cost=*/nullptr));

  const std::unordered_map<string, DeviceProperties>& device = GetDevices();
  std::unordered_map<string, int64_t> peak_mem_usage =
      estimator_->GetScheduler()->GetPeakMemoryUsage();
  for (const auto& mem_usage : peak_mem_usage) {
    const string& device_name = mem_usage.first;
    auto it = device.find(device_name);
    if (it == device.end()) {
      // It's probably the fake send/recv device. Eventually we'll need to
      // remove this fake device to ensure proper memory accounting for
      // multi-device settings.
      continue;
    }
    const DeviceProperties& dev = it->second;
    if (dev.memory_size() <= 0) {
      // Available device memory unknown
      continue;
    }
    int64_t peak_mem = mem_usage.second;
    if (peak_mem >= dev.memory_size()) {
      return errors::ResourceExhausted(
          "Graph requires ", peak_mem, " bytes of memory on device ",
          device_name, " to run ", " but device only has ", dev.memory_size(),
          " available.");
    }
  }

  return Status::OK();
}

}  // namespace grappler
}  // namespace tensorflow
