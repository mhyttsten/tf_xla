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
class MHTracer_DTPStensorflowPSpythonPSgrapplerPScluster_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSgrapplerPScluster_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSgrapplerPScluster_wrapperDTcc() {
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

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/graph_memory.h"
#include "tensorflow/core/grappler/costs/measuring_cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_level_cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

tensorflow::Status _GetOpPerformanceDataAndRunTime(
    const tensorflow::grappler::GrapplerItem& item,
    tensorflow::grappler::CostEstimator* cost_measure,
    tensorflow::OpPerformanceList* op_performance_data,
    tensorflow::grappler::Costs* costs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSgrapplerPScluster_wrapperDTcc mht_0(mht_0_v, 225, "", "./tensorflow/python/grappler/cluster_wrapper.cc", "_GetOpPerformanceDataAndRunTime");

  tensorflow::Status status = cost_measure->Initialize(item);
  if (!status.ok()) return status;

  tensorflow::RunMetadata run_metadata;
  MaybeRaiseRegisteredFromStatus(
      cost_measure->PredictCosts(item.graph, &run_metadata, costs));

  if (op_performance_data) {
    *op_performance_data = tensorflow::grappler::CostGraphToOpPerformanceData(
        run_metadata.cost_graph(), item.graph);
  }
  return tensorflow::Status::OK();
}

PYBIND11_MAKE_OPAQUE(tensorflow::grappler::Cluster);

PYBIND11_MODULE(_pywrap_tf_cluster, m) {
  py::class_<tensorflow::grappler::Cluster> grappler_cluster(
      m, "tensorflow::grappler::Cluster");

  m.def("TF_NewCluster",
        [](bool allow_soft_placement,
           bool disable_detailed_stats) -> tensorflow::grappler::Cluster* {
          // TODO(petebu): Make these named arguments with default values
          // instead.
          int num_cpu_cores =
              tensorflow::grappler::GetNumAvailableLogicalCPUCores();
          int num_gpus = tensorflow::grappler::GetNumAvailableGPUs();
          int timeout_s = 60 * 10;
          std::unique_ptr<tensorflow::grappler::Cluster> cluster =
              std::make_unique<tensorflow::grappler::SingleMachine>(
                  timeout_s, num_cpu_cores, num_gpus);
          cluster->DisableDetailedStats(disable_detailed_stats);
          cluster->AllowSoftPlacement(allow_soft_placement);
          cluster->SetNumWarmupSteps(10);
          MaybeRaiseRegisteredFromStatus(cluster->Provision());
          return cluster.release();
        });

  m.def("TF_NewVirtualCluster",
        [](const std::vector<py::bytes>& serialized_named_devices)
            -> tensorflow::grappler::Cluster* {
          std::vector<tensorflow::NamedDevice> named_devices;
          for (const auto& s : serialized_named_devices) {
            tensorflow::NamedDevice named_device;
            if (!named_device.ParseFromString(std::string(s))) {
              throw std::invalid_argument(
                  "The NamedDevice could not be parsed as a valid protocol "
                  "buffer");
            }
            named_devices.push_back(named_device);
          }

          std::unordered_map<std::string, tensorflow::DeviceProperties> devices;
          for (const auto& named_device : named_devices) {
            devices[named_device.name()] = named_device.properties();
          }
          std::unique_ptr<tensorflow::grappler::Cluster> cluster =
              std::make_unique<tensorflow::grappler::VirtualCluster>(devices);
          {
            // TODO(petebu): Do we need to hold the GIL here?
            py::gil_scoped_acquire acquire;
            MaybeRaiseRegisteredFromStatus(cluster->Provision());
          }
          return cluster.release();
        });

  m.def("TF_ShutdownCluster", [](tensorflow::grappler::Cluster* cluster) {
    // TODO(petebu): Do we need to hold the GIL here?
    py::gil_scoped_acquire acquire;
    cluster->Shutdown();
  });

  m.def("TF_ListDevices",
        [](tensorflow::grappler::Cluster* cluster) -> std::vector<py::bytes> {
          const std::unordered_map<std::string, tensorflow::DeviceProperties>&
              devices = cluster->GetDevices();
          std::vector<py::bytes> named_devices;
          for (auto& dev : devices) {
            tensorflow::NamedDevice d;
            d.set_name(dev.first);
            *d.mutable_properties() = dev.second;
            named_devices.push_back(d.SerializeAsString());
          }
          return named_devices;
        });

  m.def("TF_ListAvailableOps", []() -> std::vector<std::string> {
    tensorflow::OpRegistry* registry = tensorflow::OpRegistry::Global();
    std::vector<tensorflow::OpDef> ops;
    registry->GetRegisteredOps(&ops);
    std::vector<std::string> op_names;
    op_names.reserve(ops.size());
    for (const tensorflow::OpDef& op : ops) {
      op_names.push_back(op.name());
    }
    std::sort(op_names.begin(), op_names.end());
    return op_names;
  });

  m.def(
      "TF_GetSupportedDevices",
      [](tensorflow::grappler::Cluster* cluster,
         tensorflow::grappler::GrapplerItem* item)
          -> std::unordered_map<std::string, std::vector<std::string>> {
        if (cluster == nullptr || item == nullptr) {
          MaybeRaiseRegisteredFromStatus(tensorflow::Status(
              tensorflow::errors::Internal("You need both a cluster and an "
                                           "item to get supported devices.")));
        }
        const std::unordered_map<std::string, tensorflow::DeviceProperties>&
            devices = cluster->GetDevices();
        std::unordered_map<std::string, std::vector<std::string>> device_types;
        for (const auto& dev : devices) {
          device_types[dev.second.type()].push_back(dev.first);
        }

        std::unordered_map<std::string, std::set<std::string>>
            supported_device_types;
        std::unordered_map<std::string, std::set<std::string>>
            device_restrictions;

        for (const auto& node : item->graph.node()) {
          for (const auto& dev : device_types) {
            const std::string& type = dev.first;
            if (cluster->type() != "single_machine") {
              // The actual kernel may not be linked in this binary.
              supported_device_types[node.name()].insert(type);
            } else {
              // Check the kernel capabilities
              const tensorflow::DeviceType dev_type(type);
              tensorflow::Status s =
                  tensorflow::FindKernelDef(dev_type, node, nullptr, nullptr);
              if (s.ok()) {
                supported_device_types[node.name()].insert(type);

                // Check which inputs are restricted to reside on the host.
                // TODO: extends this to support outputs as well
                tensorflow::MemoryTypeVector inp_mtypes;
                tensorflow::MemoryTypeVector out_mtypes;
                tensorflow::Status s = tensorflow::MemoryTypesForNode(
                    tensorflow::OpRegistry::Global(), dev_type, node,
                    &inp_mtypes, &out_mtypes);
                if (s.ok()) {
                  for (size_t i = 0; i < inp_mtypes.size(); ++i) {
                    if (inp_mtypes[i] == tensorflow::HOST_MEMORY) {
                      device_restrictions[tensorflow::grappler::NodeName(
                                              node.input(i))]
                          .insert("CPU");
                      break;
                    }
                  }
                }
              }
            }
          }
        }

        std::unordered_map<std::string, std::vector<std::string>> result;
        for (const auto& supported_dev : supported_device_types) {
          const std::string& node = supported_dev.first;
          std::set<std::string> feasible;
          const auto it = device_restrictions.find(node);
          if (it != device_restrictions.end()) {
            const std::set<std::string>& candidates = supported_dev.second;
            const std::set<std::string>& valid = it->second;
            std::set_intersection(candidates.begin(), candidates.end(),
                                  valid.begin(), valid.end(),
                                  std::inserter(feasible, feasible.begin()));
          } else {
            feasible = supported_dev.second;
          }

          std::vector<std::string> device_names;
          for (const std::string& type : feasible) {
            auto it = device_types.find(type);
            DCHECK(it != device_types.end());
            for (const std::string& name : it->second) {
              device_names.push_back(name);
            }
          }
          result[node] = device_names;
        }
        return result;
      });

  m.def("TF_EstimatePerformance", [](const py::bytes& serialized_device) {
    tensorflow::NamedDevice device;
    if (!device.ParseFromString(std::string(serialized_device))) {
      throw std::invalid_argument(
          "The NamedDevice could not be parsed as a valid protocol buffer");
    }
    tensorflow::grappler::OpLevelCostEstimator estimator;
    tensorflow::grappler::DeviceInfo info =
        estimator.GetDeviceInfo(device.properties());
    return info.gigaops;
  });

  m.def("TF_MeasureCosts",
        [](tensorflow::grappler::GrapplerItem* item,
           tensorflow::grappler::Cluster* cluster, bool generate_timeline)
            -> std::tuple<std::vector<py::bytes>, double, py::bytes> {
          const int num_measurements = cluster->type() == "virtual" ? 1 : 10;
          tensorflow::grappler::MeasuringCostEstimator cost_measure(
              cluster, num_measurements, 0);

          tensorflow::OpPerformanceList op_performance_data;
          tensorflow::grappler::Costs costs;
          tensorflow::Status s = _GetOpPerformanceDataAndRunTime(
              *item, &cost_measure, &op_performance_data, &costs);
          double run_time = FLT_MAX;
          if (s.ok()) {
            run_time = static_cast<double>(costs.execution_time.count()) / 1e9;
          }
          tensorflow::StepStats step_stats;
          if (generate_timeline) {
            tensorflow::RunMetadata metadata;
            MaybeRaiseRegisteredFromStatus(
                cluster->Run(item->graph, item->feed, item->fetch, &metadata));
            step_stats = metadata.step_stats();
          }

          std::vector<py::bytes> op_perf_objs;
          op_perf_objs.resize(op_performance_data.op_performance_size());
          for (int i = 0; i < op_performance_data.op_performance_size(); i++) {
            op_perf_objs[i] =
                op_performance_data.op_performance(i).SerializeAsString();
          }

          py::bytes step_stats_str = step_stats.SerializeAsString();
          return std::make_tuple(op_perf_objs, run_time, step_stats_str);
        });

  using DurationType = tensorflow::grappler::Costs::Duration::rep;
  using MemoryUsage =
      std::tuple<std::string, int, size_t, DurationType, DurationType>;

  m.def(
      "TF_DeterminePeakMemoryUsage",
      [](tensorflow::grappler::GrapplerItem* item,
         tensorflow::grappler::Cluster* cluster)
          -> std::unordered_map<std::string,
                                std::tuple<int64_t, std::vector<MemoryUsage>>> {
        if (item == nullptr || cluster == nullptr) {
          MaybeRaiseRegisteredFromStatus(
              tensorflow::Status(tensorflow::errors::Internal(
                  "You need both a cluster and an item to determine peak "
                  "memory usage.")));
        }
        tensorflow::grappler::GraphMemory memory(*item);

        if (cluster->DetailedStatsEnabled()) {
          MaybeRaiseRegisteredFromStatus(memory.InferDynamically(cluster));
        } else {
          MaybeRaiseRegisteredFromStatus(
              memory.InferStatically(cluster->GetDevices()));
        }

        std::unordered_map<std::string,
                           std::tuple<int64_t, std::vector<MemoryUsage>>>
            result;
        for (const auto& device : cluster->GetDevices()) {
          const tensorflow::grappler::GraphMemory::MemoryUsage& usage =
              memory.GetPeakMemoryUsage(device.first);
          std::vector<MemoryUsage> per_device;
          for (size_t i = 0; i < usage.live_tensors.size(); ++i) {
            const auto& live_tensor = usage.live_tensors[i];
            per_device.push_back(std::make_tuple(
                live_tensor.node, live_tensor.output_id,
                live_tensor.memory_used, live_tensor.allocation_time.count(),
                live_tensor.deallocation_time.count()));
          }
          result[device.first] = std::make_tuple(usage.used_memory, per_device);
        }
        return result;
      });
}
