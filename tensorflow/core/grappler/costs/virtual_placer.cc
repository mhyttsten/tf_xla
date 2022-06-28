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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc() {
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

#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

VirtualPlacer::VirtualPlacer(
    const std::unordered_map<string, DeviceProperties>& devices)
    : devices_(devices),
      // Default job name for canonical device name. Needs to be set before the
      // first call to to_lfqn_or_empty()
      default_job_name_lowercase_("localhost") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/grappler/costs/virtual_placer.cc", "VirtualPlacer::VirtualPlacer");

  lfqn_map_.reserve(devices_.size());
  for (const auto& kv : devices_) {
    const auto lfqn = to_lfqn_or_empty(kv.first);
    if (lfqn.empty()) {
      LOG(ERROR) << "VirtualPlacer couldn't parse device name from cluster: "
                 << kv.first;
    } else {
      lfqn_map_[lfqn] = kv.first;
    }
  }

  if (devices_.empty()) {
    // If there are no devices in the cluster, add a single device, "UNKNOWN" to
    // the cluster.
    default_device_name_ = "UNKNOWN";
    DeviceProperties& prop = devices_["UNKNOWN"];
    prop.set_type("UNKNOWN");
  } else if (devices_.size() == 1) {
    // If there is only one device in the cluster, use it as default device,
    // whatever it is.
    default_device_name_ = devices_.begin()->first;
  } else {
    // Default device is set from the devices in the cluster in the following
    // priority: /gpu:0, /cpu:0, or any device.
    // TODO(dyoon): This logic assumes single machine with CPU and GPU devices.
    // Make it more general to support multiple machines, job types, and devices
    // other than CPU and GPU.
    std::map<int, string> cpu_devices;  // CPU device map: id -> device name.
    std::map<int, string> gpu_devices;  // GPU device map: id -> device name.
    for (const auto& kv : lfqn_map_) {
      const auto& lfqn = kv.first;
      const auto& cluster_device_name = kv.second;
      DeviceNameUtils::ParsedName parsed_name;
      bool parsed = DeviceNameUtils::ParseFullName(lfqn, &parsed_name);
      if (parsed) {
        // Parsed devices are stored to cpu_devices or gpu_devices map,
        // addressed (and ordered) by device id.
        const auto type = absl::AsciiStrToLower(parsed_name.type);
        if (type == "gpu") {
          gpu_devices[parsed_name.id] = cluster_device_name;
        } else if (type == "cpu") {
          cpu_devices[parsed_name.id] = cluster_device_name;
        }
      }
    }

    if (!gpu_devices.empty()) {
      // GPU:0 (or GPU with smallest device id).
      default_device_name_ = gpu_devices.begin()->second;
    } else if (!cpu_devices.empty()) {
      // CPU:0 (or CPU with smallest device id).
      default_device_name_ = cpu_devices.begin()->second;
    } else {
      default_device_name_ = devices_.begin()->first;  // Any device.
    }
  }
  VLOG(3) << "default device name: " << default_device_name_;

  // Scan the device names from the cluster, and if there is one job name used,
  // use it for canonical device name.
  std::unordered_set<string> job_names_from_cluster;
  for (const auto& device : lfqn_map_) {
    const auto& lfqn = device.first;
    DeviceNameUtils::ParsedName parsed_name;
    bool parsed = DeviceNameUtils::ParseFullName(lfqn, &parsed_name);
    if (parsed && !parsed_name.job.empty()) {
      job_names_from_cluster.insert(parsed_name.job);
      if (job_names_from_cluster.size() > 1) {
        break;
      }
    }
  }
  // If there is only one type of job name in all the devices in the cluster,
  // use that one as default job name; otherwise, use localhost.
  // TODO(dyoon): this should be improved, especially when the cluster is
  // composed of multiple worker, PS, and other types of jobs.
  if (job_names_from_cluster.size() == 1) {
    auto it = job_names_from_cluster.begin();
    default_job_name_lowercase_ = *it;
  }
  VLOG(3) << "default job name: " << default_job_name_lowercase_;
}

const DeviceProperties& VirtualPlacer::get_device(const NodeDef& node) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc mht_1(mht_1_v, 287, "", "./tensorflow/core/grappler/costs/virtual_placer.cc", "VirtualPlacer::get_device");

  string device = get_canonical_device_name(node);
  VLOG(3) << "node.name=" << node.name() << " node.device=" << node.device()
          << " is placed on: " << device;
  auto it = devices_.find(device);
  DCHECK(it != devices_.end());
  return it->second;
}

string VirtualPlacer::get_canonical_device_name(const NodeDef& node) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/grappler/costs/virtual_placer.cc", "VirtualPlacer::get_canonical_device_name");

  if (node.device().empty()) {
    return default_device_name_;
  }

  const auto lfqn = to_lfqn_or_empty(node.device());
  if (lfqn.empty()) {
    return default_device_name_;
  }

  const auto it = lfqn_map_.find(lfqn);
  if (it != lfqn_map_.end()) {
    return it->second;
  }

  return default_device_name_;
}

string VirtualPlacer::to_lfqn_or_empty(const string& device_name) const {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_placerDTcc mht_3(mht_3_v, 321, "", "./tensorflow/core/grappler/costs/virtual_placer.cc", "VirtualPlacer::to_lfqn_or_empty");

  DeviceNameUtils::ParsedName parsed_name;
  const auto lowercase_name = absl::AsciiStrToLower(device_name);
  bool parsed = DeviceNameUtils::ParseFullName(lowercase_name, &parsed_name);
  if (!parsed) {
    parsed = DeviceNameUtils::ParseLocalName(lowercase_name, &parsed_name);
    parsed_name.job = "localhost";
  }
  if (!parsed) {
    if (lowercase_name == "gpu" || lowercase_name == "cpu") {
      parsed_name.job = "localhost";
      parsed_name.type = lowercase_name;
      parsed = true;
    }
  }
  if (!parsed) {
    return {};
  }

  if (parsed_name.job.empty()) {
    parsed_name.job = default_job_name_lowercase_;
  }

  // Have to do this, because parser returns uppercase types for CPU and GPU.
  parsed_name.type = absl::AsciiStrToLower(parsed_name.type);

  string lfqn = strings::StrCat(
      "/job:", parsed_name.job, "/replica:", parsed_name.replica,
      "/task:", parsed_name.task, "/device:", parsed_name.type, ":",
      parsed_name.id);
  return lfqn;
}

}  // end namespace grappler
}  // end namespace tensorflow
