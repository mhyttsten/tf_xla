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

#ifndef TENSORFLOW_CORE_GRAPPLER_CLUSTERS_CLUSTER_H_
#define TENSORFLOW_CORE_GRAPPLER_CLUSTERS_CLUSTER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh() {
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


#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace grappler {

// A cluster represents of collection of hardware resources available to run
// the TensorFlow model.
// A process can only create a single cluster at a time.
class Cluster {
 public:
  explicit Cluster(int timeout_s);
  virtual ~Cluster();

  // Returns a string that represent the type of cluster that was instantiated.
  virtual string type() const = 0;

  // Provision the hardware resources needed to run TensorFlow and start a
  // TensorFlow session that can take advantage of these resources.
  // The actual resources that are leveraged depend on the type of cluster
  // instantiated.
  // Returns OK iff all the requested resources could be reserved and a
  // TensorFlow session successfully created. Returns an error otherwise.
  // There is no graceful degradation to handle the case where only a subset
  // of the requested resources are available.
  virtual Status Provision() = 0;

  // Attempts to shutdown the cluster.
  // Returns OK iff there are no pending calls to the Run() method and all the
  // resources used by the cluster could be released. Returns an error
  // otherwise.
  virtual Status Shutdown() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh mht_0(mht_0_v, 229, "", "./tensorflow/core/grappler/clusters/cluster.h", "Shutdown");
 return Status::OK(); }

  // Whether soft placement is allowed. If allow_soft_placement is true,
  // an op will be placed on CPU if there's no GPU implementation for the OP
  // or if no GPU devices are known or registered or if we need to co-locate
  // with reftype input(s) which are from CPU.
  void AllowSoftPlacement(bool soft_placement_state);

  // Update the number of inter-op threads for each per-session threadpool
  void SetNumInterOpThreads(int num_threads);

  // Set the number of steps required to warmup TensorFlow. Must be called
  // before Provision().
  void SetNumWarmupSteps(int num_steps);

  // Set executor type to instantiate
  void SetExecutorType(const string* executor_type);

  // Returns the number of warmup steps.
  int NumWarmupSteps() const;

  // Disable the collection of detailed statistics. Must be called
  // before Provision().
  void DisableDetailedStats(bool disable);

  // Returns true iff the collection of detailed statistics is enabled.
  bool DetailedStatsEnabled() const;

  // Disable the TensorFlow optimizer. This ensures that the graph that TF
  // executes is similar to the input graph. Must be called before Provision().
  void DisableOptimizer(bool disable);

  // Return the list of TensorFlow devices that are available to execute a
  // graph. This is empty until provision() is called.
  const std::unordered_map<string, DeviceProperties>& GetDevices() const {
    return devices_;
  }

  // Convenience method that returns the set of device names. These names are
  // sorted alphabetically.
  const std::vector<string> GetDeviceNames() const;

  // The DeviceSet is not always available, but when it is it contains a
  // superset of the devices listed in GetDevices/GetDeviceNames().
  virtual const DeviceSet* GetDeviceSet() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh mht_1(mht_1_v, 276, "", "./tensorflow/core/grappler/clusters/cluster.h", "GetDeviceSet");
 return nullptr; }

  // Enables collecting the allocator stats. If called, must be called before
  // Provision().
  virtual Status EnablePeakMemoryStats() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh mht_2(mht_2_v, 283, "", "./tensorflow/core/grappler/clusters/cluster.h", "EnablePeakMemoryStats");

    return errors::Unimplemented(strings ::StrCat(
        "Peak Memory Stats are not supported on ", type(), " clusters"));
  }

  // Returns peak memory of all devices during the session creation and session
  // runs.
  virtual Status GetPeakMemoryUsage(
      std::unordered_map<string, uint64>* device_peak_memory) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh mht_3(mht_3_v, 294, "", "./tensorflow/core/grappler/clusters/cluster.h", "GetPeakMemoryUsage");

    return errors::Unimplemented(
        "GetPeakMemoryUsage is not implemented for this type of cluster.");
  }

  // Prepare the session to run the specified grappler item. This include
  // initializing all the model variables.
  virtual Status Initialize(const GrapplerItem& item) = 0;

  // Run the specified graph_def and return the corresponding metadata.
  virtual Status Run(const GraphDef& graph_def,
                     const std::vector<std::pair<string, Tensor>>& feed,
                     const std::vector<string>& fetch,
                     RunMetadata* metadata) = 0;

  // Run the specified GrapplerItem and return the corresponding metadata.
  virtual Status Run(const GrapplerItem& item, RunMetadata* metadata) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSclustersPSclusterDTh mht_4(mht_4_v, 313, "", "./tensorflow/core/grappler/clusters/cluster.h", "Run");

    return Run(item.graph, item.feed, item.fetch, metadata);
  }

 protected:
  std::unordered_map<string, DeviceProperties> devices_;
  const int timeout_s_;
  SessionOptions options_;
  RunOptions run_options_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_CLUSTERS_CLUSTER_H_
