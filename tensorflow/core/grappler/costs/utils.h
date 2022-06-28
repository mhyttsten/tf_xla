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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh() {
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
#include <vector>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

// Returns a vector of InputProperties for 'node'. The vector will contain one
// entry for each input of 'node'.
// For each node in the graph, the 'name_to_cost' map stores a pointer to the
// corresponding cost graph node indexed by node name. The 'name_to_node' maps a
// node name to its node definition.
std::vector<OpInfo::TensorProperties> FindInputFeatures(
    const NodeDef& node,
    const std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    const std::unordered_map<string, const NodeDef*>& name_to_node);

// Returns the size of tensor (unit: bytes). For tensor shape with unknown rank,
// it assumes the tensor to be scalar. For any unknown dimension, it assumes
// size one.
int64_t CalculateTensorSize(const OpInfo::TensorProperties& prop);

// Returns the size of output at port_num (unit: bytes). A special case is
// port_num -1, which is for control dependency and assumed to be 4 bytes.
int64_t CalculateOutputSize(
    const std::vector<OpInfo::TensorProperties>& output_properties,
    int port_num);

// Returns the DeviceProperties of the device on which 'node' runs.
DeviceProperties GetDeviceInfo(const CostGraphDef::Node& node);
DeviceProperties GetDeviceInfo(const string& device_str);

// Return a string describing a node given a nodeinfo.
string GetOpDescription(const OpInfo& op_info);

// Builds the OpInfo for node without filling its device information, given all
// nodes in the graph and its input properties.
OpInfo BuildOpInfoWithoutDevice(
    const NodeDef& node,
    const std::unordered_map<string, const NodeDef*>& name_to_node,
    const std::vector<OpInfo::TensorProperties>& inputs);

// Gather performance data from a cost graph.
OpPerformanceList CostGraphToOpPerformanceData(const CostGraphDef& cost_graph,
                                               const GraphDef& graph);

// Simple histogram for profiling Tensor size; histogram uses logarithmic
// buckets.
class TensorSizeHistogram {
 public:
  TensorSizeHistogram() : buckets_(kMaxBuckets, 0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_0(mht_0_v, 249, "", "./tensorflow/core/grappler/costs/utils.h", "TensorSizeHistogram");
}

  void Add(const uint64 value);
  void Merge(const TensorSizeHistogram& src);
  double Average() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_1(mht_1_v, 256, "", "./tensorflow/core/grappler/costs/utils.h", "Average");

    if (num_elem_ > 0) {
      return static_cast<double>(sum_elem_) / num_elem_;
    } else {
      return 0.0;
    }
  }
  uint64 Min() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_2(mht_2_v, 266, "", "./tensorflow/core/grappler/costs/utils.h", "Min");
 return min_; }
  uint64 Max() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_3(mht_3_v, 270, "", "./tensorflow/core/grappler/costs/utils.h", "Max");
 return max_; }
  uint64 NumElem() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_4(mht_4_v, 274, "", "./tensorflow/core/grappler/costs/utils.h", "NumElem");
 return num_elem_; }
  uint64 SumElem() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_5(mht_5_v, 278, "", "./tensorflow/core/grappler/costs/utils.h", "SumElem");
 return sum_elem_; }
  string ToString() const;

 protected:
  const int Index(const uint64 value) const;
  const std::vector<uint64>& GetBuckets() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSutilsDTh mht_6(mht_6_v, 286, "", "./tensorflow/core/grappler/costs/utils.h", "GetBuckets");
 return buckets_; }

 private:
  const int kMaxBuckets = 64;
  uint64 num_elem_ = 0;
  uint64 sum_elem_ = 0;
  // min_ and max_ are initialized to a very large value and zero, respectively,
  // so that any value added can replace the initial min_ and max_.
  uint64 min_ = kuint64max;
  uint64 max_ = 0;
  // Buckets are logarithmic:
  // 0B, 1B, 2-3B, 4-7B, 8-15B, ..., 2^N - 2^(N+1)-1B, ...
  std::vector<uint64> buckets_;
};

// Helper functions for aggregating per-device stats into per-device-class
// stats.
string GetDeviceClassForNonChannelDevice(const string& device_name);
string GetDeviceClass(const string& device_name);

// Get stats in string format from RunMetadata.
string GetStatsStringFromRunMetadata(const RunMetadata& run_metadata,
                                     bool verbosity);

// This method calculates the execution time depending on whether IO can
// overlap with computation. It assumes the memory and the compute times have
// already been calculated.
void CombineCostsAndUpdateExecutionTime(bool compute_memory_overlap,
                                        Costs* costs);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_UTILS_H_
