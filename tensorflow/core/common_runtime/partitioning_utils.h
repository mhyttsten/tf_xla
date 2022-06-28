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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PARTITIONING_UTILS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PARTITIONING_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTh() {
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


#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Given a `device_set` and a `graph`, partitions the `graph` into
// `subgraphs`. `subgraphs` maps device names to the graph assigned to that
// device. `graph` must have been placed (e.g. by running Placer),
// i.e. all nodes must have an assigned_device set.
// `graph` is non-const because the underlying Partition() function transforms
// the graph to correctly partition distributed control flow.
// `get_tensor_name_attr` computes the "tensor_name" attr value of Send/Recv ops
// inserted during partitioning. Use the default one if not set. It needs to be
// thread safe if it's shared in multple threads.
Status PartitionFunctionGraph(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph,
    std::unordered_map<string, std::unique_ptr<Graph>>* subgraphs,
    std::function<string(const Edge*)> get_tensor_name_attr = nullptr);

// Inserts send/recv ops to `graph` if nodes are assigned to multiple devices.
// Returns the new graph with the added nodes.
StatusOr<std::unique_ptr<Graph>> InsertTransferOps(
    const DeviceSet& device_set, std::unique_ptr<Graph> graph);

// This function performs bookkeeping to track which `Arg` and `Retval` nodes
// were placed on a particular device / graph.
//
// More specifically, this function
//
//  (1) rewrites the indices of the `Arg` and `Retval` nodes in `graph` to be
//      consecutive.
//
//      These indices might not be consecutive after grappler's pruning
//      optimization (e.g. removing redundant Args), or graph partitioning. In
//      the latter case, the nodes in `graph` are placed on `device_type`, and
//      each such graph partition gets a subset of the arguments and return
//      values. The `index` attributes of these _Arg and _Retval nodes reflect
//      the indices of these parameters in the original function. To convert
//      `subgraph` to a function, we need to replace there original indices with
//      0, 1, 2, ... .
//
//      The argument and return value order in `graph` is determined by the
//      argument and return value order in the original function. This stability
//      is important because it enables us to treat a single-partition function
//      as having the same signature as the subgraph.
//
//  (2) records the subsets of `Arg` and `Retval` nodes assigned to the
//      device in `*_indices`, and
//  (3) records which `Arg` and `Retval` nodes live in host memory in
//      `*_alloc_attrs`. If these vectors are NULL, do nothing here. If
//      `ints_on_device` is false, int32 `Arg` and `Retval` nodes are placed on
//      host else not. This is needed because in certain special cases e.g.
//      when graph is placed on TPU/XLA device or when the `Retval` is an output
//      of an iterator, int32 tensors live on device.
Status UpdateArgAndRetvalMetadata(
    Graph* graph, std::vector<FunctionArgIndex>* arg_indices,
    std::vector<int>* ret_indices,
    std::vector<AllocatorAttributes>* arg_alloc_attrs,
    std::vector<AllocatorAttributes>* ret_alloc_attrs, bool ints_on_device);

// Utility for generating function names not present in `flib_def`, using
// given `name` as the base for the name.
class FunctionNameGenerator {
 public:
  // `flib_def` must outlive this.
  FunctionNameGenerator(const FunctionLibraryDefinition* flib_def,
                        const string& name)
      : flib_def_(flib_def), name_(name), counter_(0) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpartitioning_utilsDTh mht_0(mht_0_v, 259, "", "./tensorflow/core/common_runtime/partitioning_utils.h", "FunctionNameGenerator");
}

  // Returns a function name not present in `flib_def` using `name` as
  // the base and appending a numeric suffix.
  string GetName();

 private:
  const FunctionLibraryDefinition* flib_def_;
  const string name_;
  uint32 counter_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PARTITIONING_UTILS_H_
