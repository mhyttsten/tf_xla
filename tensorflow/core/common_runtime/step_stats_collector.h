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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh() {
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


#include <memory>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Allocator;
class AllocatorMemoryUsed;
class CostModelManager;
class Graph;
class NodeDef;
class NodeExecStats;
class OpKernelContext;
class StepStats;
class StepStatsCollector;
class Tensor;
class TrackingAllocator;

// Statistics collection interface for individual node execution.
//
// See `NodeExecStatsWrapper` for a concrete implementation of this interface
// that interfaces with the `Session` layer.
class NodeExecStatsInterface {
 public:
  virtual ~NodeExecStatsInterface() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/common_runtime/step_stats_collector.h", "~NodeExecStatsInterface");
}

  // Called when the statistics collection for the node has finished. Once this
  // method is called, the caller should not make assumptions about the validity
  // of this object.
  virtual void Done(const string& device) = 0;

  // Called immediately after this node starts being processed by the executor.
  virtual void RecordExecutorStarted() = 0;

  // Called immediately before this node's `Compute()` or `ComputeAsync()`
  // method is called.
  virtual void RecordComputeStarted() = 0;

  // Called immediately after this node's `Compute()` method returned (or, for
  // asynchronous operations, the callback passed to its `ComputeAsync()` method
  // was called).
  virtual void RecordComputeEnded() = 0;

  // Called immediately after this executor finishes processing this node.
  virtual void RecordExecutorEnded() = 0;

  // Returns `true` if this object should track memory allocations.
  virtual bool TrackAllocations() const = 0;

  // Records information about the memory allocated during the execution of this
  // node.
  //
  // Takes ownership of any `TrackingAllocator` objects stored in `ctx`.
  virtual void SetMemory(OpKernelContext* ctx) = 0;

  // Records information about the tensor produced by this node at the given
  // output slot.
  virtual void SetOutput(int slot, const Tensor* tensor) = 0;

  // Records the absolute time in nanoseconds at which this node became
  // runnable (i.e. was scheduled for execution).
  virtual void SetScheduled(int64_t nanos) = 0;
};

// Wraps NodeExecStats and adds allocation to it.
class NodeExecStatsWrapper : public NodeExecStatsInterface {
 public:
  // Does not take ownership of `node` or `step_stats_collector`.
  NodeExecStatsWrapper(const NodeDef* node,
                       StepStatsCollector* step_stats_collector);

  // Takes ownership of 'stats' but not `node` or `step_stats_collector`.
  NodeExecStatsWrapper(std::unique_ptr<NodeExecStats> stats,
                       const NodeDef* node,
                       StepStatsCollector* step_stats_collector);

  // Destructor calls Finalize() to release the TrackingAllocators.
  ~NodeExecStatsWrapper() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh mht_1(mht_1_v, 273, "", "./tensorflow/core/common_runtime/step_stats_collector.h", "~NodeExecStatsWrapper");
 Finalize(); }

  void Done(const string& device) override;
  void RecordExecutorStarted() override;
  void RecordComputeStarted() override;
  void RecordComputeEnded() override;
  void RecordExecutorEnded() override;
  bool TrackAllocations() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh mht_2(mht_2_v, 283, "", "./tensorflow/core/common_runtime/step_stats_collector.h", "TrackAllocations");
 return true; }
  void SetMemory(OpKernelContext* ctx) override;
  void SetOutput(int slot, const Tensor* tensor) override;
  void SetScheduled(int64_t nanos) override;

 private:
  friend class StepStatsCollector;

  NodeExecStats* stats() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh mht_3(mht_3_v, 294, "", "./tensorflow/core/common_runtime/step_stats_collector.h", "stats");
 return stats_.get(); }

  // Populates stats_ and releases TrackingAllocator.
  void Finalize();

  // Does not take ownership of the `allocator`.
  // Takes ownership of `tracking_allocator`.
  void AddAllocation(Allocator* allocator,
                     TrackingAllocator* tracking_allocator);

  gtl::InlinedVector<std::pair<AllocatorMemoryUsed*, TrackingAllocator*>, 2>
      allocations_;
  std::unique_ptr<NodeExecStats> stats_;
  const NodeDef* const node_;                       // Not owned.
  StepStatsCollector* const step_stats_collector_;  // Not owned.
};

// Statistics collection interface for step execution.
//
// See `StepStatsCollector` for a concrete implementation of this interface
// that interfaces with the `Session` layer.
class StepStatsCollectorInterface {
 public:
  virtual ~StepStatsCollectorInterface() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTh mht_4(mht_4_v, 320, "", "./tensorflow/core/common_runtime/step_stats_collector.h", "~StepStatsCollectorInterface");
}

  // Creates an instance of `NodeExecStatsInterface` that should be used for
  // collecting statistics about individual node execution.
  virtual NodeExecStatsInterface* CreateNodeExecStats(const NodeDef* node) = 0;

  // Generates a string reporting the currently used memory based
  // on ResourceExhausted OOM `err` message.
  // `err` message needs to contain device name and allocator name, e.g.:
  // "ResourceExhaustedError: OOM when allocating tensor ...
  // on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc"
  virtual string ReportAllocsOnResourceExhausted(const string& err) = 0;
};

// StepStatsCollector manages the collection of a StepStats object.
// The StepStats object holds multiple DeviceStats.
// Each DeviceStats object holds multiple NodeExecStats.
class StepStatsCollector : public StepStatsCollectorInterface {
 public:
  // Does not take ownership of `step_stats`.
  explicit StepStatsCollector(StepStats* step_stats);

  // BuildCostModel builds or updates a CostModel managed by cost_model_manager,
  // using the currently collected DeviceStats associated with the devices in
  // device_map.
  void BuildCostModel(
      CostModelManager* cost_model_manager,
      const std::unordered_map<string, const Graph*>& device_map);

  // Saves node statistics to the DeviceStats object associated with device.
  // Should be called before Finalize.
  void Save(const string& device, NodeExecStats* node_stats_pb);
  void Save(const string& device, NodeExecStatsWrapper* node_stats);

  // Saves thread name.
  void SaveThreadName(const string& device, const uint32 thread_id,
                      const string& thread_name);

  NodeExecStatsInterface* CreateNodeExecStats(const NodeDef* node) override;
  string ReportAllocsOnResourceExhausted(const string& err) override;

  // The following 2 Finalize methods populate the StepStats passed
  // from the constructor. Calling it more than once won't have any effect.
  // User shouldn't call Save() methods after Finalize.
  void Finalize();
  // swaps the content of StepStats* from constructor with 'ss'.
  void FinalizeAndSwap(StepStats* step_stats);

 private:
  // TODO(suharshs): Make this configurable if its not possible to find a value
  // that works for all cases.
  static constexpr uint64 kMaxCollectedNodes = 1 << 20;

  typedef std::vector<std::unique_ptr<NodeExecStatsWrapper>> NodeStatsVector;
  typedef std::unordered_map<uint32, string> ThreadNamesMap;

  void FinalizeInternal() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  bool finalized_ TF_GUARDED_BY(mu_);
  std::unordered_map<string, NodeStatsVector> dev_stats_ TF_GUARDED_BY(mu_);
  std::unordered_map<string, ThreadNamesMap> thread_names_ TF_GUARDED_BY(mu_);
  StepStats* step_stats_ TF_GUARDED_BY(mu_);
  uint64 collected_nodes_ TF_GUARDED_BY(mu_) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_STEP_STATS_COLLECTOR_H_
