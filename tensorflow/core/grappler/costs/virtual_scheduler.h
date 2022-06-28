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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh() {
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


#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_context.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/grappler_item.h"

namespace tensorflow {
namespace grappler {

ABSL_CONST_INIT extern const char kAttrInputSrc[];
ABSL_CONST_INIT extern const char kAttrSrcDevice[];
ABSL_CONST_INIT extern const char kAttrDstDevice[];
ABSL_CONST_INIT extern const char kAttrTensorName[];
ABSL_CONST_INIT extern const char kChannelDevice[];
ABSL_CONST_INIT extern const char kStreaming[];

struct NodeState {
  // A node (i.e., an op) takes a set of input:port pairs and produces
  // a set of output ports.

  // Cross references to input and output nodes from graphdef.
  std::vector<std::pair<const NodeDef*, int>> inputs;  // Input, port pairs.
  // List of output nodes (a list of nodes that takes this output port as input)
  // keyed by port_num. Note that port_num -1 is used for control dependency.
  std::unordered_map<int, std::vector<const NodeDef*>> outputs;

  // Info from GraphProperties.
  std::vector<OpInfo::TensorProperties> input_properties;
  std::vector<OpInfo::TensorProperties> output_properties;

  // Canonical device name used within VirtualScheduler.
  string device_name;

  // States updated as scheduling nodes.
  int num_inputs_ready;
  std::unordered_map<int, int> num_outputs_executed;
  Costs::Duration time_ready;
  Costs::Duration time_scheduled;
  Costs::Duration time_finished;
  // Time that all the consumers are executed (hence, no need to keep this
  // output in memory), keyed by port_num.
  std::unordered_map<int, Costs::Duration> time_no_references;

  // Note that a node may have multiple output ports. The length of outputs,
  // num_outputs_executed, and time_no_references should be
  // identical when a NodeState is fully initialized.
  // They should be 1 + output_properties.size() as we add [-1] for control
  // dependency.

  // Node will be ready to be executed at time_ready, scheduled at
  // time_scheduled, and finishes execution at time_finished.
  // Each output port uses up memory space from time_scheduled to its
  // time_no_references.

  Costs node_costs;  // Node costs per execution
  Costs TotalNodeCosts() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_0(mht_0_v, 252, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "TotalNodeCosts");

    return MultiplyCosts(node_costs, execution_count);
  }
  // How many times this node has been executed, e.g. in a while loop.
  int execution_count;

  // Output shape incompatible between shape annotation and shape inference.
  bool shape_incompatible;

  NodeState() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_1(mht_1_v, 264, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "NodeState");

    num_inputs_ready = 0;
    time_ready = Costs::Duration::max();
    time_scheduled = Costs::Duration::max();
    time_finished = Costs::Duration::max();
    execution_count = 0;
    shape_incompatible = false;
    // Note that num_outputs_executed and time_no_references are not initialized
    // here, since we don't know the size (i.e., # outputs for this node).
  }
};

struct DeviceState {
  // Nodes executed on this device in execution order.
  std::vector<const NodeDef*> nodes_executed;

  struct NodePairHash {
   public:
    const std::size_t operator()(
        const std::pair<const NodeDef*, int>& element) const {
      return std::hash<const NodeDef*>()(element.first);
    }
  };

  // Nodes currently allocated in memory: set of NodeDef* and port_num pairs
  // so that we can track which output of the node is in memory.
  std::unordered_set<std::pair<const NodeDef*, int>, NodePairHash>
      nodes_in_memory;

  // Nodes allocated in memory persistently: e.g., Variables.
  std::unordered_set<std::pair<const NodeDef*, int>, NodePairHash>
      persistent_nodes;

  // Snapshot of nodes_in_memory, when memory usage is at peak.
  // Same to nodes_in_memory, it's a set of NodeDef* and port_num pairs.
  std::unordered_set<std::pair<const NodeDef*, int>, NodePairHash>
      mem_usage_snapshot_at_peak;

  // Vector of temporary memory usage trace in execution order.
  // Each pair represents the current node name and current (accumulated)
  // temporary memory usage of the device when the node is scheduled.
  // Only enabled when mem_usage_tracking is enabled.
  // Note: CPU uses an inter-op threadpool, so the execution order on CPU may
  // not be deterministic.
  std::vector<std::pair<std::string, int64_t>> temporary_memory_usage_trace;

  Costs device_costs;
  std::map<string, Costs> op_to_cost;  // Per-op cost.

  int64_t memory_usage;      // Current temporary memory usage
  int64_t max_memory_usage;  // Max temporary memory usage

  // Shape annotation statistics.
  struct ShapeAnnotationStats {
    // Number of ops with shape annotated.
    int64_t num_ops_annotated = 0;
    // Number of ops executed multiple times (e.g. in a loop).
    int64_t num_ops_executed_more_than_once = 0;
    // Number of ops executed: account for execution count.
    int64_t num_ops_executed = 0;
    // Number of ops with dynamic shapes (e.g. shape changes in a loop).
    int64_t num_ops_with_dynamic_shapes = 0;
    // Number of ops with incompatible shapes between annotation and shape
    // inference.
    int64_t num_ops_with_incompatible_shapes = 0;
  } shape_annotation_stats;

  DeviceState() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_2(mht_2_v, 334, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "DeviceState");

    device_costs = Costs::ZeroCosts();
    device_costs.num_ops_total = 0;
    memory_usage = 0;
    max_memory_usage = 0;
  }

  Costs::Duration GetCurrTime() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_3(mht_3_v, 344, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetCurrTime");
 return device_costs.execution_time; }
};

// ReadyNodeManager (abstract class):
// Keeps ready nodes and picks the best one to be scheduled.
class ReadyNodeManager {
 public:
  ReadyNodeManager() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_4(mht_4_v, 354, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "ReadyNodeManager");
}
  virtual ~ReadyNodeManager() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_5(mht_5_v, 358, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~ReadyNodeManager");
}
  virtual Status Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_map) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_6(mht_6_v, 363, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "Init");

    return Status::OK();
  }
  virtual void AddNode(const NodeDef* node) = 0;
  virtual const NodeDef* GetCurrNode() = 0;
  virtual void RemoveCurrNode() = 0;
  virtual bool Empty() const = 0;
};

class FIFOManager : public ReadyNodeManager {
 public:
  FIFOManager() : ReadyNodeManager() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_7(mht_7_v, 377, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "FIFOManager");
}
  ~FIFOManager() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_8(mht_8_v, 381, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~FIFOManager");
}
  void AddNode(const NodeDef* node) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_9(mht_9_v, 385, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "AddNode");
 nodes_.push_back(node); }
  const NodeDef* GetCurrNode() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_10(mht_10_v, 389, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetCurrNode");

    CHECK(!nodes_.empty()) << "GetCurrNode(), but there's no ready node";
    return nodes_.front();
  }
  void RemoveCurrNode() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_11(mht_11_v, 396, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "RemoveCurrNode");
 nodes_.pop_front(); }
  bool Empty() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_12(mht_12_v, 400, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "Empty");
 return nodes_.empty(); }

 private:
  std::list<const NodeDef*> nodes_;
};

// The LIFOManager schedules nodes by returning the last one added to the
// scheduler. A node is executed and then its ready outputs are newly added to
// the scheduler, so the LIFOManager will return outputs to a node following
// that node's execution.
class LIFOManager : public ReadyNodeManager {
 public:
  LIFOManager() : ReadyNodeManager() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_13(mht_13_v, 415, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "LIFOManager");
}
  ~LIFOManager() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_14(mht_14_v, 419, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~LIFOManager");
}
  void AddNode(const NodeDef* node) override;
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_15(mht_15_v, 426, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "Empty");
 return nodes_.empty(); }

 private:
  std::list<const NodeDef*> nodes_;
  // Keep track of the current node being executed by saving its position.
  // Necessary because nodes may be added to the end of the list while a node is
  // executing, and we want to remove the correct node (the one that is
  // executing) rather than the new ones being added.
  std::list<const NodeDef*>::iterator curr_pos_ = nodes_.end();
};

// Abstract class that maintains a heap/priority queue for scheduling ready
// nodes. Derived class needs to implement the Greater() function which returns
// the comparator for the heap.
class HeapReadyManager : public ReadyNodeManager {
 public:
  HeapReadyManager();
  Status Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_map) override;
  ~HeapReadyManager() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_16(mht_16_v, 448, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~HeapReadyManager");
}
  void AddNode(const NodeDef* node) override;
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override;

 protected:
  virtual std::function<bool(const NodeDef*, const NodeDef*)> Greater() = 0;

  // nodes_ is the main queue, where we construct heap, and the front is the
  // current node.
  std::vector<const NodeDef*> nodes_;

  // Comparator functor for heap; stl heap is max heap, so we use "greater than"
  // functor for keeping the smallest time_ready node at the front of heap.
  std::function<bool(const NodeDef*, const NodeDef*)> greater_;

  // NodeState structure from SchedulerState to get time_ready of ready nodes.
  // Not owned by FirstReadyManager.
  const std::unordered_map<const NodeDef*, NodeState>* node_map_;

  // Cached curr node. Set back to nullptr from RemoveCurrNode().
  const NodeDef* curr_node_;
};

// FirstReadyManager picks a node with the minimum time_ready value.
// Behavior is deterministic when there are more than one nodes with the minimum
// time_ready value with unique node names as the tie-breaker.
class FirstReadyManager : public HeapReadyManager {
 public:
  FirstReadyManager() : HeapReadyManager() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_17(mht_17_v, 481, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "FirstReadyManager");
}
  ~FirstReadyManager() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_18(mht_18_v, 485, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~FirstReadyManager");
}

 protected:
  std::function<bool(const NodeDef*, const NodeDef*)> Greater() override;
};

// PriorityReadyManager uses the given node priorities when picking up next node
// from all the ready nodes.
class PriorityReadyManager : public HeapReadyManager {
 public:
  PriorityReadyManager() : HeapReadyManager() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_19(mht_19_v, 498, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "PriorityReadyManager");
}
  ~PriorityReadyManager() override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_20(mht_20_v, 502, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~PriorityReadyManager");
}
  void AddNode(const NodeDef* node) override;

  // Note this should be called after Init().
  Status SetPriority(const std::unordered_map<string, int>& node_priority);

 protected:
  std::function<bool(const NodeDef*, const NodeDef*)> Greater() override;

 private:
  // A map from unique node name to priority. Lower number means higher
  // priority.
  std::unordered_map<string, int> node_priority_;
};

// CompositeNodeManager has a few other NodeManagers: per-device LIFO for normal
// ops (neither _Send nor _Recv) and FirstReadyManagers for _Send ops and _Recv
// ops, and then it chooses FirstReady among the ops chosen from each
// internal NodeManagers. The objective is to maximize producer-consumer
// locality within device, while processing nodes across devices, including
// _Send and _Recv, fairly, in terms of their time_ready.
class CompositeNodeManager : public ReadyNodeManager {
 public:
  CompositeNodeManager();
  ~CompositeNodeManager() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_21(mht_21_v, 529, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "~CompositeNodeManager");
}

  Status Init(
      const std::unordered_map<const NodeDef*, NodeState>* node_map) override;
  void AddNode(const NodeDef* node) override;
  const NodeDef* GetCurrNode() override;
  void RemoveCurrNode() override;
  bool Empty() const override;

 private:
  // Internal ready node managers:
  // LIFO for normal ops to maximize producer consumer locality.
  // One LIFO per device.
  std::unordered_map<string, LIFOManager> ops_lifo_map_;
  // FirstReady for send and recv. Handle send and recv separately ensures that
  // send and recv do not block previously read ops with LIFO schedule.
  FirstReadyManager send_manager_;
  FirstReadyManager recv_manager_;

  // NodeState structure from SchedulerState to get time_ready of ready nodes.
  // Not owned by CompositeReadyManager.
  const std::unordered_map<const NodeDef*, NodeState>* node_map_;

  // Cached curr node. Set back to nullptr from RemoveCurrNode().
  const NodeDef* curr_node_;
};

// Constructs a ready node manager from the given string.
std::unique_ptr<ReadyNodeManager> ReadyNodeManagerFactory(
    const string& ready_node_manager);

// Encapsulates all of the various pieces uses to track state of a scheduler;
// enables reuse of all scheduler state-related utilities across different
// scheduler implementations.
class SchedulerState {
 public:
  SchedulerState(const bool use_static_shapes,
                 const bool use_aggressive_shape_inference, Cluster* cluster,
                 std::unique_ptr<VirtualPlacer> placer);
  // Move constructor. Explicitly defined because it otherwise gets implicitly
  // deleted. SchedulerState is a move-only class, as we have a <unique_ptr>
  // for it in VirtualScheduler. A derivative of VirtualScheduler can move a
  // <unique_ptr> SchedulerState to VirtualScheduler when it is constructed,
  // which is where this move constructor is needed.
  SchedulerState(SchedulerState&& arg) = default;
  // We explicitly delete assinment and copy operators, this is done implicitly,
  // but we state it here explicitly for clarity.
  SchedulerState& operator=(SchedulerState&& arg) = delete;
  SchedulerState(const SchedulerState&) = delete;
  SchedulerState& operator=(const SchedulerState&) = delete;
  // Destructor. Must be defined such that a derivative class can override it
  // and allow proper desctruction of the derivative class. If this is not done
  // properly, memory leaks can occur.
  virtual ~SchedulerState();
  // Sets up the graph while also performing some necessary transformations
  // initial_nodes is the set of nodes (primary inputs) discovered by Init()
  // which may be added by a ReadyNodeManager (or related/derivative scheduler)
  // to begin node schedule and graph simulation.
  Status Init(const GrapplerItem* item,
              std::vector<const NodeDef*>* initial_nodes,
              bool create_explicit_channel_device = true);

  virtual Costs Summary() const;
  // Like the above, but writes detailed stats to RunMetadata.
  // If metadata is nullptr, then just calls and return Summary().
  virtual Costs Summary(RunMetadata* metadata);
  // Generates RunMetadata's step_stats and partition_graphs fields from results
  // of the virtual execution of the graph.
  // TODO(rdegruijl) See if we can make this function and caller Summary()
  // const.
  void GenerateRunMetadata(RunMetadata* metadata);

  // Returns per device memory usage.
  const std::unordered_map<string, int64_t> GetPeakMemoryUsage() const;
  const std::unordered_map<string, int64_t> GetPersistentMemoryUsage() const;
  void enable_mem_usage_tracking() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_22(mht_22_v, 607, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "enable_mem_usage_tracking");
 track_mem_usage_snapshot_ = true; }
  // Returns (read only) device and node states.
  const std::unordered_map<string, DeviceState>* GetDeviceStates() const {
    return &device_;
  }

  const std::unordered_map<const NodeDef*, NodeState>* GetNodeStates() const {
    return &node_map_;
  }

  virtual OpContext CreateOpContext(const NodeDef* node) const;
  std::vector<const NodeDef*> MarkNodeExecuted(
      const NodeDef* node, const Costs& node_costs, const OpContext& op_context,
      bool extract_execution_count_attr = true,
      const std::string& override_device_name = "");

  // Some getter functions.
  const GrapplerItem* GetGrapplerItem() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_23(mht_23_v, 627, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetGrapplerItem");
 return grappler_item_; }
  Costs GetGraphCost() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_24(mht_24_v, 631, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetGraphCost");
 return graph_costs_; }
  Cluster* GetCluster() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_25(mht_25_v, 635, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetCluster");
 return cluster_; }
  bool GetUseStaticShape() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_26(mht_26_v, 639, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetUseStaticShape");
 return use_static_shapes_; }
  bool GetUseAggressiveShapeInference() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_27(mht_27_v, 643, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GetUseAggressiveShapeInference");

    return use_aggressive_shape_inference_;
  }
  const std::unordered_map<const NodeDef*, NodeState>& GetNodeMap() {
    return node_map_;
  }

 protected:
  // Assigns the time_scheduled in the NodeState of node to the current
  // execution_time of the device executing this node.
  void SetNodeStateTimeScheduled(const NodeDef* node);

  // This method can be used by a class derived from SchedulerState to
  // access the device state map.
  std::unordered_map<string, DeviceState>* GetMutableDeviceState() {
    return &device_;
  }

 private:
  // Methods called from Init(). Fails if initialize_ is set.

  void MaybeUpdateInputOutput(const NodeDef* node);
  NodeState& GetNodeStateOrCreateIt(const NodeDef* node);
  // Creates a Send_ and Recv_ pair between from and to. The argument
  // create_channel_device tells the function to create an explicit device for
  // the channel.
  std::pair<const NodeDef*, const NodeDef*> CreateSendRecv(
      const NodeDef* from, const NodeDef* to, const NodeDef* input_node,
      const string& input_name, bool create_channel_device);
  string DeviceName(const NodeDef* node) const;
  string SanitizedDeviceName(const NodeDef* node) const;
  string ChannelDeviceName(const NodeDef* from, const NodeDef* to) const;

  // Helper methods.
  void GetOutputNodes(const NodeDef* node, const Costs::Duration& curr_time,
                      std::vector<const NodeDef*>* output_nodes);
  // Retrieves output size from node_cost at a port_num.  If the output size has
  // not been set, defaults back to CalculateOutputSize.
  int64_t GetOrCalculateOutputSize(const NodeState& node_state,
                                   int port_num) const;

  std::unordered_map<const NodeDef*, NodeState> node_map_;
  std::unordered_map<string, DeviceState> device_;

  // Pool of NodeDefs for SendRecv and Identity ops created.
  std::vector<std::unique_ptr<NodeDef>> additional_nodes_;

  // Stats:
  // Op counts with key with input shape.
  // Example key: "[Op=AssignSub, input_shapes=[[7,1,160,160][7,1,160,160]]"
  std::map<string, int> op_counts_;
  // Individual op costs with key with input shape.
  // Integer field for execution time in micro seconds.
  // Boolean field for whether the cost is accurate.
  std::map<string, std::pair<int, bool>> op_costs_;

  Costs graph_costs_;                   // Graph cost.
  std::map<string, Costs> op_to_cost_;  // Per-op cost.

  // Auxiliary data structures for constructing NodeState and DeviceState.
  std::unique_ptr<GraphProperties> graph_properties_;  // Initialized in Init().
  Cluster* cluster_;                                   // Not owned.
  const GrapplerItem* grappler_item_;                  // Not owned.
  bool use_static_shapes_;
  bool initialized_;
  bool track_mem_usage_snapshot_;
  const bool use_aggressive_shape_inference_;
  std::unique_ptr<VirtualPlacer> placer_;
};

// The virtual scheduler emulates execution of nodes in a graph, considering
// dependencies, device, etc.
class VirtualScheduler {
 public:
  // Does not take ownership of cluster or ready_nodes.
  VirtualScheduler(const bool use_static_shapes,
                   const bool use_aggressive_shape_inference, Cluster* cluster,
                   ReadyNodeManager* ready_nodes,
                   std::unique_ptr<VirtualPlacer> placer);
  // This constructor can be called by a derivative of VirtualScheduler to
  // construct the base class. It lets VirtualScheduler take ownership of
  // a new SchedulerState or a derivative thereof.
  // Note that this constructor does not set a VirtualPlacer, in this
  // constructor the VirtialPlacer is passed as a member of the SchedulerState
  // that is passed as an argument.
  VirtualScheduler(ReadyNodeManager* ready_nodes,
                   std::unique_ptr<SchedulerState> scheduler_state);
  virtual ~VirtualScheduler();

  // Initializes the scheduler for the specific grappler item.
  // Should be called immediately after the c'tor or when the scheduler will be
  // reused for a new grappler item. All internal states of the scheduler
  // related to the previous grappler item will be reset/cleared.
  //
  // This function should be called at least once after the scheduler is
  // constructed. An uninitialized or failed-to-initialize scheduler will cause
  // undefined behavior.
  virtual Status Init(const GrapplerItem* item);

  // Gets the current scheduled node for execution; the caller of this function
  // can accordingly simulate the execution of the current scheduled node.
  virtual OpContext GetCurrNode();
  // Marks the current scheduled node as executed. Note that we should call this
  // function only after the execution of the node has been simulated;
  // node_costs_ capture the simulated costs of the node.
  // Returns true if there is any node to be scheduled.
  virtual bool MarkCurrNodeExecuted(const Costs& node_costs);

  // Prints out summary of execution (timing, memory usage, etc.)
  Costs Summary() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_28(mht_28_v, 755, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "Summary");
 return scheduler_state_->Summary(); }
  // Like the above, but writes detailed stats to RunMetadata.
  // If metadata is nullptr, then just calls and return Summary().
  Costs Summary(RunMetadata* metadata) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_29(mht_29_v, 761, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "Summary");

    return scheduler_state_->Summary(metadata);
  }
  // Generates RunMetadata's step_stats and partition_graphs fields from results
  // of the virtual execution of the graph.
  void GenerateRunMetadata(RunMetadata* metadata) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_30(mht_30_v, 769, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "GenerateRunMetadata");

    scheduler_state_->GenerateRunMetadata(metadata);
  }
  // Returns per device memory usage.
  const std::unordered_map<string, int64_t> GetPeakMemoryUsage() const {
    return scheduler_state_->GetPeakMemoryUsage();
  }
  const std::unordered_map<string, int64_t> GetPersistentMemoryUsage() const {
    return scheduler_state_->GetPersistentMemoryUsage();
  }
  // Returns VirtualScheduler (read only) device and node states.
  const std::unordered_map<string, DeviceState>* GetDeviceStates() const {
    return scheduler_state_->GetDeviceStates();
  }
  const std::unordered_map<const NodeDef*, NodeState>* GetNodeStates() const {
    return scheduler_state_->GetNodeStates();
  }
  void enable_mem_usage_tracking() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPScostsPSvirtual_schedulerDTh mht_31(mht_31_v, 789, "", "./tensorflow/core/grappler/costs/virtual_scheduler.h", "enable_mem_usage_tracking");

    scheduler_state_->enable_mem_usage_tracking();
  }

 protected:
  // The state of the scheduler and the execution of the graph is encapsulated
  // by the scheduler_state_ object.
  std::unique_ptr<SchedulerState> scheduler_state_;
  // ready_nodes_ is responsible for ordering the traversal of the graph.
  ReadyNodeManager* ready_nodes_;  // Not owned.
};

}  // namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_VIRTUAL_SCHEDULER_H_
