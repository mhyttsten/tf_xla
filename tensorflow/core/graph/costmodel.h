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

#ifndef TENSORFLOW_CORE_GRAPH_COSTMODEL_H_
#define TENSORFLOW_CORE_GRAPH_COSTMODEL_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh() {
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

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
typedef std::unordered_map<StringPiece, int32, StringPieceHasher>
    NodeNameToCostIdMap;

class StepStats;

// CostModel keeps track of the following runtime statistics for nodes
// of a single Graph:
//    * The total number of times a node has executed.
//    * The accumulated execution time (in microseconds) of a node.
//    * The accumulated size (in bytes) of each node's output.
//
// This class is NOT thread-safe.
class CostModel {
 public:
  // If "global" is true, maintains costs based on Node::cost_id, otherwise
  // maintains costs based on Node::id.
  explicit CostModel(bool is_global) : is_global_(is_global) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh mht_0(mht_0_v, 218, "", "./tensorflow/core/graph/costmodel.h", "CostModel");

    unknown_shape_.set_unknown_rank(true);
  }

  // Assigns min_count_ as a function of the median count for a Node.
  // This value is then used for suppressing the time/size costs of
  // infrequent operations.
  // NOTE(tucker): Maybe this should move to a subclass of CostModel.
  void SuppressInfrequent();

  bool is_global() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh mht_1(mht_1_v, 231, "", "./tensorflow/core/graph/costmodel.h", "is_global");
 return is_global_; }

  inline int Id(const Node* n) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh mht_2(mht_2_v, 236, "", "./tensorflow/core/graph/costmodel.h", "Id");

    if (is_global_) {
      return n->cost_id();
    } else {
      return n->id();
    }
  }

  inline int GlobalId(const Node* n, int offset) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh mht_3(mht_3_v, 247, "", "./tensorflow/core/graph/costmodel.h", "GlobalId");

    if (is_global_) {
      return n->cost_id();
    } else {
      return n->id() + offset;
    }
  }

  // Initializes cost model for 'g'.
  void InitFromGraph(const Graph& g);

  // Merges costs from cm.
  // REQUIRES: is_global_ is true for this and for "cm"
  void MergeFromGlobal(const CostModel& cm);

  // Merges costs from "cm", which has been computed relative to "g".
  // REQUIRES: is_global_ is true for this, and false for "cm".
  void MergeFromLocal(const Graph& g, const CostModel& cm);

  void MergeFromStats(const NodeNameToCostIdMap& map, const StepStats& ss);

  // Sets the number of outputs of "node".
  void SetNumOutputs(const Node* node, int num_outputs);

  // Records that "node" has executed "num_count" more times.
  void RecordCount(const Node* node, int num_count);

  // Returns how many times "node" has been executed.
  int32 TotalCount(const Node* node) const;

  // Records that "output_slot" of "node" has produced tensors of
  // aggregated "bytes".
  void RecordSize(const Node* node, int output_slot, Bytes bytes);

  // Returns total bytes of tensors produced by "node"s output slot.
  Bytes TotalBytes(const Node* node, int output_slot) const;

  // Returns a prediction for the size of the tensor at the
  // output_slot produced by one execution of "node".
  Bytes SizeEstimate(const Node* node, int output_slot) const;

  // Records that Executions of "node" have taken "time" microseconds.
  void RecordTime(const Node* node, Microseconds time);

  // Returns the total execution time for "node".
  Microseconds TotalTime(const Node* node) const;

  // Returns a prediction for one execution of "node".
  Microseconds TimeEstimate(const Node* node) const;

  // Check that an estimate is available for every OP node in graph.
  void CheckInitialized(const Graph& graph) const;

  // Records the maximum size in bytes and optionally the corresponding shape of
  // the tensor generated by "output_slot" of "node". If
  void RecordMaxMemorySize(const Node* node, int output_slot, Bytes bytes,
                           const TensorShapeProto& tensor_shape,
                           const DataType& dtype);

  // Returns the maximum size in bytes of the tensor generated by "output_slot"
  // of "node".
  Bytes MaxMemorySize(const Node* node, int output_slot) const;

  // Returns the shape corresponding to the largest memory size of the tensor
  // generated by "output_slot" of "node".
  const TensorShapeProto& MaxMemoryShape(const Node* node,
                                         int output_slot) const;

  // Returns the shape corresponding to the largest memory size of the tensor
  // generated by "output_slot" of "node".
  DataType MaxMemoryType(const Node* node, int output_slot) const;

  // Returns the size in bytes of temporary memory consumed by "node".
  Bytes TempMemorySize(const Node* node) const;

  // Returns the size of persistent memory allocated by "node".
  Bytes PersistentMemorySize(const Node* node) const;

  // Records memory stats such as temp momory and persistent memory.
  void RecordMemoryStats(const Node* node, const MemoryStats& memory_stats);

  // Records the maximum execution time (in microseconds) of "node".
  void RecordMaxExecutionTime(const Node* node, Microseconds time);

  // Returns the maximum execution time (in microseconds) of "node".
  Microseconds MaxExecutionTime(const Node* node) const;

  // Record the unique id of the tensor generated by "output_slot" of "node".
  // Any other tensor sharing the same id will be an alias, i.e. it will share
  // the same underlying memory storage area.
  void RecordAllocationId(const Node* node, int output_slot, int64_t alloc_id);

  // Return the unique id of the tensor generated by "output_slot" of "node".
  int64_t AllocationId(const Node* node, int output_slot) const;

  bool IsPersistentTensor(const Node* node, int64_t alloc_id) const;

  // Helper routines to encapsulate static estimation heuristics

  // Compute an estimate of the time to copy "b" bytes over the network,
  // given a fixed cost of "network_latency_millis" milliseconds and
  // an estimated bandwidth of "estimated_gbps" gigabits per second (note that
  // this value is in gigabits, not gigabytes).
  static Microseconds CopyTimeEstimate(Bytes b, double network_latency_millis,
                                       double estimated_gbps);
  static Microseconds ComputationTimeEstimate(int64_t mathops);

  // Add this CostModel into the CostGraphDef.
  void AddToCostGraphDef(const Graph* graph, CostGraphDef* cost_graph) const;

  // Write the contents of the CostModel to the INFO log.
  void WriteSummaryToLog() const;

  // Increment the times that the cost model is updated.
  void IncrementUpdateTimes();

  // Get the times that the cost model is updated.
  int32 GetUpdateTimes() const;

 private:
  static Bytes MinTensorMemoryUsage(const TensorShapeProto& tensor_shape,
                                    const DataType& dtype);

  const bool is_global_;

  // Resizes vectors so that they are large enough for "id" and id's outputs.
  void Ensure(int id, int num_outputs);

  // Nodes and Edges whose count is < this value
  // get type/byte estimates of 0.
  int32 min_count_ = 0;

  // The number of times the cost model is updated.
  int32 update_times_ = 0;

  // Number of times each Node has been executed.
  std::vector<int32> count_;
  // Cumulative execution time.
  std::vector<Microseconds> time_;
  // Cumulative Bytes output on each channel.
  std::vector<gtl::InlinedVector<Bytes, 2>> slot_bytes_;

  // Maximum execution time
  std::vector<Microseconds> max_exec_time_;

  // Maximum memory usage
  struct MemUsage {
    MemUsage() : temp_memory_size(0), persistent_memory_size(0) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTh mht_4(mht_4_v, 397, "", "./tensorflow/core/graph/costmodel.h", "MemUsage");
}

    // TODO(yuefengz): temp_memory_size is not being used, remove it.
    Bytes temp_memory_size;
    Bytes persistent_memory_size;

    gtl::InlinedVector<Bytes, 2> output_port_mem;
    gtl::InlinedVector<TensorShapeProto, 2> output_port_shape;
    gtl::InlinedVector<DataType, 2> output_port_type;
  };
  std::vector<MemUsage> max_mem_usage_;

  std::vector<gtl::InlinedVector<int64_t, 2>> output_port_alloc_ids_;

  std::set<int64_t> persistent_alloc_ids_;
  std::map<string, std::set<int64_t>> persistent_alloc_ids_by_devices_;

  TensorShapeProto unknown_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(CostModel);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_COSTMODEL_H_
