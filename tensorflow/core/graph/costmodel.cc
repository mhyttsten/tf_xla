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
class MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc() {
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

#include "tensorflow/core/graph/costmodel.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
const Microseconds kDefaultTimeEstimate(1);
const Microseconds kMinTimeEstimate(1);
}  // namespace

void CostModel::SuppressInfrequent() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::SuppressInfrequent");

  // Find the median of the non-zero counts, and use half of its value
  // as the cutoff for a "normal" execution mode node.
  if (count_.empty()) return;
  std::vector<int32> non_zero;
  for (auto v : count_) {
    if (v > 0) non_zero.push_back(v);
  }
  const size_t sz = non_zero.size();
  if (sz > 0) {
    std::nth_element(non_zero.begin(), non_zero.begin() + sz / 2,
                     non_zero.end());
    int32_t median_value = non_zero[sz / 2];
    min_count_ = median_value / 2;
    VLOG(1) << "num non_zero vals: " << non_zero.size() << " median_value "
            << median_value;
  } else {
    min_count_ = 1;
  }
}

void CostModel::MergeFromLocal(const Graph& g, const CostModel& cm) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MergeFromLocal");

  CHECK(is_global_);
  CHECK(!cm.is_global());
  for (const Node* n : g.nodes()) {
    const int local_id = cm.Id(n);
    const int global_id = Id(n);
    if (local_id < 0 || global_id < 0) continue;
    int num_slots = cm.slot_bytes_[local_id].size();
    Ensure(global_id, num_slots);
    count_[global_id] += cm.count_[local_id];
    time_[global_id] += cm.time_[local_id];
    if (num_slots > 0) {
      if (slot_bytes_[global_id].empty()) {
        slot_bytes_[global_id].resize(num_slots);
      } else {
        CHECK_EQ(num_slots, slot_bytes_[global_id].size());
      }
      for (int s = 0; s < num_slots; ++s) {
        slot_bytes_[global_id][s] += cm.slot_bytes_[local_id][s];
      }
    }
  }
}

void CostModel::MergeFromGlobal(const CostModel& cm) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MergeFromGlobal");

  CHECK(is_global_);
  CHECK_EQ(true, cm.is_global());
  const int num_nodes = cm.count_.size();
  for (int i = num_nodes - 1; i >= 0; --i) {
    count_[i] += cm.count_[i];
    time_[i] += cm.time_[i];
    int num_slots = cm.slot_bytes_[i].size();
    Ensure(i, num_slots);
    if (num_slots > 0) {
      if (slot_bytes_[i].empty()) {
        slot_bytes_[i].resize(num_slots);
      } else {
        CHECK_EQ(num_slots, slot_bytes_[i].size());
      }
      for (int s = 0; s < num_slots; ++s) {
        slot_bytes_[i][s] += cm.slot_bytes_[i][s];
      }
    }
  }
}

void CostModel::MergeFromStats(const NodeNameToCostIdMap& map,
                               const StepStats& ss) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_3(mht_3_v, 280, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MergeFromStats");

  CHECK(is_global_);
  for (auto& ds : ss.dev_stats()) {
    for (auto& ns : ds.node_stats()) {
      NodeNameToCostIdMap::const_iterator iter = map.find(ns.node_name());
      // We don't keep stats for nodes not in the global graph, i.e.
      // copy/send/recv nodes, feed/fetch, etc.
      if (iter == map.end()) continue;
      int32_t global_id = iter->second;
      Ensure(global_id, ns.output_size());
      int64_t elapsed_micros =
          ns.op_end_rel_micros() - ns.op_start_rel_micros();
      count_[global_id]++;
      time_[global_id] += elapsed_micros;
      for (auto& no : ns.output()) {
        int si = no.slot();
        if (static_cast<size_t>(si) >= slot_bytes_[global_id].size()) {
          slot_bytes_[global_id].resize(1 + si);
        }
        slot_bytes_[global_id][si] +=
            no.tensor_description().allocation_description().requested_bytes();
      }
    }
  }
}

void CostModel::Ensure(int id, int num_outputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_4(mht_4_v, 309, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::Ensure");

  if (slot_bytes_.size() <= static_cast<size_t>(id)) {
    slot_bytes_.resize(id + 1);
    count_.resize(id + 1);
    time_.resize(id + 1);
    max_mem_usage_.resize(id + 1);
    max_exec_time_.resize(id + 1);
    output_port_alloc_ids_.resize(id + 1);
  }
  if (num_outputs > 0) {
    auto perslot = &slot_bytes_[id];
    auto output_port_alloc_ids = &output_port_alloc_ids_[id];
    auto max_mem_usage = &max_mem_usage_[id];

    CHECK_LE(perslot->size(), num_outputs);
    DCHECK_EQ(output_port_alloc_ids->size(), perslot->size());
    DCHECK_EQ(max_mem_usage->output_port_mem.size(), perslot->size());
    DCHECK_EQ(max_mem_usage->output_port_shape.size(), perslot->size());
    DCHECK_EQ(max_mem_usage->output_port_type.size(), perslot->size());

    perslot->resize(num_outputs, Bytes(-1));
    output_port_alloc_ids->resize(num_outputs, -1);
    max_mem_usage->output_port_mem.resize(num_outputs, Bytes(-1));
    max_mem_usage->output_port_shape.resize(num_outputs, unknown_shape_);
    max_mem_usage->output_port_type.resize(num_outputs, DT_INVALID);
  }
}

void CostModel::SetNumOutputs(const Node* node, int num_outputs) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_5(mht_5_v, 340, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::SetNumOutputs");

  const int id = Id(node);
  if (id < 0) return;
  // Do not resize the number of slots before checking its existing number of
  // slots.
  Ensure(id, 0);
  auto perslot = &slot_bytes_[id];
  if (!perslot->empty()) {
    CHECK_EQ(num_outputs, perslot->size())
        << "Cannot resize slot_bytes, node=" << node->name();
  }
  Ensure(id, num_outputs);
}

void CostModel::RecordCount(const Node* node, int count) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_6(mht_6_v, 357, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordCount");

  const int id = Id(node);
  if (id < 0) return;
  CHECK_LT(id, slot_bytes_.size());
  count_[id] += count;
}

int32 CostModel::TotalCount(const Node* node) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_7(mht_7_v, 367, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::TotalCount");

  const int id = Id(node);
  if (id < 0) return 0;
  return (static_cast<size_t>(id) < slot_bytes_.size()) ? count_[id] : 0;
}

void CostModel::RecordSize(const Node* node, int slot, Bytes bytes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_8(mht_8_v, 376, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordSize");

  const int id = Id(node);
  if (id < 0) return;
  CHECK_LT(id, slot_bytes_.size());
  auto perslot = &slot_bytes_[id];
  CHECK_LT(slot, perslot->size());
  auto v = &(*perslot)[slot];
  if (*v >= 0) {
    *v += bytes;
  } else {
    *v = bytes;
  }
}

Bytes CostModel::TotalBytes(const Node* node, int slot) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_9(mht_9_v, 393, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::TotalBytes");

  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= slot_bytes_.size() ||
      slot_bytes_[id].size() <= static_cast<size_t>(slot)) {
    return Bytes(0);
  }
  return slot_bytes_[id][slot];
}

Bytes CostModel::SizeEstimate(const Node* node, int slot) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_10(mht_10_v, 405, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::SizeEstimate");

  int32_t count = TotalCount(node);
  if (count < min_count_) return Bytes(0);
  return TotalBytes(node, slot) / std::max(1, TotalCount(node));
}

void CostModel::RecordTime(const Node* node, Microseconds time) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_11(mht_11_v, 414, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordTime");

  const int id = Id(node);
  if (id < 0) return;
  DCHECK(node->IsOp()) << node->DebugString();
  Ensure(id, node->num_outputs());
  time_[id] += time;
}

Microseconds CostModel::TotalTime(const Node* node) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_12(mht_12_v, 425, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::TotalTime");

  DCHECK(node->IsOp()) << node->DebugString();
  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= time_.size() ||
      time_[id] < Microseconds(0)) {
    return Microseconds(0);
  }
  return time_[id];
}

Microseconds CostModel::TimeEstimate(const Node* node) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_13(mht_13_v, 438, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::TimeEstimate");

  int32_t count = TotalCount(node);
  if (count <= min_count_) return kMinTimeEstimate;
  return std::max(kMinTimeEstimate, TotalTime(node) / std::max(1, count));
}

void CostModel::CheckInitialized(const Graph& graph) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_14(mht_14_v, 447, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::CheckInitialized");

  for (const Node* n : graph.op_nodes()) {
    CHECK(static_cast<size_t>(n->id()) < time_.size() &&
          time_[n->id()] >= Microseconds(0))
        << ": no time estimate for " << n->DebugString();

    CHECK(static_cast<size_t>(n->id()) < slot_bytes_.size())
        << ": no size estimate for " << n->DebugString();
    const auto& perslot = slot_bytes_[n->id()];
    for (size_t i = 0; i < perslot.size(); i++) {
      CHECK_GE(perslot[i], Bytes(0)) << ": no size estimate for output# " << i
                                     << " of " << n->DebugString();
    }
  }
}

void CostModel::RecordMaxMemorySize(const Node* node, int output_slot,
                                    Bytes bytes,
                                    const TensorShapeProto& tensor_shape,
                                    const DataType& dtype) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_15(mht_15_v, 469, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordMaxMemorySize");

  const int id = Id(node);
  if (id < 0) return;
  if (output_slot >= node->num_outputs()) {
    LOG(ERROR) << "Unexpected output slot for node " << node->DebugString()
               << ". Got " << output_slot << " but its num_outputs is "
               << node->num_outputs();
    return;
  }
  Ensure(id, node->num_outputs());
  auto& current_max = max_mem_usage_[id].output_port_mem[output_slot];
  // If the memory allocator doesn't track memory usage, let's infer a lower
  // bound from the tensor shape and its data type.
  if (bytes.value() < 0) {
    bytes = MinTensorMemoryUsage(tensor_shape, dtype);
  }
  if (bytes.value() > current_max.value()) {
    current_max = bytes.value();
    max_mem_usage_[id].output_port_shape[output_slot] = tensor_shape;
    max_mem_usage_[id].output_port_type[output_slot] = dtype;
  }
}

Bytes CostModel::MaxMemorySize(const Node* node, int slot) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_16(mht_16_v, 495, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MaxMemorySize");

  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= max_mem_usage_.size() ||
      max_mem_usage_[id].output_port_mem.size() <= static_cast<size_t>(slot)) {
    return Bytes(0);
  }
  return max_mem_usage_[id].output_port_mem[slot];
}

const TensorShapeProto& CostModel::MaxMemoryShape(const Node* node,
                                                  int slot) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_17(mht_17_v, 508, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MaxMemoryShape");

  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= max_mem_usage_.size() ||
      max_mem_usage_[id].output_port_shape.size() <=
          static_cast<size_t>(slot)) {
    return unknown_shape_;
  }
  return max_mem_usage_[id].output_port_shape[slot];
}

DataType CostModel::MaxMemoryType(const Node* node, int slot) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_18(mht_18_v, 521, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MaxMemoryType");

  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= max_mem_usage_.size() ||
      max_mem_usage_[id].output_port_type.size() <= static_cast<size_t>(slot)) {
    return DT_INVALID;
  }
  return max_mem_usage_[id].output_port_type[slot];
}

Bytes CostModel::TempMemorySize(const Node* node) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_19(mht_19_v, 533, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::TempMemorySize");

  const int id = Id(node);
  if (id < 0) {
    return Bytes(0);
  }
  return max_mem_usage_[id].temp_memory_size;
}

Bytes CostModel::PersistentMemorySize(const Node* node) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_20(mht_20_v, 544, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::PersistentMemorySize");

  const int id = Id(node);
  if (id < 0) {
    return Bytes(0);
  }
  return max_mem_usage_[id].persistent_memory_size;
}

void CostModel::RecordMemoryStats(const Node* node,
                                  const MemoryStats& memory_stats) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_21(mht_21_v, 556, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordMemoryStats");

  const int id = Id(node);
  if (id < 0) return;
  max_mem_usage_[id].temp_memory_size = memory_stats.temp_memory_size();
  max_mem_usage_[id].persistent_memory_size =
      memory_stats.persistent_memory_size();
  for (int64_t alloc_id : memory_stats.persistent_tensor_alloc_ids()) {
    if (alloc_id > 0) {
      persistent_alloc_ids_.insert(alloc_id);
    }
  }
}

void CostModel::RecordMaxExecutionTime(const Node* node, Microseconds time) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_22(mht_22_v, 572, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordMaxExecutionTime");

  const int id = Id(node);
  if (id < 0) return;
  Ensure(id, node->num_outputs());
  max_exec_time_[id] = std::max(max_exec_time_[id], time);
}

Microseconds CostModel::MaxExecutionTime(const Node* node) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_23(mht_23_v, 582, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MaxExecutionTime");

  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= max_exec_time_.size()) {
    return Microseconds(0);
  }
  return max_exec_time_[id];
}

void CostModel::RecordAllocationId(const Node* node, int output_slot,
                                   int64_t alloc_id) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_24(mht_24_v, 594, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::RecordAllocationId");

  const int id = Id(node);
  if (id < 0) return;
  Ensure(id, node->num_outputs());
  output_port_alloc_ids_[id][output_slot] = alloc_id;
}

int64_t CostModel::AllocationId(const Node* node, int slot) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_25(mht_25_v, 604, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::AllocationId");

  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= output_port_alloc_ids_.size() ||
      output_port_alloc_ids_[id].size() <= static_cast<size_t>(slot)) {
    return -1;
  }
  return output_port_alloc_ids_[id][slot];
}

bool CostModel::IsPersistentTensor(const Node* node, int64_t alloc_id) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_26(mht_26_v, 616, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::IsPersistentTensor");

  if (persistent_alloc_ids_.count(alloc_id) > 0) {
    return true;
  }
  if (persistent_alloc_ids_by_devices_.find(node->assigned_device_name()) ==
      persistent_alloc_ids_by_devices_.end()) {
    return false;
  }
  return persistent_alloc_ids_by_devices_.at(node->assigned_device_name())
      .count(alloc_id);
}

Microseconds CostModel::CopyTimeEstimate(Bytes b, double network_latency_millis,
                                         double estimated_gbps) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_27(mht_27_v, 632, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::CopyTimeEstimate");

  // TODO(jeff,sanjay): estimate cost based on bandwidth along the
  // communication path and the type of transport we are using between
  // devices.
  //
  // We assume the copy time follows a linear model:
  //    copy_time = copy_bytes / rate + min_time
  int64_t copy_bytes = b.value();
  const double bytes_per_usec = estimated_gbps * 1000.0 / 8;
  const double min_micros = network_latency_millis * 1000.0;
  return Microseconds(
      static_cast<int64_t>(copy_bytes / bytes_per_usec + min_micros));
}

Microseconds CostModel::ComputationTimeEstimate(int64_t math_ops) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_28(mht_28_v, 649, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::ComputationTimeEstimate");

  // TODO(jeff,sanjay): Eventually we should pass in the type of device
  // (GPU vs. CPU) and use that to affect the estimate.

  // We estimate the microseconds using that value.  We divide
  // by 1000 to convert the madd number into microseconds (assuming
  // roughly 1000 madds per microsecond (~1 GHz for one core)).
  return Microseconds(math_ops / 1000);
}

void CostModel::IncrementUpdateTimes() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_29(mht_29_v, 662, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::IncrementUpdateTimes");
 update_times_++; }

int32 CostModel::GetUpdateTimes() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_30(mht_30_v, 667, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::GetUpdateTimes");
 return update_times_; }

// ----------------------------------------------------------------------------
// InitCostModel
// ----------------------------------------------------------------------------

namespace {

static void AddNodesToCostModel(const Graph& g, CostModel* cost_model) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_31(mht_31_v, 678, "", "./tensorflow/core/graph/costmodel.cc", "AddNodesToCostModel");

  for (Node* n : g.nodes()) {
    const int num_outputs = n->num_outputs();
    cost_model->SetNumOutputs(n, num_outputs);
    for (int output = 0; output < num_outputs; output++) {
      // Set up an initial bogus estimate for the node's outputs
      cost_model->RecordSize(n, output, Bytes(1));
    }
  }
}

static void AssignSizes(const Graph& g, CostModel* cost_model) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_32(mht_32_v, 692, "", "./tensorflow/core/graph/costmodel.cc", "AssignSizes");

  for (const Edge* e : g.edges()) {
    // Skip if it is a control edge.
    if (e->IsControlEdge()) {
      continue;
    }
    const Node* src = e->src();

    // TODO(josh11b): Get an estimate from the Op
    Bytes size(1);
    cost_model->RecordSize(src, e->src_output(), size);
  }
}

// This generates an extremely simple initial guess for the
// computation cost of each node. For ordinary Ops, its value should quickly
// be wiped out by the real runtime measurements.  For other Ops we don't
// actually generate measurements, so suppression of infrequent Ops ends up
// giving them 0 costs.  So, this is not of much consequence except perhaps
// in tests.
static Microseconds TimeEstimateForNode(CostModel* cost_model, Node* n) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_33(mht_33_v, 715, "", "./tensorflow/core/graph/costmodel.cc", "TimeEstimateForNode");

  CHECK(n->IsOp());
  VLOG(2) << "Node " << n->id() << ": " << n->name()
          << " type_string: " << n->type_string();
  if (IsConstant(n) || IsVariable(n)) {
    return Microseconds(0);
  }
  return kDefaultTimeEstimate;
}

static void EstimateComputationCosts(const Graph& g, CostModel* cost_model) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_34(mht_34_v, 728, "", "./tensorflow/core/graph/costmodel.cc", "EstimateComputationCosts");

  for (Node* n : g.nodes()) {
    if (!n->IsOp()) continue;
    cost_model->RecordTime(n, TimeEstimateForNode(cost_model, n));
  }
}

}  // namespace

void CostModel::InitFromGraph(const Graph& g) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_35(mht_35_v, 740, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::InitFromGraph");

  const int num_node_ids = g.num_node_ids();
  slot_bytes_.reserve(num_node_ids);
  count_.reserve(num_node_ids);
  time_.reserve(num_node_ids);
  max_mem_usage_.reserve(num_node_ids);
  max_exec_time_.reserve(num_node_ids);
  output_port_alloc_ids_.reserve(num_node_ids);

  AddNodesToCostModel(g, this);
  AssignSizes(g, this);
  EstimateComputationCosts(g, this);
  CheckInitialized(g);
}

void CostModel::AddToCostGraphDef(const Graph* graph,
                                  CostGraphDef* cost_graph) const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_36(mht_36_v, 759, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::AddToCostGraphDef");

  std::vector<const Edge*> inputs;
  std::vector<const Edge*> control_inputs;
  int offset = cost_graph->node_size();
  for (const Node* n : graph->nodes()) {
    CostGraphDef::Node* cnode = cost_graph->add_node();
    cnode->set_name(n->name());
    cnode->set_device(n->assigned_device_name());
    cnode->set_id(GlobalId(n, offset));

    inputs.clear();
    inputs.resize(n->num_inputs(), nullptr);
    control_inputs.clear();
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        control_inputs.push_back(e);
      } else {
        inputs[e->dst_input()] = e;
      }
    }
    std::sort(control_inputs.begin(), control_inputs.end(),
              [this](Edge const* a, Edge const* b) {
                return Id(a->src()) < Id(b->src());
              });

    for (const Edge* e : inputs) {
      CostGraphDef::Node::InputInfo* input_info = cnode->add_input_info();
      input_info->set_preceding_node(GlobalId(e->src(), offset));
      input_info->set_preceding_port(e->src_output());
    }

    for (int i = 0; i < n->num_outputs(); i++) {
      CostGraphDef::Node::OutputInfo* output_info = cnode->add_output_info();
      int64_t alloc_id = AllocationId(n, i);
      int64_t alias_to_input = -1;
      for (const Edge* e : inputs) {
        int64_t input_alloc_id = AllocationId(e->src(), e->src_output());
        if (input_alloc_id == alloc_id) {
          alias_to_input = e->dst_input();
          break;
        }
      }
      output_info->set_alias_input_port(alias_to_input);
      output_info->set_dtype(MaxMemoryType(n, i));
      *output_info->mutable_shape() = MaxMemoryShape(n, i);
      if (alias_to_input < 0 && IsPersistentTensor(n, alloc_id)) {
        output_info->set_size(0);
      } else {
        output_info->set_size(MaxMemorySize(n, i).value());
      }
    }

    for (const Edge* e : control_inputs) {
      cnode->add_control_input(GlobalId(e->src(), offset));
    }

    cnode->set_temporary_memory_size(TempMemorySize(n).value());
    cnode->set_persistent_memory_size(PersistentMemorySize(n).value());

    cnode->set_compute_cost(MaxExecutionTime(n).value());

    // For now we treat all send nodes as final.
    // TODO(yuanbyu): Send nodes for fetches shouldn't be treated as final.
    cnode->set_is_final(n->IsSend());
  }
}

void CostModel::WriteSummaryToLog() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_37(mht_37_v, 829, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::WriteSummaryToLog");

  LOG(INFO) << " min_count_=" << min_count_;
  for (size_t i = 0; i < count_.size(); ++i) {
    LOG(INFO) << "Node " << i << " count " << count_[i] << " total time "
              << time_[i] << " avg time "
              << (time_[i] / (std::max(1, count_[i])));
  }
}

Bytes CostModel::MinTensorMemoryUsage(const TensorShapeProto& tensor_shape,
                                      const DataType& dtype) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgraphPScostmodelDTcc mht_38(mht_38_v, 842, "", "./tensorflow/core/graph/costmodel.cc", "CostModel::MinTensorMemoryUsage");

  if (tensor_shape.unknown_rank()) {
    return Bytes(-1);
  }

  size_t num_coefficients = 1;
  for (const TensorShapeProto::Dim& dim : tensor_shape.dim()) {
    // If the dimension is unknown, it has to be at least 1
    num_coefficients *= std::max<size_t>(dim.size(), 1);
  }
  return Bytes(num_coefficients * DataTypeSize(dtype));
}

}  // namespace tensorflow
