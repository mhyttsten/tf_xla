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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc() {
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

#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tracking_allocator.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {
const int kMaxAllocReportNodes = 100;
const float kMaxAllocReportFraction = 0.99;

struct AllocStats {
  std::map<int64_t, std::vector<string>> nodes_by_size;
  int64_t total_bytes = 0;
  int64_t total_nodes = 0;
};

bool IsRecv(const NodeDef* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "IsRecv");

  return node->op() == "_Recv" || node->op() == "_HostRecv";
}

bool IsSend(const NodeDef* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "IsSend");

  return node->op() == "_Send" || node->op() == "_HostSend";
}

}  // namespace

NodeExecStatsWrapper::NodeExecStatsWrapper(
    const NodeDef* node, StepStatsCollector* step_stats_collector)
    : NodeExecStatsWrapper(MakeUnique<NodeExecStats>(), node,
                           step_stats_collector) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::NodeExecStatsWrapper");

  stats_->set_node_name(node->name());
}

NodeExecStatsWrapper::NodeExecStatsWrapper(
    std::unique_ptr<NodeExecStats> stats, const NodeDef* node,
    StepStatsCollector* step_stats_collector)
    : stats_(std::move(stats)),
      node_(node),
      step_stats_collector_(step_stats_collector) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::NodeExecStatsWrapper");
}

void NodeExecStatsWrapper::Done(const string& device) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_4(mht_4_v, 249, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::Done");

  // TODO(tucker): merge with the DetailText function in session.cc in a common
  // location.
  DCHECK(node_);
  string memory;
  for (auto& all : stats_->memory()) {
    int64_t tot = all.total_bytes();
    if (tot >= 0.1 * 1048576.0) {
      int64_t peak = all.peak_bytes();
      if (peak > 0) {
        memory =
            strings::StrCat(memory, "[", all.allocator_name(),
                            strings::Printf(" %.1fMB %.1fMB] ", tot / 1048576.0,
                                            peak / 1048576.0));
      } else {
        memory = strings::StrCat(memory, "[", all.allocator_name(),
                                 strings::Printf(" %.1fMB] ", tot / 1048576.0));
      }
    }
  }
  const AttrSlice attrs(*node_);
  string text;
  if (IsSend(node_)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(attrs, "tensor_name", &tensor_name));
    string recv_device;
    TF_CHECK_OK(GetNodeAttr(attrs, "recv_device", &recv_device));
    text = strings::StrCat(memory, node_->name(), " = ", node_->op(), "(",
                           tensor_name, " @", recv_device, ")");
  } else if (IsRecv(node_)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(attrs, "tensor_name", &tensor_name));
    string send_device;
    TF_CHECK_OK(GetNodeAttr(attrs, "send_device", &send_device));
    text = strings::StrCat(memory, node_->name(), " = ", node_->op(), "(",
                           tensor_name, " @", send_device, ")");
  } else {
    text = strings::StrCat(memory, node_->name(), " = ", node_->op(), "(",
                           absl::StrJoin(node_->input(), ", "), ")");
  }
  stats_->set_timeline_label(text);
  step_stats_collector_->Save(device, this);
}

void NodeExecStatsWrapper::RecordExecutorStarted() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_5(mht_5_v, 296, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::RecordExecutorStarted");

  int64_t now_nanos = Env::Default()->NowNanos();
  stats_->set_all_start_micros(now_nanos / EnvTime::kMicrosToNanos);
  stats_->set_all_start_nanos(now_nanos);
}

void NodeExecStatsWrapper::RecordComputeStarted() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::RecordComputeStarted");

  int64_t now_nanos = Env::Default()->NowNanos();
  DCHECK_NE(stats_->all_start_micros(), 0);
  DCHECK_NE(stats_->all_start_nanos(), 0);
  stats_->set_op_start_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                                  stats_->all_start_micros());
  stats_->set_op_start_rel_nanos(now_nanos - stats_->all_start_nanos());
}

void NodeExecStatsWrapper::RecordComputeEnded() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_7(mht_7_v, 317, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::RecordComputeEnded");

  int64_t now_nanos = Env::Default()->NowNanos();
  DCHECK_NE(stats_->all_start_micros(), 0);
  DCHECK_NE(stats_->all_start_nanos(), 0);
  stats_->set_op_end_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                                stats_->all_start_micros());
  stats_->set_op_end_rel_nanos(now_nanos - stats_->all_start_nanos());
}

void NodeExecStatsWrapper::RecordExecutorEnded() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_8(mht_8_v, 329, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::RecordExecutorEnded");

  int64_t now_nanos = Env::Default()->NowNanos();
  DCHECK_NE(stats_->all_start_micros(), 0);
  DCHECK_NE(stats_->all_start_nanos(), 0);
  stats_->set_all_end_rel_micros(now_nanos / EnvTime::kMicrosToNanos -
                                 stats_->all_start_micros());
  stats_->set_all_end_rel_nanos(now_nanos - stats_->all_start_nanos());
}

void NodeExecStatsWrapper::SetScheduled(int64_t nanos) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_9(mht_9_v, 341, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::SetScheduled");

  stats_->set_scheduled_micros(nanos / EnvTime::kMicrosToNanos);
  stats_->set_scheduled_nanos(nanos);
}

void NodeExecStatsWrapper::SetMemory(OpKernelContext* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_10(mht_10_v, 349, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::SetMemory");

  for (const auto& allocator_pair : ctx->ConsumeWrappedAllocators()) {
    AddAllocation(allocator_pair.first, allocator_pair.second);
  }
  auto* ms = stats_->mutable_memory_stats();
  ms->set_temp_memory_size(ctx->temp_memory_allocated());
  for (const auto& alloc_id : ctx->persistent_alloc_ids()) {
    ms->mutable_persistent_tensor_alloc_ids()->Add(alloc_id);
  }
  ms->set_persistent_memory_size(ctx->persistent_memory_allocated());
}

void NodeExecStatsWrapper::SetOutput(int slot, const Tensor* tensor) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_11(mht_11_v, 364, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::SetOutput");

  DCHECK(tensor);
  NodeOutput* node_output = stats_->add_output();
  node_output->set_slot(slot);
  tensor->FillDescription(node_output->mutable_tensor_description());
}

void NodeExecStatsWrapper::AddAllocation(
    Allocator* allocator, TrackingAllocator* tracking_allocator) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_12(mht_12_v, 375, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::AddAllocation");

  AllocatorMemoryUsed* memory = stats_->add_memory();
  memory->set_allocator_name(allocator->Name());
  auto sizes = tracking_allocator->GetSizes();
  memory->set_total_bytes(std::get<0>(sizes));
  memory->set_peak_bytes(std::get<1>(sizes));
  memory->set_live_bytes(std::get<2>(sizes));

  absl::optional<AllocatorStats> stats = allocator->GetStats();
  if (stats) {
    memory->set_allocator_bytes_in_use(stats->bytes_in_use);
  }
  allocations_.push_back(std::make_pair(memory, tracking_allocator));
}

void NodeExecStatsWrapper::Finalize() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_13(mht_13_v, 393, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "NodeExecStatsWrapper::Finalize");

  for (auto& alloc : allocations_) {
    AllocatorMemoryUsed* memory = alloc.first;
    for (auto& record : alloc.second->GetRecordsAndUnRef()) {
      auto* r = memory->add_allocation_records();
      r->set_alloc_bytes(record.alloc_bytes);
      r->set_alloc_micros(record.alloc_micros);
    }
  }
  allocations_.clear();
}

StepStatsCollector::StepStatsCollector(StepStats* step_stats)
    : finalized_(false), step_stats_(step_stats) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_14(mht_14_v, 409, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::StepStatsCollector");
}

static int ExtractGpuWithStreamAll(string device_name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_15(mht_15_v, 415, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "ExtractGpuWithStreamAll");

  // Check if the device name matches the ".*gpu:(\\d+)/stream:all$" regexp,
  // and if it does return the stream index (always positive). If it doesn't
  // return -1.

  // The best way to parse this regexp using a scanner is to parse it in
  // reverse starting from the end.
  std::reverse(device_name.begin(), device_name.end());
  strings::Scanner scanner(device_name);
  // Check that the string end with '/stream:all'
  scanner.OneLiteral("lla:maerts/");
  // Capture the digits if present
  scanner.RestartCapture().Many(strings::Scanner::DIGIT).StopCapture();
  // Check that the digits are preceded by the 'device:GPU:' string
  scanner.OneLiteral(":UPG:ecived");
  StringPiece capture;
  bool matched = scanner.GetResult(nullptr, &capture);

  if (!matched) {
    return -1;
  } else {
    // Convert the captured string into an integer. But first we need to put
    // the digits back in order
    string ordered_capture(capture);
    std::reverse(ordered_capture.begin(), ordered_capture.end());
    int gpu_id;
    CHECK(strings::safe_strto32(ordered_capture, &gpu_id));
    return gpu_id;
  }
}

static int ExtractGpuWithoutStream(string device_name) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_16(mht_16_v, 450, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "ExtractGpuWithoutStream");

  // Check if the device name matches the ".*gpu:(\\d+)$" regexp,
  // and if it does return the stream index (always positive). If it doesn't
  // return -1.

  // The best way to parse this regexp using a scanner is to parse it in
  // reverse starting from the end.
  std::reverse(device_name.begin(), device_name.end());
  strings::Scanner scanner(device_name);
  // Capture the trailing digits if present
  scanner.RestartCapture().Many(strings::Scanner::DIGIT).StopCapture();
  // Check that the digits are preceded by the 'device:GPU:' string
  scanner.OneLiteral(":UPG:ecived");
  StringPiece capture;
  bool matched = scanner.GetResult(nullptr, &capture);

  if (!matched) {
    return -1;
  } else {
    // Convert the captured string into an integer. But first we need to put
    // the digits back in order
    string ordered_capture(capture);
    std::reverse(ordered_capture.begin(), ordered_capture.end());
    int gpu_id;
    CHECK(strings::safe_strto32(ordered_capture, &gpu_id));
    return gpu_id;
  }
}

void StepStatsCollector::BuildCostModel(
    CostModelManager* cost_model_manager,
    const std::unordered_map<string, const Graph*>& device_map) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_17(mht_17_v, 484, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::BuildCostModel");

  mutex_lock lock(mu_);

  if (!finalized_) {
    FinalizeInternal();
  }
  // Hardware stats for gpu are available under a fake device named
  // "gpu:<id>/stream::all.
  // Use them instead of regular stats whenever they're available to extract
  // the execution stats of a particular node since they're more accurate.
  // However hardware traces don't record memory usage, so we still have to
  // rely on regular traces to track memory usage.
  struct DeviceStats {
    const DeviceStepStats* regular_stats;
    const DeviceStepStats* hardware_stats;
  };

  std::unordered_map<StringPiece, DeviceStats, StringPieceHasher>
      per_device_stats;
  std::unordered_map<int, const DeviceStepStats*> gpu_hardware_stats;

  for (int i = 0; i < step_stats_->dev_stats_size(); ++i) {
    const DeviceStepStats& device_stats = step_stats_->dev_stats(i);
    const string& device_name = device_stats.device();
    const int gpu_id = ExtractGpuWithStreamAll(device_name);
    if (gpu_id >= 0) {
      // These are gpu hardware stats
      gpu_hardware_stats.emplace(gpu_id, &device_stats);
    } else {
      // These are regular stats.
      per_device_stats.emplace(device_name,
                               DeviceStats{&device_stats, nullptr});
    }
  }

  for (auto& itr : per_device_stats) {
    const StringPiece device_name = itr.first;
    const int gpu_id = ExtractGpuWithoutStream(string(device_name));
    if (gpu_id >= 0) {
      // Reference the gpu hardware stats in addition to the regular stats
      // for this gpu device if they're available.
      if (gpu_hardware_stats.find(gpu_id) != gpu_hardware_stats.end()) {
        itr.second.hardware_stats = gpu_hardware_stats.find(gpu_id)->second;
      }
    }
  }

  for (const auto& itr : device_map) {
    const StringPiece device = itr.first;
    if (per_device_stats.find(device) == per_device_stats.end()) {
      continue;
    }

    const Graph* graph = itr.second;
    CostModel* cm = cost_model_manager->FindOrCreateCostModel(graph);
    cm->IncrementUpdateTimes();

    std::unordered_map<StringPiece, Node*, StringPieceHasher> name_to_node;
    for (Node* n : graph->nodes()) {
      name_to_node.emplace(n->name(), n);
    }

    const DeviceStats& dev_stats = per_device_stats.find(device)->second;

    std::unordered_map<string, NodeExecStats> name_to_hw_node_stats;
    if (dev_stats.hardware_stats) {
      for (const auto& node_stats : dev_stats.hardware_stats->node_stats()) {
        string node_name = node_stats.node_name();
        // Remove the part of op name (e.g. :Conv2D) in the end of a node name.
        size_t pos = node_name.find_first_of(':');
        if (pos != std::string::npos) {
          node_name = node_name.substr(0, pos);
        }
        // Certain ops (e.g. Conv2D) are implemented with multiple GPU kernels,
        // which results in multiple NodeExecStats with the same node name. For
        // such ops, we sum up the time for all its GPU kernels.
        if (name_to_hw_node_stats.find(node_name) !=
            name_to_hw_node_stats.end()) {
          int64_t time = name_to_hw_node_stats[node_name].op_end_rel_micros();
          name_to_hw_node_stats[node_name].set_op_end_rel_micros(
              time + node_stats.op_end_rel_micros());
        } else {
          name_to_hw_node_stats.emplace(node_name, node_stats);
        }
      }
    }

    for (int i = 0; i < dev_stats.regular_stats->node_stats_size(); ++i) {
      const NodeExecStats& stats = dev_stats.regular_stats->node_stats(i);
      const Node* node = name_to_node[stats.node_name()];
      if (node) {
        for (int i = 0; i < stats.output_size(); ++i) {
          const auto& output = stats.output(i);
          int output_slot = output.slot();
          cm->RecordMaxMemorySize(node, output_slot,
                                  Bytes(output.tensor_description()
                                            .allocation_description()
                                            .allocated_bytes()),
                                  output.tensor_description().shape(),
                                  node->output_types()[output_slot]);
          cm->RecordAllocationId(node, output_slot,
                                 output.tensor_description()
                                     .allocation_description()
                                     .allocation_id());
        }
        cm->RecordMemoryStats(node, stats.memory_stats());
        // Use hardware stats to record the execution time if they're available,
        // otherwise use the regular (less accurate) stats
        string node_name = dev_stats.regular_stats->node_stats(i).node_name();
        if (dev_stats.hardware_stats && name_to_hw_node_stats.find(node_name) !=
                                            name_to_hw_node_stats.end()) {
          const NodeExecStats& hw_stats = name_to_hw_node_stats[node_name];
          cm->RecordMaxExecutionTime(
              node, Microseconds(hw_stats.op_end_rel_micros()));
        } else {
          cm->RecordMaxExecutionTime(node,
                                     Microseconds(stats.op_end_rel_micros()));
        }
      }
    }
  }
}

void StepStatsCollector::Save(const string& device,
                              NodeExecStats* node_stats_pb) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_18(mht_18_v, 612, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::Save");

  Save(device,
       new NodeExecStatsWrapper(std::unique_ptr<NodeExecStats>(node_stats_pb),
                                nullptr, this));
}

void StepStatsCollector::Save(const string& device,
                              NodeExecStatsWrapper* node_stats) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_19(mht_19_v, 623, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::Save");

  if (!node_stats) return;
  VLOG(1) << "Save dev " << device << " node stats " << node_stats->stats();
  {
    mutex_lock l(mu_);
    if (finalized_) {
      LOG(WARNING) << "stats saved after finalize will not be collected.";
    }
    if (!step_stats_ || collected_nodes_ >= kMaxCollectedNodes) {
      VLOG(1) << "step_stats_ nullptr or already collected too many nodes.";
      delete node_stats;
      return;
    }
    auto& device_stats = dev_stats_[device];
    device_stats.push_back(std::unique_ptr<NodeExecStatsWrapper>(node_stats));
    collected_nodes_++;
  }
}

void StepStatsCollector::SaveThreadName(const string& device,
                                        const uint32 thread_id,
                                        const string& thread_name) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("device: \"" + device + "\"");
   mht_20_v.push_back("thread_name: \"" + thread_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_20(mht_20_v, 649, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::SaveThreadName");

  VLOG(1) << "Save dev " << device << " thread id " << thread_id << " name "
          << thread_name;
  {
    mutex_lock l(mu_);
    if (finalized_) {
      LOG(WARNING) << "thread_name saved after finalize will not be collected.";
    }
    auto& thread_names_map = thread_names_[device];
    thread_names_map[thread_id] = thread_name;
  }
}

NodeExecStatsInterface* StepStatsCollector::CreateNodeExecStats(
    const NodeDef* node) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_21(mht_21_v, 666, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::CreateNodeExecStats");

  // Only collect statistics for non-transfer nodes.
  if (IsSend(node) || IsRecv(node)) {
    return nullptr;
  }
  return new NodeExecStatsWrapper(node, this);
}

string StepStatsCollector::ReportAllocsOnResourceExhausted(const string& err) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("err: \"" + err + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_22(mht_22_v, 678, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::ReportAllocsOnResourceExhausted");

  mutex_lock l(mu_);
  if (err.find("OOM") == err.npos) {
    return "";
  }
  // <device, allocator> -> AllocStats
  std::map<std::pair<string, string>, AllocStats> allocs_map;
  string report = "\n";
  for (const auto& dev_stat : dev_stats_) {
    const string& device = dev_stat.first;
    // Only print the device that has OOM.
    // TODO(xpan): Extract device from err first to speed it up.
    if (err.find(device) == err.npos) {
      continue;
    }
    // NodeExecStatsWrapper*
    for (const auto& stats : dev_stat.second) {
      // std::pair<AllocatorMemoryUsed*, TrackingAllocator*>
      for (const auto& alloc : stats->allocations_) {
        // Only print the allocator that has OOM.
        // TODO(xpan): Extract device from err first to speed it up.
        if (err.find(alloc.first->allocator_name()) == err.npos) {
          continue;
        }
        auto dev_allocator =
            std::make_pair(dev_stat.first, alloc.first->allocator_name());
        AllocStats& dev_allocs_stats = allocs_map[dev_allocator];
        TrackingAllocator* tracking_alloc = alloc.second;
        gtl::InlinedVector<AllocRecord, 4> cur_records =
            tracking_alloc->GetCurrentRecords();
        int64_t cur_bytes = 0;
        for (const auto& r : cur_records) {
          cur_bytes += r.alloc_bytes;
        }
        if (cur_bytes > 0) {
          dev_allocs_stats.total_bytes += cur_bytes;
          dev_allocs_stats.total_nodes++;
          dev_allocs_stats.nodes_by_size[cur_bytes].push_back(
              stats->stats()->node_name());
        }
      }
    }
  }

  for (const auto& dev_allocs_it : allocs_map) {
    const auto& dev = dev_allocs_it.first;
    const AllocStats& dev_allocs_stats = dev_allocs_it.second;
    int64_t reported_bytes = 0;
    int64_t reported_nodes = 0;
    bool done = false;
    strings::StrAppend(&report, "\nCurrent usage from device: ", dev.first,
                       ", allocator: ", dev.second, "\n");
    // Print allocations stats of the <device, allocator> pair.
    for (auto it = dev_allocs_stats.nodes_by_size.rbegin();
         it != dev_allocs_stats.nodes_by_size.rend(); ++it) {
      for (const string& node_name : it->second) {
        reported_bytes += it->first;
        strings::StrAppend(&report, "  ",
                           strings::HumanReadableNumBytes(it->first), " from ",
                           node_name, "\n");
        if (++reported_nodes > kMaxAllocReportNodes ||
            reported_bytes >=
                dev_allocs_stats.total_bytes * kMaxAllocReportFraction) {
          done = true;
          break;
        }
      }
      if (done) break;
    }
    int64_t remain_nodes = dev_allocs_stats.total_nodes - reported_nodes;
    int64_t remain_bytes = dev_allocs_stats.total_bytes - reported_bytes;
    if (remain_nodes > 0) {
      strings::StrAppend(&report, "  Remaining ", remain_nodes, " nodes with ",
                         strings::HumanReadableNumBytes(remain_bytes), "\n");
    }
  }
  return report;
}

void StepStatsCollector::Finalize() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_23(mht_23_v, 760, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::Finalize");

  mutex_lock l(mu_);
  FinalizeInternal();
}

void StepStatsCollector::FinalizeAndSwap(StepStats* step_stats) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_24(mht_24_v, 768, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::FinalizeAndSwap");

  mutex_lock l(mu_);
  CHECK(step_stats_);
  FinalizeInternal();
  step_stats->Swap(step_stats_);
  collected_nodes_ = 0;
}

void StepStatsCollector::FinalizeInternal() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSstep_stats_collectorDTcc mht_25(mht_25_v, 779, "", "./tensorflow/core/common_runtime/step_stats_collector.cc", "StepStatsCollector::FinalizeInternal");

  if (!step_stats_ || finalized_) {
    return;
  }
  finalized_ = true;
  std::map<string, DeviceStepStats*> dev_stats_pb;
  for (auto& ds : *step_stats_->mutable_dev_stats()) {
    dev_stats_pb[ds.device()] = &ds;
  }
  for (const auto& dev_stat : dev_stats_) {
    if (dev_stats_pb.find(dev_stat.first) == dev_stats_pb.end()) {
      DeviceStepStats* ndev_stat = step_stats_->add_dev_stats();
      ndev_stat->set_device(dev_stat.first);
      dev_stats_pb[dev_stat.first] = ndev_stat;
    }
    DeviceStepStats* dss = dev_stats_pb.at(dev_stat.first);
    for (auto& stats : dev_stat.second) {
      stats->Finalize();
      stats->stats()->Swap(dss->add_node_stats());
    }
  }
  for (const auto& device_thread : thread_names_) {
    if (dev_stats_pb.find(device_thread.first) == dev_stats_pb.end()) {
      // skip device without DeviceStepStats.
      continue;
    }
    DeviceStepStats* dss = dev_stats_pb.at(device_thread.first);
    for (const auto& thread_name : device_thread.second) {
      (*dss->mutable_thread_names())[thread_name.first] = thread_name.second;
    }
  }
}
}  // namespace tensorflow
