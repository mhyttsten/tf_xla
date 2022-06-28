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
class MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc() {
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
#include "tensorflow/core/common_runtime/hierarchical_tree_broadcaster.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"

// Set true for greater intelligibility of debug mode log messages.
#define READABLE_KEYS false

namespace tensorflow {

namespace {
// Key to be used for BufRendezvous by Broadcaster.
string BroadcastBufKey(const string& exec_key, int subdiv, int src_rank,
                       int dst_rank) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("exec_key: \"" + exec_key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "BroadcastBufKey");

  if (READABLE_KEYS) {
    return strings::StrCat("broadcast(", exec_key, "):subdiv(", subdiv,
                           "):src(", src_rank, "):dst(", dst_rank, ")");
  } else {
    // TODO(b/78352018): Try a denser format, e.g. a 64 or 128 bit hash.
    return strings::StrCat(exec_key, ":", subdiv, ":", src_rank, ":", dst_rank);
  }
}
}  // namespace

HierarchicalTreeBroadcaster::HierarchicalTreeBroadcaster()
    : col_ctx_(nullptr),
      col_params_(nullptr),
      done_(nullptr),
      is_source_(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::HierarchicalTreeBroadcaster");
}

int HierarchicalTreeBroadcaster::GetDeviceTask(
    int device_rank, const std::vector<int>& dev_per_task) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::GetDeviceTask");

  int num_tasks = static_cast<int>(dev_per_task.size());
  int task_lo = 0;
  int task_hi = -1;
  for (int ti = 0; ti < num_tasks; ti++) {
    task_hi = task_lo + dev_per_task[ti];
    if (task_lo <= device_rank && device_rank < task_hi) return ti;
    task_lo = task_hi;
  }
  LOG(FATAL) << "Unexpected device rank " << device_rank << " for " << task_hi
             << " devices";
  return -1;
}

Status HierarchicalTreeBroadcaster::InitializeCollectiveParams(
    CollectiveParams* col_params) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::InitializeCollectiveParams");

  CHECK_EQ(col_params->instance.type, BROADCAST_COLLECTIVE);
  CHECK_EQ(col_params->instance.impl_details.collective_name,
           "HierarchicalTreeBroadcast");
  const string& device_name =
      col_params->group.members[col_params->default_rank].device.name();
  // Start by counting the devices in each task.
  // Precondition: device_names must be sorted so that all devices in
  // the same task are adjacent.
  std::vector<int> dev_per_task;
  const string* prior_task_name = &col_params->group.members[0].task;
  int dev_count = 1;
  for (int di = 1; di < col_params->group.group_size; ++di) {
    if (col_params->group.members[di].task != *prior_task_name) {
      dev_per_task.push_back(dev_count);
      dev_count = 1;
      prior_task_name = &col_params->group.members[di].task;
    } else {
      ++dev_count;
    }
  }
  dev_per_task.push_back(dev_count);
  CHECK_EQ(col_params->group.num_tasks, dev_per_task.size());

  if (VLOG_IS_ON(2)) {
    string dpt_buf;
    for (int dpt : dev_per_task) strings::StrAppend(&dpt_buf, dpt, ";");
    VLOG(2) << "HierarchicalTreeBroadcaster::InitializeCollectiveParams device="
            << device_name << " source_rank=" << col_params->source_rank
            << " dev_per_task=" << dpt_buf;
  }
  int num_tasks = col_params->group.num_tasks;
  // If there is just 1 task, then execute binary tree broadcast over all
  // devices.  Otherwise, the first subdiv is inter-task broadcast, and then
  // there are N more subdivs, where N is #task.
  int num_subdivs = num_tasks + (num_tasks > 1 ? 1 : 0);
  int total_num_devices = 0;
  for (int num_dev : dev_per_task) total_num_devices += num_dev;

  col_params->instance.impl_details.subdiv_permutations.resize(num_subdivs);
  col_params->subdiv_rank.reserve(num_subdivs);
  col_params->instance.impl_details.subdiv_source_rank.reserve(num_subdivs);

  // Inter-task subdiv.  Pick one device from each task - this is the source
  // device if it belongs to that task, or device 0 for that task.  If a device
  // does not participate in the subdiv, set subdiv_rank to -1.
  if (num_tasks > 1) {
    const int sdi = 0;
    std::vector<int>& perm =
        col_params->instance.impl_details.subdiv_permutations[sdi];
    CHECK_EQ(perm.size(), 0);
    int device_count = 0;
    int source_task = GetDeviceTask(col_params->source_rank, dev_per_task);
    for (int ti = 0; ti < col_params->group.num_tasks; ti++) {
      bool participate = false;
      if (source_task == ti) {
        // Source device belongs to this task.
        perm.push_back(col_params->source_rank);
        participate =
            col_params->group.members[col_params->source_rank].device.name() ==
            device_name;
      } else {
        // Source does not belong to this task, choose dev 0.
        perm.push_back(device_count);
        participate = col_params->group.members[device_count].device.name() ==
                      device_name;
      }
      if (participate) col_params->subdiv_rank.push_back(ti);
      device_count += dev_per_task[ti];
    }
    if (col_params->subdiv_rank.empty()) col_params->subdiv_rank.push_back(-1);
    col_params->instance.impl_details.subdiv_source_rank.push_back(source_task);
  }
  VLOG(2) << collective_util::SubdivPermDebugString(*col_params);

  // Intra-task subdivs.  Pick all devices in task ti for subdiv sdi.  Set
  // source to dev 0 for that task if it does not contain original source, else
  // set to rank of original source.  If a device does not participate in
  // the subdiv, set subdiv_rank to -1;
  int abs_di = 0;
  for (int ti = 0; ti < col_params->group.num_tasks; ti++) {
    const int sdi = ti + (num_tasks > 1 ? 1 : 0);
    std::vector<int>& perm =
        col_params->instance.impl_details.subdiv_permutations[sdi];
    CHECK_EQ(perm.size(), 0);
    bool participate = false;
    int subdiv_source = 0;
    for (int di = 0; di < dev_per_task[ti]; di++) {
      perm.push_back(abs_di);
      if (col_params->group.members[abs_di].device.name() == device_name) {
        participate = true;
        col_params->subdiv_rank.push_back(di);
      }
      if (abs_di == col_params->source_rank) subdiv_source = di;
      abs_di++;
    }
    if (!participate) col_params->subdiv_rank.push_back(-1);
    col_params->instance.impl_details.subdiv_source_rank.push_back(
        subdiv_source);
  }

  for (int sri = 0; sri < num_subdivs; sri++) {
    CHECK_GE(col_params->instance.impl_details.subdiv_source_rank[sri], 0);
  }

  VLOG(2) << collective_util::SubdivPermDebugString(*col_params);
  return Status::OK();
}

Status HierarchicalTreeBroadcaster::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_4(mht_4_v, 371, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::InitializeCollectiveContext");

  CHECK(col_ctx->dev_mgr);
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

void HierarchicalTreeBroadcaster::Run(StatusCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_5(mht_5_v, 383, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::Run");

  CHECK(col_ctx_);
  CHECK(col_params_);
  done_ = std::move(done);
  is_source_ = col_params_->is_source;
  RunTree();
}

// Binary tree parent/child relations are trivial to calculate, i.e.
// device at rank r is the parent of 2r+1 and 2r+2.  The one exception
// is if the source is not rank 0.  We treat that case as though the
// source is appended to the front of the rank ordering as well as
// continuing to occupy its current position.  Hence we calculate as
// though each device's rank is actually r+1, then subtract 1 again to
// get the descendent ranks.  If the source is not rank 0 then its
// descendants include both {0,1} and the descendents of its current
// position.  Where a non-0-rank source is a descendent of another
// device, no send to it is necessary.

/* static*/
int HierarchicalTreeBroadcaster::TreeRecvFrom(const CollectiveParams& cp,
                                              int subdiv) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_6(mht_6_v, 407, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::TreeRecvFrom");

  DCHECK_LT(subdiv, static_cast<int>(cp.subdiv_rank.size()));
  int my_rank = cp.subdiv_rank[subdiv];
  if (-1 == my_rank) return -1;

  const auto& impl = cp.instance.impl_details;
  DCHECK_LT(subdiv, static_cast<int>(impl.subdiv_source_rank.size()));
  int source_rank = impl.subdiv_source_rank[subdiv];
  if (my_rank == source_rank) return -1;
  if (source_rank == 0) {
    return (my_rank - 1) / 2;
  } else {
    int predecessor_rank = (my_rank / 2) - 1;
    return (predecessor_rank < 0) ? source_rank : predecessor_rank;
  }
}

/* static */
void HierarchicalTreeBroadcaster::TreeSendTo(const CollectiveParams& cp,
                                             int subdiv,
                                             std::vector<int>* targets) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_7(mht_7_v, 430, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::TreeSendTo");

  DCHECK_LT(subdiv, static_cast<int>(cp.subdiv_rank.size()));
  int my_rank = cp.subdiv_rank[subdiv];
  if (-1 == my_rank) return;

  const auto& impl = cp.instance.impl_details;
  DCHECK_LT(subdiv, static_cast<int>(impl.subdiv_source_rank.size()));
  int source_rank = impl.subdiv_source_rank[subdiv];

  int group_size = 0;
  for (int i = 0; i < impl.subdiv_permutations[subdiv].size(); i++) {
    if (impl.subdiv_permutations[subdiv][i] >= 0) {
      group_size++;
    }
  }

  targets->clear();
  int successor_rank = 0;
  if (source_rank == 0) {
    successor_rank = (2 * my_rank) + 1;
  } else {
    successor_rank = (2 * (my_rank + 1));
  }
  DCHECK_NE(successor_rank, my_rank);
  if (cp.is_source && source_rank != 0) {
    // The source sends to rank 0,1 in addition to its positional
    // descendants.
    if (group_size > 1) {
      targets->push_back(0);
    }
    if (group_size > 2 && source_rank != 1) {
      targets->push_back(1);
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (successor_rank < group_size && successor_rank != source_rank) {
      targets->push_back(successor_rank);
    }
    ++successor_rank;
  }
}

// Executes a hierarchical tree broadcast.
// Each subdiv is a broadcast between a subset of the devices.
// If there is only one task, there is one subdiv comprising a broadcast between
// all devices belonging to the task.
// If there are n tasks, n>1, then there are n+1 subdivs.  In the first (global)
// subdiv, one device from each task participates in a binary tree broadcast.
// Each task receives a copy of the tensor on one device via this broadcast.
// Subsequent subdivs correspond to intra-task broadcasts.  Subdiv i+1
// corresponds to broadcast between all devices on task i.  Thus, each task
// participates in at most 2 subdivs.
void HierarchicalTreeBroadcaster::RunTree() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_8(mht_8_v, 485, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::RunTree");

  int num_subdivs = static_cast<int>(col_params_->subdiv_rank.size());
  // TODO(b/78352018): this is easily improved when a node participates in both
  // first and second subdivision.  It would first send to its descendents in
  // the first subdiv, then wait until all pending ops are finished before
  // sending to descendents in second subdiv.  A better implementation would
  // collapse the two send blocks.
  for (int si = 0; si < num_subdivs; si++) {
    int my_rank = col_params_->subdiv_rank[si];
    // If rank is -1, this device does not participate in this subdiv.
    if (-1 == my_rank) continue;
    int source_rank = col_params_->instance.impl_details.subdiv_source_rank[si];
    if (VLOG_IS_ON(1)) {
      string subdiv_buf;
      for (int r : col_params_->instance.impl_details.subdiv_permutations[si]) {
        strings::StrAppend(&subdiv_buf, r, ",");
      }
      VLOG(1) << "Running Broadcast tree device=" << col_ctx_->device_name
              << " subdiv=" << si << " perm=" << subdiv_buf
              << " my_rank=" << my_rank << " source_rank=" << source_rank;
    }

    mutex mu;               // also guards status_ while callbacks are pending
    int pending_count = 0;  // TF_GUARDED_BY(mu)
    condition_variable all_done;

    if (my_rank >= 0 && my_rank != source_rank) {
      // Begin by receiving the value.
      profiler::TraceMe activity(
          [&] { return strings::StrCat("ReceiveValue:", si); },
          profiler::TraceMeLevel::kInfo);
      int recv_from_rank = TreeRecvFrom(*col_params_, si);
      Notification note;
      DispatchRecv(si, recv_from_rank, my_rank, col_ctx_->output,
                   [this, &mu, &note](const Status& s) {
                     mutex_lock l(mu);
                     status_.Update(s);
                     note.Notify();
                   });
      note.WaitForNotification();
    }

    // Then forward value to all descendent devices.
    {
      profiler::TraceMe activity(
          [&] { return strings::StrCat("ForwardValue:", si); },
          profiler::TraceMeLevel::kInfo);
      if (my_rank >= 0 && status_.ok()) {
        std::vector<int> send_to_ranks;
        TreeSendTo(*col_params_, si, &send_to_ranks);
        for (int i = 0; i < send_to_ranks.size(); ++i) {
          int target_rank = send_to_ranks[i];
          {
            mutex_lock l(mu);
            ++pending_count;
          }
          DispatchSend(si, target_rank, my_rank,
                       (is_source_ ? col_ctx_->input : col_ctx_->output),
                       [this, &mu, &pending_count, &all_done](const Status& s) {
                         mutex_lock l(mu);
                         status_.Update(s);
                         --pending_count;
                         if (pending_count == 0) {
                           all_done.notify_all();
                         }
                       });
        }
      }

      // For the original source device, we copy input to output if they are
      // different.
      // If there is only 1 subdiv, we do this in that subdiv.  If there is more
      // than 1 subdiv, then the original source device will participate in 2
      // subdivs - the global inter-task broadcast and one local intra-task
      // broadcast.  In this case, we perform the copy in the second subdiv for
      // this device.
      if (status_.ok() && is_source_ && (1 == num_subdivs || 0 != si)) {
        VLOG(2) << "copying input to output for device="
                << col_ctx_->device_name << " subdiv=" << si;
        if (col_ctx_->input != col_ctx_->output &&
            (DMAHelper::base(col_ctx_->input) !=
             DMAHelper::base(col_ctx_->output))) {
          {
            mutex_lock l(mu);
            ++pending_count;
          }
          DeviceContext* op_dev_ctx = col_ctx_->op_ctx->op_device_context();
          CollectiveRemoteAccessLocal::MemCpyAsync(
              op_dev_ctx, op_dev_ctx, col_ctx_->device, col_ctx_->device,
              col_ctx_->op_ctx->input_alloc_attr(0),
              col_ctx_->op_ctx->output_alloc_attr(0), col_ctx_->input,
              col_ctx_->output, 0, /*stream_index*/
              [this, &mu, &pending_count, &all_done](const Status& s) {
                mutex_lock l(mu);
                status_.Update(s);
                --pending_count;
                if (0 == pending_count) {
                  all_done.notify_all();
                }
              });
        }
      }

      // Then wait for all pending actions to complete.
      {
        mutex_lock l(mu);
        if (pending_count > 0) {
          all_done.wait(l);
        }
      }
    }
  }
  VLOG(2) << "device=" << col_ctx_->device_name << " return status " << status_;
  done_(status_);
}

void HierarchicalTreeBroadcaster::DispatchSend(int subdiv, int dst_rank,
                                               int src_rank,
                                               const Tensor* src_tensor,
                                               const StatusCallback& done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_9(mht_9_v, 607, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::DispatchSend");

  profiler::ScopedMemoryDebugAnnotation op_annotation(
      col_params_->name.data(), col_ctx_->step_id, "dynamic",
      src_tensor->dtype(),
      [src_tensor]() { return src_tensor->shape().DebugString(); });
  string send_buf_key =
      BroadcastBufKey(col_ctx_->exec_key, subdiv, src_rank, dst_rank);
  int dst_idx =
      col_params_->instance.impl_details.subdiv_permutations[subdiv][dst_rank];
  VLOG(3) << "DispatchSend " << send_buf_key << " from_device "
          << col_ctx_->device_name << " to_device "
          << col_params_->group.members[dst_idx].device.name()
          << " subdiv=" << subdiv << " dst_rank=" << dst_rank
          << " dst_idx=" << dst_idx;
  col_ctx_->col_exec->remote_access()->PostToPeer(
      col_params_->group.members[dst_idx].device.name(),
      col_params_->group.members[dst_idx].task, send_buf_key, col_ctx_->device,
      col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), src_tensor,
      col_ctx_->device_locality, col_ctx_->op_ctx->cancellation_manager(),
      done);
}

void HierarchicalTreeBroadcaster::DispatchRecv(int subdiv, int src_rank,
                                               int dst_rank, Tensor* dst_tensor,
                                               const StatusCallback& done) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePShierarchical_tree_broadcasterDTcc mht_10(mht_10_v, 635, "", "./tensorflow/core/common_runtime/hierarchical_tree_broadcaster.cc", "HierarchicalTreeBroadcaster::DispatchRecv");

  string recv_buf_key =
      BroadcastBufKey(col_ctx_->exec_key, subdiv, src_rank, dst_rank);
  int src_idx =
      col_params_->instance.impl_details.subdiv_permutations[subdiv][src_rank];
  VLOG(3) << "DispatchRecv " << recv_buf_key << " from_device "
          << col_params_->group.members[src_idx].device.name() << " to_device "
          << col_ctx_->device_name << " subdiv=" << subdiv
          << " src_rank=" << src_rank << " src_idx=" << src_idx;
  col_ctx_->col_exec->remote_access()->RecvFromPeer(
      col_params_->group.members[src_idx].device.name(),
      col_params_->group.members[src_idx].task,
      col_params_->group.members[src_idx].is_local, recv_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), dst_tensor,
      col_ctx_->device_locality, 0 /*stream_index*/,
      col_ctx_->op_ctx->cancellation_manager(), done);
}

namespace {
REGISTER_COLLECTIVE(HierarchicalTreeBroadcast, HierarchicalTreeBroadcaster);
}  // namespace

}  // namespace tensorflow
