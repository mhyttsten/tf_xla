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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/ring_gatherer.h"

#include <stdlib.h>

#include <atomic>
#include <functional>
#include <utility>

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
Status RingGatherer::InitializeCollectiveParams(CollectiveParams* col_params) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/common_runtime/ring_gatherer.cc", "RingGatherer::InitializeCollectiveParams");

  DCHECK_EQ(col_params->instance.type, GATHER_COLLECTIVE);
  DCHECK_EQ(col_params->instance.impl_details.collective_name, "RingGather");
  // TODO(tucker): Maybe add subdiv support.  It's only useful with
  // multiple NICS, and maybe gather performance isn't important enough.
  // For now, there must always be only a single subdiv at offset 0.
  if (!col_params->instance.impl_details.subdiv_offsets.empty() &&
      (col_params->instance.impl_details.subdiv_offsets.size() > 1 ||
       col_params->instance.impl_details.subdiv_offsets[0] != 0)) {
    return errors::InvalidArgument(
        "RingGather cannot take any subdiv offset other than 0.");
  }
  if (col_params->instance.impl_details.subdiv_offsets.empty()) {
    col_params->instance.impl_details.subdiv_offsets.push_back(0);
  }
  return RingAlg::InitializeCollectiveParams(col_params);
}

void RingGatherer::Run(StatusCallback done) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/common_runtime/ring_gatherer.cc", "RingGatherer::Run");

  DCHECK(col_ctx_);
  DCHECK(col_params_);
  done_ = std::move(done);
  group_size_ = col_params_->group.group_size;
  num_subdivs_ = static_cast<int>(
      col_params_->instance.impl_details.subdiv_permutations.size());
  DCHECK_GT(num_subdivs_, 0);

  if (VLOG_IS_ON(1)) {
    string buf;
    for (int r = 0; r < col_params_->group.members.size(); ++r) {
      strings::StrAppend(&buf, "dev ", r, " : ",
                         col_params_->group.members[r].device.name(), "\n");
    }
    for (int sd = 0;
         sd < col_params_->instance.impl_details.subdiv_permutations.size();
         ++sd) {
      strings::StrAppend(&buf, "\nsubdiv ", sd, " perm: ");
      for (auto x :
           col_params_->instance.impl_details.subdiv_permutations[sd]) {
        strings::StrAppend(&buf, x, ", ");
      }
    }
    VLOG(1) << "RingGatherer::Run for device " << col_ctx_->device_name
            << " default_rank " << col_params_->default_rank << "\n"
            << buf;
  }

  // Prepare to alias fields within the output.
  AllocatorAttributes attr = col_ctx_->op_ctx->output_alloc_attr(0);
  ca_.reset(MakeCollectiveAdapter(col_ctx_->output, group_size_ * num_subdivs_,
                                  col_ctx_->device->GetAllocator(attr),
                                  false /*align_chunks*/));

  // Start by copying input to the rank-specific offset of output.
  // We are running in a blockable thread and the callback can't block so
  // just wait here on the copy.
  {
    profiler::TraceMe activity("MemCpyAsync", profiler::TraceMeLevel::kInfo);
    Notification note;
    Status status;
    Tensor alias_chunk(ca_->ChunkAlias(col_params_->subdiv_rank[0]));
    CollectiveRemoteAccessLocal::MemCpyAsync(
        col_ctx_->op_ctx->op_device_context(),
        col_ctx_->op_ctx->op_device_context(), col_ctx_->device,
        col_ctx_->device, col_ctx_->op_ctx->input_alloc_attr(0),
        col_ctx_->op_ctx->output_alloc_attr(0), col_ctx_->input, &alias_chunk,
        0 /*dev_to_dev_stream_index*/, [&note, &status](const Status& s) {
          status.Update(s);
          note.Notify();
        });
    note.WaitForNotification();
    if (!status.ok()) {
      done_(status);
      return;
    }
  }
  Finish(RunAsyncParts());
}

bool RingGatherer::RunAsyncParts() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/common_runtime/ring_gatherer.cc", "RingGatherer::RunAsyncParts");

  // This function orchestrates RingGatherer actions on behalf of a
  // single device. It is entered by a blockable thread that
  // loops within it until all actions assigned to that device
  // complete. Hence function local variables are accessible only by that
  // one thread and do not require an explicit mutex.
  rfv_.clear();
  rfv_.resize(group_size_ * num_subdivs_);
  PCQueue ready_queue;
  for (int chunk_idx = 0; chunk_idx < group_size_; ++chunk_idx) {
    for (int subdiv_idx = 0; subdiv_idx < num_subdivs_; ++subdiv_idx) {
      int rf_index = (chunk_idx * num_subdivs_) + subdiv_idx;
      InitRingField(&rfv_[rf_index], chunk_idx, subdiv_idx, rf_index);
      ready_queue.Enqueue(&rfv_[rf_index]);
    }
  }
  const DeviceBase::AcceleratorDeviceInfo* gpu_info =
      col_ctx_->device->tensorflow_accelerator_device_info();
  if (gpu_info) {
    // Wait for all currently queued events on the CPU compute stream to
    // complete before proceeding.  The previous InitRingField calls allocated
    // temp memory buffers that are not guaranteed to be valid (e.g. for RDMA
    // write) unless we do.
    profiler::TraceMe activity("WaitForQueuedEvents",
                               profiler::TraceMeLevel::kInfo);
    Notification note;
    Status s = gpu_info->default_context->ThenExecute(
        col_ctx_->device, gpu_info->stream, [&note]() { note.Notify(); });
    if (s.ok()) {
      note.WaitForNotification();
    } else {
      mutex_lock l(status_mu_);
      status_ =
          errors::Internal("Failed to dispatch ThenExecute in RingGatherer");
      return false;
    }
  }

  int field_done_count = 0;
  int send_pending_count = 0;
  int recv_pending_count = 0;
  std::atomic<bool> aborted(false);

  // Loop until all RingFields have advanced to completion.
  {
    profiler::TraceMe activity("Loop", profiler::TraceMeLevel::kInfo);
    while (field_done_count < rfv_.size()) {
      VLOG(4) << FieldState();
      // Wait for a RingField to appear in the ready_queue.
      RingField* rf = ready_queue.Dequeue();
      // Advance the RingField to its next action and execute, repeating
      // until either an async action has been started or the RingField
      // is done.
      bool dispatched = false;  // true if async action was initiated
      do {
        if (aborted) {
          // Requeue this RingField to be counted off below.
          ready_queue.Enqueue(rf);
          break;
        }
        switch (rf->action) {
          case RF_INIT:
            if (rf->do_recv) {
              rf->action = RF_RECV;
              auto requeue = [this, rf, &ready_queue, &aborted](Status s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc mht_3(mht_3_v, 366, "", "./tensorflow/core/common_runtime/ring_gatherer.cc", "lambda");

                if (!s.ok()) {
                  aborted = true;
                  StartAbort(s);
                }
                ready_queue.Enqueue(rf);
              };
              DispatchRecv(rf, requeue);
              dispatched = true;
              ++recv_pending_count;
            } else {
              rf->action = RF_SEND_READY;
            }
            break;
          case RF_RECV:
            DCHECK_GT(recv_pending_count, 0);
            --recv_pending_count;
            rf->action = RF_SEND_READY;
            break;
          case RF_REDUCE:
            // Never used for Gather, so just fall through.
            TF_FALLTHROUGH_INTENDED;
          case RF_FINALIZE:
            // Never used for Gather, so just fall through.
            TF_FALLTHROUGH_INTENDED;
          case RF_SEND_READY:
            if (rf->do_send) {
              rf->action = RF_SEND;
              auto send_complete = [this, rf, &ready_queue,
                                    &aborted](Status s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_gathererDTcc mht_4(mht_4_v, 398, "", "./tensorflow/core/common_runtime/ring_gatherer.cc", "lambda");

                if (!s.ok()) {
                  aborted = true;
                  StartAbort(s);
                }
                ready_queue.Enqueue(rf);
              };
              DispatchSend(rf, send_complete);
              dispatched = true;
              ++send_pending_count;
            } else {
              rf->action = RF_DONE;
            }
            break;
          case RF_SEND:
            DCHECK_GT(send_pending_count, 0);
            --send_pending_count;
            rf->action = RF_DONE;
            break;
          case RF_DONE:
            break;
        }
        if (rf->action == RF_DONE) {
          // There's only one pass.
          ++field_done_count;
          break;  // from do while(!dispatched)
        }
      } while (!dispatched);
      if (aborted) break;
    }  // while (field_done_count < number of fields)

    if (aborted) {
      // All of the pending data actions should be aborted; field the
      // callbacks and clear the queue before quitting.
      while ((send_pending_count > 0) || (recv_pending_count > 0)) {
        RingField* rf = ready_queue.Dequeue();
        switch (rf->action) {
          case RF_RECV:
            --recv_pending_count;
            break;
          case RF_SEND:
            --send_pending_count;
            break;
          default: {
          }  // Ignore any other actions
        }
      }
    }
  }

  DCHECK_EQ(send_pending_count, 0);
  DCHECK_EQ(recv_pending_count, 0);

  VLOG(2) << this << " device=" << col_ctx_->device_name << " finish;"
          << " final value " << TensorDebugString(ca_->Value());
  return !aborted;
}

namespace {
REGISTER_COLLECTIVE(RingGather, RingGatherer);
}  // namespace

}  // namespace tensorflow
