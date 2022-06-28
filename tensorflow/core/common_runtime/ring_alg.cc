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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc() {
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
#include "tensorflow/core/common_runtime/ring_alg.h"

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

// Set true for greater intelligibility of debug mode log messages.
#define READABLE_KEYS false
// A ring algorithm exchanges chunks of tensor between devices.  The chunk size
// depends on the number of subdivisions specified in the algorithm.  If the
// user does not specify the number of subdivisions we may infer the number
// dynamically so that the resulting chunk size does not exceed
// kMaxChunkSizeBytes, empirically set at 4 MiB.
constexpr size_t kMaxChunkSizeBytes = (4 * 1024 * 1024);
// kMaxSubdivsPerDeviceDefault is used to give an upper bound on the number of
// subdivisions dynamically generated when user does not provide the parameter
// through the collectives API. A reasonable value would be a small
// multiple of the number of NICs adjacent to each device.
constexpr int kMaxSubdivsPerDeviceDefault = 2;

namespace tensorflow {
namespace {
// Each CollectiveOp implementation is free to define its own
// BufRendezvous key format.  This function produces the key used by
// RingAlg instances.  Note that the exec_key will differentiate between
// different instances consequently we don't need to further differentiate
// between subclasses of RingAlg.
string RingAlgBufKey(const string& name, const string& exec_key, int pass,
                     int section, int source_rank) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("exec_key: \"" + exec_key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_0(mht_0_v, 236, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlgBufKey");

  if (READABLE_KEYS) {
    return strings::StrCat(name, "(", exec_key, "):pass(", pass, "):section(",
                           section, "):srcrank(", source_rank, ")");
  } else {
    // TODO(b/78352018): Try out some kind of denser encoding, e.g. 128 bit
    // hash.
    return strings::StrCat(exec_key, ":", pass, ":", section, ":", source_rank);
  }
}

}  // namespace

void RingAlg::PCQueue::Enqueue(RingField* rf) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_1(mht_1_v, 252, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::PCQueue::Enqueue");

  mutex_lock l(pcq_mu_);
  deque_.push_back(rf);
  if (waiter_count_ > 0) {
    cv_.notify_one();
  }
}

RingAlg::RingField* RingAlg::PCQueue::Dequeue() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_2(mht_2_v, 263, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::PCQueue::Dequeue");

  mutex_lock l(pcq_mu_);
  if (deque_.empty()) {
    ++waiter_count_;
    while (deque_.empty()) {
      cv_.wait(l);
    }
    --waiter_count_;
  }
  RingField* rf = deque_.front();
  deque_.pop_front();
  return rf;
}

RingAlg::RingAlg(CollectiveType type, const string& name)
    : type_(type),
      name_(name),
      col_ctx_(nullptr),
      col_params_(nullptr),
      done_(nullptr),
      group_size_(-1),
      num_subdivs_(-1) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_3(mht_3_v, 288, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::RingAlg");
}

namespace {
Status GenerateSubdivsInCollectiveParams(CollectiveParams* col_params) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/common_runtime/ring_alg.cc", "GenerateSubdivsInCollectiveParams");

  // This function generates subdivision_offsets. Expect it to be empty when
  // called.
  DCHECK(col_params->instance.impl_details.subdiv_offsets.empty());

  if (col_params->instance.impl_details.max_subdivs_per_device == -1) {
    col_params->instance.impl_details.subdiv_offsets = {0};
    VLOG(2) << "Limiting to 1 subdivision as max_subdivs_per_device == -1";
    return Status::OK();
  }

  if (col_params->instance.shape.num_elements() == 0) {
    return errors::Internal("shape in CollectiveParams should be non-empty");
  }
  const int kAvgDevPerTask =
      col_params->group.group_size / col_params->group.num_tasks;
  const int max_subdivs_per_device =
      (col_params->instance.impl_details.max_subdivs_per_device > 0)
          ? col_params->instance.impl_details.max_subdivs_per_device
          : kMaxSubdivsPerDeviceDefault;
  const int kMaxNumSubdivs = max_subdivs_per_device * kAvgDevPerTask;
  if (kMaxNumSubdivs <= 0) {
    return errors::Internal("Unexpected kMaxNumSubdivs ", kMaxNumSubdivs,
                            " in ",
                            col_params->instance.impl_details.collective_name);
  }
  // NOTE(ayushd): If no subdiv_offsets have been specified, dynamically add
  // as many offsets as needed so that the size of tensor chunks <=
  // kMaxChunkSizeBytes.  Empirically, chunks that are too small or too large
  // lead to worse performance.
  int num_subdivs = 0;
  const size_t tensor_size = col_params->instance.shape.num_elements() *
                             DataTypeSize(col_params->instance.data_type);
  size_t chunk_size;
  do {
    ++num_subdivs;
    int num_chunks = col_params->group.group_size * num_subdivs;
    chunk_size = tensor_size / num_chunks;
    VLOG(2) << "num_subdivs " << num_subdivs << " num_chunks " << num_chunks
            << " chunk_size " << chunk_size;
  } while (chunk_size > kMaxChunkSizeBytes && num_subdivs < kMaxNumSubdivs);
  if (num_subdivs <= 0) {
    return errors::Internal("Unexpected num_subdivs ", num_subdivs, " in ",
                            col_params->instance.impl_details.collective_name);
  }

  int subdiv_stride = kAvgDevPerTask / num_subdivs;
  if (subdiv_stride == 0) subdiv_stride = 1;
  col_params->instance.impl_details.subdiv_offsets.reserve(num_subdivs);
  for (int sdi = 0; sdi < num_subdivs; ++sdi) {
    int subdiv_offset = subdiv_stride * sdi;
    if (sdi % 2 == 1) subdiv_offset *= -1;
    col_params->instance.impl_details.subdiv_offsets.push_back(subdiv_offset);
  }

  if (VLOG_IS_ON(2)) {
    string subdiv_buf;
    for (const int subdiv_offset :
         col_params->instance.impl_details.subdiv_offsets) {
      strings::StrAppend(&subdiv_buf, " ", subdiv_offset);
    }
    VLOG(2) << "Dynamically generated " << num_subdivs
            << " subdiv_offsets:" << subdiv_buf << " tensor_size "
            << tensor_size << " chunk_size " << chunk_size;
  }

  return Status::OK();
}
}  // namespace

Status RingAlg::InitializeCollectiveParams(CollectiveParams* col_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_5(mht_5_v, 367, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::InitializeCollectiveParams");

  const string& device_name =
      col_params->group.members[col_params->default_rank].device.name();
  // Each subdiv permutation is a ring formed by rotating each
  // single-task subsequence of devices by an offset.  This makes most
  // sense when each task has the same number of devices but we can't
  // depend on that being the case so we'll compute something that
  // works in any case.

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
  DCHECK_EQ(col_params->group.num_tasks, dev_per_task.size());

  if (col_params->instance.impl_details.subdiv_offsets.empty()) {
    TF_RETURN_IF_ERROR(GenerateSubdivsInCollectiveParams(col_params));
  }

  // Generate a ring permutation for requested offset.
  VLOG(2) << "Setting up perms for col_params " << col_params
          << " subdiv_permutations "
          << &col_params->instance.impl_details.subdiv_permutations;
  col_params->instance.impl_details.subdiv_permutations.resize(
      col_params->instance.impl_details.subdiv_offsets.size());
  col_params->subdiv_rank.resize(
      col_params->instance.impl_details.subdiv_offsets.size(), -1);
  for (int sdi = 0;
       sdi < col_params->instance.impl_details.subdiv_offsets.size(); ++sdi) {
    std::vector<int>& perm =
        col_params->instance.impl_details.subdiv_permutations[sdi];
    DCHECK_EQ(perm.size(), 0);
    int offset = col_params->instance.impl_details.subdiv_offsets[sdi];
    // A negative subdivision offset is interpreted as follows:
    //  1. Reverse the local device ordering.
    //  2. Begin the subdivision at abs(offset) in the reversed ordering.
    bool reverse = false;
    if (offset < 0) {
      offset = abs(offset);
      reverse = true;
    }
    int prior_dev_count = 0;  // sum over prior worker device counts
    for (int ti = 0; ti < col_params->group.num_tasks; ++ti) {
      for (int di = 0; di < dev_per_task[ti]; ++di) {
        int di_offset = (di + offset) % dev_per_task[ti];
        int offset_di =
            reverse ? (dev_per_task[ti] - (di_offset + 1)) : di_offset;
        // Device index in global subdivision permutation.
        int permuted_di = prior_dev_count + offset_di;
        int rank = static_cast<int>(perm.size());
        perm.push_back(permuted_di);
        if (col_params->group.members[permuted_di].device.name() ==
            device_name) {
          DCHECK_EQ(permuted_di, col_params->default_rank);
          col_params->subdiv_rank[sdi] = rank;
        }
      }
      prior_dev_count += dev_per_task[ti];
    }
    DCHECK_EQ(col_params->group.group_size, perm.size());
  }

  VLOG(2) << collective_util::SubdivPermDebugString(*col_params);
  return Status::OK();
}

Status RingAlg::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_6(mht_6_v, 449, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::InitializeCollectiveContext");

  DCHECK(col_ctx->dev_mgr);
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

string RingAlg::TensorDebugString(const Tensor& tensor) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_7(mht_7_v, 461, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::TensorDebugString");

  const DeviceBase::AcceleratorDeviceInfo* gpu_device_info =
      col_ctx_->op_ctx->device()->tensorflow_accelerator_device_info();
  if (gpu_device_info) {
    Tensor cpu_tensor(tensor.dtype(), tensor.shape());
    Status st = gpu_device_info->default_context->CopyDeviceTensorToCPUSync(
        &tensor, "" /*tensor_name*/, col_ctx_->device, &cpu_tensor);
    DCHECK(st.ok());
    return cpu_tensor.SummarizeValue(64);
  } else {
    return tensor.SummarizeValue(64);
  }
}

void RingAlg::StartAbort(const Status& s) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_8(mht_8_v, 478, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::StartAbort");

  // In abort mode we stop issuing additional ProvideBuf
  // and ConsumeBuf calls, but we need to wait for all of the
  // outstanding callbacks to be invoked before quitting.
  bool abort_started = false;
  {
    mutex_lock l(status_mu_);
    if (status_.ok()) {
      LOG(ERROR) << "Aborting Ring" << name_ << " with " << s;
      abort_started = true;
      status_.Update(s);
    }
  }
  // If this is the initial entry to abort mode and it's not a cancellation,
  // then invoke StartAbort on the CollectiveExecutor that invoked us.  That
  // should start cancellation on all of the outstanding CollectiveRemoteAccess
  // actions. If it's cancellation all pending send/recv should be cancelled as
  // well and there's then no need to abort.
  if (abort_started) {
    if (col_ctx_->op_ctx->cancellation_manager() == nullptr ||
        (!col_ctx_->op_ctx->cancellation_manager()->IsCancelled() &&
         !col_ctx_->op_ctx->cancellation_manager()->IsCancelling())) {
      col_ctx_->col_exec->StartAbort(s);
    }
  }
}

void RingAlg::Finish(bool ok) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_9(mht_9_v, 508, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::Finish");

  if (ok) {
    // Recover the output from the adaptor.
    ca_->ConsumeFinalValue(col_ctx_->output);
  }
  Status s;
  {
    mutex_lock l(status_mu_);
    s = status_;
  }
  rfv_.clear();  // Give up Refs on output tensor.
  done_(s);
}

// At the beginning of the algorithm initialize a RingField struct for
// every independent field of the tensor.
void RingAlg::InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                            int field_idx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_10(mht_10_v, 528, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::InitRingField");

  // Note on field indexing: There are group_size_ devices in the
  // instance, implying the same number of chunks per tensor, where a
  // chunk is the unit of data transferred in a time step.  However, if
  // a device can simultaneously send data by 2 or more independent
  // channels we can speed up the transfer by subdividing chunks and
  // processing multiple subdivisions at once.  So the actual number
  // of RingFields is group_size_ * num_subdivs_.
  DCHECK_EQ(field_idx, (chunk_idx * num_subdivs_) + subdiv_idx);
  rf->chunk_idx = chunk_idx;
  rf->subdiv_idx = subdiv_idx;
  rf->sc_idx = field_idx;
  rf->rank = col_params_->subdiv_rank[subdiv_idx];
  rf->second_pass = false;
  rf->action = RF_INIT;
  // Recv from the device with preceding rank within the subdivision.
  int recv_from_rank = (rf->rank + (group_size_ - 1)) % group_size_;
  int send_to_rank = (rf->rank + 1) % group_size_;
  rf->recv_dev_idx = col_params_->instance.impl_details
                         .subdiv_permutations[subdiv_idx][recv_from_rank];
  int send_dev_idx = col_params_->instance.impl_details
                         .subdiv_permutations[subdiv_idx][send_to_rank];
  rf->recv_is_remote = !col_params_->group.members[rf->recv_dev_idx].is_local;
  rf->send_is_remote = !col_params_->group.members[send_dev_idx].is_local;
  if (ca_->ChunkBytes(rf->sc_idx) > 0) {
    // In pass 0 we skip Recv when rank = chunk_idx
    rf->do_recv = (rf->chunk_idx != rf->rank);
    // In pass 0 we skip Send when rank = chunk_idx-1
    rf->do_send =
        (rf->rank != ((rf->chunk_idx + (group_size_ - 1)) % group_size_));
  }
  rf->is_final =
      (rf->rank == ((rf->chunk_idx + (group_size_ - 1)) % group_size_));
  if (rf->do_send || rf->do_recv) {
    rf->chunk = ca_->ChunkAlias(rf->sc_idx);
  }
  VLOG(2) << this << " InitRingField " << rf->DebugString() << " chunk "
          << ca_->TBounds(rf->chunk);
}

// When a RingField transitions from first to second recompute the
// do_send and do_recv values.
void RingAlg::AdvanceToSecondPass(RingField* rf) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_11(mht_11_v, 573, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::AdvanceToSecondPass");

  VLOG(3) << "IncrRingField old value " << rf->DebugString();
  DCHECK(!rf->second_pass);
  rf->second_pass = true;
  rf->action = RF_INIT;
  if (ca_->ChunkBytes(rf->sc_idx) > 0) {
    // In pass 1 the send/no-send boundary moves down 1 place.
    rf->do_recv =
        (rf->rank != ((rf->chunk_idx + (group_size_ - 1)) % group_size_));
    rf->do_send =
        (rf->rank != ((rf->chunk_idx + (group_size_ - 2)) % group_size_));
  }
  rf->is_final =
      (rf->rank == ((rf->chunk_idx + (group_size_ - 2)) % group_size_));
  VLOG(3) << "IncrRingField new value " << rf->DebugString();
}

string RingAlg::RingField::DebugString() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_12(mht_12_v, 593, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::RingField::DebugString");

  string rv = strings::StrCat("RingField rank=", rank, " chunk_idx=", chunk_idx,
                              " subdiv=", subdiv_idx, " sc_idx=", sc_idx,
                              " action=", action);
  strings::StrAppend(&rv, " pass=", second_pass);
  strings::StrAppend(&rv, " do_send=", do_send, " do_recv=", do_recv,
                     " is_final=", is_final, " recv_is_remote=", recv_is_remote,
                     " recv_dev_idx=", recv_dev_idx, " sc_idx=", sc_idx);
  return rv;
}

void RingAlg::DispatchSend(RingField* rf, const StatusCallback& done) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_13(mht_13_v, 607, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::DispatchSend");

  DCHECK(rf->do_send);
  string send_buf_key = RingAlgBufKey(name_, col_ctx_->exec_key,
                                      rf->second_pass, rf->sc_idx, rf->rank);
  VLOG(3) << "DispatchSend rank=" << col_params_->default_rank << " send key "
          << send_buf_key << " chunk " << ca_->TBounds(rf->chunk) << " sc_idx "
          << rf->sc_idx;
  int send_to_rank = (rf->rank + 1) % group_size_;
  int send_to_dev_idx = col_params_->instance.impl_details
                            .subdiv_permutations[rf->subdiv_idx][send_to_rank];
  col_ctx_->col_exec->remote_access()->PostToPeer(
      col_params_->group.members[send_to_dev_idx].device.name(),
      col_params_->group.members[send_to_dev_idx].task, send_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), &rf->chunk,
      col_ctx_->device_locality, col_ctx_->op_ctx->cancellation_manager(),
      done);
}

void RingAlg::DispatchRecv(RingField* rf, const StatusCallback& done) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_14(mht_14_v, 629, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::DispatchRecv");

  DCHECK(rf->do_recv);
  string recv_buf_key =
      RingAlgBufKey(name_, col_ctx_->exec_key, rf->second_pass, rf->sc_idx,
                    (rf->rank + (group_size_ - 1)) % group_size_);
  VLOG(3) << "DispatchRecv rank=" << col_params_->default_rank << " recv key "
          << recv_buf_key << " chunk " << ca_->TBounds(rf->chunk) << " into "
          << ((col_params_->merge_op != nullptr) ? "tmp_chunk" : "chunk");
  Tensor* dst_tensor = (!rf->second_pass && (col_params_->merge_op != nullptr))
                           ? &rf->tmp_chunk
                           : &rf->chunk;
  col_ctx_->col_exec->remote_access()->RecvFromPeer(
      col_params_->group.members[rf->recv_dev_idx].device.name(),
      col_params_->group.members[rf->recv_dev_idx].task,
      col_params_->group.members[rf->recv_dev_idx].is_local, recv_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), dst_tensor,
      col_ctx_->device_locality, rf->subdiv_idx,
      col_ctx_->op_ctx->cancellation_manager(), done);
}

string RingAlg::FieldState() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTcc mht_15(mht_15_v, 653, "", "./tensorflow/core/common_runtime/ring_alg.cc", "RingAlg::FieldState");

  string s = strings::StrCat(
      "Ring", name_, " ", strings::Hex(reinterpret_cast<uint64>(this)),
      " exec ", col_ctx_->exec_key, " step_id=", col_ctx_->step_id,
      " state of all ", rfv_.size(), " fields:");
  for (int i = 0; i < rfv_.size(); ++i) {
    s.append("\n");
    s.append(rfv_[i].DebugString());
  }
  return s;
}

}  // namespace tensorflow
