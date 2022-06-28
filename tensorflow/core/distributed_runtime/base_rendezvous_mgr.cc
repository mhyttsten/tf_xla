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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"

#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

static void StartAbortRendevous(Rendezvous* rendez, const Status& s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "StartAbortRendevous");

  rendez->StartAbort(s);
  rendez->Unref();
}

BaseRendezvousMgr::BaseRendezvousMgr(const WorkerEnv* worker_env)
    : worker_env_(worker_env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::BaseRendezvousMgr");
}

BaseRendezvousMgr::~BaseRendezvousMgr() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::~BaseRendezvousMgr");

  for (auto& p : table_) {
    auto rendez = p.second;
    StartAbortRendevous(rendez, errors::Aborted("Shutdown"));
  }
}

RemoteRendezvous* BaseRendezvousMgr::Find(int64_t step_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::Find");

  return FindOrCreate(step_id);
}

BaseRemoteRendezvous* BaseRendezvousMgr::FindOrCreate(int64_t step_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_4(mht_4_v, 243, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::FindOrCreate");

  mutex_lock l(mu_);
  auto iter = table_.find(step_id);
  if (iter == table_.end()) {
    auto rr = Create(step_id, worker_env_);
    iter = table_.insert({step_id, rr}).first;
  }
  iter->second->Ref();
  return iter->second;
}

void BaseRendezvousMgr::RecvLocalAsync(int64_t step_id,
                                       const Rendezvous::ParsedKey& parsed,
                                       Rendezvous::DoneCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::RecvLocalAsync");

  auto rendez = FindOrCreate(step_id);
  auto done_cb = [rendez, done = std::move(done)](
                     const Status& s, const Rendezvous::Args& send_args,
                     const Rendezvous::Args& recv_args, const Tensor& v,
                     bool dead) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_6(mht_6_v, 267, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "lambda");

    rendez->Unref();
    done(s, send_args, recv_args, v, dead);
  };
  rendez->RecvLocalAsync(parsed, std::move(done_cb));
}

Status BaseRendezvousMgr::RecvLocal(int64_t step_id,
                                    const Rendezvous::ParsedKey& parsed,
                                    Tensor* val, bool* is_dead) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_7(mht_7_v, 279, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::RecvLocal");

  Status ret;
  Notification n;
  RecvLocalAsync(step_id, parsed,
                 [val, is_dead, &ret, &n](const Status& s,
                                          const Rendezvous::Args& send_args,
                                          const Rendezvous::Args& recv_args,
                                          const Tensor& v, const bool dead) {
                   ret = s;
                   *val = v;
                   *is_dead = dead;
                   n.Notify();
                 });
  n.WaitForNotification();
  return ret;
}

void BaseRendezvousMgr::Cleanup(int64_t step_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_8(mht_8_v, 299, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::Cleanup");

  Rendezvous* rendez = nullptr;
  {
    mutex_lock l(mu_);
    auto iter = table_.find(step_id);
    if (iter != table_.end()) {
      rendez = iter->second;
      table_.erase(iter);
    }
  }
  if (rendez) {
    StartAbortRendevous(rendez, errors::Aborted("Cleanup ", step_id));
  }
}

void BaseRendezvousMgr::CleanupAll() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_9(mht_9_v, 317, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRendezvousMgr::CleanupAll");

  mutex_lock l(mu_);
  for (auto iter = table_.begin(); iter != table_.end(); iter++) {
    iter->second->Unref();
  }
}

BaseRemoteRendezvous::BaseRemoteRendezvous(const WorkerEnv* env,
                                           int64_t step_id)
    : env_(env),
      step_id_(step_id),
      local_(NewLocalRendezvous()),
      session_(nullptr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_10(mht_10_v, 332, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::BaseRemoteRendezvous");
}

BaseRemoteRendezvous::~BaseRemoteRendezvous() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_11(mht_11_v, 337, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::~BaseRemoteRendezvous");

  {
    mutex_lock l(calls_mu_);
    calls_.clear();
  }
  local_->Unref();
}

// Returns true if "device_name" is a valid full name of local device
// of the "worker".  This helper is purely based on the worker name
// and device name and does no lookups in the worker->device_mgr.
static bool IsLocalDevice(const StringPiece worker_name,
                          const StringPiece device_name) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_12(mht_12_v, 352, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "IsLocalDevice");

  return absl::StartsWith(device_name, worker_name);
}

Status BaseRemoteRendezvous::Initialize(WorkerSession* session) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_13(mht_13_v, 359, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::Initialize");

  CHECK_NE(session, nullptr) << "session must not be null!";
  std::vector<DeferredCall> deferred_calls;
  {
    mutex_lock l(mu_);
    if (session_ != nullptr) {
      if (session_->worker_name() == session->worker_name()) {
        VLOG(1) << "Skipping rendezvous re-initialization.";
        return Status::OK();
      }
      Status s = errors::Internal(
          "Double init! Worker names would have changed from: ",
          session_->worker_name(), " -> ", session->worker_name());
      LOG(WARNING) << s;
      return s;
    }
    session_ = session;
    std::swap(deferred_calls, deferred_calls_);
  }
  for (auto& call : deferred_calls) {
    RecvLocalAsyncInternal(call.parsed, std::move(call.done));
  }
  return Status::OK();
}

WorkerSession* BaseRemoteRendezvous::session() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_14(mht_14_v, 387, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::session");

  tf_shared_lock l(mu_);
  return session_;
}

bool BaseRemoteRendezvous::is_initialized() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_15(mht_15_v, 395, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::is_initialized");

  tf_shared_lock l(mu_);
  return is_initialized_locked();
}

Status BaseRemoteRendezvous::Send(const Rendezvous::ParsedKey& parsed,
                                  const Rendezvous::Args& args,
                                  const Tensor& val, const bool is_dead) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_16(mht_16_v, 405, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::Send");

  VLOG(1) << "BaseRemoteRendezvous Send " << this << " " << parsed.FullKey();
  WorkerSession* sess = nullptr;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) return status_;
    DCHECK(is_initialized_locked());
    sess = session_;
  }

  if (!IsLocalDevice(sess->worker_name(), parsed.src_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }

  // Buffers "val" and "device_context" in local_.
  return local_->Send(parsed, args, val, is_dead);
}

Status BaseRemoteRendezvous::ValidateDevices(const ParsedKey& parsed,
                                             bool is_src) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_17(mht_17_v, 429, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::ValidateDevices");

  // Cache session pointer to avoid repeatedly taking & releasing the lock
  // (e.g. calling session())
  WorkerSession* sess = nullptr;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) return status_;
    if (!is_initialized_locked()) {
      return errors::Internal("ValidateDevices called before initialization.");
    }
    sess = session_;
  }
  if (is_src && !IsLocalDevice(sess->worker_name(), parsed.src_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (src): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }
  if (!is_src && !IsLocalDevice(sess->worker_name(), parsed.dst_device)) {
    return errors::InvalidArgument(
        "Invalid rendezvous key (dst): ", parsed.FullKey(), " @ ",
        sess->worker_name());
  }
  return Status::OK();
}

void BaseRemoteRendezvous::SameWorkerRecvDone(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args, const Tensor& in, Tensor* out,
    StatusCallback done) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_18(mht_18_v, 460, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::SameWorkerRecvDone");

  // Do a quick copy (sharing the underlying buffer) if both tensors
  // are on host memory.
  const bool src_host =
      (send_args.alloc_attrs.on_host() || parsed.src.type == "CPU");
  const bool dst_host =
      (recv_args.alloc_attrs.on_host() || parsed.dst.type == "CPU");
  if (src_host && dst_host) {
    *out = in;
    done(Status::OK());
    return;
  }

  // This copy must involve a GPU. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DMAHelper::CanUseDMA(&in) && in.dtype() != DT_VARIANT &&
      in.dtype() != DT_RESOURCE) {
    done(errors::InvalidArgument(
        "Non-DMA-safe ", DataTypeString(in.dtype()),
        " tensor may not be copied from/to a device. Key: ", parsed.FullKey()));
    return;
  }

  WorkerSession* sess = session();
  Device* src_device;
  Status s = sess->device_mgr()->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }

  profiler::ScopedMemoryDebugAnnotation op_annotation(
      "SameWorkerRecvDone", step_id_, "dynamic", in.dtype(),
      [&in]() { return in.shape().DebugString(); });
  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
  AllocationAttributes allocation_attr;
  uint64 safe_alloc_frontier = dst_device->SafeAllocFrontier(0);
  bool sync_dst_compute = (safe_alloc_frontier == 0);
  std::function<uint64()> freed_by_func = [dst_device, &safe_alloc_frontier]() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_19(mht_19_v, 511, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "lambda");

    safe_alloc_frontier = dst_device->SafeAllocFrontier(safe_alloc_frontier);
    return safe_alloc_frontier;
  };
  if (!sync_dst_compute) {
    allocation_attr.freed_by_func = &freed_by_func;
  }
  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    Tensor copy(out_allocator, in.dtype(), in.shape(), allocation_attr);
    *out = copy;
  }

  // The following function takes care of cpu->gpu, gpu->cpu, gpu->gpu copies,
  // etc.
  CopyTensor::ViaDMA(
      parsed.edge_name, send_args.device_context, recv_args.device_context,
      src_device, dst_device, send_args.alloc_attrs, recv_args.alloc_attrs, &in,
      out, 0 /*dev_to_dev_stream_index*/, std::move(done), sync_dst_compute);
}

bool BaseRemoteRendezvous::IsSameWorker(DeviceNameUtils::ParsedName src,
                                        DeviceNameUtils::ParsedName dst) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_20(mht_20_v, 536, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::IsSameWorker");

  return DeviceNameUtils::IsSameAddressSpace(src, dst);
}

void BaseRemoteRendezvous::RecvAsync(const ParsedKey& parsed,
                                     const Rendezvous::Args& recv_args,
                                     DoneCallback done) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_21(mht_21_v, 545, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::RecvAsync");

  VLOG(1) << "RemoteRendezvous Recv " << this << " " << parsed.FullKey();
  Status s = ValidateDevices(parsed, false /*!is_src*/);
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), false);
    return;
  }

  // ValidateDevices() returns an error status if the rendezvous is not
  // initialized.
  DCHECK(is_initialized()) << "RecvAsync called when uninitialized (key: "
                           << parsed.FullKey() << ").";

  profiler::ScopedMemoryDebugAnnotation op_annotation("RecvAsync", step_id_);
  // Are src and dst in the same worker?
  if (IsSameWorker(parsed.src, parsed.dst)) {
    // Recv the tensor from local_.
    local_->RecvAsync(
        parsed, recv_args,
        [this, parsed, done](
            const Status& status, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& in, bool is_dead) {
          VLOG(2) << "RemoteRendezvous Finished Recv " << this << " "
                  << parsed.FullKey();
          Tensor* out = new Tensor;
          StatusCallback final_callback = [done, send_args, recv_args, out,
                                           is_dead](const Status& s) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_22(mht_22_v, 574, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "lambda");

            done(s, send_args, recv_args, *out, is_dead);
            delete out;
          };

          if (status.ok()) {
            SameWorkerRecvDone(parsed, send_args, recv_args, in, out,
                               std::move(final_callback));
          } else {
            final_callback(status);
          }
        });
    return;
  } else {
    RecvFromRemoteAsync(parsed, recv_args, std::move(done));
  }
}

void BaseRemoteRendezvous::RecvLocalAsync(const ParsedKey& parsed,
                                          DoneCallback done) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_23(mht_23_v, 596, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::RecvLocalAsync");

  // Test whether the rendezvous is initialized using a shared lock, to avoid
  // the need for exclusive access in the common case.
  if (TF_PREDICT_FALSE(!is_initialized())) {
    mutex_lock l(mu_);
    if (!is_initialized_locked()) {
      // RecvLocalAsync can be called (due to an incoming RecvTensor RPC from a
      // remote worker) before the RunStep (or PartialRunStep) RPC from the
      // master arrives. RecvLocalAsync thus buffers the arguments until after
      // the RemoteRendezvous is Initialize()'d, when it completes the
      // rendezvous logic. At some point after Initialize() is called, a Tensor
      // is produced locally that will then be sent in response to the incoming
      // RPC.
      DeferredCall call(parsed, std::move(done));
      deferred_calls_.push_back(call);
      return;
    }
  }
  RecvLocalAsyncInternal(parsed, std::move(done));
}

void BaseRemoteRendezvous::RecvLocalAsyncInternal(const ParsedKey& parsed,
                                                  DoneCallback done) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_24(mht_24_v, 621, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::RecvLocalAsyncInternal");

  Status s = ValidateDevices(parsed, true /* is_src */);
  if (!s.ok()) {
    done(s, Args(), Args(), Tensor(), false);
    return;
  }
  local_->RecvAsync(parsed, Args(), std::move(done));
}

void BaseRemoteRendezvous::StartAbort(const Status& s) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_25(mht_25_v, 633, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::StartAbort");

  CHECK(!s.ok());
  // If the status passed in is a cancelled or aborted error, mark it as
  // "derived" for the rendezvous. Derived status messages are ignored when
  // aggregating errors across devices: this allows us to prefer our original
  // status message over any cancellation related errors.
  Status derived_status = s;
  if (errors::IsCancelled(s) || errors::IsAborted(s)) {
    derived_status = StatusGroup::MakeDerived(s);
  }

  local_->StartAbort(derived_status);

  bool status_ok = false;
  {
    mutex_lock l(mu_);
    status_ok = status_.ok();
    if (status_ok) {
      status_ = derived_status;
    }
  }

  if (status_ok) {
    // Aborts all active RecvTensor calls.
    mutex_lock l(calls_mu_);
    for (auto& cm_and_token_and_calls : calls_) {
      for (auto& call : cm_and_token_and_calls.second.second) {
        call->StartAbort(derived_status);
      }
      auto* cm = cm_and_token_and_calls.first;
      calls_[cm].second.clear();
    }
    calls_.clear();
  }
}

void BaseRemoteRendezvous::RegisterCall(BaseRecvTensorCall* call,
                                        const Rendezvous::Args& args) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_26(mht_26_v, 673, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::RegisterCall");

  CancellationManager* cm = args.cancellation_manager;
  bool already_cancelled = false;
  {
    tf_shared_lock l(mu_);
    if (!status_.ok()) {
      call->StartAbort(status_);
      return;
    }
  }

  CancellationToken token = CancellationManager::kInvalidToken;
  if (cm != nullptr) {
    mutex_lock l(calls_mu_);
    auto it = calls_.find(cm);
    if (it == calls_.end()) {
      token = cm->get_cancellation_token();
      already_cancelled = !cm->RegisterCallback(token, [this, cm]() {
        mutex_lock l(calls_mu_);
        // Abort all the RecvTensor calls associated with thie cancellation
        // manager.
        for (const auto& call : calls_[cm].second) {
          call->StartAbort(
              errors::Cancelled("RecvFromRemoteAsync is cancelled."));
        }
      });

      if (!already_cancelled) {
        calls_.emplace(
            cm,
            std::make_pair(token, absl::flat_hash_set<BaseRecvTensorCall*>{}));
      }
    }
  }

  if (already_cancelled) {
    call->StartAbort(errors::Cancelled("RecvFromRemoteAsync is cancelled."));
  } else {
    mutex_lock l(calls_mu_);
    bool emplaced = calls_[cm].second.emplace(call).second;
    CHECK(emplaced);  // Crash OK.
  }
}

void BaseRemoteRendezvous::DeregisterCall(BaseRecvTensorCall* call,
                                          const Rendezvous::Args& args) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_27(mht_27_v, 721, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::DeregisterCall");

  auto cm = args.cancellation_manager;
  mutex_lock l(calls_mu_);
  CancellationToken token = calls_[cm].first;
  calls_[cm].second.erase(call);
  if (calls_[cm].second.empty()) {
    calls_.erase(cm);
    if (cm != nullptr) {
      cm->TryDeregisterCallback(token);
    }
  }
}

BaseRemoteRendezvous::DeferredCall::DeferredCall(const ParsedKey& parsed,
                                                 DoneCallback done)
    : parsed(parsed), done(std::move(done)) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTcc mht_28(mht_28_v, 739, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.cc", "BaseRemoteRendezvous::DeferredCall::DeferredCall");
}

}  // end namespace tensorflow
