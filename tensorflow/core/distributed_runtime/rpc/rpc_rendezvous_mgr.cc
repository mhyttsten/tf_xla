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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class RpcRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RpcRemoteRendezvous(const WorkerEnv* env, int64_t step_id)
      : BaseRemoteRendezvous(env, step_id) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "RpcRemoteRendezvous");
}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RpcRemoteRendezvous() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "~RpcRemoteRendezvous");
}

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class RpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  RpcRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "RpcRecvTensorCall");
}

  void Init(WorkerInterface* wi, int64_t step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "Init");

    wi_ = wi;
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
    req_.set_request_id(GetUniqueRequestId());
  }

  void Reset() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_4(mht_4_v, 254, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "Reset");

    // The RpcRemoteRendezvous using this object is responsible for calling
    // ReleaseWorker() before Reset().
    DCHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall::Reset().";

    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~RpcRecvTensorCall() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_5(mht_5_v, 276, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "~RpcRecvTensorCall");

    // Since only the RpcRecvTensorFreeList will delete an
    // RpcRecvTensorCall, we require that ReleaseWorker() has been called before
    // the user releases a Call object to the free list.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall destructor.";
  }

  void Start(std::function<void()> recv_done) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_6(mht_6_v, 287, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "Start");

    StartRTCall(std::move(recv_done));
  }

  void StartAbort(const Status& s) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_7(mht_7_v, 294, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "StartAbort");

    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_8(mht_8_v, 305, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "status");

    mutex_lock l(mu_);
    return status_;
  }

  void ReleaseWorker(WorkerCacheInterface* worker_cache) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_9(mht_9_v, 313, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "ReleaseWorker");

    DCHECK_NE(static_cast<WorkerInterface*>(nullptr), wi_)
        << "RpcRecvTensorCall::ReleaseWorker() called twice.";
    worker_cache->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
  }

  const Tensor& tensor() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_10(mht_10_v, 323, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "tensor");
 return resp_.tensor(); }

  bool is_dead() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_11(mht_11_v, 328, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "is_dead");
 return resp_.metadata().is_dead(); }

  Device* dst_device() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_12(mht_12_v, 333, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "dst_device");
 return dst_device_; }
  const Rendezvous::Args& recv_args() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_13(mht_13_v, 337, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "recv_args");
 return recv_args_; }
  const Rendezvous::DoneCallback& done() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_14(mht_14_v, 341, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "done");
 return done_; }

 private:
  friend class RpcRemoteRendezvous;

  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_15(mht_15_v, 350, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "StartRTCall");

    resp_.InitAlloc(dst_device_, alloc_attrs_);
    auto abort_checked = std::make_shared<Notification>();
    auto cb = [this, abort_checked,
               recv_done = std::move(recv_done)](const Status& s) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_16(mht_16_v, 357, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "lambda");

      // Make sure the Rendezvous abort checking is finished before running the
      // callback, which might destroy the current call object.
      abort_checked->WaitForNotification();
      if (!s.ok()) {
        mutex_lock l(mu_);
        status_.Update(s);
      }
      recv_done();
    };
    wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));

    // NOTE: Check if the rendezvous was aborted after sending out the RPC. The
    // ordering is important because `StartAbort` could be called right before
    // the `RecvTensorAsync` request registers its RPC cancellation to `opts_`.
    // In that case, the previous `StartAbort` would not trigger the
    // cancellation of this call.
    Status s;
    {
      mutex_lock l(mu_);
      s = status_;
    }
    if (!s.ok()) {
      opts_.StartCancel();
    }
    // Notify that the abort check has finished.
    abort_checked->Notify();
  }

  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;  // Not owned.
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRecvTensorCall);
};

class RpcRecvTensorFreeList {
 public:
  RpcRecvTensorFreeList() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_17(mht_17_v, 408, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "RpcRecvTensorFreeList");
}
  ~RpcRecvTensorFreeList() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_18(mht_18_v, 412, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "~RpcRecvTensorFreeList");

    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  RpcRecvTensorCall* New() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_19(mht_19_v, 421, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "New");

    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        RpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new RpcRecvTensorCall;
  }

  void Release(RpcRecvTensorCall* obj) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_20(mht_20_v, 436, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "Release");

    obj->Reset();
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static constexpr int kMaxObjects = 1000;

  mutex mu_;
  std::vector<RpcRecvTensorCall*> objects_ TF_GUARDED_BY(mu_);
};

static RpcRecvTensorFreeList* get_call_freelist() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_21(mht_21_v, 458, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "get_call_freelist");

  static RpcRecvTensorFreeList* call_freelist = new RpcRecvTensorFreeList();
  return call_freelist;
}

void RpcRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_22(mht_22_v, 468, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "RpcRemoteRendezvous::RecvFromRemoteAsync");

  CHECK(is_initialized());
  Status s;

  // Prepare a RecvTensor call that can handle being aborted.
  RpcRecvTensorCall* call = get_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerSession* sess = session();
  std::shared_ptr<WorkerCacheInterface> worker_cache =
      sess->GetSharedWorkerCache();
  // The worker will be released in a subsequent call to
  // `sess->worker_cache()->ReleaseWorker()` (if the call has not yet been
  // initialized) or `call->ReleaseWorker()` (if it has been initialized).
  WorkerInterface* rwi = worker_cache->GetOrCreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }

  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache()->ReleaseWorker(call->src_worker_, rwi);
    }
    get_call_freelist()->Release(call);
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));

  // Record "call" in calls_ so that it can be aborted cleanly.
  RegisterCall(call, recv_args);

  // RendezvousMgr already aborted, shouldn't send RPC call any more
  if (!call->status().ok()) {
    DeregisterCall(call, recv_args);
    // NOTE: `*sess` can potentially be deleted before we return from
    // `call->done()(...)`, so we must release the worker before calling the
    // callback.
    call->ReleaseWorker(sess->worker_cache());
    call->done()(call->status(), Args(), Args(), Tensor(), false);
    get_call_freelist()->Release(call);
    return;
  }

  // Start "call".
  Ref();
  call->Start([this, call, recv_args, worker_cache]() {
    // Removes "call" from calls_. Prevent StartAbort().
    DeregisterCall(call, recv_args);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    // NOTE: `*session()` can potentially be deleted before we return from
    // `call->done()(...)`, so we must release the worker before calling the
    // callback.
    call->ReleaseWorker(session()->worker_cache());
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    get_call_freelist()->Release(call);
    Unref();
  });
}

}  // namespace

RpcRendezvousMgr::RpcRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_23(mht_23_v, 547, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "RpcRendezvousMgr::RpcRendezvousMgr");
}

BaseRemoteRendezvous* RpcRendezvousMgr::Create(int64_t step_id,
                                               const WorkerEnv* worker_env) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgrDTcc mht_24(mht_24_v, 553, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.cc", "RpcRendezvousMgr::Create");

  return new RpcRemoteRendezvous(worker_env, step_id);
}

}  // end namespace tensorflow
