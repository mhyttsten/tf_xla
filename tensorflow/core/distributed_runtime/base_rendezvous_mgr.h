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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_BASE_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_BASE_RENDEZVOUS_MGR_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh() {
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


#include <string>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class BaseRemoteRendezvous;
class BaseRecvTensorCall;
class CancellationManager;

// RendezvousMgr keeps track of a set of local rendezvous instances.
// All tensors sent by this worker are buffered in a RendezvousMgr
// until the tensor is received.  Each global unique "step_id"
// corresponds to one local rendezvous instance managed by a
// RendezvousMgr.
//
// E.g.,
//   Rendezvous* rendez = worker_env->rendezvous_mgr->Find(0x8935);
//   fork execution of a graph executor using "rendez" on thread 1;
//   fork execution of another graph executor using "rendez" on thread 2;
//   ...
//   join threads 1 and 2;
//
// In the example above, execution in thread 1 and 2 communicates with
// each other by send/recv operations through `rendez`.
//
// Tensors sent and received through a rendezvous managed by this
// RendezvousMgr must have keys generated by Rendezvous::CreateKey().
class BaseRendezvousMgr : public RendezvousMgrInterface {
 public:
  explicit BaseRendezvousMgr(const WorkerEnv* worker_env);

  ~BaseRendezvousMgr() override;

  // Returns Rendezvous supporting send and recv among workers in the
  // "step_id".  The caller takes ownership of one reference on the
  // returned Rendezvous instance.
  //
  // Note: the caller must guarantee to eventually call Initialize on the
  // returned RemoteRendezvous
  RemoteRendezvous* Find(int64_t step_id) override;

  // Finds the local rendezvous instance for the "step_id".  Runs
  // "done" when the tensor for "key" is produced or an error occurs.
  //
  // This method is used by the rpc handler of RecvTensor.
  void RecvLocalAsync(int64_t step_id, const Rendezvous::ParsedKey& parsed,
                      Rendezvous::DoneCallback done) override;

  // Synchronous wrapper for RecvLocalAsync.
  Status RecvLocal(int64_t step_id, const Rendezvous::ParsedKey& parsed,
                   Tensor* val, bool* is_dead) override;

  // Removes rendezvous for "step_id".
  //
  // TODO(zhifengc): Have a background thread in worker that
  // periodically calls CleanupAll().
  void Cleanup(int64_t step_id) override;

  // Remove all rendezvous instances owned by the rendezvous_mgr.
  void CleanupAll() override;

 protected:
  virtual BaseRemoteRendezvous* Create(int64_t step_id,
                                       const WorkerEnv* worker_env) = 0;

 private:
  // Maps step_id to rendezvous.
  typedef absl::flat_hash_map<int64_t, BaseRemoteRendezvous*> Table;

  // Not owned.
  const WorkerEnv* const worker_env_;

  mutex mu_;
  Table table_ TF_GUARDED_BY(mu_);

  BaseRemoteRendezvous* FindOrCreate(int64_t step_id);

  TF_DISALLOW_COPY_AND_ASSIGN(BaseRendezvousMgr);
};

// RemoteRendezvous is a Rendezvous which can handle either
// the producer or consumer being in a remote process.
//
// Buffering of Tensor values is delegated to a "local" Rendezvous
// obtained from NewLocalRendezvous().  This class just adds
// functionality to coordinate with remote workers.
class BaseRemoteRendezvous : public RemoteRendezvous {
 public:
  BaseRemoteRendezvous(const WorkerEnv* env, int64_t step_id);

  // Upgrades the BaseRemoteRendezvous to full initialization.
  Status Initialize(WorkerSession* session) override;

  void SetRemoteEagerContextDefault() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh mht_0(mht_0_v, 297, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.h", "SetRemoteEagerContextDefault");

    remote_eager_context_default_ = true;
  }
  bool IsRemoteEagerContextDefault() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh mht_1(mht_1_v, 303, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.h", "IsRemoteEagerContextDefault");

    return remote_eager_context_default_;
  }

  // Forwards to local_, where the Tensor "val" will be buffered and
  // any waiting callback stored.
  Status Send(const ParsedKey& key, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;

  // This method is called only by the RecvOp.  It tests to see
  // whether the value will be produced by a local or remote device
  // and handles accordingly.  In the local case it forwards to
  // local_, in the remote case it initiates an RPC request.
  void RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                 DoneCallback done) override;

  void StartAbort(const Status& status) override;

  // This method is called only by the local Worker, forwarded through
  // the same method on RendezvousMgr.  This occurs when the Worker
  // has received a RecvTensor request, either locally or over the
  // network.  In either case it needs to retrieve a locally buffered
  // value from local_, and give it to its caller.
  //
  // Runs "done" as soon as the tensor for "parsed" is available or an error
  // is detected.
  //
  // REQUIRES: "parsed" is one that will be Saved into the local rendezvous.
  void RecvLocalAsync(const ParsedKey& parsed, DoneCallback done);

 protected:
  virtual void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                                   const Rendezvous::Args& args,
                                   DoneCallback done) = 0;

  // Returns true if "src" and "dst" are located in the same worker,
  // and hence may use a local rendezvous.
  virtual bool IsSameWorker(DeviceNameUtils::ParsedName src,
                            DeviceNameUtils::ParsedName dst);

  // If aborted, aborts "call". Otherwise, adds "call" into calls_.
  void RegisterCall(BaseRecvTensorCall* call, const Rendezvous::Args& args);

  // Removes "call" from calls_ if "call" is in calls_.
  void DeregisterCall(BaseRecvTensorCall* call, const Rendezvous::Args& args);

  WorkerSession* session();

  bool is_initialized();

  ~BaseRemoteRendezvous() override;

  const WorkerEnv* const env_;  // Not owned.
  const int64_t step_id_;

 private:
  Rendezvous* local_;  // Owns a Ref on this object.
  // Indicates whether this remote rendezvous instance is used as the default
  // rendezvous for remote eager op-by-op execution. Errors in eager op-by-op
  // execution should not abort the rendezvous since it is a context-wide
  // instance and needs to be reused; instead, the errors are propagated through
  // eager executors.
  bool remote_eager_context_default_ = false;

  mutable mutex mu_;
  mutable mutex calls_mu_;

  // Status given by StartAbort() if any.
  Status status_ TF_GUARDED_BY(mu_);

  WorkerSession* session_ TF_GUARDED_BY(mu_);  // Not owned.

  // Data structures to handle calls when partially initialized.
  struct DeferredCall {
    const ParsedKey parsed;
    DoneCallback done;

    DeferredCall(const ParsedKey& parsed, DoneCallback done);
  };
  std::vector<DeferredCall> deferred_calls_ TF_GUARDED_BY(mu_);

  // "CancellationToken" is stored here so that when there's no active
  // RecvTensorCalls, we can de-register the callback in the cancellation
  // manager.
  //
  // Note: pointer to CancellationManager can be nullptr in certain use cases.
  absl::flat_hash_map<
      CancellationManager*,
      std::pair<CancellationToken, absl::flat_hash_set<BaseRecvTensorCall*>>>
      calls_ TF_GUARDED_BY(calls_mu_);

  bool is_initialized_locked() TF_SHARED_LOCKS_REQUIRED(mu_) {
    return session_ != nullptr;
  }

  // If "is_src" is true, checks that the rendezvous key "parsed"'s
  // source is in this process. If "is_src" is false, checks that the
  // rendezvous key "parsed"'s destination is in this process.
  Status ValidateDevices(const Rendezvous::ParsedKey& parsed, bool is_src);

  // Callback handling the case when a rendezvous has been
  // accomplished in local_ and the consumer is local to this process.
  // Tensor "in" will be copied into "out". The key "parsed" encodes
  // the src and dst devices.
  void SameWorkerRecvDone(const Rendezvous::ParsedKey& parsed,
                          const Rendezvous::Args& in_args,
                          const Rendezvous::Args& out_args, const Tensor& in,
                          Tensor* out, StatusCallback done);

  // Must be called only if fully initialized.
  void RecvLocalAsyncInternal(const ParsedKey& parsed, DoneCallback done);

  TF_DISALLOW_COPY_AND_ASSIGN(BaseRemoteRendezvous);
};

class BaseRecvTensorCall {
 public:
  BaseRecvTensorCall() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh mht_2(mht_2_v, 423, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.h", "BaseRecvTensorCall");
}
  virtual ~BaseRecvTensorCall() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSbase_rendezvous_mgrDTh mht_3(mht_3_v, 427, "", "./tensorflow/core/distributed_runtime/base_rendezvous_mgr.h", "~BaseRecvTensorCall");
}

  virtual void Start(std::function<void()> recv_done) = 0;

  virtual void StartAbort(const Status& s) = 0;

  virtual Status status() const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BaseRecvTensorCall);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_BASE_RENDEZVOUS_MGR_H_
