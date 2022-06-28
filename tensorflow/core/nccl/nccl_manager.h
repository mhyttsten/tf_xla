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
#ifndef TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
#define TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
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
class MHTracer_DTPStensorflowPScorePSncclPSnccl_managerDTh {
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
   MHTracer_DTPStensorflowPScorePSncclPSnccl_managerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSncclPSnccl_managerDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

// TODO(rmlarsen): Get rid of this workaround. "gpu_assert" is defined when
// setting EIGEN_USE_THREADS. But when defining EIGEN_USE_THREADS here,
// incAtomic and other CUDA specific symbols are no longer recognized.
#ifndef gpu_assert
#define gpu_assert(x)
#endif

#include "absl/container/flat_hash_map.h"
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#endif
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// NCCL manager is used to make the asynchronous communicator calls and to
// manage the per-device streams used for communication.
//
// See nccl_ops.cc for example usage, including description of memory
// management and stream synchronization.
class NcclManager {
 public:
  typedef std::function<void(Status)> DoneCallback;
  NcclManager();
  ~NcclManager();

  static NcclManager* instance();

#if TENSORFLOW_USE_ROCM
  static int instance_count;
#endif

  // Calls `ncclGetUniqueId` and returns the id as a string.  The returned value
  // may be shared with other participants on different nodes and passed in to
  // multi-node collective invocations.
  string GenerateCommunicatorKey();

  // A participant in a Collective.
  struct Participant {
    Participant(se::StreamExecutor* executor, se::Stream* tensor_stream,
                const DeviceBase::AcceleratorDeviceInfo* info,
                const Tensor* input, Tensor* output, int global_rank,
                DoneCallback done_callback)
        : executor(executor),
          tensor_stream(tensor_stream),
          event_mgr(info->event_mgr),
          gpu_device_id(info->gpu_id),
#if TENSORFLOW_USE_ROCM
          context(static_cast<GPUDeviceContext*>(info->default_context)),
#endif
          input(input),
          output(output),
          global_rank(global_rank),
          done_callback(std::move(done_callback)),
          root(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_managerDTh mht_0(mht_0_v, 252, "", "./tensorflow/core/nccl/nccl_manager.h", "Participant");

      DCHECK(executor != nullptr);
      DCHECK(event_mgr != nullptr);
      DCHECK(tensor_stream != nullptr);
    }

    // StreamExecutor for the device. Expected to be live for process lifetime.
    se::StreamExecutor* const executor = nullptr;

    // `tensor_stream` is the stream that should be waited on to ensure
    // `input`'s data is available on the GPU for the communication stream to
    // access. It is also the stream that will use the produced data;
    // `done_callback` is not called until the next kernel launched on `stream`
    // would see the data. Owned by the caller, who must keep it live until
    // `done_callback` is called.
    se::Stream* const tensor_stream;

    // EventMgr which polls on executor.
    // Owned by the caller, who must keep it live until `done_callback` is
    // called.
    EventMgr* const event_mgr;

    const int gpu_device_id;

#if TENSORFLOW_USE_ROCM
    GPUDeviceContext* const context;
#endif

    // Owned by the caller, who must keep it live until `done_callback` is
    // called. Is NULL for participants that only receive data.
    const Tensor* input;

    // Owned by the caller, who must keep it live until `done_callback` is
    // called. Is NULL for participants that only send data.
    Tensor* output;

    // Rank across all devices and all nodes.
    // `global_rank` is not required for single-node collectives.
    const int global_rank;

    // The callback which is called at the completion of the NCCL operation.
    // When called, `output` has been set to the result of the operation. (note:
    // the stream may not yet have been synced)
    DoneCallback done_callback;

    // True if this is the root of the collective, e.g. source of broadcast.
    bool root;
  };

  // Data that provides context for the collective operation, including the
  // operation key, number of participants, and communicator key.
  struct Context {
    Context(const string& collective_key, int num_local_devices,
            int num_global_devices, const string& communicator_key,
            int source_rank)
        : collective_key(collective_key),
          num_local_devices(num_local_devices),
          num_global_devices(num_global_devices),
          communicator_key(communicator_key),
          source_rank(source_rank) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("collective_key: \"" + collective_key + "\"");
   mht_1_v.push_back("communicator_key: \"" + communicator_key + "\"");
   MHTracer_DTPStensorflowPScorePSncclPSnccl_managerDTh mht_1(mht_1_v, 316, "", "./tensorflow/core/nccl/nccl_manager.h", "Context");
}

    // Unique key for this collective instance
    const string& collective_key;

    // Devices local to this node
    int num_local_devices;

    // Devices across all nodes
    int num_global_devices;

    // In order to use NCCL across nodes, the callee first has to generate a
    // `communicator_key` via `GenerateCommunicatorKey()` function and share
    // this with all the other nodes.  Each node should pass in this
    // `communicator_key` to the `NcclManager` functions.
    // `communicator_key` is not required for single-node collectives and can be
    // empty.
    const string& communicator_key;

    // Rank of broadcast source.
    int source_rank;
  };

  // Adds one participant to an all-reduce.
  void AddToAllReduce(std::unique_ptr<Participant> participant,
                      const Context& context, ncclRedOp_t reduction_op);

  // Adds one participant to an all-gather.
  void AddToAllGather(std::unique_ptr<Participant> participant,
                      const Context& context);

  // AddBroadcastSend and AddBroadcastRecv combine to send data from one sender
  // to all receivers.
  void AddBroadcastSend(std::unique_ptr<Participant> participant,
                        const Context& context);
  void AddBroadcastRecv(std::unique_ptr<Participant> participant,
                        const Context& context);

  // AddReduceSend and AddReduceRecv combine to send data from all senders
  // to one receiver.
  void AddReduceSend(std::unique_ptr<Participant> participant,
                     const Context& context, ncclRedOp_t reduction_op);
  void AddReduceRecv(std::unique_ptr<Participant> participant,
                     const Context& context, ncclRedOp_t reduction_op);

  // Signals that the `Collective` corresponding to `key` is ready to launch
  // across all nodes participating in this multi-node collective operation.
  //
  // This should only be called for multi-node collectives; single-node
  // collectives are implicitly ready when all participants have called Add*
  // function.
  void SignalMultiNodeReady(const string& collective_key);

  // Aborts all collectives. After abortion, no further collectives can be
  // launched with this NcclManager.
  void StartAbort(const Status& s);

  // Resets a previously aborted NcclManager, making it available for future
  // collectives.
  void Reset();

 private:
  enum CollectiveType {
    kAllReduce = 1,
    kBroadcast = 2,
    kReduce = 3,
    kAllGather = 4,
  };
  struct Collective;
  struct Communicator;
  struct CommunicatorMember;
  struct NcclStream;

  // Gets the `Communicator` object that will be used to enqueue NCCL kernels
  // for `collective`, and returns it via `communicator`.
  //
  // This may involve creating CUDA streams and NCCL initialization.  If a NCCL
  // or CUDA error occurs in the process, this returns an INTERNAL error with
  // the corresponding NCCL/CUDA error string.
  Status GetCommunicator(Collective* collective, Communicator** communicator);

  // Adds a participant device to the local `Collective` instance corresponding
  // to `collective_key`.  Launches the `Collective` if it is ready, which it
  // checks by calling `CheckReady()`.  Also performs consistency and sanity
  // checks before launching.
  void AddParticipant(std::unique_ptr<Participant> participant,
                      const Context& context, CollectiveType collective_type,
                      ncclRedOp_t reduction_op);

  // If `collective` is ready to run, removes it from the `collectives_` map and
  // returns true.  Otherwise returns false.
  // Assumes `collective_key` corresponds to `collective`.
  //
  // A collective is ready to run when all local participants have called Add*
  // function, and the collective is signalled globally ready via
  // `SetMultiNodeReady`.
  bool CheckReady(const string& collective_key, Collective* collective)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Run <collective>.  This calls takes ownership of <collective>.
  void RunCollective(Collective* collective);
  void LoopKernelLaunches(NcclStream* stream);

  mutex mu_;

  // Maps key to collectives currently being assembled or run.
  absl::flat_hash_map<string, Collective*> collectives_ TF_GUARDED_BY(mu_);

  // Maps a device to the communication streams that make up its collective.
  // This is used to share the stream across different communicators that
  // include the same device.
  absl::flat_hash_map<se::StreamExecutor*, std::vector<NcclStream*>>
      device_to_comm_streams_ TF_GUARDED_BY(mu_);

  std::vector<std::unique_ptr<Communicator>> communicators_ TF_GUARDED_BY(mu_);

  Status status_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(NcclManager);
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
