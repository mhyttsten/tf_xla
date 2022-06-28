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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_COPY_NODE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_COPY_NODE_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh() {
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


#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace eager {

// This node supports copying a tensor in the following way:
// - Remote -> Local:
//   We don't block on the remote _Send op and start executing the local
//   _Recv immediately after issuing the remote _Send. The local _Recv
//   kernel (or rather the special _Recv handling in KernelAndDeviceOp::Run)
//   blocks until the tensor is received. If the remote _Send (or some op
//   before it) fails, the local callback we give to EnqueueAsync will run
//   and call CancellationManager.StartCancel(). The blocked local _Recv will
//   get this notification and return with a cancelled error.
//
// - Local -> Remote:
//   The local _Send op is synchronous and non-blocking, thus it should complete
//   quickly. We issue remote _Recv RPC only after local _Send completes
//   successfully. At this point, the tensor to be sent is in the local
//   Rendezvous, hence, remote _Recv op will not deadlock waiting for the tensor
//   to appear.
//   When ctx->UseSendTensorRPC() is true, we use EagerService::Enqueue
//   SendTensor instead of _Send/_Recv.
//
// - Remote -> Remote:
//   We could issue both remote ops asynchronously, but if remote _Send (or some
//   op before it) fails, we don't have a good way of cancelling the remote
//   _Recv. The remote _Recv will deadlock in this case. The current approach
//   to deal with this issue is to wait for remote _Send to complete before
//   issuing remote _Recv RPC. Another option is to close the whole streaming
//   RPC that contains the deadlocked remote _Recv. This would not unblock the
//   deadlocked RPC on the remote machine without some extra code. Luckily, the
//   remote -> remote case seems to be fairly rare at this point. So, the
//   current partially synchronous approach seems fine.
//
// To copy a tensor within a host, please use copy_to_device_node instead.
class RemoteCopyNode : public AsyncEagerNode {
 public:
  RemoteCopyNode(EagerContext* ctx, EagerExecutor* executor, TensorHandle* src,
                 TensorHandle* dst, Device* recv_device, uint64 recv_op_id);

  ~RemoteCopyNode() override;

  Status Prepare() override;

  void RunAsync(StatusCallback done) override;

  void Abort(Status status) override;

  string DebugString() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_0(mht_0_v, 243, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "DebugString");

    string out = "[RemoteCopyNode]";
    strings::StrAppend(&out, " send_device: ", send_device_->name());
    strings::StrAppend(&out, ", recv_device: ", recv_device_->name());
    strings::StrAppend(&out, ", send_tensor: ", src_->DebugString());
    strings::StrAppend(
        &out, ", recv_tensor: ", captured_state_->dst()->DebugString());
    return out;
  }

 private:
  // Runs the _Send operation locally or remotely.
  // StartSend() makes sure that captured_state_->send_status_ is set to the
  // final _Send status after captured_state->send_done_.WaitForNotification()
  // returns.
  void StartSend();

  // Synchronously runs local send `op` and returns its status.
  Status RunLocalSend(EagerOperation* op);

  // Runs the _Recv operation locally or remotely.
  // An error return value indicates that _Recv did not run successfully. It
  // does not indicate that _Send op has completed since StartRecv could have
  // encountered an error before waiting for _Send's completion.
  // An OK return value does NOT necessarily indicate that _Recv has completed
  // successfully (it does now, but won't when streaming RPCs are turned on).
  // StartRecv() makes sure that dst_ tensor handle is handled correctly
  // (potentially after this methods returns); a tensor is set in the local
  // case, a remote shape is set in the remote case, the dst_ handle is
  // poisoned in either case if there is an error.
  void StartRecv(StatusCallback done);

  // Synchronously runs local receive `op` and returns its status.
  // Does not wait for the send to complete before running receive.
  Status RunLocalRecv(EagerOperation* op, std::vector<Tensor>* outputs);

  // Waits for send to complete, then issues remote receive `op` and
  // returns its status.
  void RunRemoteRecv(EagerOperation* op, StatusCallback done);

  // When !ctx->UseSendTensorRPC(), then tensors are shipped between remote
  // devices by the receiver invoking the WorkerService.RecvTensor RPC *on the
  // sender* (Rendezvous::RecvAsync() invoked by the _Recv kernel).
  //
  // However, in some configurations the node that has the tensor to be copied
  // isn't running a server (WorkerService RPC interface). For such cases,
  // this function enables sending tensors using the EagerService.Enqueue
  // SendTensor RPC *on the receiver*.
  void StartRemoteSendTensor(StatusCallback done);

  // Send a local packed TensorHandle to a remote device.
  void StartSendPackedHandle(StatusCallback done);

  // State that is captured by Send and/or Recv callbacks (depending on which
  // one(s) is remote) and outlives this node in the case of remote->remote
  // copy.
  class CapturedSharedState {
   public:
    explicit CapturedSharedState(TensorHandle* d) : dst_(d) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_1(mht_1_v, 304, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "CapturedSharedState");
 dst_->Ref(); }
    ~CapturedSharedState() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_2(mht_2_v, 308, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "~CapturedSharedState");
 dst_->Unref(); }

    void SetSendStatus(Status status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_3(mht_3_v, 313, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "SetSendStatus");

      send_status_.Update(status);
      send_done_.Notify();
    }

    Status GetSendStatus() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_4(mht_4_v, 321, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "GetSendStatus");

      send_done_.WaitForNotification();
      return send_status_;
    }

    // src_shape_ is not thread-safe. It should only be set in one thread.
    void SetSrcShape(const TensorShape& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_5(mht_5_v, 330, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "SetSrcShape");
 src_shape_ = shape; }

    const TensorShape& GetSrcShape() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_6(mht_6_v, 335, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "GetSrcShape");
 return src_shape_; }

    TensorHandle* dst() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_7(mht_7_v, 340, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "dst");
 return dst_; }
    CancellationManager* recv_cancellation() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTh mht_8(mht_8_v, 344, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.h", "recv_cancellation");
 return &recv_cancellation_; }

   private:
    TensorHandle* const dst_;
    CancellationManager recv_cancellation_;
    // send_status_ is safe to read only after send_done_.WaitForNotification()
    // has returned.
    Status send_status_;
    Notification send_done_;
    TensorShape src_shape_;
  };

  TensorHandle* const src_;
  EagerContext* const ctx_;
  EagerExecutor* const executor_;
  Device* const send_device_;
  Device* const recv_device_;
  const string wire_id_;
  const uint64 recv_op_id_;

  std::shared_ptr<CapturedSharedState> captured_state_;
  bool started_;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_COPY_NODE_H_
