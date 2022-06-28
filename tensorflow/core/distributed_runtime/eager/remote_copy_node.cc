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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc() {
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

#include "tensorflow/core/distributed_runtime/eager/remote_copy_node.h"

#include <functional>

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace eager {

namespace {

void PrepareRemoteOp(eager::Operation* remote_op, EagerOperation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "PrepareRemoteOp");

  remote_op->set_name(op->Name());

  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(op->DeviceName());
}

Status CreateUncachedKernelAndDeviceOp(
    EagerOperation* op, core::RefCountPtr<KernelAndDevice>* kernel) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "CreateUncachedKernelAndDeviceOp");

  EagerContext& ctx = op->EagerContext();
  Device* device = absl::get<Device*>(op->Device());

  FunctionLibraryRuntime* flr = ctx.func_lib(device);
  if (flr == nullptr) {
    return errors::Unavailable(
        "Unable to find a FunctionLibraryRuntime corresponding to device ",
        device->name());
  }

  auto runner = (flr->runner() != nullptr) ? flr->runner() : ctx.runner();
  kernel->reset(new KernelAndDeviceOp(ctx.GetRendezvous(), ctx.LogMemory(), flr,
                                      runner, ctx.GetCollectiveExecutorHandle(),
                                      ctx.HostCPU()));

  const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
  return kernel->get()->Init(ctx.LogDevicePlacement(), ndef,
                             /*graph_collector=*/nullptr);
}

// This gets a unique wire ID. We add a random identifier so that if the
// worker has other clients that it is servicing, we don't have any collision.
string GetUniqueWireID() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "GetUniqueWireID");

  static tensorflow::uint64 random_seed = random::New64();
  static tensorflow::mutex wireid_mutex(tensorflow::LINKER_INITIALIZED);
  static std::atomic<int64_t> wire_id;
  return strings::StrCat(random_seed, "_", wire_id++);
}

}  // namespace

RemoteCopyNode::RemoteCopyNode(EagerContext* ctx, EagerExecutor* executor,
                               TensorHandle* src, TensorHandle* dst,
                               Device* recv_device, uint64 recv_op_id)
    : AsyncEagerNode(),
      src_(src),
      ctx_(ctx),
      executor_(executor),
      send_device_(src->DeviceOrHostCPU(*ctx)),
      recv_device_(recv_device),
      wire_id_(GetUniqueWireID()),
      recv_op_id_(recv_op_id),
      captured_state_(std::make_shared<CapturedSharedState>(dst)),
      started_(false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::RemoteCopyNode");

  DCHECK(!send_device_->IsLocal() || !recv_device_->IsLocal());
  src_->Ref();
  ctx_->Ref();
}

RemoteCopyNode::~RemoteCopyNode() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_4(mht_4_v, 275, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::~RemoteCopyNode");

  src_->Unref();
  ctx_->Unref();
}

Status RemoteCopyNode::RunLocalSend(EagerOperation* op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::RunLocalSend");

  TF_RETURN_IF_ERROR(executor_->status());

  TF_RETURN_IF_ERROR(op->AddInput(src_));

  core::RefCountPtr<KernelAndDevice> kernel;
  TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(op, &kernel));

  EagerKernelArgs args(1);
  Device* d = ctx_->CanonicalDevice(absl::get<Device*>(op->Device()));
  TF_RETURN_IF_ERROR(src_->TensorValue(d, args.MutableInput(0)));
  CoordinationServiceAgent* coord_agent = nullptr;
  if (ctx_->GetDistributedManager() != nullptr)
    coord_agent = ctx_->GetDistributedManager()->GetCoordinationServiceAgent();

  return kernel->Run(/*step_container=*/nullptr, args, /*outputs=*/nullptr,
                     /*cancellation_manager=*/nullptr,
                     /*remote_func_params=*/absl::nullopt,
                     /*stack_trace=*/absl::nullopt, coord_agent);
}

void RemoteCopyNode::StartSend() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_6(mht_6_v, 307, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::StartSend");

  // TODO(gjn): We should consider just using the low-level SendOp::Compute()
  // functionality here instead of constructing an Op.
  EagerOperation op(ctx_);
  Status status = op.Reset("_Send", /*raw_device_name=*/nullptr,
                           /*remote=*/false, /*executor=*/nullptr);
  if (!status.ok()) {
    captured_state_->SetSendStatus(status);
    return;
  }

  op.SetDevice(send_device_);

  op.MutableAttrs()->Set("tensor_name", wire_id_);
  op.MutableAttrs()->Set("send_device", send_device_->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64_t>(send_device_->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device_->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("T", src_->dtype);

  DCHECK(send_device_ != nullptr);

  if (send_device_->IsLocal()) {
    status = RunLocalSend(&op);
    captured_state_->SetSendStatus(status);
    return;
  } else {
    // Prepare the request
    EnqueueRequest request;
    request.set_context_id(ctx_->GetContextId());
    auto* remote_op = request.add_queue()->mutable_operation();
    status = ctx_->RemoteMgr()->SerializeRemoteTensorHandle(
        src_, /*wait_until_ready=*/false,
        remote_op->add_op_inputs()->mutable_remote_handle(), src_->device(),
        src_->DeviceOrHostCPU(*ctx_)->name());
    if (!status.ok()) {
      captured_state_->SetSendStatus(status);
      return;
    }

    PrepareRemoteOp(remote_op, &op);
    remote_op->set_id(ctx_->RemoteMgr()->NextOpId());

    // Issue the RPC
    core::RefCountPtr<eager::EagerClient> eager_client;
    status = ctx_->GetClient(send_device_, &eager_client);
    if (!status.ok()) {
      captured_state_->SetSendStatus(status);
      return;
    }

    const std::shared_ptr<CapturedSharedState>& captured_state =
        captured_state_;
    EnqueueResponse* response = new EnqueueResponse;
    // If StartRecv fails very quickly, `this` can be destroyed before the
    // callback below is executed. So, we can't capture `this`.
    eager_client->StreamingEnqueueAsync(
        /*call_opts=*/nullptr, &request, response,
        [response, captured_state](const Status& s) {
          captured_state->SetSendStatus(s);
          if (!s.ok()) {
            captured_state->recv_cancellation()->StartCancel();
          }
          delete response;
        });
  }
}

Status RemoteCopyNode::RunLocalRecv(EagerOperation* op,
                                    std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_7(mht_7_v, 382, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::RunLocalRecv");

  TF_RETURN_IF_ERROR(executor_->status());

  core::RefCountPtr<KernelAndDevice> kernel;
  TF_RETURN_IF_ERROR(CreateUncachedKernelAndDeviceOp(op, &kernel));

  EagerKernelArgs args;
  std::vector<EagerKernelRet> rets;
  CoordinationServiceAgent* coord_agent = nullptr;
  if (ctx_->GetDistributedManager() != nullptr)
    coord_agent = ctx_->GetDistributedManager()->GetCoordinationServiceAgent();
  TF_RETURN_IF_ERROR(kernel->Run(/*step_container*/ nullptr, args, &rets,
                                 captured_state_->recv_cancellation(),
                                 /*remote_func_params=*/absl::nullopt,
                                 /*stack_trace=*/absl::nullopt, coord_agent));
  outputs->clear();
  for (const auto& ret : rets) {
    if (ret.index() == 0) {
      outputs->push_back(absl::get<Tensor>(ret));
    } else {
      return errors::Internal(
          "Expect to receive a Tensor but got a TensorShape.");
    }
  }
  return Status::OK();
}

void RemoteCopyNode::RunRemoteRecv(EagerOperation* op, StatusCallback done) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_8(mht_8_v, 412, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::RunRemoteRecv");

  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  auto* remote_op = request.add_queue()->mutable_operation();
  PrepareRemoteOp(remote_op, op);
  remote_op->set_id(recv_op_id_);
  uint64 context_view_id = ctx_->GetContextViewId();

  core::RefCountPtr<eager::EagerClient> eager_client;
  Status status = ctx_->GetClient(recv_device_, &eager_client);
  if (!status.ok()) {
    captured_state_->dst()->PoisonRemote(status, recv_device_, context_view_id);
    done(status);
    return;
  }

  // Don't issue the recv until send has completed.
  //  - local send will complete very quickly.
  //  - remote send will take some time, but remote->remote copy is
  //    probably rare enough that we don't care much.
  // Blocks until send has completed.
  Status send_status = captured_state_->GetSendStatus();
  if (!send_status.ok()) {
    captured_state_->dst()->PoisonRemote(status, recv_device_, context_view_id);
    done(send_status);
    return;
  }

  EnqueueResponse* response = new EnqueueResponse;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  Device* recv_device = recv_device_;
  eager_client->StreamingEnqueueAsync(
      /*call_opts=*/nullptr, &request, response,
      [captured_state, response, recv_device, context_view_id,
       done](const Status& s) {
        if (s.ok()) {
          Status status = captured_state->dst()->SetRemoteShape(
              response->queue_response(0).shape(0), recv_device,
              context_view_id);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by remote Recv op: "
                       << status.ToString()
                       << "\nThis should never happen. "
                          "Please file an issue with the TensorFlow Team.";
          }
        } else {
          captured_state->dst()->PoisonRemote(s, recv_device, context_view_id);
        }
        done(s);
        delete response;
      });
}

void RemoteCopyNode::StartRecv(StatusCallback done) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_9(mht_9_v, 470, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::StartRecv");

  // TODO(gjn): We should consider just using the low-level RecvOp::Compute()
  // functionality here instead of constructing an Op.
  EagerOperation op(ctx_);
  Status status = op.Reset("_Recv", /*raw_device_name=*/nullptr,
                           /*remote=*/false, /*executor=*/nullptr);
  Device* recv_device = ctx_->CanonicalDevice(recv_device_);
  if (!status.ok()) {
    captured_state_->dst()->Poison(status, recv_device);
    done(status);
    return;
  }

  op.SetDevice(recv_device_);

  op.MutableAttrs()->Set("tensor_name", wire_id_);
  op.MutableAttrs()->Set("send_device", send_device_->name());
  op.MutableAttrs()->Set(
      "send_device_incarnation",
      static_cast<int64_t>(send_device_->attributes().incarnation()));
  op.MutableAttrs()->Set("recv_device", recv_device_->name());
  op.MutableAttrs()->Set("client_terminated", false);

  op.MutableAttrs()->Set("tensor_type", src_->dtype);

  if (recv_device_->IsLocal()) {
    std::vector<Tensor> outputs(1);
    status = RunLocalRecv(&op, &outputs);
    if (!status.ok()) {
      captured_state_->dst()->Poison(status, recv_device);
      done(status);
      return;
    }
    status =
        captured_state_->dst()->SetTensor(std::move(outputs[0]), recv_device);
    done(status);
  } else {
    // Handles captured_state_->dst_ internally.
    RunRemoteRecv(&op, std::move(done));
  }
}

Status SerializePackedHandle(const uint64 op_id, TensorHandle* packed_handle,
                             const Device* target_device, EagerContext* ctx,
                             SendPackedHandleOp* op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_10(mht_10_v, 517, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "SerializePackedHandle");

  op->set_op_id(op_id);
  op->set_device_name(packed_handle->DeviceOrHostCPU(*ctx)->name());
  for (int i = 0; i < packed_handle->NumPackedHandles(); ++i) {
    TensorHandle* h = nullptr;
    TF_RETURN_IF_ERROR(packed_handle->ExtractPackedHandle(i, &h));
    if (h->Type() == TensorHandle::LOCAL) {
      // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence
      // copy it to the CPU before copying it out.
      Tensor tensor;
      TF_RETURN_IF_ERROR(h->CopyToDevice(*ctx, ctx->HostCPU(), &tensor));
      auto* local_handle = op->add_handles()->mutable_local_handle();
      local_handle->set_device(h->op_device() ? h->op_device()->name()
                                              : ctx->HostCPU()->name());
      tensor.AsProtoTensorContent(local_handle->mutable_tensor());
    } else if (h->Type() == TensorHandle::REMOTE) {
      // Only serialize the resource dtype and shape of the first handle, since
      // all handles are of the same resource dtype and shape.
      // If src_device is on the same task of target_device, the handle is a
      // local handle on the target device, which means the resource dtype and
      // shape are known on the target device.
      Device* src_device = h->device();
      const bool serialize_resource_dtype_and_shape =
          (i == 0) && (h->dtype == DT_RESOURCE) &&
          (!ctx->OnSameTask(src_device, target_device));
      // For a remote component function, a function execution request and an
      // input generation request may come from different workers. We need to
      // guarantee that the input generation request is processed before the
      // function execution request, so wait until the underlying remote handles
      // are ready before sending a packed handle to the function device.
      TF_RETURN_IF_ERROR(ctx->RemoteMgr()->SerializeRemoteTensorHandle(
          h, /*wait_until_ready=*/true,
          op->add_handles()->mutable_remote_handle(), src_device,
          h->DeviceOrHostCPU(*ctx)->name(),
          serialize_resource_dtype_and_shape));
    } else {
      return errors::InvalidArgument("Nested packed handles are not supported");
    }
  }
  return Status::OK();
}

void RemoteCopyNode::StartSendPackedHandle(StatusCallback done) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_11(mht_11_v, 562, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::StartSendPackedHandle");

  Status s;
  const uint64 context_view_id = ctx_->GetContextViewId();
  if (!send_device_->IsLocal()) {
    s = errors::InvalidArgument(
        "Copy a packed handle from a remote device is not supported");
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }

  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  s = SerializePackedHandle(recv_op_id_, src_, recv_device_, ctx_,
                            request.add_queue()->mutable_send_packed_handle());
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }

  TensorShape shape;
  s = src_->Shape(&shape);
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }
  captured_state_->SetSrcShape(shape);

  core::RefCountPtr<eager::EagerClient> eager_client;
  s = ctx_->GetClient(recv_device_, &eager_client);
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }

  EnqueueResponse* response = new EnqueueResponse;
  Device* recv_device = recv_device_;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  eager_client->StreamingEnqueueAsync(
      /*call_opts=*/nullptr, &request, response,
      [captured_state, response, recv_device, context_view_id,
       done](const Status& s) {
        if (s.ok()) {
          Status status = captured_state->dst()->SetRemoteShape(
              captured_state->GetSrcShape(), recv_device, context_view_id);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by SendPackedHadnle rpc: "
                       << status.ToString();
          }
        } else {
          captured_state->dst()->PoisonRemote(s, recv_device, context_view_id);
        }
        done(s);
        delete response;
      });
}

void RemoteCopyNode::StartRemoteSendTensor(StatusCallback done) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_12(mht_12_v, 627, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::StartRemoteSendTensor");

  Status s;
  EnqueueRequest request;
  uint64 context_id = ctx_->GetContextId();
  request.set_context_id(context_id);
  auto* send_tensor = request.add_queue()->mutable_send_tensor();
  send_tensor->set_op_id(recv_op_id_);
  send_tensor->set_device_name(recv_device_->name());
  uint64 context_view_id = ctx_->GetContextViewId();

  // AsProtoTensorContent doesn't work when the tensor is on the GPU, hence
  // copy it to the CPU before copying it out.
  // TODO(fishx): Make CopyToDevice asynchronous.
  Tensor tensor;
  s = src_->CopyToDevice(*ctx_, ctx_->HostCPU(), &tensor);
  if (!s.ok()) {
    done(s);
    return;
  }
  tensor.AsProtoTensorContent(send_tensor->add_tensors());

  core::RefCountPtr<eager::EagerClient> eager_client;
  s = ctx_->GetClient(recv_device_, &eager_client);
  if (!s.ok()) {
    captured_state_->dst()->PoisonRemote(s, recv_device_, context_view_id);
    done(s);
    return;
  }
  EnqueueResponse* response = new EnqueueResponse;
  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  captured_state->SetSrcShape(tensor.shape());
  Device* recv_device = recv_device_;
  eager_client->StreamingEnqueueAsync(
      /*call_opts=*/nullptr, &request, response,
      [captured_state, response, recv_device, context_view_id,
       done](const Status& s) {
        if (s.ok()) {
          Status status = captured_state->dst()->SetRemoteShape(
              captured_state->GetSrcShape(), recv_device, context_view_id);
          if (!status.ok()) {
            LOG(ERROR) << "Ignoring an error encountered when setting remote "
                          "shape of tensor received by SendTensor rpc: "
                       << status.ToString();
          }
        } else {
          captured_state->dst()->PoisonRemote(s, recv_device, context_view_id);
        }
        done(s);
        delete response;
      });
}

Status RemoteCopyNode::Prepare() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_13(mht_13_v, 682, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::Prepare");

  TF_RETURN_IF_ERROR(captured_state_->dst()->CopyInferenceShape(src_));
  return Status::OK();
}

void RemoteCopyNode::RunAsync(StatusCallback done) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_14(mht_14_v, 690, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::RunAsync");

  started_ = true;
  if (src_->Type() == TensorHandle::PACKED) {
    return StartSendPackedHandle(std::move(done));
  }

  if ((ctx_->UseSendTensorRPC()) && send_device_->IsLocal() &&
      !recv_device_->IsLocal()) {
    return StartRemoteSendTensor(std::move(done));
  }
  StartSend();

  const std::shared_ptr<CapturedSharedState>& captured_state = captured_state_;
  auto done_wrapper = [captured_state,
                       done = std::move(done)](const Status& s) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_15(mht_15_v, 707, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "lambda");

    if (!s.ok() && errors::IsCancelled(s)) {
      Status send_status = captured_state->GetSendStatus();
      if (!send_status.ok()) {
        // In this case, Recv is cancelled because the Send op failed.
        // Return the status of the Send op instead.
        done(send_status);
      }
    } else {
      done(s);
    }
  };

  // StartRecv() takes care of doing the right thing to dst handle.
  // No need to poison it after this point.
  StartRecv(std::move(done_wrapper));
}

void RemoteCopyNode::Abort(Status status) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_copy_nodeDTcc mht_16(mht_16_v, 728, "", "./tensorflow/core/distributed_runtime/eager/remote_copy_node.cc", "RemoteCopyNode::Abort");

  if (!started_) {
    uint64 context_view_id = ctx_->GetContextViewId();
    captured_state_->dst()->PoisonRemote(status, recv_device_, context_view_id);
  }
}

}  // namespace eager
}  // namespace tensorflow
