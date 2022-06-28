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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc() {
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
#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"

#include <memory>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/cancellable_call.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/protobuf_internal.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

namespace {

class RecvBufCall : public CancellableCall {
 public:
  RecvBufCall(int64_t step_id, const string& peer_device,
              const string& peer_task, const string& key, Device* to_device,
              DeviceContext* to_device_ctx,
              const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
              const DeviceLocality& client_locality,
              const DeviceAttributes& server_attributes,
              CancellationManager* cancel_mgr, WorkerCacheInterface* wc)
      : CancellableCall(cancel_mgr, peer_task, wc) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("peer_device: \"" + peer_device + "\"");
   mht_0_v.push_back("peer_task: \"" + peer_task + "\"");
   mht_0_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "RecvBufCall");

    req_.set_step_id(step_id);
    req_.set_buf_rendezvous_key(key);
    *req_.mutable_client_locality() = client_locality;
    *req_.mutable_server_locality() = server_attributes.locality();
    req_.set_num_bytes(to_tensor->TotalBytes());
    req_.set_buf_ptr(reinterpret_cast<int64_t>(DMAHelper::base(to_tensor)));
    req_.set_src_device(peer_device);
    req_.set_src_incarnation(server_attributes.incarnation());
    req_.set_dst_device(to_device->name());
    req_.set_request_id(GetUniqueRequestId());
  }

  ~RecvBufCall() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_1(mht_1_v, 236, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "~RecvBufCall");
}

  void IssueCall(const StatusCallback& done) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "IssueCall");

    wi_->RecvBufAsync(&opts_, &req_, &resp_, done);
  }

  RecvBufRequest req_;
  RecvBufResponse resp_;
};

void PopulateTensorFromExtra(const RecvBufRespExtra& extra,
                             Tensor* cpu_tensor) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "PopulateTensorFromExtra");

  char* head = reinterpret_cast<char*>(DMAHelper::base(cpu_tensor));
  for (const auto& tensor_content_chunk : extra.tensor_content()) {
    memcpy(head, std::string(tensor_content_chunk).data(),
           tensor_content_chunk.size());
    head += tensor_content_chunk.size();
  }
}

Status PopulateTensorFromResponse(const RecvBufResponse& response,
                                  Tensor* cpu_tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "PopulateTensorFromResponse");

  const bool has_transport_options = response.has_transport_options();

  // If there are no transport options, then the tensor has already been
  // copied into request.buf_ptr.
  if (!has_transport_options) return Status::OK();

  const int64_t total_bytes = cpu_tensor->TotalBytes();
  int64_t num_bytes = 0;
  RecvBufRespExtra extra;
  response.transport_options().UnpackTo(&extra);
  for (const auto& chunk : extra.tensor_content()) {
    num_bytes += chunk.size();
  }

  if (num_bytes != total_bytes) {
    return errors::Internal("Tensor Size Mismatch: RecvBufResponse returned ",
                            num_bytes,
                            " bytes, expected: ", cpu_tensor->TotalBytes());
  }
  PopulateTensorFromExtra(extra, cpu_tensor);
  return Status::OK();
}

}  // namespace

void CollectiveRemoteAccessDistributed::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("peer_device: \"" + peer_device + "\"");
   mht_5_v.push_back("peer_task: \"" + peer_task + "\"");
   mht_5_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_5(mht_5_v, 303, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "CollectiveRemoteAccessDistributed::RecvFromPeer");

  if (peer_is_local) {
    CollectiveRemoteAccessLocal::RecvFromPeer(
        peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
        to_alloc_attr, to_tensor, client_locality, dev_to_dev_stream_index,
        cancellation_manager, done);
    return;
  }

  // State that needs to be threaded through a couple of async calls
  // in order to make this function completely non-blocking.
  struct State {
    DeviceAttributes server_attributes;
    std::unique_ptr<RecvBufCall> call;
    std::unique_ptr<Tensor> cpu_tensor;
  };
  State* state = new State;

  DeviceAttributes server_attributes;
  Status s = dev_resolver_->GetDeviceAttributes(peer_device,
                                                &state->server_attributes);
  if (!s.ok()) {
    delete state;
    done(s);
    return;
  }

  Tensor* dst_tensor = nullptr;
  Device* cpu_dev = nullptr;
  if (to_device->tensorflow_accelerator_device_info()) {
    // Move the bytes into a CPU tensor then use tensor-to-tensor copy.
    // Use GPU-registered memory for the CPU tensor so the transfer
    // goes faster.

    Status status = dev_mgr_->LookupDevice("CPU:0", &cpu_dev);
    if (!status.ok()) {
      delete state;
      done(s);
      return;
    }
    AllocatorAttributes cpu_attr;
    cpu_attr.set_gpu_compatible(true);
    profiler::ScopedMemoryDebugAnnotation op_annotation(
        "CollectiveRemoteAccessDistributed::RecvFromPeer"
        "::recv_buf_callback",
        step_id_, "dynamic", to_tensor->dtype(),
        [to_tensor]() { return to_tensor->shape().DebugString(); });

    state->cpu_tensor =
        std::make_unique<Tensor>(cpu_dev->GetAllocator(cpu_attr),
                                 to_tensor->dtype(), to_tensor->shape());
    dst_tensor = state->cpu_tensor.get();
  } else {
    dst_tensor = to_tensor;
  }

  // Logic to be executed on the RecvBufAsync callback.
  auto recv_buf_callback =
      [this, state, to_device, to_alloc_attr, to_device_ctx, to_tensor, cpu_dev,
       dev_to_dev_stream_index, dst_tensor, done](const Status& s) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_6(mht_6_v, 365, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "lambda");

        if (s.ok()) {
          // In this generic implementation the bytes come back in one of 2
          // ways:
          // 1. In the response protobuf transport_options field (OR)
          // 2. It has already been copied over into RecvBufCall::req_.buf_ptr()
          // provided in request. buf_ptr is set to dst_tensor and points to
          // either the temporary cpu_tensor in case to_device is a GPU device
          // OR directly to to_tensor if to_device is not a GPU device.
          //
          // PopulateTensorFromResponse handles both cases.
          // (NOP in 2nd case) In case the final to_tensor is on GPU, buf_ptr
          // points to a tmp CPU buffer and needs to be copied over to
          // to_tensor.
          Status status =
              PopulateTensorFromResponse(state->call->resp_, dst_tensor);
          if (!status.ok()) {
            done(status);
            delete state;
            return;
          }

          if (to_device->tensorflow_accelerator_device_info()) {
            AllocatorAttributes cpu_attr;
            cpu_attr.set_gpu_compatible(true);
            CopyTensor::ViaDMA("",  // edge name (non-existent)
                               nullptr /*send_dev_ctx*/, to_device_ctx, cpu_dev,
                               to_device, cpu_attr, to_alloc_attr, dst_tensor,
                               to_tensor, dev_to_dev_stream_index,
                               [this, state, done](const Status& s) {
                                 delete state;
                                 // This callback must not block, so execute
                                 // done in another thread.
                                 work_queue_->Schedule([s, done] { done(s); });
                               });
            return;
          }
        }
        delete state;
        done(s);
      };

  state->call.reset(new RecvBufCall(
      step_id_, peer_device, peer_task, key, to_device, to_device_ctx,
      to_alloc_attr, dst_tensor, client_locality, state->server_attributes,
      cancellation_manager, worker_cache_));
  CancellationToken abortion_token =
      abortion_cancel_mgr_.get_cancellation_token();
  bool already_aborted = !abortion_cancel_mgr_.RegisterCallback(
      abortion_token, [state] { state->call->Cancel(); });
  if (already_aborted) {
    recv_buf_callback(errors::Cancelled("collective ops already aborted"));
  } else {
    state->call->Start([this, abortion_token,
                        done = std::move(recv_buf_callback)](const Status& s) {
      abortion_cancel_mgr_.DeregisterCallback(abortion_token);
      done(s);
    });
  }
}

void CollectiveRemoteAccessDistributed::CheckPeerHealth(
    const string& peer_task, int64_t timeout_in_ms,
    const StatusCallback& done) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("peer_task: \"" + peer_task + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_7(mht_7_v, 432, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "CollectiveRemoteAccessDistributed::CheckPeerHealth");

  if (peer_task == task_name_) {
    // Fast path if the peer is the worker itself.
    done(Status::OK());
    return;
  }
  // We send a GetStatus RPC to check the health of a peer task. If the RPC
  // succeeds, we verify if the peer_device incarnation matches the local record
  // if we have it. Note that DeviceResolverInterface always caches the device
  // attributes.
  WorkerInterface* wi = worker_cache_->GetOrCreateWorker(peer_task);
  if (wi == nullptr) {
    done(errors::InvalidArgument(peer_task,
                                 " not found. It's probably in valid. The "
                                 "valid form is /job:xxx/replica:0/task:N"));
    return;
  }
  auto opts = new CallOptions();
  opts->SetTimeout(timeout_in_ms);
  auto req = new GetStatusRequest();
  auto resp = new GetStatusResponse();
  // Note that fail_fast is not always respected, so we set a timeout as well.
  // We're not using CancellableCall since check health shouldn't need to be
  // cancelled.
  wi->GetStatusAsync(
      opts, req, resp, /*fail_fast*/ true,
      [this, opts, req, resp, wi, peer_task, done](Status s) {
        std::vector<DeviceAttributes> cached_attrs;
        if (s.ok()) {
          s = dev_resolver_->GetAllDeviceAttributes(peer_task, &cached_attrs);
        }
        if (s.ok()) {
          absl::flat_hash_set<uint64> remote_incarnations;
          for (const DeviceAttributes& da : resp->device_attributes()) {
            remote_incarnations.insert(da.incarnation());
          }
          for (const DeviceAttributes& attr : cached_attrs) {
            if (!remote_incarnations.contains(attr.incarnation())) {
              s = errors::FailedPrecondition(
                  attr.name(), " with incarnation ", attr.incarnation(),
                  " is not available. This usually means ", peer_task,
                  " has restarted");
              break;
            }
          }
        } else if (errors::IsNotFound(s)) {
          // Skip validating device incarnation if we don't know what the
          // incarnation should be. The device attribute is cached after the
          // first collective.
          s = Status::OK();
        }
        delete opts;
        delete req;
        delete resp;
        worker_cache_->ReleaseWorker(peer_task, wi);
        done(s);
      });
}

void CollectiveRemoteAccessDistributed::StartAbort(const Status& s) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributedDTcc mht_8(mht_8_v, 494, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed.cc", "CollectiveRemoteAccessDistributed::StartAbort");

  CollectiveRemoteAccessLocal::StartAbort(s);
  abortion_cancel_mgr_.StartCancel();
}

}  // namespace tensorflow
