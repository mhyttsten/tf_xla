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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc() {
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
#include "tensorflow/core/common_runtime/collective_rma_local.h"

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

void CollectiveRemoteAccessLocal::StartAbort(const Status& s) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/common_runtime/collective_rma_local.cc", "CollectiveRemoteAccessLocal::StartAbort");

  buf_rendezvous_.StartAbort(s);
}

void CollectiveRemoteAccessLocal::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("peer_device: \"" + peer_device + "\"");
   mht_1_v.push_back("peer_task: \"" + peer_task + "\"");
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/common_runtime/collective_rma_local.cc", "CollectiveRemoteAccessLocal::RecvFromPeer");

  VLOG(1) << "RecvFromPeer " << this << " from " << peer_device << " key "
          << key;
  if (!peer_is_local) {
    done(
        errors::Internal("CollectiveRemoteAccessLocal::RecvFromPeer "
                         "called with peer_is_local=false"));
    return;
  }

  Device* from_device;
  Status status = dev_mgr_->LookupDevice(peer_device, &from_device);
  if (!status.ok()) {
    done(status);
    return;
  }

  auto consumer_callback = [to_tensor, to_device_ctx, to_device, to_alloc_attr,
                            dev_to_dev_stream_index,
                            done](const Status& status,
                                  BufRendezvous::Hook* hook) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/common_runtime/collective_rma_local.cc", "lambda");

    Status s = status;
    if (s.ok()) {
      if (hook == nullptr) {
        s = errors::Internal("Invalid null hook in ConsumeBuf callback");
      }
    } else {
      if (hook != nullptr) {
        LOG(ERROR) << "Got hook " << hook << " with status " << s
                   << " from ConsumeBuf";
      }
    }

    if (s.ok()) {
      int64_t recv_bytes = to_tensor->TotalBytes();
      CHECK_EQ(recv_bytes, hook->prod_value->TotalBytes());
      MemCpyAsync(hook->prod_ctx,    // src DeviceContext
                  to_device_ctx,     // dst DeviceContext
                  hook->prod_dev,    // src Device
                  to_device,         // dst Device
                  hook->prod_attr,   // src AllocatorAttributes
                  to_alloc_attr,     // dst AllocatorAttributes
                  hook->prod_value,  // src Tensor*
                  to_tensor,         // dst Tensor*
                  dev_to_dev_stream_index,
                  [hook, done](const Status& memcpy_status) {
                    // This callback may be executing in the GPUEventMgr
                    // pool in which case it must be very short duration
                    // and non-blocking (except e.g. for queue insertion).
                    // It would be safer, though expensive, to transfer
                    // to another thread here.
                    done(memcpy_status);
                    BufRendezvous::DoneWithHook(hook);
                  });
    } else {
      done(s);
      if (hook != nullptr) {
        BufRendezvous::DoneWithHook(hook);
      }
    }
  };
  buf_rendezvous_.ConsumeBuf(key, from_device->name(),
                             from_device->attributes().incarnation(),
                             consumer_callback, cancellation_manager);
}

void CollectiveRemoteAccessLocal::PostToPeer(
    const string& peer_device, const string& peer_task, const string& key,
    Device* from_device, DeviceContext* from_device_ctx,
    const AllocatorAttributes& from_alloc_attr, const Tensor* from_tensor,
    const DeviceLocality& client_locality,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("peer_device: \"" + peer_device + "\"");
   mht_3_v.push_back("peer_task: \"" + peer_task + "\"");
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/common_runtime/collective_rma_local.cc", "CollectiveRemoteAccessLocal::PostToPeer");

  VLOG(1) << "PostToPeer " << this << " key " << key
          << " step_id_=" << step_id_;
  buf_rendezvous_.ProvideBuf(key, from_device, from_device_ctx, from_tensor,
                             from_alloc_attr, done, cancellation_manager);
}

void CollectiveRemoteAccessLocal::CheckPeerHealth(const string& peer_task,
                                                  int64_t timeout_in_ms,
                                                  const StatusCallback& done) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("peer_task: \"" + peer_task + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/common_runtime/collective_rma_local.cc", "CollectiveRemoteAccessLocal::CheckPeerHealth");

  // Assume local devices are always healthy.
  done(errors::Internal(
      "CheckPeerHealth is not supposed to be called for local collectives"));
}

/*static*/
void CollectiveRemoteAccessLocal::MemCpyAsync(
    DeviceContext* src_dev_ctx, DeviceContext* dst_dev_ctx, Device* src_dev,
    Device* dst_dev, const AllocatorAttributes& src_attr,
    const AllocatorAttributes& dst_attr, const Tensor* src, Tensor* dst,
    int dev_to_dev_stream_index, const StatusCallback& done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_rma_localDTcc mht_5(mht_5_v, 313, "", "./tensorflow/core/common_runtime/collective_rma_local.cc", "CollectiveRemoteAccessLocal::MemCpyAsync");

  // We want a real copy to happen, i.e. the bytes inside of src should be
  // transferred to the buffer backing dst.  If src and dst are on different
  // devices then CopyTensor::ViaDMA will do just that.  But if they're both
  // the same CPU, then it will actually just reset dst to point to src.
  // Since this routine is used for copying between devices and within a
  // device, we need to detect and bypass the wrong-semantics case.
  const DeviceType src_device_type(
      src_attr.on_host() ? DEVICE_CPU : src_dev->attributes().device_type());
  const DeviceType dst_device_type(
      dst_attr.on_host() ? DEVICE_CPU : dst_dev->attributes().device_type());
  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);
  // For GPU devices when only one compute stream is used (the default)
  // the OpKernelContext does not supply a DeviceContext.  It's assumed
  // that all nodes use the default context.
  if (src_dev_ctx == nullptr && src_device_type == DEVICE_GPU) {
    const DeviceBase::AcceleratorDeviceInfo* dev_info =
        src_dev->tensorflow_accelerator_device_info();
    CHECK(dev_info);
    src_dev_ctx = dev_info->default_context;
  }
  if (dst_dev_ctx == nullptr && dst_device_type == DEVICE_GPU) {
    const DeviceBase::AcceleratorDeviceInfo* dev_info =
        src_dev->tensorflow_accelerator_device_info();
    CHECK(dev_info);
    dst_dev_ctx = dev_info->default_context;
  }
  if (non_cpu_src) CHECK(src_dev_ctx);
  if (non_cpu_dst) CHECK(dst_dev_ctx);
  if (non_cpu_src || non_cpu_dst) {
    CopyTensor::ViaDMA("",  // edge name (non-existent)
                       src_dev_ctx, dst_dev_ctx, src_dev, dst_dev, src_attr,
                       dst_attr, src, dst, dev_to_dev_stream_index, done);
  } else {
    int64_t bytes = src->TotalBytes();
    DCHECK_EQ(dst->TotalBytes(), bytes);
    memcpy(DMAHelper::base(dst), DMAHelper::base(src), bytes);
    done(Status::OK());
  }
}

}  // namespace tensorflow
