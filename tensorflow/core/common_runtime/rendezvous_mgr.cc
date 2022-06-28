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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/rendezvous_mgr.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

namespace {
void SameWorkerRecvDone(const DeviceMgr* device_mgr,
                        const Rendezvous::ParsedKey& parsed,
                        const Rendezvous::Args& send_args,
                        const Rendezvous::Args& recv_args, const Tensor& in,
                        Tensor* out, StatusCallback done) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "SameWorkerRecvDone");

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

  // This copy must involve a non-CPU device. Hence, "in" must support DMA
  // (e.g., string tensors do not work on GPU).  Variant copy DMA
  // checks happen inside CopyTensor::ViaDMA.
  if (!DataTypeCanUseMemcpy(in.dtype()) && in.dtype() != DT_VARIANT &&
      in.dtype() != DT_RESOURCE) {
    done(errors::InvalidArgument(
        "Non-DMA-safe ", DataTypeString(in.dtype()),
        " tensor may not be copied from/to a device. Key: ", parsed.FullKey()));
    return;
  }

  Device* src_device;
  Status s = device_mgr->LookupDevice(parsed.src_device, &src_device);
  if (!s.ok()) {
    done(s);
    return;
  }
  Device* dst_device;
  s = device_mgr->LookupDevice(parsed.dst_device, &dst_device);
  if (!s.ok()) {
    done(s);
    return;
  }

  profiler::ScopedMemoryDebugAnnotation op_annotation(
      "SameWorkerRecvDone", 0, "dynamic", in.dtype(),
      [&in]() { return in.shape().DebugString(); });
  AllocatorAttributes attr = recv_args.alloc_attrs;
  attr.set_gpu_compatible(send_args.alloc_attrs.gpu_compatible() ||
                          recv_args.alloc_attrs.gpu_compatible());
  Allocator* out_allocator = dst_device->GetAllocator(attr);
  bool sync_dst_compute = true;
  if (in.dtype() != DT_VARIANT) {
    // Variants are handled by CopyTensor::ViaDMA.
    AllocationAttributes aa;
    uint64 safe_alloc_frontier = dst_device->SafeAllocFrontier(0);
    std::function<uint64()> freed_by_func = [dst_device,
                                             &safe_alloc_frontier]() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_1(mht_1_v, 264, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "lambda");

      safe_alloc_frontier = dst_device->SafeAllocFrontier(safe_alloc_frontier);
      return safe_alloc_frontier;
    };
    if ((parsed.dst.type == "GPU" ||
         DeviceFactory::IsPluggableDevice(parsed.dst.type)) &&
        safe_alloc_frontier > 0) {
      // There's a timestamped allocator at work, so use it instead
      // of sync_dst_compute.
      aa.freed_by_func = &freed_by_func;
      sync_dst_compute = false;
    }
    Tensor copy(out_allocator, in.dtype(), in.shape(), aa);
    *out = copy;
    if (in.shape().num_elements() > 0 && out->data() == nullptr) {
      done(tensorflow::errors::ResourceExhausted(
          "SameWorkerRecvDone unable to allocate output tensor. Key: ",
          parsed.FullKey()));
      return;
    }
  }

  CopyTensor::ViaDMA(
      parsed.edge_name, send_args.device_context, recv_args.device_context,
      src_device, dst_device, send_args.alloc_attrs, recv_args.alloc_attrs, &in,
      out, 0 /*dev_to_dev_stream_index*/, std::move(done), sync_dst_compute);
}

void IntraProcessRecvAsyncImpl(const DeviceMgr* device_mgr,
                               LocalRendezvous* local,
                               const RendezvousInterface::ParsedKey& parsed,
                               const Rendezvous::Args& recv_args,
                               RendezvousInterface::DoneCallback done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_2(mht_2_v, 299, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "IntraProcessRecvAsyncImpl");

  VLOG(1) << "IntraProcessRendezvous Recv " << local << " " << parsed.FullKey();

  profiler::ScopedMemoryDebugAnnotation op_annotation("RecvAsync");
  // Recv the tensor from local_.
  local->RecvAsync(
      parsed, recv_args,
      [device_mgr, parsed, done = std::move(done)](
          const Status& status, const Rendezvous::Args& send_args,
          const Rendezvous::Args& recv_args, const Tensor& in,
          bool is_dead) mutable {
        // If "in" is an uninitialized tensor, do copy-construction to
        // preserve the uninitialized state, along with data type and shape
        // info, which is useful for debugger purposes.
        Tensor* out = in.IsInitialized() ? new Tensor : new Tensor(in);

        auto final_callback = [send_args, recv_args, out, is_dead,
                               done = std::move(done)](const Status& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_3(mht_3_v, 319, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "lambda");

          done(s, send_args, recv_args, *out, is_dead);
          delete out;
        };

        if (status.ok() && in.IsInitialized()) {
          SameWorkerRecvDone(device_mgr, parsed, send_args, recv_args, in, out,
                             std::move(final_callback));
        } else {
          final_callback(status);
        }
      });
}

}  // namespace

RefCountedIntraProcessRendezvous::RefCountedIntraProcessRendezvous(
    const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(this) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_4(mht_4_v, 340, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "RefCountedIntraProcessRendezvous::RefCountedIntraProcessRendezvous");
}

RefCountedIntraProcessRendezvous::~RefCountedIntraProcessRendezvous() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_5(mht_5_v, 345, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "RefCountedIntraProcessRendezvous::~RefCountedIntraProcessRendezvous");
}

Status RefCountedIntraProcessRendezvous::Send(const ParsedKey& key,
                                              const Rendezvous::Args& args,
                                              const Tensor& val,
                                              const bool is_dead) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_6(mht_6_v, 353, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "RefCountedIntraProcessRendezvous::Send");

  VLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
  return local_.Send(key, args, val, is_dead);
}

void RefCountedIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
                                                 const Rendezvous::Args& args,
                                                 DoneCallback done) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_7(mht_7_v, 363, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "RefCountedIntraProcessRendezvous::RecvAsync");

  VLOG(1) << "IntraProcessRendezvous Recv " << this << " " << key.FullKey();
  IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
}

void RefCountedIntraProcessRendezvous::StartAbort(const Status& s) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_8(mht_8_v, 371, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "RefCountedIntraProcessRendezvous::StartAbort");

  local_.StartAbort(s);
}

Status RefCountedIntraProcessRendezvous::GetLocalRendezvousStatus() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_9(mht_9_v, 378, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "RefCountedIntraProcessRendezvous::GetLocalRendezvousStatus");

  return local_.status();
}

PrivateIntraProcessRendezvous::PrivateIntraProcessRendezvous(
    const DeviceMgr* device_mgr)
    : device_mgr_(device_mgr), local_(nullptr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_10(mht_10_v, 387, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "PrivateIntraProcessRendezvous::PrivateIntraProcessRendezvous");
}

PrivateIntraProcessRendezvous::~PrivateIntraProcessRendezvous() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_11(mht_11_v, 392, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "PrivateIntraProcessRendezvous::~PrivateIntraProcessRendezvous");
}

Status PrivateIntraProcessRendezvous::Send(const ParsedKey& key,
                                           const Rendezvous::Args& args,
                                           const Tensor& val,
                                           const bool is_dead) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_12(mht_12_v, 400, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "PrivateIntraProcessRendezvous::Send");

  DVLOG(1) << "IntraProcessRendezvous Send " << this << " " << key.FullKey();
  return local_.Send(key, args, val, is_dead);
}

void PrivateIntraProcessRendezvous::RecvAsync(const ParsedKey& key,
                                              const Rendezvous::Args& args,
                                              DoneCallback done) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_13(mht_13_v, 410, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "PrivateIntraProcessRendezvous::RecvAsync");

  DVLOG(1) << "StackAllocatedIntraProcessRendezvous Recv " << this << " "
           << key.FullKey();
  IntraProcessRecvAsyncImpl(device_mgr_, &local_, key, args, std::move(done));
}

void PrivateIntraProcessRendezvous::StartAbort(const Status& s) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrendezvous_mgrDTcc mht_14(mht_14_v, 419, "", "./tensorflow/core/common_runtime/rendezvous_mgr.cc", "PrivateIntraProcessRendezvous::StartAbort");

  local_.StartAbort(s);
}

}  // end namespace tensorflow
