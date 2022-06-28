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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc() {
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

#include "tensorflow/core/common_runtime/copy_tensor.h"

#include <atomic>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {
namespace {

struct RegistrationInfo {
  RegistrationInfo(DeviceType s, DeviceType r, CopyTensor::CopyFunction cf,
                   bool is_pluggable_device)
      : sender_device_type(std::move(s)),
        receiver_device_type(std::move(r)),
        copy_function(cf),
        is_pluggable_device(is_pluggable_device) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "RegistrationInfo");
}
  DeviceType sender_device_type;
  DeviceType receiver_device_type;
  CopyTensor::CopyFunction copy_function;
  bool is_pluggable_device;
};

// We use a vector instead of a map since we expect there to be very
// few registrations.
std::vector<RegistrationInfo>* MutableRegistry() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "MutableRegistry");

  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}

void CopyHostToDevice(const Tensor* input, Allocator* cpu_allocator,
                      Allocator* out_allocator, StringPiece edge_name,
                      Device* dst, Tensor* output,
                      DeviceContext* recv_dev_context, StatusCallback done,
                      bool sync_dst_compute) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "CopyHostToDevice");

  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = [dst, recv_dev_context, out_allocator, status_cb,
                   cpu_allocator, edge_name, sync_dst_compute,
                   wrapped_done = std::move(wrapped_done)](const Tensor& from,
                                                           Tensor* to) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      if (from.dtype() == DT_VARIANT) {
        status_cb->Ref();
        CopyHostToDevice(&from, cpu_allocator, out_allocator, edge_name, dst,
                         to, recv_dev_context, wrapped_done, sync_dst_compute);
        return Status::OK();
      } else {
        if (!DMAHelper::CanUseDMA(&from)) {
          Status err = errors::InvalidArgument(
              "During Variant Host->Device Copy: "
              "non-DMA-copy attempted of tensor type: ",
              DataTypeString(from.dtype()));
          status_cb->UpdateStatus(err);
          return err;
        }
        if (status_cb->ok()) {
          status_cb->Ref();
          *to = Tensor(out_allocator, from.dtype(), from.shape());
          recv_dev_context->CopyCPUTensorToDevice(&from, dst, to, wrapped_done,
                                                  sync_dst_compute);
          return Status::OK();
        } else {
          return status_cb->status();
        }
      }
    };

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64_t i = 0; i < input->NumElements(); ++i) {
      s_copy_init = VariantDeviceCopy(
          VariantDeviceCopyDirection::HOST_TO_DEVICE, v[i], &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else if (input->dtype() == DT_RESOURCE) {
    *output = *input;
    done(Status::OK());
  } else {
    recv_dev_context->CopyCPUTensorToDevice(input, dst, output, std::move(done),
                                            sync_dst_compute);
  }
}

void CopyDeviceToDevice(CopyTensor::CopyFunction copy_function,
                        Allocator* cpu_allocator, Allocator* out_allocator,
                        DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        int dev_to_dev_stream_index, StatusCallback done) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_5(mht_5_v, 313, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "CopyDeviceToDevice");

  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_6(mht_6_v, 322, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = [copy_function, cpu_allocator, src, dst, src_alloc_attr,
                   dst_alloc_attr, recv_dev_context, send_dev_context,
                   out_allocator, status_cb, dev_to_dev_stream_index,
                   wrapped_done = std::move(wrapped_done)](
                      // Begin unbound arguments
                      const Tensor& from, Tensor* to) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_7(mht_7_v, 334, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      if (from.dtype() == DT_VARIANT) {
        status_cb->Ref();
        CopyDeviceToDevice(copy_function, cpu_allocator, out_allocator,
                           send_dev_context, recv_dev_context, src, dst,
                           src_alloc_attr, dst_alloc_attr, &from, to,
                           dev_to_dev_stream_index, wrapped_done);
        return Status::OK();
      } else {
        if (!DMAHelper::CanUseDMA(&from)) {
          Status err = errors::InvalidArgument(
              "During Variant Device->Device Copy: ", src->name(), " to ",
              dst->name(), " non-DMA-copy attempted of tensor type: ",
              DataTypeString(from.dtype()));
          status_cb->UpdateStatus(err);
          return err;
        }
        if (status_cb->ok()) {
          status_cb->Ref();
          *to = Tensor(out_allocator, from.dtype(), from.shape());
          copy_function(send_dev_context, recv_dev_context, src, dst,
                        src_alloc_attr, dst_alloc_attr, &from, to,
                        dev_to_dev_stream_index, wrapped_done);
          return Status::OK();
        } else {
          return status_cb->status();
        }
      }
    };

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64_t i = 0; i < input->NumElements(); ++i) {
      s_copy_init =
          VariantDeviceCopy(VariantDeviceCopyDirection::DEVICE_TO_DEVICE, v[i],
                            &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else if (input->dtype() == DT_RESOURCE) {
    *output = *input;
    done(Status::OK());
  } else {
    copy_function(send_dev_context, recv_dev_context, src, dst, src_alloc_attr,
                  dst_alloc_attr, input, output, dev_to_dev_stream_index,
                  std::move(done));
  }
}

}  // namespace

// static
void CopyTensor::ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        int dev_to_dev_stream_index, StatusCallback done,
                        bool sync_dst_compute) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_8(mht_8_v, 401, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "CopyTensor::ViaDMA");

  profiler::ScopedAnnotation annotation(
      [&] { return absl::StrCat("#edge_name=", edge_name, "#"); });
  VLOG(1) << "Copy " << edge_name;

  const DeviceType src_device_type(
      src_alloc_attr.on_host() ? DEVICE_CPU : src->attributes().device_type());
  const DeviceType dst_device_type(
      dst_alloc_attr.on_host() ? DEVICE_CPU : dst->attributes().device_type());
  const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
  const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);

  // TODO(phawkins): choose an allocator optimal for both the src and dst
  // devices, not just the src device.
  AllocatorAttributes host_alloc_attrs;
  host_alloc_attrs.set_gpu_compatible(true);
  host_alloc_attrs.set_on_host(true);
  Allocator* cpu_allocator = src->GetAllocator(host_alloc_attrs);
  Allocator* out_allocator = dst->GetAllocator(dst_alloc_attr);

  // E.g., gpu -> gpu
  if (non_cpu_src && non_cpu_dst) {
    // Device to device copy.  Look through registry for an appropriate
    // CopyFunction.
    std::vector<RegistrationInfo>* registry = MutableRegistry();
    // TODO(penpornk): Revisit the lookup mechanism after PR #43611 (device
    // alias) is resolved.
    const bool src_device_is_pluggable =
        DeviceFactory::IsPluggableDevice(src_device_type.type_string());
    for (const RegistrationInfo& ri : *registry) {
      if (ri.sender_device_type == src_device_type &&
          ri.receiver_device_type == dst_device_type) {
        if (src_device_is_pluggable && !ri.is_pluggable_device) continue;
        CopyDeviceToDevice(ri.copy_function, cpu_allocator, out_allocator,
                           send_dev_context, recv_dev_context, src, dst,
                           src_alloc_attr, dst_alloc_attr, input, output,
                           dev_to_dev_stream_index, std::move(done));
        return;
      }
    }

    // Fall back to copying via the host.
    VLOG(1) << "No function registered to copy from devices of type "
            << src_device_type.type() << " to devices of type "
            << dst_device_type.type()
            << ". Falling back to copying via the host.";

    Tensor* cpu_tensor =
        new Tensor(cpu_allocator, input->dtype(), input->shape());
    auto delete_and_done = [cpu_tensor,
                            done = std::move(done)](const Status& status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_9(mht_9_v, 454, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      delete cpu_tensor;
      done(status);
    };
    auto then_copy_to_other_device =
        [delete_and_done = std::move(delete_and_done), recv_dev_context,
         cpu_tensor, cpu_allocator, out_allocator, edge_name, dst, output,
         sync_dst_compute](Status status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_10(mht_10_v, 464, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

          if (!status.ok()) {
            delete_and_done(status);
            return;
          }
          CopyHostToDevice(cpu_tensor, cpu_allocator, out_allocator, edge_name,
                           dst, output, recv_dev_context,
                           std::move(delete_and_done), sync_dst_compute);
        };
    CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                     cpu_tensor, send_dev_context,
                     std::move(then_copy_to_other_device));
    return;
  }

  // E.g., gpu -> cpu
  if (non_cpu_src && !non_cpu_dst) {
    // Device to host copy.
    CopyDeviceToHost(input, cpu_allocator, out_allocator, edge_name, src,
                     output, send_dev_context, std::move(done));
    return;
  }

  // E.g., cpu -> gpu
  if (!non_cpu_src && non_cpu_dst) {
    // Host to Device copy.
    CopyHostToDevice(input, cpu_allocator, out_allocator, edge_name, dst,
                     output, recv_dev_context, std::move(done),
                     sync_dst_compute);
    return;
  }

  // cpu -> cpu
  CHECK(!non_cpu_src && !non_cpu_dst);
  *output = *input;
  done(Status::OK());
}

// static
Status CopyTensor::Register(DeviceType sender_device_type,
                            DeviceType receiver_device_type,
                            CopyFunction copy_function,
                            bool is_pluggable_device) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_11(mht_11_v, 509, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "CopyTensor::Register");

  std::vector<RegistrationInfo>* registry = MutableRegistry();
  registry->emplace_back(sender_device_type, receiver_device_type,
                         copy_function, is_pluggable_device);
  return Status::OK();
}

namespace {

// The following registrations enable a DT_VARIANT tensor element that contains
// a wrapped `tensorflow::Tensor` to be copied between devices.
static Status WrappedTensorDeviceCopy(
    const Tensor& from, Tensor* to,
    const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_12(mht_12_v, 525, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "WrappedTensorDeviceCopy");

  if (DMAHelper::CanUseDMA(&from)) {
    TF_RETURN_IF_ERROR(copy(from, to));
  } else {
    *to = from;
  }

  return Status::OK();
}

#define REGISTER_WRAPPED_TENSOR_COPY(DIRECTION)         \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      Tensor, DIRECTION, WrappedTensorDeviceCopy)

REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_WRAPPED_TENSOR_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

}  // namespace

void CopyDeviceToHost(const Tensor* input, Allocator* cpu_allocator,
                      Allocator* out_allocator, StringPiece edge_name,
                      Device* src, Tensor* output,
                      DeviceContext* send_dev_context, StatusCallback done) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_13(mht_13_v, 551, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "CopyDeviceToHost");

  if (input->dtype() == DT_VARIANT) {
    Tensor copy(cpu_allocator, DT_VARIANT, input->shape());
    auto* status_cb = new ReffedStatusCallback(std::move(done));
    core::ScopedUnref status_cb_unref(status_cb);

    auto wrapped_done = [status_cb](const Status& s) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_14(mht_14_v, 560, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      status_cb->UpdateStatus(s);
      status_cb->Unref();
    };
    auto copier = [edge_name, src, send_dev_context, out_allocator, status_cb,
                   cpu_allocator, wrapped_done = std::move(wrapped_done)](
                      const Tensor& from, Tensor* to) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScopy_tensorDTcc mht_15(mht_15_v, 569, "", "./tensorflow/core/common_runtime/copy_tensor.cc", "lambda");

      if (from.dtype() == DT_VARIANT) {
        status_cb->Ref();
        CopyDeviceToHost(&from, cpu_allocator, out_allocator, edge_name, src,
                         to, send_dev_context, wrapped_done);
        return Status::OK();
      } else {
        if (!DMAHelper::CanUseDMA(&from)) {
          Status err = errors::InvalidArgument(
              "During Variant Device->Host Copy: "
              "non-DMA-copy attempted of tensor type: ",
              DataTypeString(from.dtype()));
          status_cb->UpdateStatus(err);
          return err;
        }
        if (status_cb->ok()) {
          status_cb->Ref();
          *to = Tensor(out_allocator, from.dtype(), from.shape());
          send_dev_context->CopyDeviceTensorToCPU(&from, edge_name, src, to,
                                                  wrapped_done);
          return Status::OK();
        } else {
          return status_cb->status();
        }
      }
    };

    const Variant* v = input->flat<Variant>().data();
    Variant* v_out = copy.flat<Variant>().data();
    Status s_copy_init;
    for (int64_t i = 0; i < input->NumElements(); ++i) {
      s_copy_init = VariantDeviceCopy(
          VariantDeviceCopyDirection::DEVICE_TO_HOST, v[i], &v_out[i], copier);
      if (!s_copy_init.ok()) {
        status_cb->UpdateStatus(s_copy_init);
        break;
      }
    }
    if (s_copy_init.ok()) {
      *output = std::move(copy);
    }
  } else if (input->dtype() == DT_RESOURCE) {
    *output = *input;
    done(Status::OK());
  } else {
    send_dev_context->CopyDeviceTensorToCPU(input, edge_name, src, output,
                                            std::move(done));
  }
}

}  // namespace tensorflow
