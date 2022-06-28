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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_device_context.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_launch_util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace tensorflow {

// The allocator used for Tensors assigned to the XLA device.
XlaDeviceAllocator::XlaDeviceAllocator(
    stream_executor::StreamExecutor* stream_executor)
    : stream_executor_(stream_executor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceAllocator::XlaDeviceAllocator");
}

XlaDeviceAllocator::~XlaDeviceAllocator() = default;

string XlaDeviceAllocator::Name() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceAllocator::Name");
 return "xla"; }

void* XlaDeviceAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_2(mht_2_v, 221, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceAllocator::AllocateRaw");

  // We always return an empty XlaTensor object, encoded as an opaque tagged
  // pointer. We can return an empty object and ignore num_bytes here because we
  // have control over all of the uses of this device tensor, and can lazily
  // allocate memory when used. This allows us to also know the shape of the
  // allocated Tensor, which is useful if the device's tensor representation
  // differs from the host.
  return XlaTensor::ToOpaquePointer(new XlaTensor());
}

void XlaDeviceAllocator::DeallocateRaw(void* ptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceAllocator::DeallocateRaw");

  delete XlaTensor::FromOpaquePointer(ptr);
}

absl::optional<AllocatorStats> XlaDeviceAllocator::GetStats() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_4(mht_4_v, 241, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceAllocator::GetStats");

  absl::optional<stream_executor::AllocatorStats> se_stats =
      stream_executor_->GetAllocatorStats();
  if (!se_stats) {
    return absl::nullopt;
  }

  tensorflow::AllocatorStats tf_stats;
  tf_stats.num_allocs = se_stats->num_allocs;
  tf_stats.bytes_in_use = se_stats->bytes_in_use;
  tf_stats.peak_bytes_in_use = se_stats->peak_bytes_in_use;
  tf_stats.largest_alloc_size = se_stats->largest_alloc_size;
  tf_stats.bytes_limit = se_stats->bytes_limit;
  tf_stats.bytes_reserved = se_stats->bytes_reserved;
  tf_stats.peak_bytes_reserved = se_stats->peak_bytes_reserved;
  tf_stats.bytes_reservable_limit = se_stats->bytes_reservable_limit;
  tf_stats.largest_free_block_bytes = se_stats->largest_free_block_bytes;
  return tf_stats;
}

bool XlaDeviceAllocator::ClearStats() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_5(mht_5_v, 264, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceAllocator::ClearStats");

  if (!stream_executor_->SynchronizeAllActivity()) {
    return false;
  }
  return stream_executor_->ClearAllocatorStats();
}

XlaDeviceContext::XlaDeviceContext(
    std::shared_ptr<se::Stream> compute_stream,
    std::shared_ptr<se::Stream> host_to_device_stream,
    std::shared_ptr<se::Stream> device_to_host_stream,
    std::vector<std::shared_ptr<se::Stream>> device_to_device_streams,
    xla::LocalClient* client,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    thread::ThreadPool* thread_pool)
    : stream_(std::move(compute_stream)),
      host_to_device_stream_(std::move(host_to_device_stream)),
      device_to_host_stream_(std::move(device_to_host_stream)),
      device_to_device_streams_(std::move(device_to_device_streams)),
      client_(client),
      transfer_manager_(client->backend().transfer_manager()),
      shape_determination_fns_(std::move(shape_determination_fns)),
      thread_pool_(thread_pool) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_6(mht_6_v, 289, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceContext::XlaDeviceContext");

  CHECK(host_to_device_stream_ != nullptr);
  CHECK(stream_ != nullptr);
}

void XlaDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                              Device* device,
                                              Tensor* output_tensor,
                                              StatusCallback done) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_7(mht_7_v, 300, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceContext::CopyTensorInSameDevice");

  done(errors::Unimplemented("XLA->XLA same-device copies not implemented."));
}

void XlaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done,
                                             bool sync_dst_compute) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_8(mht_8_v, 311, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceContext::CopyCPUTensorToDevice");

  if (cpu_tensor->NumElements() == 0) {
    VLOG(2) << "CopyCPUTensorToDevice empty tensor";
    done(Status::OK());
    return;
  }

  VLOG(2) << "CopyCPUTensorToDevice " << this << " "
          << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
          << " "
          << reinterpret_cast<const void*>(device_tensor->tensor_data().data())
          << " " << cpu_tensor->NumElements() << " "
          << cpu_tensor->shape().DebugString() << " "
          << device_tensor->shape().DebugString();

  XlaTensor* xla_tensor = XlaTensor::FromTensor(device_tensor);
  CHECK(xla_tensor);

  XlaLayoutPreference layout_preference =
      shape_determination_fns_.layout_preference_fn(
          device_tensor->shape(), device_tensor->dtype(), absl::nullopt);
  Status status = [&]() -> Status {
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        shape_determination_fns_.shape_representation_fn(
                            device_tensor->shape(), device_tensor->dtype(),
                            /*fast_mem=*/false, layout_preference));

    // The device tensor should always be fresh.
    TF_RET_CHECK(!xla_tensor->has_shaped_buffer());

    TF_RETURN_IF_ERROR(
        xla_tensor->AllocateShapedBuffer(device_tensor->dtype(), shape, client_,
                                         stream_->parent()->device_ordinal()));

    // The cpu_tensor and literal that we created here hold the data of host
    // tensor in descending layout. The layout could be different from layout in
    // device_tensor (but the logical shape has to be the same). The
    // transfer_manager is responsible to do corresponding transposing when
    // transferring the data to device.
    xla::BorrowingLiteral literal(
        static_cast<const char*>(DMAHelper::base(cpu_tensor)),
        xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions()));

    VLOG(2) << "Transfer to device as literal: " << literal.ToString() << " "
            << xla_tensor->shaped_buffer().ToString();
    if (UseMultipleStreams() &&
        !transfer_manager_->CanShapedBufferBeAccessedNow(
            stream_->parent(), xla_tensor->shaped_buffer())) {
      // Initially wait for the compute stream so that memory allocations are
      // synchronized.
      host_to_device_stream_->ThenWaitFor(stream_.get());
    }

    TF_RETURN_IF_ERROR(transfer_manager_->TransferLiteralToDeviceAsync(
        host_to_device_stream_.get(), literal, xla_tensor->shaped_buffer()));

    if (UseMultipleStreams()) {
      auto event = std::make_shared<se::Event>(stream_->parent());
      TF_RET_CHECK(event->Init()) << "Event failed to initialize!";
      host_to_device_stream_->ThenRecordEvent(event.get());
      xla_tensor->ResetDefinitionEvent(std::move(event),
                                       host_to_device_stream_.get());
    }

    return Status::OK();
  }();
  if (!status.ok()) {
    done(status);
    return;
  }

  // Create a reference to hold onto cpu_tensor until after the literal has
  // been transferred
  TensorReference ref(*cpu_tensor);
  if (UseMultipleStreams()) {
    // Unref the host tensor when the transfer completes.
    // We don't defer the call to done() onto the stream here, and the reasons
    // why this is correct are subtle. We assume that:
    // a) all consumers of the device tensor will wait for its definition event.
    // b) if the tensor is destroyed, then the memory allocator will not hand
    //    out the same buffers until the transfer has completed.
    host_to_device_stream_->ThenDoHostCallback([ref]() { ref.Unref(); });
    done(status);
  } else {
    host_to_device_stream_->ThenDoHostCallback([ref, done]() {
      ref.Unref();
      done(Status::OK());
    });
  }
}

void XlaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             absl::string_view tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("tensor_name: \"" + std::string(tensor_name.data(), tensor_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_9(mht_9_v, 409, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceContext::CopyDeviceTensorToCPU");

  if (device_tensor->NumElements() == 0) {
    VLOG(2) << "CopyDeviceTensorToCPU empty tensor";
    done(Status::OK());
    return;
  }
  VLOG(2) << "CopyDeviceTensorToCPU "
          << reinterpret_cast<const void*>(device_tensor->tensor_data().data())
          << " "
          << reinterpret_cast<const void*>(cpu_tensor->tensor_data().data())
          << " " << device_tensor->NumElements() << " "
          << cpu_tensor->shape().DebugString() << " "
          << device_tensor->shape().DebugString();

  std::shared_ptr<se::Stream> device_to_host_stream;
  if (device_to_host_stream_) {
    device_to_host_stream = device_to_host_stream_;
  } else {
    stream_executor::port::StatusOr<xla::StreamPool::Ptr> ptr_or_status =
        client_->mutable_backend()->BorrowStream(
            stream_->parent()->device_ordinal());
    if (!ptr_or_status.status().ok()) {
      done(ptr_or_status.status());
      return;
    }
    device_to_host_stream =
        std::shared_ptr<se::Stream>(std::move(ptr_or_status.ValueOrDie()));
  }

  XlaTensor* xla_tensor = XlaTensor::FromTensor(device_tensor);
  xla_tensor->WaitForDefinitionEventOnStream(device_to_host_stream.get());

  // Transfer manager requires the shape of the shaped buffer to be the same as
  // literal shape except for the layout.  Set the literal to use xla_tensor's
  // shape as it is derived from the cpu_tensor's shape using
  // shape_representation_fn_.
  xla::MutableBorrowingLiteral literal;
  TF_CHECK_OK(HostTensorToMutableBorrowingLiteral(
      xla::LayoutUtil::GetWithDefaultLayout(
          xla_tensor->shaped_buffer().on_host_shape()),
      cpu_tensor, &literal));

  TensorReference ref(*device_tensor);
  const bool device_allows_sync_on_completion =
      device->AllowsSyncOnCompletion();
  // Explicitly capture device_to_host_stream to make sure the stream is alive
  // before the transfer finishes.
  transfer_manager_->TransferLiteralFromDevice(
      device_to_host_stream.get(), xla_tensor->shaped_buffer(), literal,
      [this, ref, xla_tensor, done, device_to_host_stream,
       device_allows_sync_on_completion](xla::Status status) {
        Status done_status = status;
        VLOG(2) << "Transfer from device as literal: "
                << xla_tensor->shaped_buffer().ToString();
        // For devices don't allow sync on completion, the device execution is
        // deferred. We check the execution stream status here to avoid wrong
        // results from a failed stream being propagated to following
        // host-side ops.
        if (!device_allows_sync_on_completion) {
          done_status.Update(xla_tensor->RefreshStatusOfStreams());
        }
        done(done_status);
        ref.Unref();
        // If a stream is in a bad state, it gets deleted when it's returned to
        // the stream pool, i.e. when it leaves this scope. However, a stream
        // deleting itself in a host callback on itself can cause bad behaviors
        // on some platforms. Releasing it in another stream to avoid that.
        if (!device_allows_sync_on_completion &&
            !device_to_host_stream->RefreshStatus().ok()) {
          auto status_or_new_stream = client_->mutable_backend()->BorrowStream(
              stream_->parent()->device_ordinal());
          if (status_or_new_stream.ok()) {
            status_or_new_stream.ValueOrDie()->ThenDoHostCallback(
                [device_to_host_stream] {});
          }
        }
      });
}

se::Stream* XlaDeviceContext::GetDeviceToDeviceStream() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_10(mht_10_v, 491, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceContext::GetDeviceToDeviceStream");

  DCHECK_GT(device_to_device_streams_.size(), 0);
  absl::MutexLock lock(&mu_);
  int stream = next_stream_;
  next_stream_ = (next_stream_ + 1) % device_to_device_streams_.size();
  return device_to_device_stream(stream);
}

Status XlaDeviceContext::ThenExecute(Device* device,
                                     stream_executor::Stream* stream,
                                     std::function<void()> func) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_device_contextDTcc mht_11(mht_11_v, 504, "", "./tensorflow/compiler/jit/xla_device_context.cc", "XlaDeviceContext::ThenExecute");

  VLOG(2) << "XlaDeviceContext::ThenExecute";
  stream->ThenDoHostCallback(std::move(func));
  return Status::OK();
}

}  // namespace tensorflow
