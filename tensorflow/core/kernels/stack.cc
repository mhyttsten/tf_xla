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
class MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc() {
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

#include "tensorflow/core/kernels/stack.h"

#include <limits.h>
#include <atomic>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Stack : public ResourceBase {
 public:
  static std::atomic<int64_t> stack_counter;

  struct TensorAndAllocation {
    Tensor tensor;
    AllocatorAttributes alloc_attrs;
    bool swapped_to_cpu;
  };

  Stack(const DataType& elem_type, const string& stack_name, int max_size)
      : elem_type_(elem_type),
        stack_name_(stack_name),
        max_size_(max_size),
        closed_(false) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("stack_name: \"" + stack_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/kernels/stack.cc", "Stack");
}

  Status Push(const TensorAndAllocation& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/stack.cc", "Push");

    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CheckNotClosed());
    int stack_size = stack_.size();
    if (max_size_ >= 0 && stack_size >= max_size_) {
      return errors::InvalidArgument("Stack[", stack_name_, "] overflowed ",
                                     "its max_size (", max_size_, ")");
    }
    stack_.push_back(value);
    return Status::OK();
  }

  Status Pop(TensorAndAllocation* value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/stack.cc", "Pop");

    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(CheckNotClosed());
    if (stack_.empty()) {
      return errors::InvalidArgument("Stack[", stack_name_,
                                     "] is empty when calling Pop().");
    }
    *value = stack_.back();
    stack_.pop_back();
    return Status::OK();
  }

  // We don't swap the first tensor on the stack and any subsequent tensors
  // that share the buffer with the first tensor.
  bool IsUsefulToSwap(const Tensor& tensor) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/kernels/stack.cc", "IsUsefulToSwap");

    mutex_lock l(mu_);
    if (stack_.empty()) {
      return false;
    }
    const Tensor& first = stack_.front().tensor;
    return !tensor.SharesBufferWith(first);
  }

  void Close() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/kernels/stack.cc", "Close");

    mutex_lock l(mu_);
    stack_.clear();
    closed_ = true;
  }

  DataType ElemType() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/kernels/stack.cc", "ElemType");
 return elem_type_; }

  string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_6(mht_6_v, 288, "", "./tensorflow/core/kernels/stack.cc", "DebugString");

    mutex_lock l(mu_);
    return strings::StrCat("Stack[", stack_name_, "]");
  }

  const string& stack_name() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_7(mht_7_v, 296, "", "./tensorflow/core/kernels/stack.cc", "stack_name");
 return stack_name_; }

 private:
  friend class StackOp;
  mutex* mu() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_8(mht_8_v, 303, "", "./tensorflow/core/kernels/stack.cc", "mu");
 return &mu_; }

  mutable mutex mu_;
  DataType elem_type_;
  const string stack_name_;
  Tensor handle_;
  int max_size_;
  bool closed_ TF_GUARDED_BY(mu_);
  std::vector<TensorAndAllocation> stack_ TF_GUARDED_BY(mu_);

  Status CheckNotClosed() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (closed_) {
      return errors::InvalidArgument("Stack[", stack_name_,
                                     "] has already been closed.");
    }
    return Status::OK();
  }
};

Status GetStack(OpKernelContext* ctx, Stack** stack) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_9(mht_9_v, 325, "", "./tensorflow/core/kernels/stack.cc", "GetStack");

  if (ctx->input_dtype(0) == DT_RESOURCE) {
    return LookupResource(ctx, HandleFromInput(ctx, 0), stack);
  } else {
    Tensor Tstack_handle = ctx->mutable_input(0, false);
    if (Tstack_handle.NumElements() != 2) {
      return errors::InvalidArgument(
          "Stack handle must have two elements, but had shape: ",
          Tstack_handle.shape().DebugString());
    }
    const string& container = Tstack_handle.flat<tstring>()(0);
    const string& stack_name = Tstack_handle.flat<tstring>()(1);
    string key = strings::StrCat(container, stack_name);
    ResourceMgr* rm = ctx->resource_manager();
    if (rm == nullptr) {
      return errors::Internal("No resource manager.");
    }
    auto* step_container = ctx->step_container();
    if (step_container == nullptr) {
      return errors::Internal("No step container.");
    }
    TF_RETURN_IF_ERROR(step_container->Lookup(rm, key, stack));
    return Status::OK();
  }
}

std::atomic<int64_t> Stack::stack_counter{0};

// StackOp

StackOp::StackOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_10(mht_10_v, 358, "", "./tensorflow/core/kernels/stack.cc", "StackOp::StackOp");

  OP_REQUIRES_OK(context, context->GetAttr("elem_type", &elem_type_));
  OP_REQUIRES_OK(context, context->GetAttr("stack_name", &stack_name_));
  if (stack_name_.empty()) stack_name_ = name();
}

void StackOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_11(mht_11_v, 367, "", "./tensorflow/core/kernels/stack.cc", "StackOp::Compute");

  int32_t size = std::numeric_limits<int32>::max();
  if (ctx->num_inputs() > 0) {
    const Tensor* tensor_size;
    OP_REQUIRES_OK(ctx, ctx->input("max_size", &tensor_size));

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(tensor_size->shape()),
        errors::InvalidArgument("Stack size must be a scalar, but had shape: ",
                                tensor_size->shape().DebugString()));

    int32_t size_value = tensor_size->scalar<int32>()();
    if (size_value >= 0) {
      size = size_value;
    }
  }

  static const char kContainer[] = "_stacks";
  auto stack_id = Stack::stack_counter.fetch_add(1);
  string stack_name = strings::StrCat(stack_name_, "_", stack_id);
  // Store the handle in a per-step container.
  ResourceMgr* rm = ctx->resource_manager();
  OP_REQUIRES(ctx, rm != nullptr, errors::Internal("No resource manager."));
  string key = strings::StrCat(kContainer, stack_name);
  auto* step_container = ctx->step_container();
  OP_REQUIRES(ctx, step_container != nullptr,
              errors::Internal("No step container."));
  Stack* stack = new Stack(elem_type_, stack_name, size);
  OP_REQUIRES_OK(ctx, step_container->Create(rm, key, stack));
  if (IsRefType(ctx->expected_output_dtype(0))) {
    // Create the stack handle.
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(tensorflow::DT_STRING,
                                           tensorflow::TensorShape({2}),
                                           &stack->handle_, alloc_attr));
    auto handle = stack->handle_.flat<tstring>();
    handle(0) = kContainer;
    handle(1) = std::move(stack_name);
    ctx->set_output_ref(0, stack->mu(), &stack->handle_);
  } else {
    Tensor* handle;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &handle));
    handle->flat<ResourceHandle>()(0) =
        ctx->step_container()->MakeResourceHandle<Stack>(key, *ctx->device());
  }
}

// StackPushOp

StackPushOp::StackPushOp(OpKernelConstruction* context, bool allow_swapping)
    : AsyncOpKernel(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_12(mht_12_v, 421, "", "./tensorflow/core/kernels/stack.cc", "StackPushOp::StackPushOp");

  if (allow_swapping) {
    OP_REQUIRES_OK(context, context->GetAttr("swap_memory", &swap_memory_));
  }
}

void StackPushOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_13(mht_13_v, 430, "", "./tensorflow/core/kernels/stack.cc", "StackPushOp::ComputeAsync");

  // Get the stack from the handle.
  Stack* stack = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, GetStack(ctx, &stack), done);
  core::ScopedUnref unref(stack);

  if (ctx->input_dtype(1) != stack->ElemType()) {
    ctx->CtxFailure(errors::InvalidArgument("Must have type ",
                                            stack->ElemType(), " but got ",
                                            ctx->input_dtype(1)));
    done();
    return;
  }

  // Push the tensor onto the stack. Swap the tensor to CPU if instructed.
  const Tensor& tensor = ctx->input(1);
  AllocatorAttributes alloc_attrs = ctx->input_alloc_attr(1);
  // For now, we use a simple heuristic for swapping: A GPU tensor is moved
  // to CPU if the tensor has more than kCopyThreshold bytes and the GPU
  // allocator says more than kOccupancy of the memory is in use.
  static constexpr int kCopyThreshold = 2048;
  static constexpr double kOccupancy = 0.7;
  if (swap_memory_ && !alloc_attrs.on_host() &&
      tensor.TotalBytes() > kCopyThreshold && stack->IsUsefulToSwap(tensor)) {
    DeviceContext* device_ctxt = ctx->op_device_context();
    auto device = static_cast<tensorflow::Device*>(ctx->device());
    Allocator* allocator = device->GetAllocator(alloc_attrs);
    absl::optional<AllocatorStats> stats = allocator->GetStats();
    if (stats && *stats->bytes_limit &&
        stats->bytes_in_use > (*stats->bytes_limit * kOccupancy)) {
      // Asynchronously copy the tensor from GPU to CPU memory.
      // TODO(yuanbyu): Swap the oldest tensor first.
      AllocatorAttributes host_alloc_attrs;
      host_alloc_attrs.set_gpu_compatible(true);
      host_alloc_attrs.set_on_host(true);
      Allocator* cpu_allocator = device->GetAllocator(host_alloc_attrs);
      Tensor* cpu_tensor =
          new Tensor(cpu_allocator, tensor.dtype(), tensor.shape());
      device_ctxt->CopyDeviceTensorToCPU(
          &tensor, "StackPush", device, cpu_tensor,
          [cpu_tensor, stack, ctx, done](const Status& s) {
            ctx->SetStatus(s);
            if (s.ok()) {
              AllocatorAttributes alloc_attrs = ctx->input_alloc_attr(1);
              ctx->SetStatus(stack->Push({*cpu_tensor, alloc_attrs, true}));
            }
            if (ctx->status().ok()) {
              ctx->set_output(0, *cpu_tensor);
            }
            done();
            delete cpu_tensor;
          });
      return;
    }
  }

  // Execute synchronously if not swapped.
  OP_REQUIRES_OK_ASYNC(ctx, stack->Push({tensor, alloc_attrs, false}), done);
  ctx->set_output(0, tensor);
  done();
}

bool StackPushOp::IsExpensive() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_14(mht_14_v, 495, "", "./tensorflow/core/kernels/stack.cc", "StackPushOp::IsExpensive");
 return false; }

// StackPopOp

StackPopOp::StackPopOp(OpKernelConstruction* context)
    : AsyncOpKernel(context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_15(mht_15_v, 503, "", "./tensorflow/core/kernels/stack.cc", "StackPopOp::StackPopOp");
}

void StackPopOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_16(mht_16_v, 508, "", "./tensorflow/core/kernels/stack.cc", "StackPopOp::ComputeAsync");

  // Get the stack from the handle.
  Stack* stack = nullptr;
  OP_REQUIRES_OK_ASYNC(ctx, GetStack(ctx, &stack), done);
  core::ScopedUnref unref(stack);

  // Pop the tensor. Transfer the tensor back to device if it was
  // swapped out to CPU.
  Stack::TensorAndAllocation value;
  OP_REQUIRES_OK_ASYNC(ctx, stack->Pop(&value), done);
  if (value.swapped_to_cpu) {
    // Asynchronously copy the tensor back from CPU to GPU memory.
    DeviceContext* device_ctxt = ctx->op_device_context();
    Device* device = static_cast<Device*>(ctx->device());
    Tensor* cpu_tensor = &value.tensor;
    Allocator* gpu_allocator = device->GetAllocator(value.alloc_attrs);
    Tensor* device_tensor =
        new Tensor(gpu_allocator, cpu_tensor->dtype(), cpu_tensor->shape());
    device_ctxt->CopyCPUTensorToDevice(
        cpu_tensor, device, device_tensor,
        [device_tensor, ctx, done](const Status& s) {
          ctx->SetStatus(s);
          if (s.ok()) {
            ctx->set_output(0, *device_tensor);
          }
          done();
          delete device_tensor;
        });
  } else {
    // Execute synchronously if not swapped.
    ctx->set_output(0, value.tensor);
    done();
  }
}

bool StackPopOp::IsExpensive() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_17(mht_17_v, 546, "", "./tensorflow/core/kernels/stack.cc", "StackPopOp::IsExpensive");
 return false; }

// StackCloseOp

StackCloseOp::StackCloseOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_18(mht_18_v, 553, "", "./tensorflow/core/kernels/stack.cc", "StackCloseOp::StackCloseOp");
}

void StackCloseOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_19(mht_19_v, 558, "", "./tensorflow/core/kernels/stack.cc", "StackCloseOp::Compute");

  Stack* stack = nullptr;
  OP_REQUIRES_OK(ctx, GetStack(ctx, &stack));
  core::ScopedUnref unref(stack);
  stack->Close();
}

bool StackCloseOp::IsExpensive() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstackDTcc mht_20(mht_20_v, 568, "", "./tensorflow/core/kernels/stack.cc", "StackCloseOp::IsExpensive");
 return false; }

}  // namespace tensorflow
