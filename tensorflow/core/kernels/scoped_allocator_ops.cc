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
class MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc() {
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

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/scoped_allocator.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class ScopedAllocatorOp : public OpKernel {
 public:
  explicit ScopedAllocatorOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/scoped_allocator_ops.cc", "ScopedAllocatorOp");

    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("sa_name", &name_));
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    OP_REQUIRES_OK(context, context->GetAttr("expected_call_count",
                                             &expected_call_count_));
    device_ = context->device();
    // Precalculate the size of the backing tensor and the offsets of
    // the subtensors to be allocated from it, taking into account
    // alignment considerations.
    ScopedAllocatorMgr::PopulateFields(id_, shapes_, dtype_, &fields_);
    size_t num_bytes = fields_.back().offset + fields_.back().bytes_allocated;
    num_elements_ = num_bytes / DataTypeSize(dtype_);
    OP_REQUIRES(context, num_bytes % DataTypeSize(dtype_) == 0,
                errors::InvalidArgument(
                    "Number of bytes ", num_bytes,
                    " must be divisible by size of datatype ", dtype_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/scoped_allocator_ops.cc", "Compute");

    ScopedAllocatorMgr* sam = device_->GetScopedAllocatorMgr();
    if (!sam) {
      context->SetStatus(errors::Internal(
          "ScopedAllocatorMgr not supported on device ", device_->name()));
      return;
    }
    Tensor* backing_tensor = nullptr;
    AllocatorAttributes attr = context->output_alloc_attr(0);
    Status s =
        context->allocate_output(0, {num_elements_}, &backing_tensor, attr);
    VLOG(1) << "_ScopedAllocatorOp " << context->op_kernel().name()
            << " new backing tensor size " << backing_tensor->TotalBytes()
            << " num_elements_ " << num_elements_ << " buffer "
            << DMAHelper::buffer(backing_tensor) << " base addr "
            << DMAHelper::base(backing_tensor);
    if (s.ok()) {
      s = sam->AddScopedAllocator(*backing_tensor, context->step_id(), id_,
                                  name_, fields_, expected_call_count_);
    }
    if (!s.ok()) {
      context->SetStatus(s);
    }
  }

 private:
  std::vector<TensorShape> shapes_;
  DataType dtype_;
  int64_t num_elements_;
  std::vector<ScopedAllocator::Field> fields_;
  string name_;
  int32 id_;
  int32 expected_call_count_;
  DeviceBase* device_;
};

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocator").Device(DEVICE_CPU),
                        ScopedAllocatorOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocator").Device(DEVICE_GPU),
                        ScopedAllocatorOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocator").Device(DEVICE_DEFAULT),
                        ScopedAllocatorOp);

class ScopedAllocatorConcatOp : public OpKernel {
 public:
  explicit ScopedAllocatorConcatOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/kernels/scoped_allocator_ops.cc", "ScopedAllocatorConcatOp");

    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("reshape", &reshape_));
    // These attributes are just for debugging.
    OP_REQUIRES_OK(context, context->GetAttr("sa_name", &name_));
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    device_ = context->device();
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/kernels/scoped_allocator_ops.cc", "Compute");

    const Tensor& backing_tensor = context->input(0);
    // Check that type matches.
    OP_REQUIRES(context, backing_tensor.dtype() == dtype_,
                errors::InvalidArgument("Backing tensor type ",
                                        DataTypeString(backing_tensor.dtype()),
                                        " does not match expected type ",
                                        DataTypeString(dtype_)));
    // Check that backing tensor is at least as large as the shape of the
    // output.
    OP_REQUIRES(context, backing_tensor.NumElements() >= shape_.num_elements(),
                errors::InvalidArgument("Backing tensor num elements ",
                                        backing_tensor.NumElements(),
                                        " is not >= to expected ",
                                        shape_.num_elements()));
    Tensor output(dtype_);
    if (reshape_) {
      CHECK(output.CopyFrom(backing_tensor, shape_));
    } else {
      CHECK(output.CopyFrom(backing_tensor, backing_tensor.shape()));
    }
    context->set_output(0, output);
    const TensorBuffer* backing_buf = DMAHelper::buffer(&output);
    const void* backing_tensor_lb = backing_buf->data();
    const void* backing_tensor_ub = static_cast<const void*>(
        static_cast<const char*>(backing_tensor_lb) + backing_buf->size());
    // Check that all inputs lie entirely within the backing tensor.
    for (int i = 1; i < context->num_inputs(); ++i) {
      const TensorBuffer* input_buf = DMAHelper::buffer(&context->input(i));
      const void* input_lb = input_buf->data();
      const void* input_ub = static_cast<const void*>(
          static_cast<const char*>(input_lb) + input_buf->size());
      OP_REQUIRES(
          context, input_lb >= backing_tensor_lb,
          errors::InvalidArgument(
              "Lower bound check fail for input ", i, " from node ",
              context->op_kernel().requested_input(i), " to node ",
              context->op_kernel().name(), " input bounds = [", input_lb, ", ",
              input_ub, "]", " backing_tensor bounds = [", backing_tensor_lb,
              ", ", backing_tensor_ub, "]"));
      OP_REQUIRES(
          context, input_ub <= backing_tensor_ub,
          errors::InvalidArgument(
              "Upper bound check fail for input ", i, " from node ",
              context->op_kernel().requested_input(i), " to node ",
              context->op_kernel().name(), " input bounds = [", input_lb, ", ",
              input_ub, "]", " backing_tensor bounds = [", backing_tensor_lb,
              ", ", backing_tensor_ub, "]"));
    }
    VLOG(1) << "_ScopedAllocatorConcatOp outputting backing tensor at "
            << backing_buf;
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  string name_;
  int32 id_;
  bool reshape_;
  DeviceBase* device_;
};

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorConcat").Device(DEVICE_CPU),
                        ScopedAllocatorConcatOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorConcat").Device(DEVICE_GPU),
                        ScopedAllocatorConcatOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorConcat").Device(DEVICE_DEFAULT),
                        ScopedAllocatorConcatOp);

class ScopedAllocatorSplitOp : public OpKernel {
 public:
  explicit ScopedAllocatorSplitOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc mht_4(mht_4_v, 363, "", "./tensorflow/core/kernels/scoped_allocator_ops.cc", "ScopedAllocatorSplitOp");

    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype_));
    // This stuff is just for debugging
    OP_REQUIRES_OK(context, context->GetAttr("sa_name", &name_));
    OP_REQUIRES_OK(context, context->GetAttr("id", &id_));
    device_ = context->device();
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscoped_allocator_opsDTcc mht_5(mht_5_v, 374, "", "./tensorflow/core/kernels/scoped_allocator_ops.cc", "Compute");

    Tensor backing_copy(context->input(0));
    // Check that type matches.
    OP_REQUIRES(context, backing_copy.dtype() == dtype_,
                errors::InvalidArgument("Backing tensor type ",
                                        DataTypeString(backing_copy.dtype()),
                                        " does not match expected type ",
                                        DataTypeString(dtype_)));
    const TensorBuffer* backing_buf = DMAHelper::buffer(&backing_copy);
    const void* backing_tensor_lb = backing_buf->data();
    const void* backing_tensor_ub = static_cast<const void*>(
        static_cast<const char*>(backing_tensor_lb) + backing_buf->size());
    for (int i = 1; i < context->num_inputs(); ++i) {
      VLOG(1) << "_ScopedAllocatorSplitOp assigning input " << i
              << " to output " << i - 1 << " buf addr "
              << DMAHelper::base(&context->input(i));
      Tensor copy(context->input(i));
      OP_REQUIRES(context, copy.dtype() == dtype_,
                  errors::InvalidArgument("Input ", i, " tensor type ",
                                          DataTypeString(copy.dtype()),
                                          " does not match expected type ",
                                          DataTypeString(dtype_)));
      context->set_output(i - 1, copy);
      const TensorBuffer* input_buf = DMAHelper::buffer(&copy);
      const void* input_lb = input_buf->data();
      OP_REQUIRES(
          context, input_lb >= backing_tensor_lb,
          errors::InvalidArgument("Lower bound check fail for input ", i,
                                  " to node ", context->op_kernel().name()));
      const void* input_ub = static_cast<const void*>(
          static_cast<const char*>(input_lb) + input_buf->size());
      OP_REQUIRES(
          context, input_ub <= backing_tensor_ub,
          errors::InvalidArgument("Upper bound check fail for input ", i,
                                  " to node ", context->op_kernel().name()));
    }
  }

 private:
  DataType dtype_;
  string name_;
  int32 id_;
  DeviceBase* device_;
};

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorSplit").Device(DEVICE_CPU),
                        ScopedAllocatorSplitOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorSplit").Device(DEVICE_GPU),
                        ScopedAllocatorSplitOp);

REGISTER_KERNEL_BUILDER(Name("_ScopedAllocatorSplit").Device(DEVICE_DEFAULT),
                        ScopedAllocatorSplitOp);

}  // namespace tensorflow
