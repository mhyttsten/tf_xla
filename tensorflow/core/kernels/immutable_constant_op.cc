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
class MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc() {
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

#include "tensorflow/core/kernels/immutable_constant_op.h"

#include <unordered_set>

#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

namespace {
class MemmappedTensorAllocator : public Allocator {
 public:
  MemmappedTensorAllocator() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "MemmappedTensorAllocator");
}

  Status InitializeFromRegion(const string& name, Env* env) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "InitializeFromRegion");

    const auto status =
        env->NewReadOnlyMemoryRegionFromFile(name, &memory_region_);
    if (!status.ok()) {
      return status;
    }
    return Status::OK();
  }
  string Name() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "Name");
 return "MemmappedTensorAllocator"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_3(mht_3_v, 218, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "AllocateRaw");

    if ((reinterpret_cast<intptr_t>(memory_region_->data())) % alignment != 0) {
      allocation_status_ =
          errors::Internal("Readonly memory region has wrong alignment");
      return nullptr;
    }
    if (num_bytes > memory_region_->length()) {
      allocation_status_ = errors::Internal(
          "Readonly memory region has wrong length (", memory_region_->length(),
          ") when allocating ", num_bytes);
      return nullptr;
    }
    return const_cast<void*>(memory_region_->data());
  }

  void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_4(mht_4_v, 236, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "DeallocateRaw");

    if (ptr != memory_region_->data()) {
      LOG(ERROR)
          << "Deallocating not allocated region for readonly memory region";
    }
    if (delete_on_deallocate_) {
      delete this;
    }
  }
  const Status& allocation_status() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_5(mht_5_v, 248, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "allocation_status");
 return allocation_status_; }

  void set_delete_on_deallocate() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_6(mht_6_v, 253, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "set_delete_on_deallocate");
 delete_on_deallocate_ = true; }

  // Make sure tensors or complex types (strings, variants, resources) don't get
  // their constructor called via a placement new since that would require
  // writing to immutable data.
  // See also: tensorflow/core/framework/typed_allocator.h
  bool AllocatesOpaqueHandle() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_7(mht_7_v, 262, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "AllocatesOpaqueHandle");
 return true; }

 private:
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region_;
  // If there is an error during allocation we keep it in this status.
  Status allocation_status_;

  // When the allocator is owned by TensorBuffer it will be deleted on
  // de-allocation.
  bool delete_on_deallocate_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(MemmappedTensorAllocator);
};
}  // namespace

ImmutableConstantOp::ImmutableConstantOp(OpKernelConstruction* context)
    : OpKernel(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_8(mht_8_v, 281, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "ImmutableConstantOp::ImmutableConstantOp");

  OP_REQUIRES_OK(context,
                 context->GetAttr(kMemoryRegionNameAttr, &region_name_));
  OP_REQUIRES_OK(context, context->GetAttr(kDTypeAttr, &dtype_));
  OP_REQUIRES(context, dtype_ != DT_RESOURCE && dtype_ != DT_VARIANT,
              errors::InvalidArgument(
                  "Resource and variant dtypes are invalid for this op."));
  OP_REQUIRES_OK(context, context->GetAttr(kShapeAttr, &shape_));
}

void ImmutableConstantOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_9(mht_9_v, 294, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "ImmutableConstantOp::Compute");

  std::unique_ptr<MemmappedTensorAllocator> allocator(
      new MemmappedTensorAllocator());

  OP_REQUIRES_OK(ctx,
                 allocator->InitializeFromRegion(region_name_, ctx->env()));
  OP_REQUIRES(ctx, dtype_ != DT_STRING,
              errors::Unimplemented("Sorry, DT_STRING is not currently "
                                    "supported for ImmutableConstOp."));
  ctx->set_output(0, Tensor(allocator.get(), dtype_, shape_));
  OP_REQUIRES_OK(ctx, allocator->allocation_status());
  // Allocator is owned by the tensor from this point.
  allocator.release()->set_delete_on_deallocate();
}

ImmutableConstantOp::~ImmutableConstantOp() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSimmutable_constant_opDTcc mht_10(mht_10_v, 312, "", "./tensorflow/core/kernels/immutable_constant_op.cc", "ImmutableConstantOp::~ImmutableConstantOp");
}
constexpr char const* ImmutableConstantOp::kDTypeAttr;
constexpr char const* ImmutableConstantOp::kShapeAttr;
constexpr char const* ImmutableConstantOp::kMemoryRegionNameAttr;

REGISTER_KERNEL_BUILDER(Name("ImmutableConst").Device(DEVICE_CPU),
                        ImmutableConstantOp);
}  // namespace tensorflow
