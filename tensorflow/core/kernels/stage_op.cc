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
class MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc() {
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

#include <cstddef>
#include <deque>
#include <mutex>
#include <numeric>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {

class Buffer : public ResourceBase {
 public:
  using Tuple = std::vector<Tensor>;

  explicit Buffer(std::size_t capacity, std::size_t memory_limit)
      : capacity_(capacity), memory_limit_(memory_limit), current_bytes_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/stage_op.cc", "Buffer");
}

  // the Buffer takes ownership of the Tuple
  Status Put(Tuple* tuple) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/kernels/stage_op.cc", "Put");

    std::unique_lock<std::mutex> lock(mu_);

    std::size_t tuple_bytes = GetTupleBytes(*tuple);

    // Sanity check so that we don't block for ever below
    if (memory_limit_ > 0 && tuple_bytes > memory_limit_) {
      return Status(
          errors::ResourceExhausted("Attempted to insert "
                                    "tensors with combined size of '",
                                    tuple_bytes,
                                    "' bytes into "
                                    "Staging Area with a memory limit of '",
                                    memory_limit_, "'."));
    }

    // If buffer capacity is bounded wait until elements have been removed
    if (IsBounded()) {
      full_cond_var_.wait(lock, [tuple_bytes, this]() {
        // If there's a memory limit, check if there's space for insertion
        bool memory_limit_valid =
            memory_limit_ > 0 ? !WouldExceedMemoryLimit(tuple_bytes) : true;
        // If we're configured for capacity check if there's space for insertion
        bool capacity_valid = capacity_ > 0 ? !IsCapacityFull() : true;

        // Stop waiting upon success for both conditions
        return capacity_valid && memory_limit_valid;
      });
    }

    // Update bytes in the Staging Area
    current_bytes_ += tuple_bytes;

    // Store tuple
    buf_.push_back(std::move(*tuple));

    lock.unlock();
    // Notify all removers. Removers
    // may be peeking at a specific element or waiting
    // for the element at the front of the deque.
    // As we don't know the appropriate one to wake up
    // we should wake them all.
    non_empty_cond_var_.notify_all();

    return Status::OK();
  }

  // Get tuple at front of the buffer
  void Get(Tuple* tuple) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/kernels/stage_op.cc", "Get");
  // TODO(zhifengc): Support cancellation.
    std::unique_lock<std::mutex> lock(mu_);

    // Wait for data if the buffer is empty
    non_empty_cond_var_.wait(lock, [this]() { return !buf_.empty(); });

    // Move data into the output tuple
    *tuple = std::move(buf_.front());
    buf_.pop_front();

    // Update bytes in the Staging Area
    current_bytes_ -= GetTupleBytes(*tuple);

    notify_inserters_if_bounded(&lock);
  }

  // Return tuple at index
  Status Peek(std::size_t index, Tuple* tuple) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_3(mht_3_v, 284, "", "./tensorflow/core/kernels/stage_op.cc", "Peek");

    std::unique_lock<std::mutex> lock(mu_);

    // Wait if the requested index is not available
    non_empty_cond_var_.wait(
        lock, [index, this]() { return index < this->buf_.size(); });

    // Place tensors in the output tuple
    for (const auto& tensor : buf_[index]) {
      tuple->push_back(tensor);
    }

    return Status::OK();
  }

  // Buffer size
  size_t Size() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_4(mht_4_v, 303, "", "./tensorflow/core/kernels/stage_op.cc", "Size");

    std::unique_lock<std::mutex> lock(mu_);
    return buf_.size();
  }

  void Clear() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_5(mht_5_v, 311, "", "./tensorflow/core/kernels/stage_op.cc", "Clear");

    std::unique_lock<std::mutex> lock(mu_);
    buf_.clear();
    current_bytes_ = 0;

    notify_inserters_if_bounded(&lock);
  }

  string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_6(mht_6_v, 322, "", "./tensorflow/core/kernels/stage_op.cc", "DebugString");

    std::unique_lock<std::mutex> lock(mu_);
    return strings::StrCat("Staging size: ", buf_.size());
  }

 private:
  // If the buffer is configured for bounded capacity, notify
  // waiting inserters that space is now available
  void notify_inserters_if_bounded(std::unique_lock<std::mutex>* lock) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_7(mht_7_v, 333, "", "./tensorflow/core/kernels/stage_op.cc", "notify_inserters_if_bounded");

    if (IsBounded()) {
      lock->unlock();
      // Notify all inserters. The removal of an element
      // may make memory available for many inserters
      // to insert new elements
      full_cond_var_.notify_all();
    }
  }

  // Are there a limit number of elements or a memory limit
  // configured on this buffer?
  bool IsBounded() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_8(mht_8_v, 348, "", "./tensorflow/core/kernels/stage_op.cc", "IsBounded");
 return capacity_ > 0 || memory_limit_ > 0; }

  bool IsCapacityFull() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_9(mht_9_v, 353, "", "./tensorflow/core/kernels/stage_op.cc", "IsCapacityFull");
 return buf_.size() >= capacity_; }

  bool WouldExceedMemoryLimit(std::size_t bytes) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_10(mht_10_v, 358, "", "./tensorflow/core/kernels/stage_op.cc", "WouldExceedMemoryLimit");

    return bytes + current_bytes_ > memory_limit_;
  }

  std::size_t GetTupleBytes(const Tuple& tuple) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_11(mht_11_v, 365, "", "./tensorflow/core/kernels/stage_op.cc", "GetTupleBytes");

    return std::accumulate(tuple.begin(), tuple.end(), 0,
                           [](const std::size_t& lhs, const Tensor& rhs) {
                             return lhs + rhs.TotalBytes();
                           });
  }

  std::size_t capacity_;
  std::size_t memory_limit_;
  std::size_t current_bytes_;
  mutable std::mutex mu_;
  std::condition_variable non_empty_cond_var_;
  std::condition_variable full_cond_var_;
  std::deque<Tuple> buf_;
};

Status GetBuffer(OpKernelContext* ctx, const NodeDef& ndef, Buffer** buf) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_12(mht_12_v, 384, "", "./tensorflow/core/kernels/stage_op.cc", "GetBuffer");

  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;

  // Lambda for creating the Staging Area
  auto create_fn = [&ndef](Buffer** ret) -> Status {
    int64_t capacity;
    int64_t memory_limit;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "memory_limit", &memory_limit));
    *ret = new Buffer(capacity, memory_limit);
    return Status::OK();
  };

  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<Buffer>(cinfo.container(), cinfo.name(),
                                                buf, create_fn));
  return Status::OK();
}

}  // namespace

class StageOp : public OpKernel {
 public:
  explicit StageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_13(mht_13_v, 411, "", "./tensorflow/core/kernels/stage_op.cc", "StageOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_14(mht_14_v, 416, "", "./tensorflow/core/kernels/stage_op.cc", "Compute");

    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);
    Buffer::Tuple tuple;
    tuple.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      tuple.push_back(ctx->input(i));
    }
    OP_REQUIRES_OK(ctx, buf->Put(&tuple));
  }
};

REGISTER_KERNEL_BUILDER(Name("Stage").Device(DEVICE_CPU), StageOp);
REGISTER_KERNEL_BUILDER(Name("Stage").Device(DEVICE_DEFAULT), StageOp);

class UnstageOp : public OpKernel {
 public:
  explicit UnstageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_15(mht_15_v, 437, "", "./tensorflow/core/kernels/stage_op.cc", "UnstageOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_16(mht_16_v, 444, "", "./tensorflow/core/kernels/stage_op.cc", "Compute");

    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);
    Buffer::Tuple tuple;

    buf->Get(&tuple);

    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));

    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Unstage").Device(DEVICE_CPU), UnstageOp);
REGISTER_KERNEL_BUILDER(Name("Unstage").Device(DEVICE_DEFAULT), UnstageOp);

class StagePeekOp : public OpKernel {
 public:
  explicit StagePeekOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_17(mht_17_v, 471, "", "./tensorflow/core/kernels/stage_op.cc", "StagePeekOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_18(mht_18_v, 478, "", "./tensorflow/core/kernels/stage_op.cc", "Compute");

    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);
    Buffer::Tuple tuple;

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ctx->input(0).shape()),
                errors::InvalidArgument("index must be scalar"));
    std::size_t index = ctx->input(0).scalar<int>()();

    OP_REQUIRES_OK(ctx, buf->Peek(index, &tuple));

    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));

    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StagePeek").Device(DEVICE_CPU), StagePeekOp);
REGISTER_KERNEL_BUILDER(
    Name("StagePeek").HostMemory("index").Device(DEVICE_DEFAULT), StagePeekOp);

class StageSizeOp : public OpKernel {
 public:
  explicit StageSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_19(mht_19_v, 510, "", "./tensorflow/core/kernels/stage_op.cc", "StageSizeOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_20(mht_20_v, 517, "", "./tensorflow/core/kernels/stage_op.cc", "Compute");

    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);

    // Allocate size output tensor
    Tensor* size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(buf->Size());
  }
};

REGISTER_KERNEL_BUILDER(Name("StageSize").Device(DEVICE_CPU), StageSizeOp);
REGISTER_KERNEL_BUILDER(
    Name("StageSize").HostMemory("size").Device(DEVICE_DEFAULT), StageSizeOp);

class StageClearOp : public OpKernel {
 public:
  explicit StageClearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_21(mht_21_v, 540, "", "./tensorflow/core/kernels/stage_op.cc", "StageClearOp");
}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstage_opDTcc mht_22(mht_22_v, 547, "", "./tensorflow/core/kernels/stage_op.cc", "Compute");

    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);

    buf->Clear();
  }
};

REGISTER_KERNEL_BUILDER(Name("StageClear").Device(DEVICE_CPU), StageClearOp);
REGISTER_KERNEL_BUILDER(Name("StageClear").Device(DEVICE_DEFAULT),
                        StageClearOp);

}  // namespace tensorflow
