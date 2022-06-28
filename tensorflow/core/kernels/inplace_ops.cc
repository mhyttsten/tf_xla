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
class MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename Device, typename T>
Status DoParallelConcatUpdate(const Device& d, const Tensor& value, int32_t loc,
                              Tensor* output) {
  auto Tvalue = value.shaped<T, 2>({1, value.NumElements()});
  auto Toutput = output->flat_outer_dims<T>();
  auto nrows = Toutput.dimension(0);
  auto r = (loc % nrows + nrows) % nrows;  // Guard index range.
  Toutput.template chip<0>(r).device(d) = Tvalue.template chip<0>(0);
  return Status::OK();
}

template <>
Status DoParallelConcat(const CPUDevice& d, const Tensor& value, int32_t loc,
                        Tensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoParallelConcat");

  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                  \
  case DataTypeToEnum<type>::value: \
    return DoParallelConcatUpdate<CPUDevice, type>(d, value, loc, output);
    TF_CALL_POD_TYPES(CASE);
    TF_CALL_tstring(CASE);
    TF_CALL_variant(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(value.dtype()));
  }
}

}  // end namespace functor

namespace {

template <typename Device>
class ParallelConcatUpdate : public OpKernel {
 public:
  explicit ParallelConcatUpdate(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/kernels/inplace_ops.cc", "ParallelConcatUpdate");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/kernels/inplace_ops.cc", "Compute");

    auto value = ctx->input(0);
    // Value should be at least rank 1. Also the 0th dimension should be
    // at least loc_.
    OP_REQUIRES(ctx, value.dims() >= 1,
                errors::InvalidArgument("value should be at least rank 1."));
    OP_REQUIRES(
        ctx, value.dim_size(0) > loc_,
        errors::InvalidArgument("0th dimension of value = ", value.dim_size(0),
                                " is less than loc_=", loc_));

    auto update = ctx->input(1);

    OP_REQUIRES(
        ctx, value.dims() == update.dims(),
        errors::InvalidArgument("value and update shape doesn't match: ",
                                value.shape().DebugString(), " vs. ",
                                update.shape().DebugString()));
    for (int i = 1; i < value.dims(); ++i) {
      OP_REQUIRES(
          ctx, value.dim_size(i) == update.dim_size(i),
          errors::InvalidArgument("value and update shape doesn't match ",
                                  value.shape().DebugString(), " vs. ",
                                  update.shape().DebugString()));
    }
    OP_REQUIRES(ctx, 1 == update.dim_size(0),
                errors::InvalidArgument("update shape doesn't match: ",
                                        update.shape().DebugString()));

    Tensor output = value;  // This creates an alias intentionally.
    const auto& d = ctx->eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, ::tensorflow::functor::DoParallelConcat(d, update, loc_, &output));
    ctx->set_output(0, output);
  }

 private:
  int32 loc_;
};

template <typename Device, typename T>
class ParallelConcatStart : public OpKernel {
 public:
  explicit ParallelConcatStart(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_3(mht_3_v, 293, "", "./tensorflow/core/kernels/inplace_ops.cc", "ParallelConcatStart");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_4(mht_4_v, 300, "", "./tensorflow/core/kernels/inplace_ops.cc", "Compute");

    Tensor* out = nullptr;
    // We do not know whether the output will be used on GPU. Setting it to be
    // gpu-compatible for now.
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &out, attr));
  }

 private:
  TensorShape shape_;
};

class FailureKernel : public OpKernel {
 public:
  explicit FailureKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_5(mht_5_v, 318, "", "./tensorflow/core/kernels/inplace_ops.cc", "FailureKernel");

    OP_REQUIRES_OK(ctx,
                   errors::Internal("Found instance of parallel_stack which "
                                    "could not be properly replaced."));
  }

  void Compute(OpKernelContext*) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_6(mht_6_v, 327, "", "./tensorflow/core/kernels/inplace_ops.cc", "Compute");
}
};

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<CPUDevice>);
TF_CALL_POD_STRING_TYPES(REGISTER)
#undef REGISTER

#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#define REGISTER_PARALLEL_CONCAT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_POD_STRING_TYPES(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_PARALLEL_CONCAT_START(type)                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_PARALLEL_CONCAT_START)
#undef REGISTER_PARALLEL_CONCAT_START

#define REGISTER_PARALLEL_CONCAT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<GPUDevice>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER)
#undef REGISTER

// Register versions that operate on int32 data on the CPU even though the op
// has been placed on the GPU

REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ParallelConcatUpdate<CPUDevice>);
#endif

class InplaceOpBase : public OpKernel {
 public:
  explicit InplaceOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_7(mht_7_v, 398, "", "./tensorflow/core/kernels/inplace_ops.cc", "InplaceOpBase");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_8(mht_8_v, 403, "", "./tensorflow/core/kernels/inplace_ops.cc", "Compute");

    auto x = ctx->input(0);
    auto i = ctx->input(1);
    auto v = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(i.shape()),
                errors::InvalidArgument("i must be a vector. ",
                                        i.shape().DebugString()));
    OP_REQUIRES(ctx, x.dims() == v.dims(),
                errors::InvalidArgument(
                    "x and v shape doesn't match (ranks differ): ",
                    x.shape().DebugString(), " vs. ", v.shape().DebugString()));
    for (int i = 1; i < x.dims(); ++i) {
      OP_REQUIRES(
          ctx, x.dim_size(i) == v.dim_size(i),
          errors::InvalidArgument("x and v shape doesn't match at index ", i,
                                  " : ", x.shape().DebugString(), " vs. ",
                                  v.shape().DebugString()));
    }
    OP_REQUIRES(ctx, i.dim_size(0) == v.dim_size(0),
                errors::InvalidArgument(
                    "i and x shape doesn't match at index 0: ",
                    i.shape().DebugString(), " vs. ", v.shape().DebugString()));

    Tensor y = x;  // This creates an alias intentionally.
    // Skip processing if tensors are empty.
    if (x.NumElements() > 0 && v.NumElements() > 0) {
      OP_REQUIRES_OK(ctx, DoCompute(ctx, i, v, &y));
    }
    ctx->set_output(0, y);
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& i,
                           const Tensor& v, Tensor* y) = 0;
};

}  // end namespace

namespace functor {

template <typename T>
void DoInplaceOp(const CPUDevice& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_9(mht_9_v, 449, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoInplaceOp");

  auto Ti = i.flat<int32>();
  auto Tv = v.flat_outer_dims<T>();
  auto Ty = y->flat_outer_dims<T>();
  auto nrows = Ty.dimension(0);
  for (int64_t j = 0; j < Ti.size(); ++j) {
    auto r = (Ti(j) % nrows + nrows) % nrows;  // Guard index range.
    switch (op) {
      case I_UPDATE:
        Ty.template chip<0>(r).device(d) = Tv.template chip<0>(j);
        break;
      case I_ADD:
        Ty.template chip<0>(r).device(d) += Tv.template chip<0>(j);
        break;
      case I_SUB:
        Ty.template chip<0>(r).device(d) -= Tv.template chip<0>(j);
        break;
    }
  }
}

// String type only supports inplace update.
void DoInplaceStringUpdateOp(const CPUDevice& d, const Tensor& i,
                             const Tensor& v, Tensor* y) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_10(mht_10_v, 475, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoInplaceStringUpdateOp");

  auto Ti = i.flat<int32>();
  auto Tv = v.flat_outer_dims<tstring>();
  auto Ty = y->flat_outer_dims<tstring>();
  auto nrows = Ty.dimension(0);
  for (int64_t j = 0; j < Ti.size(); ++j) {
    auto r = (Ti(j) % nrows + nrows) % nrows;  // Guard index range.
    Ty.template chip<0>(r).device(d) = Tv.template chip<0>(j);
  }
}

template <>
Status DoInplace(const CPUDevice& device, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_11(mht_11_v, 491, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoInplace");

  CHECK_EQ(v.dtype(), y->dtype());
  if (op == I_UPDATE) {
    if (v.dtype() == DT_STRING) {
      DoInplaceStringUpdateOp(device, i, v, y);
      return Status::OK();
    } else if (v.dtype() == DT_BOOL) {
      DoInplaceOp<bool>(device, op, i, v, y);
      return Status::OK();
    }
  }
  switch (v.dtype()) {
#define CASE(type)                          \
  case DataTypeToEnum<type>::value:         \
    DoInplaceOp<type>(device, op, i, v, y); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(v.dtype()));
  }
  return Status::OK();
}

}  // end namespace functor

namespace {
template <typename Device, functor::InplaceOpType op>
class InplaceOp : public InplaceOpBase {
 public:
  explicit InplaceOp(OpKernelConstruction* ctx) : InplaceOpBase(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_12(mht_12_v, 525, "", "./tensorflow/core/kernels/inplace_ops.cc", "InplaceOp");
}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& i, const Tensor& v,
                   Tensor* y) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_13(mht_13_v, 532, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoCompute");

    const auto& d = ctx->eigen_device<Device>();
    return ::tensorflow::functor::DoInplace(d, op, i, v, y);
  }
};

class CopyOpBase : public OpKernel {
 public:
  explicit CopyOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_14(mht_14_v, 543, "", "./tensorflow/core/kernels/inplace_ops.cc", "CopyOpBase");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_15(mht_15_v, 548, "", "./tensorflow/core/kernels/inplace_ops.cc", "Compute");

    auto x = ctx->input(0);
    Tensor* y;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, DoCompute(ctx, x, y));
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& x,
                           Tensor* y) = 0;
};

template <typename Device>
class CopyOp : public CopyOpBase {
 public:
  explicit CopyOp(OpKernelConstruction* ctx) : CopyOpBase(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_16(mht_16_v, 566, "", "./tensorflow/core/kernels/inplace_ops.cc", "CopyOp");
}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& x, Tensor* y) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_17(mht_17_v, 572, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoCompute");

    const auto& d = ctx->eigen_device<Device>();
    return ::tensorflow::functor::DoCopy(d, x, y);
  }
};

}  // end namespace

namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <>
Status DoCopy(const CPUDevice& device, const Tensor& x, Tensor* y) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_18(mht_18_v, 588, "", "./tensorflow/core/kernels/inplace_ops.cc", "DoCopy");

  CHECK_EQ(x.dtype(), y->dtype());
  switch (x.dtype()) {
#define CASE(type)                                   \
  case DataTypeToEnum<type>::value:                  \
    y->flat<type>().device(device) = x.flat<type>(); \
    break;

    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_bool(CASE);
    TF_CALL_tstring(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(x.dtype()));
  }
  return Status::OK();
}

}  // end namespace functor

namespace {
template <typename Device, typename T>
class EmptyOp : public OpKernel {
 public:
  explicit EmptyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_19(mht_19_v, 616, "", "./tensorflow/core/kernels/inplace_ops.cc", "EmptyOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("init", &init_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_opsDTcc mht_20(mht_20_v, 623, "", "./tensorflow/core/kernels/inplace_ops.cc", "Compute");

    const Tensor& shape = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape.shape()),
        errors::InvalidArgument("shape must be a vector of int32, got shape ",
                                shape.shape().DebugString()));
    auto dims = shape.flat<int32>();
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            reinterpret_cast<const int32*>(dims.data()),
                            dims.size(), &out_shape));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (init_) {
      functor::SetZeroFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                           out->flat<T>());
    }
  }

 private:
  bool init_;
};

REGISTER_KERNEL_BUILDER(Name("InplaceUpdate").Device(DEVICE_CPU),
                        InplaceOp<CPUDevice, functor::I_UPDATE>);
REGISTER_KERNEL_BUILDER(Name("InplaceAdd").Device(DEVICE_CPU),
                        InplaceOp<CPUDevice, functor::I_ADD>);
REGISTER_KERNEL_BUILDER(Name("InplaceSub").Device(DEVICE_CPU),
                        InplaceOp<CPUDevice, functor::I_SUB>);
REGISTER_KERNEL_BUILDER(Name("DeepCopy").Device(DEVICE_CPU), CopyOp<CPUDevice>);

#define REGISTER_EMPTY(type, dev)                             \
  REGISTER_KERNEL_BUILDER(Name("Empty")                       \
                              .Device(DEVICE_##dev)           \
                              .HostMemory("shape")            \
                              .TypeConstraint<type>("dtype"), \
                          EmptyOp<dev##Device, type>)

REGISTER_EMPTY(float, CPU)
REGISTER_EMPTY(bfloat16, CPU)
REGISTER_EMPTY(double, CPU)
REGISTER_EMPTY(Eigen::half, CPU)
REGISTER_EMPTY(tstring, CPU)
REGISTER_EMPTY(int32, CPU)
REGISTER_EMPTY(int64_t, CPU)
REGISTER_EMPTY(bool, CPU)
REGISTER_EMPTY(uint8, CPU)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      InplaceOp<GPUDevice, functor::I_UPDATE>);                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceAdd").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      InplaceOp<GPUDevice, functor::I_ADD>);                              \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceSub").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      InplaceOp<GPUDevice, functor::I_SUB>);                              \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DeepCopy").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),      \
      CopyOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(
    Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<bool>("T"),
    InplaceOp<GPUDevice, functor::I_UPDATE>);
REGISTER(float);
REGISTER(double);
REGISTER(Eigen::half);
REGISTER(int64_t);

REGISTER_EMPTY(float, GPU);
REGISTER_EMPTY(double, GPU);
REGISTER_EMPTY(Eigen::half, GPU);
REGISTER_EMPTY(int64_t, GPU);
REGISTER_EMPTY(int32, GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER_KERNEL_BUILDER(Name("InplaceUpdate")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("i")
                            .HostMemory("v")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_UPDATE>);
REGISTER_KERNEL_BUILDER(Name("InplaceAdd")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("i")
                            .HostMemory("v")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_ADD>);
REGISTER_KERNEL_BUILDER(Name("InplaceSub")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("i")
                            .HostMemory("v")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_SUB>);

REGISTER_KERNEL_BUILDER(Name("DeepCopy")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        CopyOp<CPUDevice>);

}  // end namespace
}  // end namespace tensorflow
