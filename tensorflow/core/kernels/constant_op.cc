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
class MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/kernels/constant_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/lib/scoped_memory_debug_annotation.h"

namespace tensorflow {

namespace {

NodeDef StripTensorDataFromNodeDef(OpKernelConstruction* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/constant_op.cc", "StripTensorDataFromNodeDef");

  const NodeDef& original = ctx->def();
  if (std::is_base_of<protobuf::Message, NodeDef>()) {
    DCHECK_EQ(reinterpret_cast<const protobuf::Message*>(&original)
                  ->GetDescriptor()
                  ->field_count(),
              7)
        << "The NodeDef format has changed, and the attr-stripping code may "
           "need to be updated.";
  }
  NodeDef ret;
  ret.set_name(original.name());
  ret.set_op(original.op());
  ret.set_device(original.device());
  // Strip the "value" attr from the returned NodeDef.
  // NOTE(mrry): The present implementation of `OpKernel::OpKernel()` only uses
  // attrs that affect the cardinality of list-typed inputs and outputs, so it
  // is safe to drop other attrs from the NodeDef.
  AddNodeAttr("dtype", ctx->output_type(0), &ret);
  MergeDebugInfo(original, &ret);
  if (original.has_experimental_type()) {
    *ret.mutable_experimental_type() = original.experimental_type();
  }
  return ret;
}

}  // namespace

ConstantOp::ConstantOp(OpKernelConstruction* ctx)
    : OpKernel(ctx, StripTensorDataFromNodeDef(ctx), false),
      tensor_(ctx->output_type(0)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/kernels/constant_op.cc", "ConstantOp::ConstantOp");

  const TensorProto* proto = nullptr;
  profiler::ScopedMemoryDebugAnnotation op_annotation(name_view().data());
  OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
  OP_REQUIRES_OK(ctx, ctx->device()->MakeTensorFromProto(
                          *proto, AllocatorAttributes(), &tensor_));
  OP_REQUIRES(
      ctx, ctx->output_type(0) == tensor_.dtype(),
      errors::InvalidArgument("Type mismatch between value (",
                              DataTypeString(tensor_.dtype()), ") and dtype (",
                              DataTypeString(ctx->output_type(0)), ")"));
}

void ConstantOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_2(mht_2_v, 265, "", "./tensorflow/core/kernels/constant_op.cc", "ConstantOp::Compute");

  ctx->set_output(0, tensor_);
  if (TF_PREDICT_FALSE(ctx->track_allocations())) {
    ctx->record_persistent_memory_allocation(tensor_.AllocatedBytes());
  }
}

ConstantOp::~ConstantOp() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/kernels/constant_op.cc", "ConstantOp::~ConstantOp");
}

REGISTER_KERNEL_BUILDER(Name("Const").Device(DEVICE_CPU), ConstantOp);
REGISTER_KERNEL_BUILDER(Name("Const").Device(DEVICE_TPU_SYSTEM), ConstantOp);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_KERNEL(D, TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Const").Device(DEVICE_##D).TypeConstraint<TYPE>("dtype"), \
      ConstantOp);
REGISTER_KERNEL(GPU, Eigen::half);
REGISTER_KERNEL(GPU, bfloat16);
REGISTER_KERNEL(GPU, float);
REGISTER_KERNEL(GPU, double);
REGISTER_KERNEL(GPU, uint8);
REGISTER_KERNEL(GPU, int8);
REGISTER_KERNEL(GPU, qint8);
REGISTER_KERNEL(GPU, uint16);
REGISTER_KERNEL(GPU, int16);
REGISTER_KERNEL(GPU, qint16);
REGISTER_KERNEL(GPU, quint16);
REGISTER_KERNEL(GPU, uint32);
REGISTER_KERNEL(GPU, qint32);
REGISTER_KERNEL(GPU, int64_t);
REGISTER_KERNEL(GPU, uint64);
REGISTER_KERNEL(GPU, complex64);
REGISTER_KERNEL(GPU, complex128);
REGISTER_KERNEL(GPU, bool);
REGISTER_KERNEL(GPU, Variant);
#undef REGISTER_KERNEL
#endif

#define REGISTER_DEFAULT_KERNEL(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Const").Device(DEVICE_DEFAULT).TypeConstraint<TYPE>("dtype"), \
      ConstantOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_DEFAULT_KERNEL);
TF_CALL_QUANTIZED_TYPES(REGISTER_DEFAULT_KERNEL);
TF_CALL_qint16(REGISTER_DEFAULT_KERNEL);
TF_CALL_quint16(REGISTER_DEFAULT_KERNEL);
TF_CALL_bool(REGISTER_DEFAULT_KERNEL);
TF_CALL_variant(REGISTER_DEFAULT_KERNEL);
#undef REGISTER_DEFAULT_KERNEL

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Index>
class FillOp : public OpKernel {
 public:
  explicit FillOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_4(mht_4_v, 329, "", "./tensorflow/core/kernels/constant_op.cc", "FillOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_5(mht_5_v, 334, "", "./tensorflow/core/kernels/constant_op.cc", "Compute");

    const Tensor& Tdims = context->input(0);
    OP_REQUIRES(
        context,
        // TODO(rmlarsen): Disallow legacy use of scalars to represent shape.
        (TensorShapeUtils::IsVector(Tdims.shape()) ||
         TensorShapeUtils::IsScalar(Tdims.shape())),
        errors::InvalidArgument("dims must represent a vector, got shape ",
                                Tdims.shape().DebugString()));
    const Tensor& Tvalue = context->input(1);
    OP_REQUIRES(
        context,
        // TODO(rmlarsen): Disallow legacy use of length-1 vector to represent
        // scalar.
        TensorShapeUtils::IsScalar(Tvalue.shape()) ||
            (TensorShapeUtils::IsVector(Tvalue.shape()) &&
             Tvalue.shape().dim_size(0) == 1),
        errors::InvalidArgument("value must represent a scalar, got shape ",
                                Tvalue.shape().DebugString()));
    auto dims = Tdims.flat<Index>();
    TensorShape shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                reinterpret_cast<const Index*>(dims.data()),
                                dims.size(), &shape));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &out));
    functor::FillFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), out->flat<T>(),
            Tvalue.scalar<T>());
  }
};

#define REGISTER_KERNEL(D, TYPE)                                     \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<TYPE>("T")             \
                              .TypeConstraint<int32>("index_type")   \
                              .HostMemory("dims"),                   \
                          FillOp<D##Device, TYPE, int32>);           \
  REGISTER_KERNEL_BUILDER(Name("Fill")                               \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<TYPE>("T")             \
                              .TypeConstraint<int64_t>("index_type") \
                              .HostMemory("dims"),                   \
                          FillOp<D##Device, TYPE, int64>);

#define REGISTER_CPU_KERNEL(TYPE) REGISTER_KERNEL(CPU, TYPE)
TF_CALL_ALL_TYPES(REGISTER_CPU_KERNEL);
// TODO(b/28917570): Add a test for this. Currently python 3 is not happy about
// the conversion from uint8 to quint8.
REGISTER_KERNEL(CPU, quint8);
REGISTER_KERNEL(CPU, quint16);
REGISTER_KERNEL(CPU, qint8);
REGISTER_KERNEL(CPU, qint16);
REGISTER_KERNEL(CPU, qint32);
#undef REGISTER_CPU_KERNEL

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
REGISTER_KERNEL(GPU, Eigen::half);
REGISTER_KERNEL(GPU, bfloat16);
REGISTER_KERNEL(GPU, float);
REGISTER_KERNEL(GPU, double);
REGISTER_KERNEL(GPU, complex64);
REGISTER_KERNEL(GPU, complex128);
REGISTER_KERNEL(GPU, uint8);
REGISTER_KERNEL(GPU, int8);
REGISTER_KERNEL(GPU, uint16);
REGISTER_KERNEL(GPU, int16);
REGISTER_KERNEL(GPU, int64_t);
REGISTER_KERNEL(GPU, bool);
// Currently we do not support filling strings on GPU

// A special DEVICE_DEFAULT kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Fill")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("index_type")
                            .HostMemory("dims")
                            .HostMemory("value")
                            .HostMemory("output"),
                        FillOp<CPUDevice, int32, int32>);
#endif

#undef REGISTER_KERNEL

template <typename Device, typename T>
class ZerosLikeOp : public OpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_6(mht_6_v, 428, "", "./tensorflow/core/kernels/constant_op.cc", "ZerosLikeOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_7(mht_7_v, 433, "", "./tensorflow/core/kernels/constant_op.cc", "Compute");

    const Tensor& input = ctx->input(0);
    const Device& d = ctx->eigen_device<Device>();
    if (std::is_same<T, Variant>::value) {
      OP_REQUIRES(
          ctx, input.dims() == 0,
          errors::InvalidArgument("ZerosLike non-scalar Tensor with "
                                  "dtype=DT_VARIANT is not supported."));
      const Variant& v = input.scalar<Variant>()();
      // DT_VARIANT tensors must be allocated on CPU since they wrap C++
      // objects which can not be efficiently represented in GPU memory.
      int numa_node = ctx->device()->NumaNode();
      Tensor out(cpu_allocator(numa_node), DT_VARIANT, TensorShape({}));
      Variant* out_v = &(out.scalar<Variant>()());
      OP_REQUIRES_OK(ctx, UnaryOpVariant<Device>(
                              ctx, ZEROS_LIKE_VARIANT_UNARY_OP, v, out_v));
      ctx->set_output(0, out);
    } else {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, input.shape(), &out));
      functor::SetZeroFunctor<Device, T> f;
      f(d, out->flat<T>());
    }
  }
};

#define REGISTER_KERNEL(type, dev)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ZerosLike").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      ZerosLikeOp<dev##Device, type>)

#define REGISTER_CPU(type) REGISTER_KERNEL(type, CPU)
TF_CALL_POD_STRING_TYPES(REGISTER_CPU);
REGISTER_CPU(Variant);
#undef REGISTER_CPU

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER_KERNEL(bool, GPU);
REGISTER_KERNEL(Eigen::half, GPU);
REGISTER_KERNEL(float, GPU);
REGISTER_KERNEL(double, GPU);
REGISTER_KERNEL(int64_t, GPU);
#endif

REGISTER_KERNEL(bfloat16, GPU);
REGISTER_KERNEL(complex64, GPU);
REGISTER_KERNEL(complex128, GPU);
REGISTER_KERNEL(Variant, GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNEL

REGISTER_KERNEL_BUILDER(Name("ZerosLike")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("y"),
                        ZerosLikeOp<CPUDevice, int32>);

template <typename Device, typename T>
class OnesLikeOp : public OpKernel {
 public:
  explicit OnesLikeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_8(mht_8_v, 499, "", "./tensorflow/core/kernels/constant_op.cc", "OnesLikeOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_9(mht_9_v, 504, "", "./tensorflow/core/kernels/constant_op.cc", "Compute");

    const Tensor& input = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {0}, 0, input.shape(), &out));
    functor::SetOneFunctor<Device, T> f;
    f(ctx->eigen_device<Device>(), out->flat<T>());
  }
};

#define REGISTER_KERNEL(type, dev)                                     \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("OnesLike").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      OnesLikeOp<dev##Device, type>)

#define REGISTER_CPU(type) REGISTER_KERNEL(type, CPU)
TF_CALL_POD_TYPES(REGISTER_CPU);
#undef REGISTER_CPU

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER_KERNEL(bool, GPU);
REGISTER_KERNEL(Eigen::half, GPU);
REGISTER_KERNEL(float, GPU);
REGISTER_KERNEL(double, GPU);
REGISTER_KERNEL(int64_t, GPU);
#endif
REGISTER_KERNEL(bfloat16, GPU);
REGISTER_KERNEL(complex64, GPU);
REGISTER_KERNEL(complex128, GPU);
REGISTER_KERNEL_BUILDER(Name("OnesLike")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("y"),
                        OnesLikeOp<CPUDevice, int32>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_KERNEL

PlaceholderOp::PlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
}

void PlaceholderOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_opDTcc mht_10(mht_10_v, 551, "", "./tensorflow/core/kernels/constant_op.cc", "PlaceholderOp::Compute");

  if (expected_shape_.dims() > 0) {
    OP_REQUIRES(ctx, false,
                errors::InvalidArgument(
                    "You must feed a value for placeholder tensor '", name(),
                    "' with dtype ", DataTypeString(output_type(0)),
                    " and shape ", expected_shape_.DebugString()));
  } else {
    OP_REQUIRES(ctx, false,
                errors::InvalidArgument(
                    "You must feed a value for placeholder tensor '", name(),
                    "' with dtype ", DataTypeString(output_type(0))));
  }
}

REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE_CPU), PlaceholderOp);
REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE_CPU),
                        PlaceholderOp);
// The following GPU/Default kernel registration is used to address the
// situation that a placeholder is added in a GPU device context and soft
// placement is false. Since a placeholder should never be executed, adding
// these GPU kernels has no effect on graph execution.
REGISTER_KERNEL_BUILDER(Name("Placeholder").Device(DEVICE_DEFAULT),
                        PlaceholderOp);
REGISTER_KERNEL_BUILDER(Name("PlaceholderV2").Device(DEVICE_DEFAULT),
                        PlaceholderOp);
}  // namespace tensorflow
