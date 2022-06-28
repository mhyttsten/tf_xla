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
class MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc() {
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

#include "tensorflow/core/kernels/stateless_random_ops_v2.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/random_ops_util.h"
#include "tensorflow/core/kernels/random_poisson_op.h"
#include "tensorflow/core/kernels/stateless_random_ops.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

#if EIGEN_COMP_GNUC && __cplusplus > 199711L
#define DISABLE_FLOAT_EQUALITY_WARNING \
  _Pragma("GCC diagnostic push")       \
      _Pragma("GCC diagnostic ignored \"-Wfloat-equal\"")
#define ENABLE_FLOAT_EQUALITY_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_FLOAT_EQUALITY_WARNING
#define ENABLE_FLOAT_EQUALITY_WARNING
#endif

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "GetScalar");

  auto dtype = DataTypeToEnum<T>::v();
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("input ", std::to_string(input_idx),
                                   " (0-based) must have shape [], not ",
                                   tensor.shape().DebugString());
  }
  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument("dtype of input ", std::to_string(input_idx),
                                   " (0-based) must be ", DataTypeString(dtype),
                                   ", not ", DataTypeString(tensor.dtype()));
  }
  *result = tensor.flat<T>()(0);
  return Status::OK();
}

class StatelessRandomOpBase : public OpKernel {
 public:
  explicit StatelessRandomOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "StatelessRandomOpBase");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Compute");

    // Sanitize input
    const Tensor& shape_t = ctx->input(0);
    const Tensor& key_t = ctx->input(1);
    const Tensor& counter_t = ctx->input(2);
    const int alg_input_idx = 3;
    const Tensor& alg_t = ctx->input(alg_input_idx);

    int alg_id;
    OP_REQUIRES_OK(ctx, GetScalar(alg_t, alg_input_idx, &alg_id));
    Algorithm alg = Algorithm(alg_id);
    if (alg == RNG_ALG_AUTO_SELECT) {
      alg = RNG_ALG_PHILOX;
    }

    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
    OP_REQUIRES_OK(ctx,
                   CheckKeyCounterShape(alg, key_t.shape(), counter_t.shape()));

    // Allocate output
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) {
      return;
    }

    // Fill in the random numbers
    Fill(ctx, alg, key_t, counter_t, output);
  }

  // The part of Compute that depends on device, type, and distribution
  virtual void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
                    const Tensor& counter, Tensor* output) = 0;
};

template <typename Device, typename Distribution>
class StatelessRandomOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_3(mht_3_v, 291, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Fill");

    typedef typename Distribution::ResultElementType T;
    auto flat = output->flat<T>();
    if (alg == RNG_ALG_PHILOX) {
      // Reuse the compute kernels from the stateful random ops
      auto key_data = key.flat<uint64>().data();
      auto counter_data = counter.flat<uint64>().data();
      functor::FillPhiloxRandom<Device, Distribution>()(
          ctx, ctx->eigen_device<Device>(), key_data, counter_data,
          random::PhiloxRandom() /*dummy*/, flat.data(), flat.size(),
          Distribution());
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

template <typename Device, typename IntType>
class StatelessRandomUniformIntOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_4(mht_4_v, 318, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Fill");

    const Tensor& minval = ctx->input(4);
    const Tensor& maxval = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval.  Note that we'll never reach this point for
    // empty output.  Zero impossible things are fine.
    const auto lo = minval.scalar<IntType>()();
    const auto hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

    auto flat = output->flat<IntType>();
    if (alg == RNG_ALG_PHILOX) {
      // Reuse the compute kernels from the stateful random ops
      auto key_data = key.flat<uint64>().data();
      auto counter_data = counter.flat<uint64>().data();
      functor::FillPhiloxRandom<Device, Distribution>()(
          ctx, ctx->eigen_device<Device>(), key_data, counter_data,
          random::PhiloxRandom() /*dummy*/, flat.data(), flat.size(), dist);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

template <typename Device, typename IntType>
class StatelessRandomUniformFullIntOp : public StatelessRandomOpBase {
 public:
  using StatelessRandomOpBase::StatelessRandomOpBase;

  void Fill(OpKernelContext* ctx, Algorithm alg, const Tensor& key,
            const Tensor& counter, Tensor* output) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_5(mht_5_v, 365, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Fill");

    // Build distribution
    typedef random::UniformFullIntDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist;

    auto flat = output->flat<IntType>();
    if (alg == RNG_ALG_PHILOX) {
      // Reuse the compute kernels from the stateful random ops
      auto key_data = key.flat<uint64>().data();
      auto counter_data = counter.flat<uint64>().data();
      functor::FillPhiloxRandom<Device, Distribution>()(
          ctx, ctx->eigen_device<Device>(), key_data, counter_data,
          random::PhiloxRandom() /*dummy*/, flat.data(), flat.size(), dist);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

class GetKeyCounterAlgOp : public OpKernel {
 public:
  explicit GetKeyCounterAlgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_6(mht_6_v, 391, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "GetKeyCounterAlgOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_7(mht_7_v, 396, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Compute");

    const Tensor& seed_t = ctx->input(0);
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));
    // Allocate outputs
    Tensor* key_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({RNG_KEY_SIZE}), &key_output));
    Tensor* counter_output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({RNG_MAX_COUNTER_SIZE}),
                                        &counter_output));
    Tensor* alg_output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &alg_output));

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));
    WriteKeyToMem(key, key_output->flat<uint64>().data());
    WriteCounterToMem(counter, counter_output->flat<uint64>().data());
    alg_output->flat<int>()(0) = RNG_ALG_PHILOX;
  }
};

class GetKeyCounterOp : public OpKernel {
 public:
  explicit GetKeyCounterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_8(mht_8_v, 426, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "GetKeyCounterOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_9(mht_9_v, 431, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Compute");

    const Tensor& seed_t = ctx->input(0);
    OP_REQUIRES(ctx, seed_t.dims() == 1 && seed_t.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_t.shape().DebugString()));
    // Allocate outputs
    Tensor* key_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({RNG_KEY_SIZE}), &key_output));
    Tensor* counter_output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, TensorShape({RNG_MAX_COUNTER_SIZE}),
                                        &counter_output));

    random::PhiloxRandom::Key key;
    random::PhiloxRandom::ResultType counter;
    OP_REQUIRES_OK(ctx, GenerateKey(seed_t, &key, &counter));
    WriteKeyToMem(key, key_output->flat<uint64>().data());
    WriteCounterToMem(counter, counter_output->flat<uint64>().data());
  }
};

class GetAlgOp : public OpKernel {
 public:
  explicit GetAlgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_10(mht_10_v, 458, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "GetAlgOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateless_random_ops_v2DTcc mht_11(mht_11_v, 463, "", "./tensorflow/core/kernels/stateless_random_ops_v2.cc", "Compute");

    Tensor* alg_output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &alg_output));
    alg_output->flat<int>()(0) = RNG_ALG_PHILOX;
  }
};

#define REGISTER(DEVICE, TYPE)                                              \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessRandomUniformV2")                                      \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("alg")                                                \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<DEVICE##Device, random::UniformDistribution<        \
                                            random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessRandomNormalV2")                                       \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("alg")                                                \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<DEVICE##Device, random::NormalDistribution<         \
                                            random::PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("StatelessTruncatedNormalV2")                                    \
          .Device(DEVICE_##DEVICE)                                          \
          .HostMemory("shape")                                              \
          .HostMemory("alg")                                                \
          .TypeConstraint<TYPE>("dtype"),                                   \
      StatelessRandomOp<                                                    \
          DEVICE##Device,                                                   \
          random::TruncatedNormalDistribution<                              \
              random::SingleSampleAdapter<random::PhiloxRandom>, TYPE> >)

#define REGISTER_FULL_INT(DEVICE, TYPE)       \
  REGISTER_KERNEL_BUILDER(                    \
      Name("StatelessRandomUniformFullIntV2") \
          .Device(DEVICE_##DEVICE)            \
          .HostMemory("shape")                \
          .HostMemory("alg")                  \
          .TypeConstraint<TYPE>("dtype"),     \
      StatelessRandomUniformFullIntOp<DEVICE##Device, TYPE>)

#define REGISTER_INT(DEVICE, TYPE)                            \
  REGISTER_FULL_INT(DEVICE, TYPE);                            \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomUniformIntV2") \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("shape")            \
                              .HostMemory("alg")              \
                              .HostMemory("minval")           \
                              .HostMemory("maxval")           \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatelessRandomUniformIntOp<DEVICE##Device, TYPE>)

#define REGISTER_CPU(TYPE) REGISTER(CPU, TYPE)
#define REGISTER_GPU(TYPE) REGISTER(GPU, TYPE)
#define REGISTER_INT_CPU(TYPE) REGISTER_INT(CPU, TYPE)
#define REGISTER_INT_GPU(TYPE) REGISTER_INT(GPU, TYPE)
#define REGISTER_FULL_INT_CPU(TYPE) REGISTER_FULL_INT(CPU, TYPE)
#define REGISTER_FULL_INT_GPU(TYPE) REGISTER_FULL_INT(GPU, TYPE)

TF_CALL_half(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);
TF_CALL_int32(REGISTER_INT_CPU);
TF_CALL_int64(REGISTER_INT_CPU);
TF_CALL_uint32(REGISTER_FULL_INT_CPU);
TF_CALL_uint64(REGISTER_FULL_INT_CPU);

#define REGISTER_GET_KCA(DEVICE)                                               \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGetKeyCounterAlg")              \
                              .Device(DEVICE_##DEVICE)                         \
                              .HostMemory("seed")                              \
                              .HostMemory("key")                               \
                              .HostMemory("counter")                           \
                              .HostMemory("alg"),                              \
                          GetKeyCounterAlgOp)                                  \
  REGISTER_KERNEL_BUILDER(Name("StatelessRandomGetKeyCounter")                 \
                              .Device(DEVICE_##DEVICE)                         \
                              .HostMemory("seed")                              \
                              .HostMemory("key")                               \
                              .HostMemory("counter"),                          \
                          GetKeyCounterOp)                                     \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("StatelessRandomGetAlg").Device(DEVICE_##DEVICE).HostMemory("alg"), \
      GetAlgOp)

REGISTER_GET_KCA(CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_int32(REGISTER_INT_GPU);
TF_CALL_int64(REGISTER_INT_GPU);
TF_CALL_uint32(REGISTER_FULL_INT_GPU);
TF_CALL_uint64(REGISTER_FULL_INT_GPU);

REGISTER_GET_KCA(GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER
#undef REGISTER_INT
#undef REGISTER_CPU
#undef REGISTER_GPU
#undef REGISTER_INT_CPU
#undef REGISTER_INT_GPU
#undef REGISTER_FULL_INT_CPU
#undef REGISTER_FULL_INT_GPU

#undef REGISTER_GET_KCA

}  // namespace

}  // namespace tensorflow
