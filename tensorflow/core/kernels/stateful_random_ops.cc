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
class MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/rng_alg.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/random_op_cpu.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace functor {

template <typename Distribution>
struct UpdateVariableAndFill_Philox<CPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const CPUDevice& device,
                  Distribution dist, UpdateVariableAndFill_Philox_Arg* arg,
                  typename Distribution::ResultElementType* output_data)
      TF_UNLOCK_FUNCTION() {
    int64_t output_size = arg->output_size;
    int64_t alg_tag_skip = arg->alg_tag_skip;
    ScopedUnlockUnrefVar* state_var_guard = arg->state_var_guard;
    Tensor* state_tensor = arg->state_tensor;

    auto state_tensor_flat = state_tensor->flat<StateElementType>();
    auto state_data = state_tensor_flat.data();
    // Delegates to PhiloxRandom to do the actual increasing.
    auto philox = GetPhiloxRandomFromMem(state_data + alg_tag_skip);
    UpdateMemWithPhiloxRandom(philox, output_size, state_data + alg_tag_skip);
    // No longer needs the lock.
    state_var_guard->Release();
    functor::FillPhiloxRandom<CPUDevice, Distribution>()(
        ctx, device, /*key=*/nullptr, /*counter=*/nullptr, philox, output_data,
        output_size, dist);
  }
};

}  // end namespace functor

Status CheckState(const Tensor& state) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_0(mht_0_v, 225, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "CheckState");

  if (state.dtype() != STATE_ELEMENT_DTYPE) {
    return errors::InvalidArgument("dtype of RNG state variable must be ",
                                   DataTypeString(STATE_ELEMENT_DTYPE),
                                   ", not ", DataTypeString(state.dtype()));
  }
  if (state.dims() != 1) {
    return errors::InvalidArgument(
        "RNG state must have one and only one dimension, not ", state.dims());
  }
  return Status::OK();
}

Status CheckPhiloxState(const Tensor& state, int64_t alg_tag_skip = 0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "CheckPhiloxState");

  static_assert(std::is_same<StateElementType, int64_t>::value,
                "StateElementType must be int64");
  static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                "PhiloxRandom::ResultElementType must be uint32");
  auto min_size = alg_tag_skip + PHILOX_MIN_STATE_SIZE;
  if (state.NumElements() < min_size) {
    return errors::InvalidArgument(
        "For the Philox algorithm, the size of state"
        " must be at least ",
        min_size, "; got ", state.NumElements());
  }
  return Status::OK();
}

template <typename Device, typename Distribution>
Status UpdateVariableAndFill(
    OpKernelContext* ctx, Distribution dist, int state_input_idx,
    bool read_alg_from_state, Algorithm alg, int64_t output_size,
    typename Distribution::ResultElementType* output_data) {
  Var* var = nullptr;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, state_input_idx), &var));
  // Use `ScopedUnlockUnrefVar` here instead of `mutex_lock` and `ScopedUnref`
  // because the former supports early releasing which is needed by
  // `UpdateVariableAndFill_Philox<CPU>` to avoid holding the lock while
  // filling.
  ScopedUnlockUnrefVar state_var_guard(var);
  Tensor* var_tensor = var->tensor();
  TF_RETURN_IF_ERROR(CheckState(*var_tensor));
  auto var_tensor_flat = var_tensor->flat<StateElementType>();
  int64_t alg_tag_skip = 0;
  if (read_alg_from_state) {
    alg_tag_skip = 1;
    if (var_tensor_flat.size() < 1) {
      return errors::InvalidArgument("Size of tensor must be at least 1");
    }
    alg = Algorithm(var_tensor_flat(0));
  }
  if (alg == RNG_ALG_PHILOX) {
    TF_RETURN_IF_ERROR(CheckPhiloxState(*var_tensor, alg_tag_skip));
    TF_RETURN_IF_ERROR(PrepareToUpdateVariable<Device, StateElementType>(
        ctx, var_tensor, var->copy_on_read_mode.load()));

    UpdateVariableAndFill_Philox_Arg arg;
    arg.output_size = output_size;
    arg.alg_tag_skip = alg_tag_skip;
    arg.state_var_guard = &state_var_guard;
    arg.state_tensor = var_tensor;
    functor::UpdateVariableAndFill_Philox<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), dist, &arg, output_data);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unsupported algorithm id: ", alg);
  }
}

// Precondition: input(0) is an existing resource.
template <typename Device, class Distribution>
void StatefulRandomCompute(OpKernelContext* ctx, Distribution dist,
                           int state_input_idx, int shape_input_idx,
                           bool read_alg_from_state, Algorithm alg) {
  using T = typename Distribution::ResultElementType;
  const Tensor& shape_t = ctx->input(shape_input_idx);
  TensorShape shape;
  OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  auto output_flat = output->flat<T>();
  OP_REQUIRES_OK(ctx, UpdateVariableAndFill<Device>(
                          ctx, dist, state_input_idx, read_alg_from_state, alg,
                          output_flat.size(), output_flat.data()));
}

template <typename Device, class Distribution>
class StatefulRandomOp : public OpKernel {
 public:
  explicit StatefulRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_2(mht_2_v, 321, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "StatefulRandomOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_3(mht_3_v, 326, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "Compute");

    StatefulRandomCompute<Device>(ctx, Distribution(), 0, 1, true,
                                  RNG_ALG_PHILOX /*dummy*/);
  }
};

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_4(mht_4_v, 336, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "GetScalar");

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

template <typename AlgEnumType>
Status GetAlg(OpKernelContext* ctx, int input_idx, Algorithm* alg) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_5(mht_5_v, 356, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "GetAlg");

  AlgEnumType alg_id;
  TF_RETURN_IF_ERROR(GetScalar(ctx->input(input_idx), input_idx, &alg_id));
  *alg = Algorithm(alg_id);
  return Status::OK();
}

template <typename Device, class Distribution>
class StatefulRandomOpV2 : public OpKernel {
 public:
  explicit StatefulRandomOpV2(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_6(mht_6_v, 369, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "StatefulRandomOpV2");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_7(mht_7_v, 374, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "Compute");

    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<int64_t>(ctx, 1, &alg));
    StatefulRandomCompute<Device>(ctx, Distribution(), /*state_input_idx=*/0,
                                  /*shape_input_idx=*/2,
                                  /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, class IntType>
class StatefulUniformIntOp : public OpKernel {
 public:
  explicit StatefulUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_8(mht_8_v, 389, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "StatefulUniformIntOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_9(mht_9_v, 394, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "Compute");

    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<int64_t>(ctx, 1, &alg));
    const Tensor& minval = ctx->input(3);
    const Tensor& maxval = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval.  This check intentionally happens after the
    // early exit for empty output.  Zero impossible things are fine.
    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

    StatefulRandomCompute<Device>(ctx, dist, /*state_input_idx=*/0,
                                  /*shape_input_idx=*/2,
                                  /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, class IntType>
class StatefulUniformFullIntOp : public OpKernel {
 public:
  explicit StatefulUniformFullIntOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_10(mht_10_v, 432, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "StatefulUniformFullIntOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_11(mht_11_v, 437, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "Compute");

    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<int64_t>(ctx, 1, &alg));
    StatefulRandomCompute<Device>(
        ctx,
        random::UniformFullIntDistribution<random::PhiloxRandom, IntType>(),
        /*state_input_idx=*/0, /*shape_input_idx=*/2,
        /*read_alg_from_state=*/false, alg);
  }
};

namespace functor {

template <>
struct RngSkip_Philox<CPUDevice> {
  void operator()(const CPUDevice& device, const StateElementType* in_data,
                  uint64 delta, StateElementType* out_data) {
    // Delegates to PhiloxRandom to do the actual increasing.
    auto counter = GetCounterFromMem(reinterpret_cast<const uint64*>(in_data));
    UpdateCounterMemWithPhiloxRandom(counter, delta, out_data);
  }
};

}  // end namespace functor

template <typename Device, typename AlgEnumType = int64_t,
          typename DeltaType = int64_t, bool read_old_value = false>
class RngSkipOp : public OpKernel {
 public:
  explicit RngSkipOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_12(mht_12_v, 469, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "RngSkipOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_13(mht_13_v, 474, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "Compute");

    auto state_input_idx = 0;
    auto alg_input_idx = 1;
    auto delta_input_idx = 2;
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetAlg<AlgEnumType>(ctx, alg_input_idx, &alg));
    DeltaType delta_;
    OP_REQUIRES_OK(
        ctx, GetScalar(ctx->input(delta_input_idx), delta_input_idx, &delta_));
    uint64 delta = static_cast<uint64>(delta_);
    Var* var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, state_input_idx), &var));
    ScopedUnlockUnrefVar state_var_guard(var);
    Tensor* var_tensor = var->tensor();
    OP_REQUIRES_OK(ctx, CheckState(*var_tensor));
    using T = StateElementType;
    OP_REQUIRES_OK(ctx, PrepareToUpdateVariable<Device, T>(
                            ctx, var_tensor, var->copy_on_read_mode.load()));
    if (read_old_value) {
      Tensor* output;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output(0, {RNG_MAX_COUNTER_SIZE + RNG_KEY_SIZE},
                                    &output));
      auto output_flat = output->flat<T>();
      if (RNG_MAX_COUNTER_SIZE > GetCounterSize(alg)) {
        functor::SetZeroFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                             output_flat);
      }
      functor::DenseUpdate<Device, T, ASSIGN>()(
          ctx->eigen_device<Device>(), output_flat,
          const_cast<const Tensor*>(var_tensor)->flat<T>());
    }
    if (alg == RNG_ALG_PHILOX) {
      OP_REQUIRES_OK(ctx, CheckPhiloxState(*var_tensor));
      // var_tensor layout is counter+key, so var_tensor data is also counter
      // data.
      auto counter_data = var_tensor->flat<T>().data();
      functor::RngSkip_Philox<Device>()(ctx->eigen_device<Device>(),
                                        counter_data, delta, counter_data);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

template <typename T>
class NonDeterministicIntsOp : public OpKernel {
 public:
  explicit NonDeterministicIntsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_14(mht_14_v, 527, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "NonDeterministicIntsOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_opsDTcc mht_15(mht_15_v, 534, "", "./tensorflow/core/kernels/stateful_random_ops.cc", "Compute");

    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, tensor::MakeShape(shape_t, &shape));
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    switch (dtype_) {
      case DT_INT32:
      case DT_UINT32:
      case DT_INT64:
      case DT_UINT64: {
        auto output_flat = output->flat<T>();
        auto data = output_flat.data();
        for (int64_t i = 0; i < output_flat.size(); ++i) {
          data[i] = static_cast<T>(random::New64());
        }
        break;
      }
      default:
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("Unsupported dtype: ",
                                            DataTypeString(dtype_)));
    }
  }

 private:
  DataType dtype_;
};

// So far the 'Distribution' type parameter is only used when the algorithm is
// philox, so 'NormalDistribution<PhiloxRandom, ...>' is fine for now.
#define REGISTER_FloatOps(DEVICE, TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StatefulStandardNormalV2")                                       \
          .Device(DEVICE_##DEVICE)                                           \
          .HostMemory("resource")                                            \
          .HostMemory("algorithm")                                           \
          .HostMemory("shape")                                               \
          .TypeConstraint<TYPE>("dtype"),                                    \
      StatefulRandomOpV2<DEVICE##Device,                                     \
                         random::NormalDistribution<PhiloxRandom, TYPE> >);  \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StatefulUniform")                                                \
          .Device(DEVICE_##DEVICE)                                           \
          .HostMemory("resource")                                            \
          .HostMemory("algorithm")                                           \
          .HostMemory("shape")                                               \
          .TypeConstraint<TYPE>("dtype"),                                    \
      StatefulRandomOpV2<DEVICE##Device,                                     \
                         random::UniformDistribution<PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StatefulTruncatedNormal")                                        \
          .Device(DEVICE_##DEVICE)                                           \
          .HostMemory("resource")                                            \
          .HostMemory("algorithm")                                           \
          .HostMemory("shape")                                               \
          .TypeConstraint<TYPE>("dtype"),                                    \
      StatefulRandomOpV2<                                                    \
          DEVICE##Device,                                                    \
          random::TruncatedNormalDistribution<                               \
              random::SingleSampleAdapter<PhiloxRandom>, TYPE> >);

// CPU also has the deprecated 'StatefulStandardNormal' op for backward
// compatibility.
#define REGISTER_FloatOps_CPU(TYPE)                     \
  REGISTER_FloatOps(CPU, TYPE) REGISTER_KERNEL_BUILDER( \
      Name("StatefulStandardNormal")                    \
          .Device(DEVICE_CPU)                           \
          .HostMemory("resource")                       \
          .HostMemory("shape")                          \
          .TypeConstraint<TYPE>("dtype"),               \
      StatefulRandomOp<CPUDevice,                       \
                       random::NormalDistribution<PhiloxRandom, TYPE> >);

#define REGISTER_FloatOps_GPU(TYPE) REGISTER_FloatOps(GPU, TYPE)

TF_CALL_half(REGISTER_FloatOps_CPU);
TF_CALL_bfloat16(REGISTER_FloatOps_CPU);
TF_CALL_float(REGISTER_FloatOps_CPU);
TF_CALL_double(REGISTER_FloatOps_CPU);

#define REGISTER_StatefulUniformInt(DEVICE, TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("StatefulUniformInt")          \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .HostMemory("minval")           \
                              .HostMemory("maxval")           \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatefulUniformIntOp<DEVICE##Device, TYPE>);

#define REGISTER_StatefulUniformInt_CPU(TYPE) \
  REGISTER_StatefulUniformInt(CPU, TYPE)
#define REGISTER_StatefulUniformInt_GPU(TYPE) \
  REGISTER_StatefulUniformInt(GPU, TYPE)

TF_CALL_int32(REGISTER_StatefulUniformInt_CPU);
TF_CALL_int64(REGISTER_StatefulUniformInt_CPU);

#define REGISTER_StatefulUniformFullInt(DEVICE, TYPE)         \
  REGISTER_KERNEL_BUILDER(Name("StatefulUniformFullInt")      \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatefulUniformFullIntOp<DEVICE##Device, TYPE>);

#define REGISTER_StatefulUniformFullInt_CPU(TYPE) \
  REGISTER_StatefulUniformFullInt(CPU, TYPE)
#define REGISTER_StatefulUniformFullInt_GPU(TYPE) \
  REGISTER_StatefulUniformFullInt(GPU, TYPE)

TF_CALL_int32(REGISTER_StatefulUniformFullInt_CPU);
TF_CALL_int64(REGISTER_StatefulUniformFullInt_CPU);
TF_CALL_uint32(REGISTER_StatefulUniformFullInt_CPU);
TF_CALL_uint64(REGISTER_StatefulUniformFullInt_CPU);

// TODO(wangpeng): Remove `HostMemory("delta")` for RngReadAndSkip
#define REGISTER_RngSkip(DEVICE)                       \
  REGISTER_KERNEL_BUILDER(Name("RngSkip")              \
                              .Device(DEVICE_##DEVICE) \
                              .HostMemory("resource")  \
                              .HostMemory("algorithm") \
                              .HostMemory("delta"),    \
                          RngSkipOp<DEVICE##Device>);  \
  REGISTER_KERNEL_BUILDER(Name("RngReadAndSkip")       \
                              .Device(DEVICE_##DEVICE) \
                              .HostMemory("resource")  \
                              .HostMemory("alg")       \
                              .HostMemory("delta"),    \
                          RngSkipOp<DEVICE##Device, int32, uint64, true>);

REGISTER_RngSkip(CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_half(REGISTER_FloatOps_GPU);
TF_CALL_float(REGISTER_FloatOps_GPU);
TF_CALL_double(REGISTER_FloatOps_GPU);
TF_CALL_int32(REGISTER_StatefulUniformInt_GPU);
TF_CALL_int64(REGISTER_StatefulUniformInt_GPU);
TF_CALL_int32(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_int64(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_uint32(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_uint64(REGISTER_StatefulUniformFullInt_GPU);
REGISTER_RngSkip(GPU);

#endif  // GOOGLE_CUDA

#undef REGISTER_StatefulUniformFullInt_GPU
#undef REGISTER_StatefulUniformFullInt_CPU
#undef REGISTER_StatefulUniformFullInt
#undef REGISTER_StatefulUniformInt_GPU
#undef REGISTER_StatefulUniformInt_CPU
#undef REGISTER_StatefulUniformInt
#undef REGISTER_FloatOps_GPU
#undef REGISTER_FloatOps_CPU
#undef REGISTER_FloatOps

#define REGISTER_NonDeterministicInts(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("NonDeterministicInts")        \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          NonDeterministicIntsOp<TYPE>);

TF_CALL_int32(REGISTER_NonDeterministicInts);
TF_CALL_uint32(REGISTER_NonDeterministicInts);
TF_CALL_int64(REGISTER_NonDeterministicInts);
TF_CALL_uint64(REGISTER_NonDeterministicInts);

#undef REGISTER_NonDeterministicInts

// TODO(wangpeng): Add RNG ops for other distributions.

}  // end namespace tensorflow
