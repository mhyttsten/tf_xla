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

#ifndef TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_OP_H_
#define TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_OP_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh() {
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


#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conditional_accumulator_base.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

typedef std::function<void()> DoneCallback;

namespace tensorflow {

/**
 * Defines a ConditionalAccumulatorBaseOp, which constructs a
 * ConditionalAccumulatorBase (via sub-class's Creator) and returns its handle.
 */
class ConditionalAccumulatorBaseOp : public OpKernel {
 public:
  explicit ConditionalAccumulatorBaseOp(OpKernelConstruction* context)
      : OpKernel(context), accumulator_set_(false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ConditionalAccumulatorBaseOp");

    OP_REQUIRES_OK(context, context->allocate_temp(DT_STRING, TensorShape({2}),
                                                   &accumulator_));
    OP_REQUIRES_OK(context, context->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("reduction_type", &reduction_type_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_1(mht_1_v, 229, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "Compute");

    mutex_lock l(mu_);
    if (!accumulator_set_) {
      OP_REQUIRES_OK(ctx, SetAccumulatorHandle(ctx));
    }
    SetHandleToOutput(ctx);
  }

 protected:
  ~ConditionalAccumulatorBaseOp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_2(mht_2_v, 241, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "~ConditionalAccumulatorBaseOp");

    // If the accumulator object was not shared, delete it.
    if (accumulator_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK((cinfo_.resource_manager()
                       ->template Delete<ConditionalAccumulatorBase>(
                           cinfo_.container(), cinfo_.name())));
    }
  }

 protected:
  virtual void SetHandleToOutput(OpKernelContext* ctx)
      TF_SHARED_LOCKS_REQUIRED(mu_) = 0;

  virtual Status CheckSignature(OpKernelContext* ctx) = 0;

 protected:
  typedef std::function<Status(ConditionalAccumulatorBase**)> Creator;

  // Subclasses must override this
  virtual Creator GetCreator() const = 0;

  // Variables required to construct ConditionalAccumulator
  DataType dtype_;
  PartialTensorShape shape_;
  ContainerInfo cinfo_;
  string reduction_type_;
  mutex mu_;
  Tensor accumulator_ TF_GUARDED_BY(mu_);
  bool accumulator_set_ TF_GUARDED_BY(mu_);

 private:
  Status SetAccumulatorHandle(OpKernelContext* ctx)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_3(mht_3_v, 276, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "SetAccumulatorHandle");

    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));

    // Check input signature
    TF_RETURN_IF_ERROR(CheckSignature(ctx));

    Creator creator = GetCreator();
    ConditionalAccumulatorBase* accumulator;
    TF_RETURN_IF_ERROR(
        (cinfo_.resource_manager()
             ->template LookupOrCreate<ConditionalAccumulatorBase>(
                 cinfo_.container(), cinfo_.name(), &accumulator, creator)));
    core::ScopedUnref unref_me(accumulator);

    // Verify that the shared accumulator is compatible
    // with the requested arguments.
    TF_RETURN_IF_ERROR(accumulator->MatchesNodeDef(def()));
    auto h = accumulator_.template flat<tstring>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    accumulator_set_ = true;
    return Status::OK();
  }
};

// ------------------Sync kernels ------------------------------------------

/**
 * General OpKernel for ConditionalAccumulatorBase-related ops.
 */
class ConditionalAccumulatorBaseSyncOpKernel : public OpKernel {
 public:
  explicit ConditionalAccumulatorBaseSyncOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_4(mht_4_v, 312, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ConditionalAccumulatorBaseSyncOpKernel");
}

  void Compute(OpKernelContext* ctx) final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_5(mht_5_v, 317, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "Compute");

    ConditionalAccumulatorBase* accumulator;
    OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "handle", &accumulator));
    Compute(ctx, accumulator);
    accumulator->Unref();
  }

 protected:
  virtual void Compute(OpKernelContext* ctx,
                       ConditionalAccumulatorBase* accumulator) = 0;

  virtual DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) = 0;

  virtual void CheckSignature(OpKernelContext* ctx,
                              ConditionalAccumulatorBase* accumulator) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_6(mht_6_v, 335, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "CheckSignature");

    // Check input signature
    DataTypeVector expected_inputs = GetExpectedInputs(accumulator);
    OP_REQUIRES_OK(ctx, ctx->MatchSignature(expected_inputs, {}));
  }
};

/**
 * Defines a AccumulateGradientOp, the execution of which adds a gradient to the
 * given ConditionalAccumulator.
 */
class ConditionalAccumulatorBaseApplyGradientOp
    : public ConditionalAccumulatorBaseSyncOpKernel {
 public:
  explicit ConditionalAccumulatorBaseApplyGradientOp(
      OpKernelConstruction* context)
      : ConditionalAccumulatorBaseSyncOpKernel(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_7(mht_7_v, 354, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ConditionalAccumulatorBaseApplyGradientOp");
}

 protected:
  void Compute(OpKernelContext* ctx,
               ConditionalAccumulatorBase* accumulator) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_8(mht_8_v, 361, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "Compute");

    // Check input signature
    CheckSignature(ctx, accumulator);

    // Get input local_step
    const Tensor* local_step_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("local_step", &local_step_tensor));
    if (!TensorShapeUtils::IsScalar(local_step_tensor->shape())) {
      ctx->CtxFailureWithWarning(errors::InvalidArgument(
          "Argument local_step must be scalar, but had bad shape ",
          local_step_tensor->shape().DebugString()));
    }

    // Actually try to apply gradient now
    accumulator->TryApplyGrad(local_step_tensor->scalar<int64_t>()(), ctx);
  }
};

// -------------------- Async kernels --------------------------------------
/**
 * General OpKernel for ConditionalAccumulatorBase-related ops.
 */
class ConditionalAccumulatorBaseAsyncOpKernel : public AsyncOpKernel {
 public:
  explicit ConditionalAccumulatorBaseAsyncOpKernel(
      OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_9(mht_9_v, 390, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ConditionalAccumulatorBaseAsyncOpKernel");
}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_10(mht_10_v, 395, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ComputeAsync");

    ConditionalAccumulatorBase* accumulator;
    OP_REQUIRES_OK_ASYNC(
        ctx, GetResourceFromContext(ctx, "handle", &accumulator), callback);
    ComputeAsync(ctx, accumulator, [callback, accumulator]() {
      accumulator->Unref();
      callback();
    });
  }

 protected:
  virtual void ComputeAsync(OpKernelContext* ctx,
                            ConditionalAccumulatorBase* accumulator,
                            DoneCallback callback) = 0;

  virtual DataTypeVector GetExpectedInputs(
      ConditionalAccumulatorBase* accumulator) = 0;

  virtual void CheckSignature(OpKernelContext* ctx,
                              ConditionalAccumulatorBase* accumulator,
                              DoneCallback callback) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_11(mht_11_v, 418, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "CheckSignature");

    // Check input signature
    OP_REQUIRES_OK_ASYNC(ctx,
                         ctx->MatchSignature(GetExpectedInputs(accumulator),
                                             {accumulator->dtype()}),
                         callback);
  }
};

/**
 * Defines a TakeAccumulatedGradientOp, the execution of which adds a gradient
 * to the given ConditionalAccumulator.
 */
class ConditionalAccumulatorBaseTakeGradientOp
    : public ConditionalAccumulatorBaseAsyncOpKernel {
 public:
  explicit ConditionalAccumulatorBaseTakeGradientOp(
      OpKernelConstruction* context)
      : ConditionalAccumulatorBaseAsyncOpKernel(context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_12(mht_12_v, 439, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ConditionalAccumulatorBaseTakeGradientOp");
}

 protected:
  void ComputeAsync(OpKernelContext* ctx,
                    ConditionalAccumulatorBase* accumulator,
                    DoneCallback callback) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_base_opDTh mht_13(mht_13_v, 447, "", "./tensorflow/core/kernels/conditional_accumulator_base_op.h", "ComputeAsync");

    // Check signature
    CheckSignature(ctx, accumulator, callback);

    // Get input num_required
    const Tensor* num_required_tensor;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("num_required", &num_required_tensor),
                         callback);
    if (!TensorShapeUtils::IsScalar(num_required_tensor->shape())) {
      ctx->CtxFailureWithWarning(errors::InvalidArgument(
          "Argument num_required must be scalar, but had bad shape ",
          num_required_tensor->shape().DebugString()));
      callback();
    }

    // Actually try to take gradient now
    accumulator->TryTakeGrad(num_required_tensor->scalar<int32>()(), ctx,
                             callback);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_OP_H_
