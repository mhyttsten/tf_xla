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

#ifndef TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_H_
#define TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh() {
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


#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/typed_conditional_accumulator_base.h"

namespace tensorflow {

/**
 * An aggregation object for adding dense gradients.
 *
 * The two main methods of this class are TryApplyGrad and TryTakeGrad.
 *
 * TryApplyGrad tries add a gradient to the accumulator. The attempt is
 * successful if local_step >= global_step, i.e., if the gradient is not stale,
 * having been computed using up-to-date information. Otherwise, the gradient is
 * silently dropped.
 *
 * TryTakeGrad logs an attempt to read the average gradient. The attempt is
 * blocked until the number of gradients accumulated (via TryApplyGrad) is equal
 * or exceeds the number requested by TryTakeGrad.
 * Once this condition is satisfied, the following actions are taken:
 * (1) the value of the average gradient is returned
 * (2) the count of accumulated gradients is reset to 0
 * (3) the internal global_step value (current_global_step_) is incremented by 1
 *
 * ConditionalAccumulator is the datatype-dependent templated sub-class of
 * ConditionalAccumulatorBase. It implements the virtual arithmetic methods that
 * are used by for aggregating, averaging, allocating, returning dense Tensors.
 */
template <typename Device, typename T>
class ConditionalAccumulator
    : public TypedConditionalAccumulatorBase<const Tensor> {
 public:
  // Args:
  //   dtype: The datatype of the gradients to be accumulated.
  //   shape: The shape of the accumulated gradients.
  //   name:  A name to use for the ConditionalAccumulator.
  //   reduction_type: The reduction type, i.e., MEAN or SUM
  ConditionalAccumulator(const DataType& dtype, const PartialTensorShape& shape,
                         const string& name, const string& reduction_type)
      : TypedConditionalAccumulatorBase<const Tensor>(dtype, shape, name,
                                                      reduction_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("reduction_type: \"" + reduction_type + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_0(mht_0_v, 229, "", "./tensorflow/core/kernels/conditional_accumulator.h", "ConditionalAccumulator");
}
  ~ConditionalAccumulator() override{
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_1(mht_1_v, 233, "", "./tensorflow/core/kernels/conditional_accumulator.h", "~ConditionalAccumulator");
};

 protected:
  // accum_grad is the tensor that holds the aggregate gradient.
  // It is initialized the first time ApplyGrad is called.
  Tensor accum_grad_;

  functor::SetZeroFunctor<Device, T> set_zero_functor_;

  Status ValidateShape(const Tensor* tensor)
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/conditional_accumulator.h", "ValidateShape");

    // Must be compatible with accumulated gradient if available
    if (counter_ > 0) {
      if (!accum_grad_.shape().IsSameSize(tensor->shape())) {
        return errors::InvalidArgument("Shape mismatch: expected ",
                                       accum_grad_.shape().DebugString(),
                                       ", got ", tensor->shape().DebugString());
      }
    }
    // Must also be compatible with given shape
    if (!shape_.IsCompatibleWith(tensor->shape())) {
      return errors::InvalidArgument("Shape mismatch: expected ",
                                     shape_.DebugString(), ", got ",
                                     tensor->shape().DebugString());
    }
    return Status::OK();
  }

  void AllocateAndAssignToAccumGradFunction(OpKernelContext* ctx,
                                            const Tensor* grad) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/conditional_accumulator.h", "AllocateAndAssignToAccumGradFunction");

    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    ctx->allocate_temp(dtype_, grad->shape(), &accum_grad_).IgnoreError();
    accum_grad_.flat<T>().device(ctx->template eigen_device<Device>()) =
        grad->flat<T>();
  }

  void AddToAccumGradFunction(OpKernelContext* ctx,
                              const Tensor* grad) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_4(mht_4_v, 279, "", "./tensorflow/core/kernels/conditional_accumulator.h", "AddToAccumGradFunction");

    accum_grad_.flat<T>().device(ctx->template eigen_device<Device>()) +=
        grad->flat<T>();
  }

  void DivideAccumGradByCounter(OpKernelContext* ctx) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    Tensor c(DataTypeToEnum<T>::value, {});
    c.scalar<T>()() = TypeConverter<T, int>::ConvertUToT(this->counter_);
    this->accum_grad_.template flat<T>().device(
        ctx->template eigen_device<Device>()) =
        this->accum_grad_.template flat<T>() / c.scalar<T>()();
  }

  bool SetOutput(OpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_5(mht_5_v, 296, "", "./tensorflow/core/kernels/conditional_accumulator.h", "SetOutput");

    ctx->set_output(0, accum_grad_);
    return true;
  }

  bool GetAndValidateTensorInputForApplyGrad(OpKernelContext* ctx,
                                             const Tensor** tensor) override
      TF_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {
    // Get input gradient tensor
    const Tensor* grad_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx, ctx->input("gradient", &grad_tensor));
    *tensor = grad_tensor;
    OP_REQUIRES_OK_BOOLEAN(ctx, this->ValidateShape(*tensor));
    return true;
  }

  void CleanUpGradTensor(const Tensor* tensor) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulatorDTh mht_6(mht_6_v, 315, "", "./tensorflow/core/kernels/conditional_accumulator.h", "CleanUpGradTensor");

    // do nothing
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ConditionalAccumulator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_H_
