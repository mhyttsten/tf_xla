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

#ifndef TENSORFLOW_CORE_KERNELS_TYPED_CONDITIONAL_ACCUMULATOR_BASE_H_
#define TENSORFLOW_CORE_KERNELS_TYPED_CONDITIONAL_ACCUMULATOR_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPStyped_conditional_accumulator_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_conditional_accumulator_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStyped_conditional_accumulator_baseDTh() {
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


#include "tensorflow/core/kernels/conditional_accumulator_base.h"

namespace tensorflow {

/*
 * TypedConditionalAccumulatorBase is a templated companion of
 * ConditionalAccumulatorBase which allows for subclasses to use different
 * types for the input gradients. (See ConditionalAccumulator and
 * SparseConditionalAccumulator.)
 *
 * TypedConditionalAccumulatorBase defines virtual methods and implements
 * methods which depend on the gradient type. These are mainly methods that are
 * used for adding a new gradient to the accumulator.
 */
template <typename GradientTensorType>
class TypedConditionalAccumulatorBase : public ConditionalAccumulatorBase {
 public:
  TypedConditionalAccumulatorBase(const DataType& dtype,
                                  const PartialTensorShape& shape,
                                  const string& name,
                                  const string& reduction_type)
      : ConditionalAccumulatorBase(dtype, shape, name, reduction_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("reduction_type: \"" + reduction_type + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_conditional_accumulator_baseDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/typed_conditional_accumulator_base.h", "TypedConditionalAccumulatorBase");
}

  /**
   * Attempts to add a gradient to the accumulator. An ApplyGrad attempt is
   * successful (i.e., has its gradient applied) if its local_step >=
   * current_global_step_ at the time the attempt is processed. Otherwise, if
   * local_step < current_global_step_, the stale gradient is silently dropped.
   *
   * local_step: Time-step at which the gradient was computed.
   * grad:       Gradient tensor to be added to the accumulator.
   * ctx:        Context in which the op is executed.
   */
  void TryApplyGrad(int64_t local_step, OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStyped_conditional_accumulator_baseDTh mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/typed_conditional_accumulator_base.h", "TryApplyGrad");

    {
      mutex_lock l(mu_);
      if (local_step >= current_global_step_) {
        GradientTensorType* grad = nullptr;
        bool is_valid = GetAndValidateTensorInputForApplyGrad(ctx, &grad);
        if (is_valid) {
          if (counter_ > 0) {
            AddToAccumGradFunction(ctx, grad);
          } else {
            AllocateAndAssignToAccumGradFunction(ctx, grad);
          }
          counter_++;
        }
        CleanUpGradTensor(grad);
      }
    }
    FlushUnlocked();
  }

 protected:
  // Virtual methods to be implemented by sub-classes for different datatypes.
  // Implements arithmetic operations specific to datatype.
  virtual void AllocateAndAssignToAccumGradFunction(
      OpKernelContext* ctx, GradientTensorType* grad) = 0;

  virtual void AddToAccumGradFunction(OpKernelContext* ctx,
                                      GradientTensorType* grad) = 0;

  // Method for extracting and validating input provided in an OpKernelContext.
  // Returns true if input was successfully retrieved and is valid.
  // Gradient is returned via the GradientTensorType** tensor.
  virtual bool GetAndValidateTensorInputForApplyGrad(
      OpKernelContext* ctx, GradientTensorType** tensor)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  // Method for cleaning up any memory allocated in
  // GetAndValidateTensorInputForApplyGrad
  virtual void CleanUpGradTensor(GradientTensorType* tensor) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TYPED_CONDITIONAL_ACCUMULATOR_BASE_H_
