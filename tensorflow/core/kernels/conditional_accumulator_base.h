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

#ifndef TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_H_
#define TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh() {
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


#include <deque>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

/**
 * ConditionalAccumulator/ConditionalAccumulatorBase implements an aggregation
 * object for adding gradients.
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
 */
class ConditionalAccumulatorBase : public ResourceBase {
 public:
  // Args:
  //   dtype: The datatype of the gradients to be accumulated.
  //   shape: The shape of the accumulated gradients.
  //   name:  A name to use for the ConditionalAccumulator.
  ConditionalAccumulatorBase(const DataType& dtype,
                             const PartialTensorShape& shape,
                             const string& name, const string& reduction_type);

  typedef AsyncOpKernel::DoneCallback DoneCallback;

  virtual void TryApplyGrad(int64_t local_step, OpKernelContext* ctx) = 0;
  void TryTakeGrad(int num_required, OpKernelContext* ctx,
                   DoneCallback callback);

  // Accessor methods
  uint32 num_accumulated() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_0(mht_0_v, 233, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "num_accumulated");

    mutex_lock lock(mu_);
    return counter_;
  }

  const DataType& dtype() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_1(mht_1_v, 241, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "dtype");
 return dtype_; }

  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "DebugString");
 return "A conditional accumulator"; }

  // SetGlobalStep is a modifier method for current_global_step.
  // It returns an InvalidArgument error if the new_global_step is less than
  // current_global_step.
  Status SetGlobalStep(int64_t new_global_step);

  Status MatchesNodeDef(const NodeDef& node_def);

 protected:
  // Virtual methods to be implemented by sub-classes for different datatypes.
  // Implements arithmetic operations specific to datatype.
  virtual void DivideAccumGradByCounter(OpKernelContext* ctx)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;
  virtual bool SetOutput(OpKernelContext* ctx) = 0;

  enum RunResult { kNoProgress, kComplete };

  // Helper struct holding information about a TakeGrad attempt
  struct Attempt;
  typedef std::function<RunResult(Attempt*)> RunCallback;
  struct Attempt {
    int elements_requested;
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationManager* cancellation_manager;  // not owned
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;

    Attempt(int elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationManager* cancellation_manager,
            CancellationToken cancellation_token, RunCallback run_callback)
        : elements_requested(elements_requested),
          done_callback(std::move(done_callback)),
          context(context),
          cancellation_manager(cancellation_manager),
          cancellation_token(cancellation_token),
          run_callback(std::move(run_callback)),
          is_cancelled(false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_3(mht_3_v, 288, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "Attempt");
}
  };

  // Helper struct for deregistration of a cancellation token and executing a
  // DoneCallback after a TakeGrad attempt is complete.
  struct CleanUp {
    CleanUp(DoneCallback&& f, CancellationToken ct, CancellationManager* cm)
        : finished(f), to_deregister(ct), cm(cm) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_4(mht_4_v, 298, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "CleanUp");
}
    DoneCallback finished;
    CancellationToken to_deregister;
    CancellationManager* cm;
  };

  // Fields

  const DataType dtype_;
  const PartialTensorShape shape_;
  const string name_;
  const string reduction_type_;
  mutex mu_;
  int counter_ TF_GUARDED_BY(mu_);
  int64_t current_global_step_ TF_GUARDED_BY(mu_);

  std::deque<Attempt> takegrad_attempts_ TF_GUARDED_BY(mu_);

  // Methods

  // Helper function for creating cancellation callback
  void Cancel(CancellationManager* cancellation_manager,
              CancellationToken token);

  // Helper functions to process TakeGrad attempts.
  // FlushUnlocked is called at the end of each TryApplyGrad and TryTakeGrad
  // calls to try to clear the TakeGrad attempts. This in turn calls
  // TryAttemptLocked, which then executes the RunCallback of the logged
  // attempts.
  // Both functions are modeled after core/kernels/queue_base.
  // Note: ApplyGrad attempts never block -- unlike in a queue with limited
  //       capacity, we can always add the newest gradient to our accumulator
  //       (if it is not stale) or drop it silently (if it is stale).
  void FlushUnlocked();
  bool TryAttemptLocked(std::vector<CleanUp>* clean_up)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper methods
  //  void DeepCopy(Tensor* dst);
  bool TakeGradLockedHelper(OpKernelContext* ctx, DoneCallback callback)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
};

/*
 * Modifications to convenience macros defined in core/framework/op_kernel.h.
 * The below macros return a boolean if the test fails, so that the calling
 * function can get an indication that a failure has occurred.
 */
#define OP_REQUIRES_BOOLEAN(CTX, EXP, STATUS)          \
  do {                                                 \
    if (!TF_PREDICT_TRUE(EXP)) {                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
      return false;                                    \
    }                                                  \
  } while (0)

#define OP_REQUIRES_OK_BOOLEAN(CTX, STATUS)                 \
  do {                                                      \
    ::tensorflow::Status _s(STATUS);                        \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      return false;                                         \
    }                                                       \
  } while (0)

/*
 * Convenience classes for helping to convert between numeric types.
 * The specialization for Eigen::half here simplifies specialization of
 * ConditionalAccumulator classes later.
 */
template <typename T, typename U>
class TypeConverter {
 public:
  static T ConvertUToT(U c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_5(mht_5_v, 374, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "ConvertUToT");
 return c; /* implicit conversion */ }
};

template <typename U>
class TypeConverter<Eigen::half, U> {
 public:
  static Eigen::half ConvertUToT(U c) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTh mht_6(mht_6_v, 383, "", "./tensorflow/core/kernels/conditional_accumulator_base.h", "ConvertUToT");
 return static_cast<Eigen::half>(c); }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONDITIONAL_ACCUMULATOR_BASE_H_
