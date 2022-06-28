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
class MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc() {
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

#include "tensorflow/core/kernels/conditional_accumulator_base.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

ConditionalAccumulatorBase::ConditionalAccumulatorBase(
    const DataType& dtype, const PartialTensorShape& shape, const string& name,
    const string& reduction_type)
    : dtype_(dtype),
      shape_(shape),
      name_(name),
      reduction_type_(reduction_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("reduction_type: \"" + reduction_type + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::ConditionalAccumulatorBase");

  counter_ = 0;
  current_global_step_ = 0;
}

Status ConditionalAccumulatorBase::MatchesNodeDef(const NodeDef& node_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::MatchesNodeDef");

  // TODO(xinghao@): implement the checks for the node definition
  return Status::OK();
}

/**
 * Sets the time step of the accumulator to be in line with the global time
 * step. Logs warning if the accumulator's time step is already larger than the
 * provided time step.
 */
Status ConditionalAccumulatorBase::SetGlobalStep(int64_t new_global_step) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::SetGlobalStep");

  mutex_lock lock(mu_);
  if (new_global_step < current_global_step_) {
    LOG(WARNING) << "Attempt to set current_global_step_ to smaller value: "
                 << "current_global_step_ = " << current_global_step_
                 << " >= " << new_global_step << " = new_global_step.";
  }
  current_global_step_ = new_global_step;
  return Status::OK();
}

/**
 * Logs an attempt to extract the average gradient, and tries to flush all
 * TakeGrad attempts.
 * A TakeGrad attempt is blocked until num_required > counter_, i.e.,
 * sufficient gradients have been accumulated.
 *
 * num_required: Number of gradients that needs to be accumulated before the
 *               attempt is unblocked.
 * ctx:          Context in which the op is executed.
 * callback:     A callback to be executed after the attempt has been completed.
 */
void ConditionalAccumulatorBase::TryTakeGrad(int num_required,
                                             OpKernelContext* ctx,
                                             DoneCallback callback) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::TryTakeGrad");

  if (num_required <= 0) {
    ctx->CtxFailureWithWarning(errors::InvalidArgument(
        "Argument num_required must be positive, but was ", num_required));
    callback();
  } else {
    CancellationManager* cm = ctx->cancellation_manager();
    CancellationToken token = cm->get_cancellation_token();
    bool already_cancelled;
    {
      mutex_lock l(mu_);
      already_cancelled = !cm->RegisterCallback(
          token, [this, cm, token]() { Cancel(cm, token); });
      if (!already_cancelled) {
        takegrad_attempts_.emplace_back(
            num_required, callback, ctx, cm, token,
            [this](Attempt* attempt) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
              if (counter_ >= attempt->elements_requested) {
                bool successful_take_grad = TakeGradLockedHelper(
                    attempt->context, attempt->done_callback);
                if (successful_take_grad) {
                  return kComplete;
                } else {
                  // Try again
                  return kNoProgress;
                }
              } else {
                return kNoProgress;
              }
            });
      }
    }
    if (!already_cancelled) {
      FlushUnlocked();
    } else {
      ctx->SetStatus(errors::Cancelled("TakeGrad operation was cancelled"));
      callback();
    }
  }
}

/**
 * Cancellation callback.
 */
void ConditionalAccumulatorBase::Cancel(
    CancellationManager* cancellation_manager, CancellationToken token) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::Cancel");

  DoneCallback callback = nullptr;
  {
    mutex_lock lock(mu_);

    for (Attempt& attempt : takegrad_attempts_) {
      if (attempt.cancellation_manager == cancellation_manager &&
          attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          attempt.context->SetStatus(
              errors::Cancelled("TakeGrad operation was cancelled"));
          std::swap(callback, attempt.done_callback);
        }
        break;
      }
    }
  }
  if (callback) {
    callback();
    FlushUnlocked();
  }
}

/**
 * Try to flush logged, blocked TakeGrad attempts.
 */
bool ConditionalAccumulatorBase::TryAttemptLocked(
    std::vector<CleanUp>* clean_up) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_5(mht_5_v, 325, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::TryAttemptLocked");

  bool progress = false;
  bool done = false;
  while (!done && !takegrad_attempts_.empty()) {
    if (takegrad_attempts_.front().is_cancelled) {
      VLOG(1) << "Skipping cancelled TakeGrad attempt";
      takegrad_attempts_.pop_front();
    } else {
      Attempt* cur_attempt = &takegrad_attempts_.front();
      switch (cur_attempt->run_callback(cur_attempt)) {
        case kNoProgress:
          done = true;
          break;
        case kComplete:
          progress = true;
          clean_up->emplace_back(std::move(cur_attempt->done_callback),
                                 cur_attempt->cancellation_token,
                                 cur_attempt->context->cancellation_manager());
          takegrad_attempts_.pop_front();
          break;
      }
    }
  }
  return progress;
}

/**
 * Try to flush logged, blocked TakeGrad attempts.
 */
void ConditionalAccumulatorBase::FlushUnlocked() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_6(mht_6_v, 357, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::FlushUnlocked");

  std::vector<CleanUp> clean_up;
  Ref();
  {
    mutex_lock lock(mu_);
    bool changed;
    do {
      changed = TryAttemptLocked(&clean_up);
    } while (changed);
  }
  Unref();
  for (const auto& to_clean : clean_up) {
    if (to_clean.to_deregister != CancellationManager::kInvalidToken) {
      // NOTE(mrry): We can safely ignore the return value of
      // DeregisterCallback because the mutex mu_ ensures that the
      // cleanup action only executes once.
      to_clean.cm->DeregisterCallback(to_clean.to_deregister);
    }
    to_clean.finished();
  }
}

bool ConditionalAccumulatorBase::TakeGradLockedHelper(OpKernelContext* ctx,
                                                      DoneCallback callback) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconditional_accumulator_baseDTcc mht_7(mht_7_v, 383, "", "./tensorflow/core/kernels/conditional_accumulator_base.cc", "ConditionalAccumulatorBase::TakeGradLockedHelper");

  // At this point, the conditional should have been passed

  // Implicitly increment global_step
  current_global_step_++;

  // Average the accumulated gradient
  if (reduction_type_ == "MEAN") {
    DivideAccumGradByCounter(ctx);
  }

  // Set output for accumulated gradient tensor
  bool successful_set_output = SetOutput(ctx);

  // Reset counter
  if (successful_set_output) counter_ = 0;

  return successful_set_output;
}

}  // namespace tensorflow
