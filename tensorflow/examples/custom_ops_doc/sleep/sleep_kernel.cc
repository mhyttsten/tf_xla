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
class MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc {
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
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/threadpool.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

class AsyncSleepOp : public AsyncOpKernel {
 public:
  explicit AsyncSleepOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc mht_0(mht_0_v, 201, "", "./tensorflow/examples/custom_ops_doc/sleep/sleep_kernel.cc", "AsyncSleepOp");
}
  AsyncSleepOp(const AsyncSleepOp& other) = delete;
  AsyncSleepOp& operator=(const AsyncSleepOp& other) = delete;
  ~AsyncSleepOp() override = default;

  // Implementations of ComputeAsync() must ensure that `done` is (eventually)
  // called exactly once to signal the completion of the computation. The
  // implementation of ComputeAsync() must not block on the execution of another
  // OpKernel. `done` may be called by the current thread, or by another thread.
  // `context` is guaranteed to stay alive until the `done` callback starts.
  // For example, use OP_REQUIRES_ASYNC which takes the `done` paramater
  // as an input and calls `done` for the case of exiting early with an error
  // (instead of OP_REQUIRES).
  //
  // Since it is possible that the unblocking kernel may never run (due to an
  // error or cancellation), in most cases the AsyncOpKernel should implement
  // cancellation support via `context->cancellation_manager()`.
  // TODO (schwartzedward): should this use cancellation support?
  //
  // WARNING: As soon as the `done` callback starts, `context` and `this` may be
  // deleted. No code depending on these objects should execute after the call
  // to `done`.
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc mht_1(mht_1_v, 226, "", "./tensorflow/examples/custom_ops_doc/sleep/sleep_kernel.cc", "ComputeAsync");

    const auto& delay_tensor = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, ::tensorflow::TensorShapeUtils::IsScalar(delay_tensor.shape()),
        InvalidArgument("Input `delay` must be a scalar."),
        done);  // Important: call `done` in every execution path
    const float delay = delay_tensor.flat<float>()(0);
    OP_REQUIRES_ASYNC(ctx, delay >= 0.0,
                      InvalidArgument("Input `delay` must be non-negative."),
                      done);  // Important: call `done` in every execution path
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    OP_REQUIRES_ASYNC(ctx, thread_pool != nullptr,
                      Internal("No thread_pool found."),
                      done);  // Important: call `done` in every execution path

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, delay_tensor.shape(), &output_tensor),
        done);  // Important: call `done` in every execution path

    absl::Time now = absl::Now();
    absl::Time when = now + absl::Seconds(delay);
    VLOG(1) << "BEFORE ASYNC SLEEP " << ctx->op_kernel().name() << " now "
            << now << " when " << when;
    thread_pool->Schedule([this, output_tensor, when, done] {
      this->sleeper(output_tensor, when, done);
    });
    // Note that `done` is normaly called by sleeper(), it is not normally
    // called by this function.
  }

 private:
  void sleeper(Tensor* output_tensor, absl::Time when, DoneCallback done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc mht_2(mht_2_v, 261, "", "./tensorflow/examples/custom_ops_doc/sleep/sleep_kernel.cc", "sleeper");

    absl::Time now = absl::Now();
    int64_t delay_us = 0;
    if (now < when) {
      delay_us = absl::ToInt64Microseconds(when - now);
      VLOG(1) << "MIDDLE ASYNC SLEEP " << delay_us;
      absl::SleepFor(when - now);
      VLOG(1) << "AFTER ASYNC SLEEP " << delay_us;
    } else {
      VLOG(1) << "MIDDLE/AFTER ASYNC SKIP SLEEP";
    }
    auto output = output_tensor->template flat<float>();
    output(0) = static_cast<float>(delay_us) / 1000000.0;
    done();  // Important: call `done` in every execution path
  }
};

class SyncSleepOp : public OpKernel {
 public:
  explicit SyncSleepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc mht_3(mht_3_v, 283, "", "./tensorflow/examples/custom_ops_doc/sleep/sleep_kernel.cc", "SyncSleepOp");
}
  SyncSleepOp(const SyncSleepOp& other) = delete;
  SyncSleepOp& operator=(const SyncSleepOp& other) = delete;
  ~SyncSleepOp() override = default;

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSexamplesPScustom_ops_docPSsleepPSsleep_kernelDTcc mht_4(mht_4_v, 291, "", "./tensorflow/examples/custom_ops_doc/sleep/sleep_kernel.cc", "Compute");

    const auto& delay_tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                ::tensorflow::TensorShapeUtils::IsScalar(delay_tensor.shape()),
                InvalidArgument("Input `delay` must be a scalar."));
    const float delay = delay_tensor.flat<float>()(0);
    OP_REQUIRES(ctx, delay >= 0.0,
                InvalidArgument("Input `delay` must be non-negative."));
    VLOG(1) << "BEFORE SYNC SLEEP" << ctx->op_kernel().name();
    absl::SleepFor(absl::Seconds(delay));
    VLOG(1) << "AFTER SYNC SLEEP" << ctx->op_kernel().name();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, delay_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<float>();
    output(0) = delay;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Examples>AsyncSleep").Device(::tensorflow::DEVICE_CPU), AsyncSleepOp)
REGISTER_KERNEL_BUILDER(
    Name("Examples>SyncSleep").Device(::tensorflow::DEVICE_CPU), SyncSleepOp)

}  // namespace custom_op_examples
}  // namespace tensorflow
