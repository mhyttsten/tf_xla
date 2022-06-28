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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc() {
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

#include <atomic>
#include <utility>

#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

class FunctionLibraryRuntimeTest : public ::testing::Test {
 protected:
  void Init(const std::vector<FunctionDef>& flib,
            thread::ThreadPool* default_thread_pool) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "Init");

    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 3});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(
        options, "/job:localhost/replica:0/task:0", &devices));

    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    OptimizerOptions opts;
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    pflr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts, default_thread_pool,
        /*parent=*/nullptr, /*session_metadata=*/nullptr,
        Rendezvous::Factory{
            [](const int64_t, const DeviceMgr* device_mgr, Rendezvous** r) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "lambda");

              *r = new IntraProcessRendezvous(device_mgr);
              return Status::OK();
            }}));
    flr0_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  }

  Status Run(FunctionLibraryRuntime* flr, FunctionLibraryRuntime::Handle handle,
             FunctionLibraryRuntime::Options opts,
             const std::vector<Tensor>& args, std::vector<Tensor*> rets,
             bool add_runner = true) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "Run");

    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "lambda");

          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };
    if (add_runner) {
      opts.runner = &runner;
    } else {
      opts.runner = nullptr;
    }
    Notification done;
    std::vector<Tensor> out;
    Status status;
    flr->Run(opts, handle, args, &out, [&status, &done](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }
    CHECK_EQ(rets.size(), out.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = out[i];
    }

    if (add_runner) {
      EXPECT_GE(call_count, 1);  // Test runner is used.
    }

    return Status::OK();
  }

  Status Instantiate(FunctionLibraryRuntime* flr, const string& name,
                     test::function::Attrs attrs,
                     FunctionLibraryRuntime::Handle* handle) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "Instantiate");

    return flr->Instantiate(name, attrs, handle);
  }

  Status Instantiate(FunctionLibraryRuntime* flr, const string& name,
                     test::function::Attrs attrs,
                     const FunctionLibraryRuntime::InstantiateOptions& options,
                     FunctionLibraryRuntime::Handle* handle) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_5(mht_5_v, 310, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "Instantiate");

    return flr->Instantiate(name, attrs, options, handle);
  }

  Status InstantiateAndRun(FunctionLibraryRuntime* flr, const string& name,
                           test::function::Attrs attrs,
                           const std::vector<Tensor>& args,
                           std::vector<Tensor*> rets, bool add_runner = true) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_6(mht_6_v, 321, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "InstantiateAndRun");

    return InstantiateAndRun(flr, name, attrs,
                             FunctionLibraryRuntime::InstantiateOptions(), args,
                             std::move(rets), add_runner);
  }

  Status InstantiateAndRun(
      FunctionLibraryRuntime* flr, const string& name,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const std::vector<Tensor>& args, std::vector<Tensor*> rets,
      bool add_runner = true) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_7(mht_7_v, 336, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "InstantiateAndRun");

    FunctionLibraryRuntime::Handle handle;
    Status status = flr->Instantiate(name, attrs, options, &handle);
    if (!status.ok()) {
      return status;
    }
    FunctionLibraryRuntime::Options opts;
    status = Run(flr, handle, opts, args, rets, add_runner);
    if (!status.ok()) return status;

    // Release the handle and try running again. It should not succeed.
    status = flr->ReleaseHandle(handle);
    if (!status.ok()) return status;

    Status status2 = Run(flr, handle, opts, args, std::move(rets));
    EXPECT_TRUE(errors::IsNotFound(status2));
    EXPECT_TRUE(absl::StrContains(status2.error_message(), "Handle"));
    EXPECT_TRUE(absl::StrContains(status2.error_message(), "not found"));

    return status;
  }

  Status Run(FunctionLibraryRuntime* flr, FunctionLibraryRuntime::Handle handle,
             FunctionLibraryRuntime::Options opts, CallFrameInterface* frame,
             bool add_runner = true) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_8(mht_8_v, 363, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "Run");

    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSfunction_threadpool_testDTcc mht_9(mht_9_v, 369, "", "./tensorflow/core/common_runtime/function_threadpool_test.cc", "lambda");

          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };
    if (add_runner) {
      opts.runner = &runner;
    } else {
      opts.runner = nullptr;
    }
    Notification done;
    Status status;
    flr->Run(opts, handle, frame, [&status, &done](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }

    if (add_runner) {
      EXPECT_GE(call_count, 1);  // Test runner is used.
    }

    return Status::OK();
  }

  FunctionLibraryRuntime* flr0_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
};

TEST_F(FunctionLibraryRuntimeTest, DefaultThreadpool) {
  using test::function::blocking_op_state;
  using test::function::BlockingOpState;

  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "FLRTest", 1);
  Init({test::function::BlockingOpFn(), test::function::XTimesTwo()}, tp);

  auto x = test::AsScalar<float>(1.3);
  Tensor y;
  blocking_op_state = new BlockingOpState();

  thread::ThreadPool* tp1 = new thread::ThreadPool(Env::Default(), "tp1", 5);
  bool finished_running = false;
  tp1->Schedule([&x, &y, &finished_running, this]() {
    TF_CHECK_OK(InstantiateAndRun(flr0_, "BlockingOpFn", {}, {x}, {&y},
                                  false /* add_runner */));
    finished_running = true;
  });

  // InstantiateAndRun shouldn't finish because BlockingOpFn should be blocked.
  EXPECT_FALSE(finished_running);

  FunctionLibraryRuntime::Handle h;
  TF_CHECK_OK(Instantiate(flr0_, "XTimesTwo", {{"T", DT_FLOAT}}, &h));

  auto x1 = test::AsTensor<float>({1, 2, 3, 4});
  std::atomic<int32> num_done(0);
  FunctionLibraryRuntime::Options opts;
  for (int i = 0; i < 4; ++i) {
    tp1->Schedule([&h, &x1, &opts, &num_done, this]() {
      Tensor y1;
      TF_CHECK_OK(Run(flr0_, h, opts, {x1}, {&y1}, false /* add_runner */));
      num_done.fetch_add(1);
    });
  }
  // All the 4 Run() calls should be blocked because the runner is occupied.
  EXPECT_EQ(0, num_done.load());

  blocking_op_state->AwaitState(1);
  blocking_op_state->MoveToState(1, 2);
  // Now the runner should be unblocked and all the other Run() calls should
  // proceed.
  blocking_op_state->AwaitState(3);
  blocking_op_state->MoveToState(3, 0);
  delete tp1;
  EXPECT_TRUE(finished_running);
  EXPECT_EQ(4, num_done.load());

  delete blocking_op_state;
  blocking_op_state = nullptr;
  delete tp;
}

}  // namespace
}  // namespace tensorflow
