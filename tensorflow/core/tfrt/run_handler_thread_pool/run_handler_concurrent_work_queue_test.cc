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
class MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queue_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queue_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queue_testDTcc() {
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
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h"

#include <cstdio>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/mutex.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

const int kNumMainThreads = 1;
const int kNumComplementaryThreads = 1;

class RunHandlerThreadWorkQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queue_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue_test.cc", "SetUp");

    RunHandlerThreadWorkQueue::Options options;
    options.num_complementary_threads = kNumComplementaryThreads;
    options.num_main_threads = kNumMainThreads;
    options.init_timeout_ms = 100;
    pool_ = std::make_unique<RunHandlerThreadWorkQueue>(options);

    // decoded_diagnostic_handler does nothing.
    auto decoded_diagnostic_handler = [&](const DecodedDiagnostic& diag) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSrun_handler_thread_poolPSrun_handler_concurrent_work_queue_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue_test.cc", "lambda");
};

    std::unique_ptr<ConcurrentWorkQueue> work_queue =
        CreateSingleThreadedWorkQueue();
    std::unique_ptr<HostAllocator> host_allocator = CreateMallocAllocator();
    host_ = std::make_unique<HostContext>(decoded_diagnostic_handler,
                                          std::move(host_allocator),
                                          std::move(work_queue));
    RequestContextBuilder req_ctx_builder{host_.get(),
                                          /*resource_context=*/nullptr};
    tensorflow::thread::ThreadPoolInterface* intra_op_threadpool = nullptr;
    auto queue =
        pool_->InitializeRequest(&req_ctx_builder, &intra_op_threadpool);
    TF_CHECK_OK(queue.status());
    queue_ = std::move(*queue);
    auto req_ctx = std::move(req_ctx_builder).build();
    ASSERT_TRUE(static_cast<bool>(req_ctx));
    exec_ctx_ = std::make_unique<ExecutionContext>(std::move(*req_ctx));
  }

  std::unique_ptr<RunHandlerThreadWorkQueue> pool_;
  std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface> queue_;
  std::unique_ptr<HostContext> host_;
  std::unique_ptr<ExecutionContext> exec_ctx_;
};

TEST_F(RunHandlerThreadWorkQueueTest, RunningBlockingTask) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    ASSERT_FALSE(pool_->AddBlockingTask(TaskFunction([&n, &m] {
                                          tensorflow::mutex_lock lock(m);
                                          ++n;
                                        }),
                                        true));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningBlockingTaskNoExecCtx) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    pool_->AddBlockingTask(TaskFunction([&n, &m] {
                             tensorflow::mutex_lock lock(m);
                             ++n;
                           }),
                           true);
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningBlockingTaskNoQueueing) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    ASSERT_FALSE(pool_->AddBlockingTask(TaskFunction([&n, &m] {
                                          tensorflow::mutex_lock lock(m);
                                          ++n;
                                        }),
                                        false));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningNonBlockingTask) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    queue_->AddTask(TaskFunction([&n, &m] {
      tensorflow::mutex_lock lock(m);
      ++n;
    }));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningNonBlockingTaskWithNoExecCtx) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    pool_->AddTask(TaskFunction([&n, &m] {
      tensorflow::mutex_lock lock(m);
      ++n;
    }));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningMixedTask) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    queue_->AddTask(TaskFunction([&n, &m] {
      tensorflow::mutex_lock lock(m);
      ++n;
    }));
    ASSERT_FALSE(pool_->AddBlockingTask(TaskFunction([&n, &m] {
                                          tensorflow::mutex_lock lock(m);
                                          ++n;
                                        }),
                                        true));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 20);
}

TEST_F(RunHandlerThreadWorkQueueTest, NameReturnsValidString) {
  EXPECT_EQ(queue_->name(), "run_handler");
}

TEST_F(RunHandlerThreadWorkQueueTest, GetParallelismLevelOk) {
  EXPECT_EQ(queue_->GetParallelismLevel(),
            kNumComplementaryThreads + kNumMainThreads);
}

TEST_F(RunHandlerThreadWorkQueueTest, IsWorkerThreadOk) {
  EXPECT_TRUE(queue_->IsInWorkerThread());
}

TEST_F(RunHandlerThreadWorkQueueTest, NoHandlerReturnsError) {
  RunHandlerThreadWorkQueue::Options options;
  options.num_complementary_threads = 0;
  options.num_main_threads = 0;
  options.init_timeout_ms = 1;
  options.max_concurrent_handler = 0;
  auto queue = std::make_unique<RunHandlerThreadWorkQueue>(options);
  tensorflow::thread::ThreadPoolInterface* interface;
  tfrt::RequestContextBuilder ctx_builder(nullptr, nullptr);
  EXPECT_THAT(
      queue->InitializeRequest(&ctx_builder, &interface),
      tensorflow::testing::StatusIs(
          tensorflow::error::INTERNAL,
          "Could not obtain RunHandler for request after waiting for 1 ms."));
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
