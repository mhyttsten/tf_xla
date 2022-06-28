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
class MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc {
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
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/client/client_session.h"

#include <vector>

#include "absl/synchronization/barrier.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_options.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

using ops::Add;
using ops::BatchMatMul;
using ops::Const;
using ops::Mul;
using ops::Placeholder;
using ops::Sub;

class CustomThreadPoolImpl : public thread::ThreadPoolInterface {
 public:
  explicit CustomThreadPoolImpl(int numThreads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/cc/client/client_session_test.cc", "CustomThreadPoolImpl");

    underlying_threadpool_.reset(new thread::ThreadPool(
        tensorflow::Env::Default(), "custom_threadpool", numThreads));
    num_schedule_called_ = 0;
  }

  void Schedule(std::function<void()> fn) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/cc/client/client_session_test.cc", "Schedule");

    num_schedule_called_ += 1;
    underlying_threadpool_->Schedule(std::move(fn));
  }

  void ScheduleWithHint(std::function<void()> fn, int start, int end) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/cc/client/client_session_test.cc", "ScheduleWithHint");

    num_schedule_called_ += 1;
    underlying_threadpool_->ScheduleWithHint(std::move(fn), start, end);
  }

  void Cancel() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/cc/client/client_session_test.cc", "Cancel");
}

  int NumThreads() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_4(mht_4_v, 243, "", "./tensorflow/cc/client/client_session_test.cc", "NumThreads");

    return underlying_threadpool_->NumThreads();
  }

  int CurrentThreadId() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_5(mht_5_v, 250, "", "./tensorflow/cc/client/client_session_test.cc", "CurrentThreadId");

    return underlying_threadpool_->CurrentThreadId();
  }

  int GetNumScheduleCalled() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_session_testDTcc mht_6(mht_6_v, 257, "", "./tensorflow/cc/client/client_session_test.cc", "GetNumScheduleCalled");
 return num_schedule_called_; }

 private:
  int num_schedule_called_;
  std::unique_ptr<tensorflow::thread::ThreadPool> underlying_threadpool_;
};

TEST(ClientSessionTest, Basic) {
  Scope root = Scope::NewRootScope();
  auto c = Const(root, {{1, 1}});
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({1, 1}, {1, 2}));
}

TEST(ClientSessionTest, Feed) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  auto b = Placeholder(root, DT_INT32);
  auto c = Add(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({{a, 1}, {b, 41}}, {c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({42}, {}));
}

TEST(ClientSessionTest, Extend) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32, Placeholder::Shape({2}));
  auto c = Add(root, a, {2, 2});
  ClientSession session(root);
  std::vector<Tensor> outputs;

  TF_EXPECT_OK(session.Run({{a, {1, 1}}}, {c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({3, 3}, {2}));

  auto d = Add(root, c, {39, 39});
  outputs.clear();
  TF_EXPECT_OK(session.Run({{a, {-10, 1}}}, {d}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({31, 42}, {2}));
}

TEST(ClientSessionTest, MultiThreadedWithDefaultThreadpool) {
  Scope root = Scope::NewRootScope();
  auto a = Add(root, {1, 2}, {3, 4});
  auto b = Mul(root, {1, 2}, {3, 4});
  ClientSession session(root);
  {
    thread::ThreadPool thread_pool(Env::Default(), "pool", 2);
    thread_pool.Schedule([&session, a]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run({a}, &outputs));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({4, 6}, {2}));
    });
    thread_pool.Schedule([&session, b]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run({b}, &outputs));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({3, 8}, {2}));
    });
  }
  auto c = Sub(root, b, a);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run({c}, &outputs));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({-1, 2}, {2}));
}

TEST(ClientSessionTest, MultiThreadedWithCustomThreadpool) {
  Scope root = Scope::NewRootScope();
  int num_threads = 3;
  auto a = Add(root, {1, 2}, {3, 4});
  auto b = Mul(root, {1, 2}, {3, 4});
  ClientSession session(root);

  auto inter_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(inter_op_threadpool->GetNumScheduleCalled(), 0);

  auto intra_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(intra_op_threadpool->GetNumScheduleCalled(), 0);

  tensorflow::thread::ThreadPoolOptions threadPoolOptions;
  threadPoolOptions.inter_op_threadpool = inter_op_threadpool.get();
  threadPoolOptions.intra_op_threadpool = intra_op_threadpool.get();

  {
    thread::ThreadPool thread_pool(Env::Default(), "pool", 2);
    thread_pool.Schedule([&session, a]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run(RunOptions(), ClientSession::FeedType{}, {a}, {},
                               &outputs, nullptr, thread::ThreadPoolOptions()));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({4, 6}, {2}));
    });
    thread_pool.Schedule([&session, b]() {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(session.Run(RunOptions(), ClientSession::FeedType{}, {b}, {},
                               &outputs, nullptr, thread::ThreadPoolOptions()));
      test::ExpectTensorEqual<int>(outputs[0],
                                   test::AsTensor<int>({3, 8}, {2}));
    });
  }
  auto c = Sub(root, b, a);
  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session.Run(RunOptions(), ClientSession::FeedType{}, {c}, {},
                           &outputs, nullptr, thread::ThreadPoolOptions()));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({-1, 2}, {2}));
}

TEST(ClientSessionTest, CallableWithDefaultThreadPool) {
  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  auto b = Placeholder(root, DT_INT32);
  auto c = Add(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  CallableOptions options;
  options.add_feed(a.node()->name());
  options.add_feed(b.node()->name());
  options.add_fetch(c.node()->name());
  ClientSession::CallableHandle callable;
  TF_CHECK_OK(session.MakeCallable(options, &callable));
  TF_EXPECT_OK(session.RunCallable(
      callable, {test::AsTensor<int>({1}, {}), test::AsTensor<int>({41}, {})},
      &outputs, nullptr));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({42}, {}));
  TF_EXPECT_OK(session.ReleaseCallable(callable));
}

TEST(ClientSessionTest, CallableWithCustomThreadPool) {
  Scope root = Scope::NewRootScope();
  int num_threads = 3;

  TensorShape data_shape({1, 1});
  auto a = Placeholder(root, DT_INT32, Placeholder::Shape(data_shape));
  auto b = Placeholder(root, DT_INT32, Placeholder::Shape(data_shape));
  auto c = BatchMatMul(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  auto inter_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(inter_op_threadpool->GetNumScheduleCalled(), 0);

  auto intra_op_threadpool =
      absl::make_unique<CustomThreadPoolImpl>(num_threads);
  ASSERT_EQ(intra_op_threadpool->GetNumScheduleCalled(), 0);

  tensorflow::thread::ThreadPoolOptions threadPoolOptions;
  threadPoolOptions.inter_op_threadpool = inter_op_threadpool.get();
  threadPoolOptions.intra_op_threadpool = intra_op_threadpool.get();

  CallableOptions options;
  options.add_feed(a.node()->name());
  options.add_feed(b.node()->name());
  options.add_fetch(c.node()->name());
  ClientSession::CallableHandle callable;
  TF_CHECK_OK(session.MakeCallable(options, &callable));

  // This is needed to have BatchMatMul computation be scheduled in the
  // intra_op_threadpool.
  absl::Barrier barrier(num_threads + 1);
  for (int i = 0; i < num_threads; i++) {
    intra_op_threadpool->Schedule([&barrier, num_threads]() {
      tensorflow::SetPerThreadMaxParallelism(num_threads - 1);
      barrier.Block();
    });
  }
  barrier.Block();

  TF_EXPECT_OK(session.RunCallable(
      callable,
      {test::AsTensor<int>({2}, {1, 1}), test::AsTensor<int>({10}, {1, 1})},
      &outputs, nullptr, threadPoolOptions));
  test::ExpectTensorEqual<int>(outputs[0], test::AsTensor<int>({20}, {1, 1}));
  TF_EXPECT_OK(session.ReleaseCallable(callable));
  ASSERT_GT(inter_op_threadpool->GetNumScheduleCalled(), 0);
  ASSERT_GT(intra_op_threadpool->GetNumScheduleCalled(), 0);

  // Free intra_op_threadpool and wait for its threads to exit before freeing
  // other objects (e.g. barrier). This is needed to avoid data race.
  intra_op_threadpool.reset();
}

}  // namespace
}  // namespace tensorflow
