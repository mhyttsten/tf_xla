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
class MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc {
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
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc() {
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

#include "tensorflow/cc/training/queue_runner.h"

#include <string>
#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/training/coordinator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace {

using error::Code;
using ops::Assign;
using ops::Const;
using ops::CountUpTo;
using ops::FIFOQueue;
using ops::QueueClose;
using ops::QueueDequeue;
using ops::QueueEnqueue;
using ops::RandomNormal;
using ops::Square;
using ops::Variable;

constexpr char kAssignOpName[] = "assign";
constexpr char kCancelOp0[] = "cancel0";
constexpr char kCancelOp1[] = "cancel1";
constexpr char kCloseOp0[] = "close0";
constexpr char kCloseOp1[] = "close1";
constexpr char kCountUpToOpName[] = "count";
constexpr char kDequeueOp0[] = "dequeue0";
constexpr char kDequeueOp1[] = "dequeue1";
constexpr char kEnqueueOp0[] = "enqueue0";
constexpr char kEnqueueOp1[] = "enqueue1";
constexpr char kIllegalOpName1[] = "would fail";
constexpr char kIllegalOpName2[] = "fail again";
constexpr char kQueueName[] = "unit_test";
constexpr char kQueueName0[] = "q0";
constexpr char kQueueName1[] = "q1";
constexpr char kSquareOpName[] = "square";
constexpr char kVarOpName[] = "var";

GraphDef BuildSimpleGraph() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc mht_0(mht_0_v, 238, "", "./tensorflow/cc/training/queue_runner_test.cc", "BuildSimpleGraph");

  Scope root = Scope::NewRootScope();
  auto init_value = Const(root, 0);
  auto var = Variable(root.WithOpName(kVarOpName), TensorShape({}),
                      DataType::DT_INT32);
  auto assign = Assign(root.WithOpName(kAssignOpName), var, init_value);
  auto count = CountUpTo(root.WithOpName(kCountUpToOpName), var, 10);
  Square(root.WithOpName(kSquareOpName), var);  // NOLINT

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));
  return graph_def;
}

QueueRunnerDef BuildQueueRunnerDef(
    const std::string& queue_name, const std::vector<std::string>& enqueue_ops,
    const std::string& close_op, const std::string& cancel_op,
    const std::vector<Code>& queue_closed_error_codes) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("queue_name: \"" + queue_name + "\"");
   mht_1_v.push_back("close_op: \"" + close_op + "\"");
   mht_1_v.push_back("cancel_op: \"" + cancel_op + "\"");
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc mht_1(mht_1_v, 261, "", "./tensorflow/cc/training/queue_runner_test.cc", "BuildQueueRunnerDef");

  QueueRunnerDef queue_runner_def;
  *queue_runner_def.mutable_queue_name() = queue_name;
  for (const std::string& enqueue_op : enqueue_ops) {
    *queue_runner_def.mutable_enqueue_op_name()->Add() = enqueue_op;
  }
  *queue_runner_def.mutable_close_op_name() = close_op;
  *queue_runner_def.mutable_cancel_op_name() = cancel_op;
  for (const auto& error_code : queue_closed_error_codes) {
    *queue_runner_def.mutable_queue_closed_exception_types()->Add() =
        error_code;
  }
  return queue_runner_def;
}

std::unique_ptr<Session> BuildSessionAndInitVariable(
    const GraphDef& graph_def) {
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  TF_CHECK_OK(session->Run({}, {}, {kAssignOpName}, nullptr));
  return session;
}

TEST(QueueRunnerTest, BasicTest) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kCountUpToOpName}, kSquareOpName, "", {});

  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_CHECK_OK(qr->Start(session.get()));
  TF_EXPECT_OK(qr->Join());

  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session->Run({}, {kSquareOpName}, {}, &outputs));
  int square_value = *outputs[0].scalar<int>().data();
  EXPECT_EQ(square_value, 100);
}

TEST(QueueRunnerTest, QueueClosedCode) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  // Start two queues so that multiple threads are in Run.
  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kCountUpToOpName, kCountUpToOpName}, kSquareOpName, "",
      {Code::OUT_OF_RANGE, Code::CANCELLED});

  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_EXPECT_OK(qr->Start(session.get()));
  TF_EXPECT_OK(qr->Join());

  std::vector<Tensor> outputs;
  TF_EXPECT_OK(session->Run({}, {kSquareOpName}, {}, &outputs));
  int square_value = *outputs[0].scalar<int>().data();
  EXPECT_EQ(square_value, 100);
}

TEST(QueueRunnerTest, QueueCloseFails) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {kCountUpToOpName}, kIllegalOpName1, "",
                          {Code::OUT_OF_RANGE});

  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_EXPECT_OK(qr->Start(session.get()));
  auto status = qr->Join();
  EXPECT_EQ(status.code(), Code::NOT_FOUND) << status;
}

TEST(QueueRunnerTest, CatchErrorInJoin) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kIllegalOpName1, kIllegalOpName2}, kCountUpToOpName, "", {});

  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_EXPECT_OK(qr->Start(session.get()));
  EXPECT_EQ(qr->Join().code(), Code::NOT_FOUND);
}

GraphDef BuildDoubleQueueGraph() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc mht_2(mht_2_v, 355, "", "./tensorflow/cc/training/queue_runner_test.cc", "BuildDoubleQueueGraph");

  Scope root = Scope::NewRootScope();
  auto q0 = FIFOQueue(root.WithOpName(kQueueName0), {DataType::DT_INT32});
  auto ten = Const(root, 10);
  auto enqueue0 = QueueEnqueue(root.WithOpName(kEnqueueOp0), q0, {ten});
  auto close0 = QueueClose(root.WithOpName(kCloseOp0), q0);
  auto cancel0 = QueueClose(root.WithOpName(kCancelOp0), q0,
                            QueueClose::CancelPendingEnqueues(true));
  auto q1 = FIFOQueue(root.WithOpName(kQueueName1), {DataType::DT_INT32},
                      FIFOQueue::Capacity(3));
  auto dequeue0 =
      QueueDequeue(root.WithOpName(kDequeueOp0), q0, {DataType::DT_INT32});
  auto enqueue1 = QueueEnqueue(root.WithOpName(kEnqueueOp1), q1, {dequeue0[0]});
  auto dequeue1 =
      QueueDequeue(root.WithOpName(kDequeueOp1), q1, {DataType::DT_INT32});
  auto close1 = QueueClose(root.WithOpName(kCloseOp1), q1);
  auto cancel1 = QueueClose(root.WithOpName(kCancelOp1), q1,
                            QueueClose::CancelPendingEnqueues(true));

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));
  return graph_def;
}

TEST(QueueRunnerTest, RealEnqueueDequeue) {
  auto graph_def = BuildDoubleQueueGraph();

  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {kEnqueueOp1}, kCloseOp1, "", {});
  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_CHECK_OK(qr->Start(session.get()));

  TF_EXPECT_OK(session->Run({}, {}, {kEnqueueOp0}, nullptr));
  TF_EXPECT_OK(session->Run({}, {}, {kEnqueueOp0}, nullptr));
  // Closing queue 0 would also close the queue runner.
  TF_EXPECT_OK(session->Run({}, {}, {kCloseOp0}, nullptr));

  TF_EXPECT_OK(qr->Join());
  std::vector<Tensor> dq1;
  TF_EXPECT_OK(session->Run({}, {kDequeueOp1}, {}, &dq1));
  EXPECT_EQ(*dq1[0].scalar<int>().data(), 10);
  std::vector<Tensor> dq2;
  TF_EXPECT_OK(session->Run({}, {kDequeueOp1}, {}, &dq2));
  EXPECT_EQ(*dq2[0].scalar<int>().data(), 10);

  EXPECT_EQ(session->Run({}, {kDequeueOp1}, {}, nullptr).code(),
            Code::OUT_OF_RANGE);
}

void JoinThread(QueueRunner* queue_runner, bool* join_succeeded,
                Notification* join_done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPStrainingPSqueue_runner_testDTcc mht_3(mht_3_v, 413, "", "./tensorflow/cc/training/queue_runner_test.cc", "JoinThread");

  EXPECT_EQ(queue_runner->Join().code(), Code::CANCELLED);
  *join_succeeded = true;
  join_done->Notify();
}

TEST(QueueRunnerTest, SessionCloseCancelPendingEnqueue) {
  auto graph_def = BuildDoubleQueueGraph();

  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName1, {kEnqueueOp1}, kCloseOp1, kCancelOp1, {});
  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_CHECK_OK(qr->Start(session.get()));

  TF_EXPECT_OK(session->Run({}, {}, {kEnqueueOp0}, nullptr));

  std::vector<Tensor> dq1;
  TF_EXPECT_OK(session->Run({}, {kDequeueOp1}, {}, &dq1));
  EXPECT_EQ(*dq1[0].scalar<int>().data(), 10);

  // The expected behavior is the QueueRunner::Join() call is blocked until
  // Session::Close() is called.
  bool join_succeeded = false;
  Notification join_done;
  Env::Default()->SchedClosure(
      std::bind(&JoinThread, qr.get(), &join_succeeded, &join_done));

  Env::Default()->SleepForMicroseconds(10000000);
  EXPECT_EQ(join_succeeded, false);

  // Closing the session is required to cancel pending enqueue nodes.
  TF_EXPECT_OK(session->Close());

  join_done.WaitForNotification();
  EXPECT_EQ(join_succeeded, true);
}

TEST(QueueRunnerTest, EmptyEnqueueOps) {
  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {}, kCountUpToOpName, "", {});

  std::unique_ptr<QueueRunner> qr;
  EXPECT_EQ(QueueRunner::New(queue_runner_def, &qr).code(),
            Code::INVALID_ARGUMENT);
}

TEST(QueueRunnerTest, StartTimeout) {
  GraphDef graph_def = BuildDoubleQueueGraph();
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName1, {kEnqueueOp1}, kCloseOp1, kCancelOp1, {});

  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  // This will timeout since queue0 is not fed and queue1 is fetching data from
  // queue0.
  EXPECT_EQ(qr->Start(session.get(), 1).code(), Code::DEADLINE_EXCEEDED);
  TF_EXPECT_OK(session->Close());
}

TEST(QueueRunnerTest, TestCoordinatorStop) {
  auto graph_def = BuildDoubleQueueGraph();
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));

  QueueRunnerDef queue_runner0 =
      BuildQueueRunnerDef(kQueueName0, {kEnqueueOp0}, kCloseOp0, kCancelOp0,
                          {Code::OUT_OF_RANGE, Code::CANCELLED});
  QueueRunnerDef queue_runner1 =
      BuildQueueRunnerDef(kQueueName1, {kEnqueueOp1}, kCloseOp1, kCancelOp1,
                          {Code::OUT_OF_RANGE, Code::CANCELLED});

  Coordinator coord;
  std::unique_ptr<QueueRunner> qr0;
  TF_EXPECT_OK(QueueRunner::New(queue_runner0, &coord, &qr0));
  TF_CHECK_OK(qr0->Start(session.get()));
  std::unique_ptr<QueueRunner> qr1;
  TF_EXPECT_OK(QueueRunner::New(queue_runner1, &coord, &qr1));
  TF_CHECK_OK(qr1->Start(session.get()));

  TF_EXPECT_OK(coord.RegisterRunner(std::move(qr0)));
  TF_EXPECT_OK(coord.RegisterRunner(std::move(qr1)));

  std::vector<Tensor> dq;
  TF_EXPECT_OK(session->Run({}, {kDequeueOp1}, {}, &dq));
  EXPECT_EQ(*dq[0].scalar<int>().data(), 10);

  TF_EXPECT_OK(coord.RequestStop());
  TF_EXPECT_OK(coord.Join());
}

TEST(QueueRunnerTest, CallbackCalledOnError) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kIllegalOpName1, kIllegalOpName2}, kCountUpToOpName, "", {});

  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  bool error_caught = false;
  qr->AddErrorCallback([&error_caught](const Status&) { error_caught = true; });
  TF_EXPECT_OK(qr->Start(session.get()));
  EXPECT_FALSE(qr->Join().ok());
  EXPECT_TRUE(error_caught);
}

TEST(QueueRunnerTest, RunMetaDataTest) {
  Scope root = Scope::NewRootScope();
  auto q0 = FIFOQueue(root.WithOpName(kQueueName), {DataType::DT_FLOAT});
  Output rnd = RandomNormal(root.WithOpName("rnd"), {1, 1}, DataType::DT_FLOAT);
  Output square = Square(root.WithOpName(kSquareOpName), rnd);
  auto enqueue0 = QueueEnqueue(root.WithOpName(kEnqueueOp0), q0, {square});
  auto close0 = QueueClose(root.WithOpName(kCloseOp0), q0);
  auto cancel0 = QueueClose(root.WithOpName(kCancelOp0), q0,
                            QueueClose::CancelPendingEnqueues(true));
  auto dequeue0 =
      QueueDequeue(root.WithOpName(kDequeueOp0), q0, {DataType::DT_FLOAT});

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));
  for (auto& node : *graph_def.mutable_node()) {
    node.set_device("/cpu:0");
  }
  SessionOptions sess_options;
  sess_options.config.mutable_graph_options()->set_build_cost_model(1);
  std::unique_ptr<Session> session(NewSession(sess_options));

  TF_CHECK_OK(session->Create(graph_def));

  QueueRunnerDef queue_runner_def =
      BuildQueueRunnerDef(kQueueName, {kEnqueueOp0}, kCloseOp0, kCancelOp0, {});
  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  RunOptions run_options;
  TF_CHECK_OK(qr->StartAndCollectCostGraph(session.get(), run_options));

  // Make sure there was at least one element enqueued in q0: this prevents a
  // race condition where we close the queue before it was populated.
  std::vector<Tensor> dq0;
  TF_EXPECT_OK(session->Run({}, {kDequeueOp0}, {}, &dq0));
  // Second call to run dequeue op is to make sure the cost graph has been
  // stored.
  TF_EXPECT_OK(session->Run({}, {kDequeueOp0}, {}, &dq0));

  CostGraphDef cost_graph;
  TF_CHECK_OK(qr->ExportCostGraph(&cost_graph));
  EXPECT_TRUE(cost_graph.node_size() > 0);

  qr->Stop(session.get());
}

TEST(QueueRunnerTest, NoRunMetaDataTest) {
  GraphDef graph_def = BuildSimpleGraph();
  auto session = BuildSessionAndInitVariable(graph_def);

  QueueRunnerDef queue_runner_def = BuildQueueRunnerDef(
      kQueueName, {kCountUpToOpName}, kSquareOpName, "", {});
  std::unique_ptr<QueueRunner> qr;
  TF_EXPECT_OK(QueueRunner::New(queue_runner_def, &qr));
  TF_CHECK_OK(qr->Start(session.get()));

  TF_EXPECT_OK(qr->Join());
  CostGraphDef cost_graph;
  EXPECT_EQ(qr->ExportCostGraph(&cost_graph).code(),
            error::FAILED_PRECONDITION);
}

}  // namespace
}  // namespace tensorflow
