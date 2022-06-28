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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdirect_session_with_tracking_alloc_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdirect_session_with_tracking_alloc_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdirect_session_with_tracking_alloc_testDTcc() {
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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

TEST(DirectSessionWithTrackingAllocTest, CostModelTest) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Node* y_neg = test::graph::Unary(&graph, "Neg", y);
  y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  options.config.mutable_graph_options()->set_build_cost_model(1);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_min_graph_nodes(-1);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_pin_to_host_optimization(RewriterConfig::OFF);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<string> target_nodes = {y_neg->name()};
  std::vector<Tensor> outputs;
  const int64_t start_micros = Env::Default()->NowMicros();
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  const int64_t run_duration_micros =
      Env::Default()->NowMicros() - start_micros;
  TF_ASSERT_OK(s);

  DirectSession* ds = static_cast<DirectSession*>(session.get());
  int graph_cnt = 0;
  CostModelManager::CostModelMap cost_models;
  ds->ExportCostModels(&cost_models);
  for (auto& it : cost_models) {
    const Graph* g = (it).first;
    const CostModel* cm = (it).second;
    for (Node* node : g->nodes()) {
      if (node->name() == y->name() || node->name() == y_neg->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
        TensorShapeProto shape = cm->MaxMemoryShape(node, 0);
        EXPECT_EQ(2, shape.dim_size());
        EXPECT_EQ(2, shape.dim(0).size());
        EXPECT_EQ(1, shape.dim(1).size());
        // if MKL is used, it goes through additional
        // graph rewrite pass on top of Tensorflow.
        // In TF, every time a graph pass
        // happens, "constant" nodes are allocated
        // and deallocated. Each allocation calls the
        // (FindChunkPtr of BFCAllocator),
        // which increments the value of AllocationId.
        // Thus AllocationId of MKL can differ with TF if
        // someone changes the relevant codes in BFCAllocator.
        // Currently they are the same.
        if (node->name() == y->name()) {
          EXPECT_EQ(3, cm->AllocationId(node, 0));
        } else {
          EXPECT_EQ(4, cm->AllocationId(node, 0));
        }
      }
      EXPECT_LE(0, cm->MaxExecutionTime(node));
      EXPECT_GE(run_duration_micros, cm->MaxExecutionTime(node));
    }
    graph_cnt++;
  }
  // We should have 2 cost models since we have 2 cpu devices.
  ASSERT_EQ(2, graph_cnt);
}

TEST(DirectSessionWithTrackingAllocTest, CostModelWarmup) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_FLOAT, TensorShape({}));
  vx.scalar<float>()() = 1.0;
  Node* x = test::graph::Constant(&g, vx);

  int warmup_steps = 10;
  int measure_steps = 15;
  SessionOptions options;
  options.config.mutable_graph_options()->set_build_cost_model(1);
  options.config.mutable_graph_options()->set_build_cost_model_after(
      warmup_steps);
  std::unique_ptr<Session> session(NewSession(options));

  GraphDef def;
  test::graph::ToGraphDef(&g, &def);
  TF_ASSERT_OK(session->Create(def));
  std::vector<Tensor> outputs;

  for (int i = 0; i < warmup_steps + measure_steps; i++) {
    TF_EXPECT_OK(session->Run({}, {x->name() + ":0"}, {}, &outputs));
  }

  DirectSession* ds = static_cast<DirectSession*>(session.get());
  CostModelManager::CostModelMap cost_models;
  ds->ExportCostModels(&cost_models);
  CHECK_GE(cost_models.size(), 1);
  const CostModel* cm = (*cost_models.begin()).second;
  EXPECT_EQ(measure_steps, cm->GetUpdateTimes());
}

static void TestHWAccelerator(bool enableHWTrace) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdirect_session_with_tracking_alloc_testDTcc mht_0(mht_0_v, 336, "", "./tensorflow/core/common_runtime/direct_session_with_tracking_alloc_test.cc", "TestHWAccelerator");

  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");

  Node* y_neg = test::graph::Unary(&graph, "Neg", y);
  y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 1;
  (*options.config.mutable_device_count())["GPU"] = 1;
  options.config.set_allow_soft_placement(true);
  options.config.mutable_graph_options()->set_build_cost_model(1);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<string> target_nodes = {y_neg->name()};
  std::vector<Tensor> outputs;
  const int64_t start_micros = Env::Default()->NowMicros();

  RunOptions run_options;
  if (enableHWTrace) {
    run_options.set_trace_level(RunOptions::FULL_TRACE);
  }
  RunMetadata run_metadata;
  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  const int64_t run_duration_micros =
      Env::Default()->NowMicros() - start_micros;
  TF_ASSERT_OK(s);

  DirectSession* ds = static_cast<DirectSession*>(session.get());
  int graph_cnt = 0;
  CostModelManager::CostModelMap cost_models;
  ds->ExportCostModels(&cost_models);
  for (auto& it : cost_models) {
    const Graph* g = (it).first;
    const CostModel* cm = (it).second;
    for (Node* node : g->nodes()) {
      if (node->name() == y->name() || node->name() == y_neg->name()) {
        EXPECT_LE(8, cm->MaxMemorySize(node, 0));
        TensorShapeProto shape = cm->MaxMemoryShape(node, 0);
        EXPECT_EQ(2, shape.dim_size());
        EXPECT_EQ(2, shape.dim(0).size());
        EXPECT_EQ(1, shape.dim(1).size());
      }
      EXPECT_LE(0, cm->MaxExecutionTime(node));
      EXPECT_GE(run_duration_micros, cm->MaxExecutionTime(node));
    }
    graph_cnt++;
  }
  // We should have 2 cost models since we requested 1 cpu and 1 gpu. However
  // since the placement is soft, we might end up placing everything on cpu.
  ASSERT_GE(2, graph_cnt);
  ASSERT_LE(1, graph_cnt);
}

TEST(DirectSessionWithTrackingAllocTest, CostModelForAccelerator) {
  TestHWAccelerator(false);
}

TEST(DirectSessionWithTrackingAllocTest, CostModelWithHardwareStats) {
  TestHWAccelerator(true);
}

TEST(DirectSessionWithTrackingAllocTest, CostGraph) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Node* y_neg = test::graph::Unary(&graph, "Neg", y);
  y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  options.config.mutable_graph_options()->set_build_cost_model(1);
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  RunOptions run_options;
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<string> target_nodes = {y_neg->name()};
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  const int64_t start_micros = Env::Default()->NowMicros();
  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  const int64_t run_duration_micros =
      Env::Default()->NowMicros() - start_micros;
  TF_ASSERT_OK(s);

  EXPECT_LE(2, run_metadata.cost_graph().node_size());
  for (const auto& node : run_metadata.cost_graph().node()) {
    if (node.name() == y->name() || node.name() == y_neg->name()) {
      EXPECT_EQ(1, node.output_info_size());
      EXPECT_LE(8, node.output_info(0).size());
      const TensorShapeProto& shape = node.output_info(0).shape();
      EXPECT_EQ(2, shape.dim_size());
      EXPECT_EQ(2, shape.dim(0).size());
      EXPECT_EQ(1, shape.dim(1).size());
      const DataType& dtype = node.output_info(0).dtype();
      EXPECT_EQ(DT_FLOAT, dtype);
    }
    EXPECT_LE(0, node.compute_cost());
    EXPECT_GE(run_duration_micros, node.compute_cost());
  }
}

TEST(DirectSessionWithTrackingAllocTest, TrackMemoryAllocation) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {1, 1});
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  std::unique_ptr<Session> session(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;

  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  Status s = session->Run(run_options, inputs, output_names, {}, &outputs,
                          &run_metadata);
  TF_ASSERT_OK(s);

  for (const auto& dev_stat : run_metadata.step_stats().dev_stats()) {
    for (const auto& node_stat : dev_stat.node_stats()) {
      if (node_stat.node_name() == y->name()) {
        EXPECT_LT(0, node_stat.memory(0).total_bytes());
        EXPECT_LT(0, node_stat.memory(0).live_bytes());
        EXPECT_LT(0, node_stat.memory(0).peak_bytes());
      }
    }
  }
}
}  // namespace
}  // namespace tensorflow
