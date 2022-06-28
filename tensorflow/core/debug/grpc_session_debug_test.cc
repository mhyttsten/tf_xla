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
class MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"

#include <memory>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/debug.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {
namespace {

SessionOptions Devices(int num_cpus, int num_gpus) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "Devices");

  SessionOptions result;
  (*result.config.mutable_device_count())["CPU"] = num_cpus;
  (*result.config.mutable_device_count())["GPU"] = num_gpus;
  return result;
}

void CreateGraphDef(GraphDef* graph_def, string node_names[3]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "CreateGraphDef");

  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({1, 2}));
  test::FillValues<float>(&a_tensor, {1.0, 2.0});
  Node* a = test::graph::Constant(&graph, a_tensor);
  node_names[0] = a->name();

  Tensor b_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&b_tensor, {2.0, 1.0});
  Node* b = test::graph::Constant(&graph, b_tensor);
  node_names[1] = b->name();

  // c = a * b
  Node* c = test::graph::Matmul(&graph, a, b, false, false);
  node_names[2] = c->name();

  test::graph::ToGraphDef(&graph, graph_def);
}

// Asserts that "val" is a single float tensor. The only float is
// "expected_val".
void IsSingleFloatValue(const Tensor& val, float expected_val) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "IsSingleFloatValue");

  ASSERT_EQ(val.dtype(), DT_FLOAT);
  ASSERT_EQ(val.NumElements(), 1);
  ASSERT_EQ(val.flat<float>()(0), expected_val);
}

SessionOptions Options(const string& target, int placement_period) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "Options");

  SessionOptions options;
  // NOTE(mrry): GrpcSession requires a grpc:// scheme prefix in the target
  // string.
  options.target = strings::StrCat("grpc://", target);
  options.config.set_placement_period(placement_period);
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  return options;
}

std::unique_ptr<Session> NewRemote(const SessionOptions& options) {
  return std::unique_ptr<Session>(CHECK_NOTNULL(NewSession(options)));
}

class GrpcSessionDebugTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_4(mht_4_v, 283, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "SetUp");
 CreateDumpDir(); }

  void TearDown() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_5(mht_5_v, 288, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "TearDown");
 DeleteDumpDir(); }

  void DeleteDumpDir() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "DeleteDumpDir");

    if (Env::Default()->IsDirectory(dump_dir_).ok()) {
      int64_t undeleted_files = 0;
      int64_t undeleted_dirs = 0;
      ASSERT_TRUE(
          Env::Default()
              ->DeleteRecursively(dump_dir_, &undeleted_files, &undeleted_dirs)
              .ok());
      ASSERT_EQ(0, undeleted_files);
      ASSERT_EQ(0, undeleted_dirs);
    }
  }

  const string GetDebugURL() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_7(mht_7_v, 309, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "GetDebugURL");
 return debug_url_; }

  void LoadTensorDumps(const string& subdir, std::vector<Tensor>* tensors) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("subdir: \"" + subdir + "\"");
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_8(mht_8_v, 315, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "LoadTensorDumps");

    const string dirpath = io::JoinPath(dump_dir_, subdir);
    if (!(Env::Default()->IsDirectory(dirpath).ok())) {
      return;
    }

    std::vector<string> filenames;
    TF_ASSERT_OK(Env::Default()->GetChildren(dirpath, &filenames));

    for (const string& filename : filenames) {
      Event event;
      TF_ASSERT_OK(ReadEventFromFile(io::JoinPath(dirpath, filename), &event));
      if (event.summary().value().size() == 1) {
        Tensor tensor;
        ASSERT_TRUE(tensor.FromProto(event.summary().value(0).tensor()));
        tensors->push_back(tensor);
      }
    }
  }

 private:
  void CreateDumpDir() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_9(mht_9_v, 339, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "CreateDumpDir");

    char dir_template[] = "/tmp/tfdbg_grpc_sessions_XXXXXX";
    dump_dir_ = mkdtemp(dir_template);
    debug_url_ = strings::StrCat("file://", dump_dir_);
  }

  string dump_dir_;
  string debug_url_;
};

TEST_F(GrpcSessionDebugTest, FileDebugURL) {
  GraphDef graph;
  string node_names[3];
  CreateGraphDef(&graph, node_names);

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  auto session = NewRemote(Options(cluster->targets()[0], 1));
  TF_CHECK_OK(session->Create(graph));

  // Iteration 0: No watch.
  // Iterations 1 and 2: Watch one Tensor.
  // Iterations 3 and 4: Watch two Tensors.
  // Iteration 5: No watch.
  for (size_t i = 0; i < 6; ++i) {
    RunOptions options;
    if (i >= 1 && i < 5) {
      DebugOptions* debug_options = options.mutable_debug_options();
      DebugTensorWatch* watch = debug_options->add_debug_tensor_watch_opts();
      watch->set_node_name(node_names[0]);
      watch->set_output_slot(0);
      watch->add_debug_ops("DebugIdentity");
      watch->add_debug_urls(GetDebugURL());

      if (i >= 3) {
        watch = debug_options->add_debug_tensor_watch_opts();
        watch->set_node_name(node_names[1]);
        watch->set_output_slot(0);
        watch->add_debug_ops("DebugIdentity");
        watch->add_debug_urls(GetDebugURL());
      }
    }

    RunMetadata metadata;
    std::vector<Tensor> outputs;
    TF_CHECK_OK(
        session->Run(options, {}, {node_names[2]}, {}, &outputs, &metadata));
    ASSERT_EQ(1, outputs.size());
    IsSingleFloatValue(outputs[0], 4.0);

    std::vector<Tensor> dumped_tensors;
    LoadTensorDumps(io::JoinPath(DebugNodeKey::DeviceNameToDevicePath(
                                     cluster->devices()[0].name()),
                                 "n"),
                    &dumped_tensors);

    if (i == 0 || i == 5) {
      ASSERT_EQ(0, dumped_tensors.size());
    } else {
      if (i == 1 || i == 2) {
        ASSERT_EQ(1, dumped_tensors.size());
        ASSERT_EQ(TensorShape({1, 2}), dumped_tensors[0].shape());
        ASSERT_EQ(1.0, dumped_tensors[0].flat<float>()(0));
        ASSERT_EQ(2.0, dumped_tensors[0].flat<float>()(1));
      } else {
        ASSERT_EQ(2, dumped_tensors.size());
      }
      DeleteDumpDir();
    }
  }
  TF_CHECK_OK(session->Close());
}

void SetDevice(GraphDef* graph, const string& name, const string& dev) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   mht_10_v.push_back("dev: \"" + dev + "\"");
   MHTracer_DTPStensorflowPScorePSdebugPSgrpc_session_debug_testDTcc mht_10(mht_10_v, 418, "", "./tensorflow/core/debug/grpc_session_debug_test.cc", "SetDevice");

  for (size_t i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).name() == name) {
      graph->mutable_node(i)->set_device(dev);
      return;
    }
  }
  LOG(FATAL) << "Name '" << name << "' not found.";
}

TEST_F(GrpcSessionDebugTest, MultiDevices_String) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 1), 2, &cluster));
  auto session = NewRemote(Options(cluster->targets()[0], 1000));

  // b = a
  Graph graph(OpRegistry::Global());
  Tensor a_tensor(DT_STRING, TensorShape({2, 2}));
  for (size_t i = 0; i < 4; ++i) {
    a_tensor.flat<tstring>()(i) = "hello, world";
  }
  Node* a = test::graph::Constant(&graph, a_tensor);
  Node* b = test::graph::Identity(&graph, a);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  // In this test, we force each node (a, b) on every possible device.
  // We test all possible cases.
  for (const auto& a_dev : cluster->devices()) {
    for (const auto& b_dev : cluster->devices()) {
      LOG(INFO) << "a: " << a_dev.name() << " b: " << b_dev.name();
      SetDevice(&def, a->name(), a_dev.name());
      SetDevice(&def, b->name(), b_dev.name());

      Status s = session->Create(def);
      if (s.ok()) {
        std::vector<Tensor> outputs;

        RunOptions options;
        DebugOptions* debug_options = options.mutable_debug_options();
        DebugTensorWatch* watch = debug_options->add_debug_tensor_watch_opts();
        watch->set_node_name(a->name());
        watch->set_output_slot(0);
        watch->add_debug_ops("DebugIdentity");
        watch->add_debug_urls(GetDebugURL());

        RunMetadata metadata;
        TF_CHECK_OK(
            session->Run(options, {}, {b->name()}, {}, &outputs, &metadata));
        ASSERT_EQ(1, outputs.size());
        ASSERT_EQ(outputs[0].dtype(), DT_STRING);
        ASSERT_EQ(outputs[0].NumElements(), 4);
        for (size_t i = 0; i < outputs[0].NumElements(); ++i) {
          EXPECT_EQ(outputs[0].flat<tstring>()(i), "hello, world");
        }
        TF_CHECK_OK(session->Close());

        std::vector<Tensor> dumped_tensors;
        LoadTensorDumps(
            io::JoinPath(DebugNodeKey::DeviceNameToDevicePath(a_dev.name()),
                         "n"),
            &dumped_tensors);
        ASSERT_EQ(1, dumped_tensors.size());
        ASSERT_EQ(TensorShape({2, 2}), dumped_tensors[0].shape());
        for (size_t i = 0; i < 4; ++i) {
          ASSERT_EQ("hello, world", dumped_tensors[0].flat<tstring>()(i));
        }

        DeleteDumpDir();
      } else {
        // The CUDA device does not have an Identity op for strings
        LOG(ERROR) << "Error: " << s;
        ASSERT_TRUE((a_dev.device_type() == DEVICE_GPU) ||
                    (b_dev.device_type() == DEVICE_GPU));
        ASSERT_FALSE(s.ok());
      }
    }
  }
}

}  // namespace
}  // namespace tensorflow
