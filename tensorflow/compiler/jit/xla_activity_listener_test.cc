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
class MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_activity_listener.h"

#include <cstdlib>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestListener : public XlaActivityListener {
 public:
  Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "Listen");

    auto_clustering_activity_ = auto_clustering_activity;
    return Status::OK();
  }

  Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "Listen");

    jit_compilation_activity_ = jit_compilation_activity;
    return Status::OK();
  }

  Status Listen(const XlaOptimizationRemark& optimization_remark) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "Listen");

    return Status::OK();
  }

  ~TestListener() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_3(mht_3_v, 228, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "~TestListener");
}

  const XlaAutoClusteringActivity& auto_clustering_activity() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_4(mht_4_v, 233, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "auto_clustering_activity");

    return auto_clustering_activity_;
  }
  const XlaJitCompilationActivity& jit_compilation_activity() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_5(mht_5_v, 239, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "jit_compilation_activity");

    return jit_compilation_activity_;
  }

 private:
  XlaAutoClusteringActivity auto_clustering_activity_;
  XlaJitCompilationActivity jit_compilation_activity_;
};

class XlaActivityListenerTest : public ::testing::Test {
 protected:
  XlaActivityListenerTest() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_6(mht_6_v, 253, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "XlaActivityListenerTest");

    auto listener = absl::make_unique<TestListener>();
    listener_ = listener.get();
    RegisterXlaActivityListener(std::move(listener));
  }

  TestListener* listener() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_7(mht_7_v, 262, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "listener");
 return listener_; }

 private:
  TestListener* listener_;
};

GraphDef CreateGraphDef() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_8(mht_8_v, 271, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "CreateGraphDef");

  Scope root = Scope::NewRootScope().ExitOnError().WithAssignedDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);
  for (int i = 0; i < 5; i++) {
    a = ops::MatMul(root.WithOpName(absl::StrCat("matmul_", i)), a, a);
    a = ops::Add(root.WithOpName(absl::StrCat("add_", i)), a, a);
  }

  GraphDef graph_def;
  root.graph()->ToGraphDef(&graph_def);
  return graph_def;
}

TEST_F(XlaActivityListenerTest, Test) {
  GraphDef graph_def = CreateGraphDef();
  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  std::unique_ptr<Session> session(NewSession(options));

  TF_ASSERT_OK(session->Create(graph_def));

  std::vector<std::string> output_names = {std::string("add_4:0")};

  Tensor tensor_2x2(DT_FLOAT, TensorShape({2, 2}));
  for (int i = 0; i < 4; i++) {
    tensor_2x2.matrix<float>()(i / 2, i % 2) = 5 * i;
  }

  Tensor tensor_3x3(DT_FLOAT, TensorShape({3, 3}));
  for (int i = 0; i < 9; i++) {
    tensor_3x3.matrix<float>()(i / 3, i % 3) = 5 * i;
  }

  std::vector<std::pair<string, Tensor>> inputs_2x2 = {{"A", tensor_2x2}};

  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(inputs_2x2, output_names, /*target_node_names=*/{},
                            &outputs));

  XlaAutoClusteringActivity expected_auto_clustering_activity;
  protobuf::TextFormat::ParseFromString(
      R"(global_jit_level: ON_2
cpu_global_jit_enabled: true
summary {
  unclustered_node_count: 4
  clustered_node_count: 14
  clusters {
    name: "cluster_0"
    size: 14
    op_histogram {
      op: "Add"
      count: 1
    }
    op_histogram {
      op: "Const"
      count: 4
    }
    op_histogram {
      op: "MatMul"
      count: 5
    }
    op_histogram {
      op: "Mul"
      count: 4
    }
  }
  unclustered_op_histogram {
    op: "NoOp"
    count: 2
  }
  unclustered_op_histogram {
    op: "_Arg"
    count: 1
  }
  unclustered_op_histogram {
    op: "_Retval"
    count: 1
  }
}
)",
      &expected_auto_clustering_activity);
  EXPECT_EQ(listener()->auto_clustering_activity().DebugString(),
            expected_auto_clustering_activity.DebugString());

  EXPECT_EQ(listener()->jit_compilation_activity().cluster_name(), "cluster_0");
  EXPECT_EQ(listener()->jit_compilation_activity().compile_count(), 1);

  int64_t first_compile_time =
      listener()->jit_compilation_activity().compile_time_us();
  EXPECT_GT(first_compile_time, 0);
  EXPECT_EQ(listener()->jit_compilation_activity().cumulative_compile_time_us(),
            first_compile_time);

  std::vector<std::pair<string, Tensor>> inputs_3x3 = {{"A", tensor_3x3}};

  outputs.clear();
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK(session->Run(inputs_3x3, output_names,
                              /*target_node_names=*/{}, &outputs));
  }

  EXPECT_EQ(listener()->jit_compilation_activity().cluster_name(), "cluster_0");
  EXPECT_EQ(listener()->jit_compilation_activity().compile_count(), 2);

  EXPECT_GT(listener()->jit_compilation_activity().compile_time_us(), 0);
  EXPECT_EQ(listener()->jit_compilation_activity().cumulative_compile_time_us(),
            first_compile_time +
                listener()->jit_compilation_activity().compile_time_us());
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSxla_activity_listener_testDTcc mht_9(mht_9_v, 390, "", "./tensorflow/compiler/jit/xla_activity_listener_test.cc", "main");

  tensorflow::GetMarkForCompilationPassFlags()->tf_xla_cpu_global_jit = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
