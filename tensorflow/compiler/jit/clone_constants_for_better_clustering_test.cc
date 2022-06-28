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
class MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clustering_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clustering_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clustering_testDTcc() {
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

#include "tensorflow/compiler/jit/clone_constants_for_better_clustering.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
using ::tensorflow::testing::FindNodeByName;

Status CloneConstantsForBetterClustering(const Scope& s,
                                         std::unique_ptr<Graph>* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSclone_constants_for_better_clustering_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/compiler/jit/clone_constants_for_better_clustering_test.cc", "CloneConstantsForBetterClustering");

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.session_options = &session_options;

  // Scope::ToGraph seems to drop assigned devices, probably because it goes
  // through a GraphDef.  So explicitly maintain the device assignment.
  // std::unordered_map<string, string> assigned_device_names;
  // for (Node* n : s.graph()->nodes()) {
  //   assigned_device_names[n->name()] = n->assigned_device_name();
  // }
  GraphConstructorOptions opts;
  opts.expect_device_spec = true;
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get(), opts));
  // for (Node* n : graph->nodes()) {
  //   n->set_assigned_device_name(assigned_device_names[n->name()]);
  // }

  CloneConstantsForBetterClusteringPass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));
  *result = std::move(graph);
  return Status::OK();
}

const char* kCPU = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU = "/job:localhost/replica:0/task:0/device:GPU:0";

TEST(CloneConstantsForBetterClusteringTest, HostConstantPlacedOnCpu) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);
  Scope on_cpu = root.WithAssignedDevice(kCPU).WithDevice(kCPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(on_cpu.WithOpName("perm"), {3, 1, 2, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_NE(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, HostConstantPlacedOnGpu) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(on_gpu.WithOpName("perm"), {3, 1, 2, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_NE(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, DontCloneNonHostConstants) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm_f32 = ops::Const(on_gpu.WithOpName("perm"), {3.0, 1.0, 2.0, 0.0});
  Output perm_int0 =
      ops::Cast(on_gpu.WithOpName("perm_cast_0"), perm_f32, DT_INT32);
  Output perm_int1 =
      ops::Cast(on_gpu.WithOpName("perm_cast_1"), perm_f32, DT_INT32);

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm_int0);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm_int1);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(
      FindNodeByName(result.get(), "perm_cast_0")->input_tensor(0, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(
      FindNodeByName(result.get(), "perm_cast_1")->input_tensor(0, &tr1_perm));

  EXPECT_EQ(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, DontCloneLargeConstants) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);
  Scope on_cpu = root.WithAssignedDevice(kCPU).WithDevice(kCPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(
      on_cpu.WithOpName("perm"),
      {17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_EQ(tr0_perm.node, tr1_perm.node);
}

TEST(CloneConstantsForBetterClusteringTest, InplaceOps) {
  Scope root = Scope::NewRootScope().ExitOnError();
  Scope on_gpu = root.WithAssignedDevice(kGPU).WithDevice(kGPU);
  Scope on_cpu = root.WithAssignedDevice(kCPU).WithDevice(kCPU);

  Output in0 = ops::Placeholder(on_gpu.WithOpName("in0"), DT_FLOAT);
  Output in1 = ops::Placeholder(on_gpu.WithOpName("in1"), DT_FLOAT);

  Output perm = ops::Const(on_cpu.WithOpName("perm"), {3, 1, 2, 0});

  {
    Output tr0 = ops::Transpose(on_gpu.WithOpName("tr0"), in0, perm);
    Output tr1 = ops::Transpose(on_gpu.WithOpName("tr1"), in1, perm);
  }

  Output in_place_add =
      ops::InplaceAdd(on_cpu.WithOpName("tr0"), perm,
                      ops::Placeholder(on_cpu.WithOpName("i"), DT_INT32), perm);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(CloneConstantsForBetterClustering(root, &result));

  OutputTensor tr0_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr0")->input_tensor(1, &tr0_perm));

  OutputTensor tr1_perm;
  TF_ASSERT_OK(FindNodeByName(result.get(), "tr1")->input_tensor(1, &tr1_perm));

  EXPECT_EQ(tr0_perm.node, tr1_perm.node);
}
}  // namespace
}  // namespace tensorflow
