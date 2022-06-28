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
class MHTracer_DTPStensorflowPScorePStransformsPSgraph_transform_wrapper_testDTcc {
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
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_transform_wrapper_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStransformsPSgraph_transform_wrapper_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/transforms/graph_transform_wrapper.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace {

// Testing pass that deletes a single op from the Graph. This assumes the
// graph created below.
struct TestPass : public PassWrapper<TestPass, OperationPass<ModuleOp>> {
  TestPass() = default;
  StringRef getArgument() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_transform_wrapper_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/transforms/graph_transform_wrapper_test.cc", "getArgument");
 return "test"; }
  void runOnOperation() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_transform_wrapper_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/transforms/graph_transform_wrapper_test.cc", "runOnOperation");

    Operation* del;
    getOperation()->walk([&](Operation* op) {
      if (op->getName().getStringRef() != "tfg.TestInput") return;
      del = *op->getResult(0).getUsers().begin();
    });
    del->erase();
  }
};

}  // namespace
}  // namespace mlir

REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestRelu").Input("i: float").Output("o: float");
REGISTER_OP("NoOp");

TEST(GraphTransformWrapper, ReplacedGraph) {
  tensorflow::Graph graph(tensorflow::OpRegistry::Global());
  {
    tensorflow::GraphDefBuilder b(
        tensorflow::GraphDefBuilder::kFailImmediately);
    tensorflow::Node* input =
        tensorflow::ops::SourceOp("TestInput", b.opts().WithName("in"));
    tensorflow::ops::UnaryOp("TestRelu", tensorflow::ops::NodeOut(input, 0),
                             b.opts().WithName("n1"));
    tensorflow::ops::UnaryOp("TestRelu", tensorflow::ops::NodeOut(input, 1),
                             b.opts().WithName("n2"));
    TF_EXPECT_OK(tensorflow::GraphDefBuilderToGraph(b, &graph));
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::tfg::TFGraphDialect>();

  auto create_pass = [&]() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStransformsPSgraph_transform_wrapper_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/transforms/graph_transform_wrapper_test.cc", "lambda");
 return std::make_unique<mlir::TestPass>(); };

  TF_QCHECK_OK(mlir::tfg::RunTransformOnGraph(&graph, {create_pass}));

  EXPECT_EQ(4, graph.num_nodes());
  EXPECT_TRUE(
      absl::StrContains(graph.ToGraphDefDebug().ShortDebugString(), "\"n2\""));
}
