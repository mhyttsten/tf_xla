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
class MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondef_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondef_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondef_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/graph_to_functiondef.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {
namespace {

FunctionDef RemoveDebugInfo(const FunctionDef& def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondef_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/framework/graph_to_functiondef_test.cc", "RemoveDebugInfo");

  FunctionDef copy = def;
  for (auto& node_def : *copy.mutable_node_def()) {
    node_def.clear_experimental_debug_info();
  }
  return copy;
}

bool EqualFunctionDef(const FunctionDef& a, const FunctionDef& b,
                      string* diff) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSgraph_to_functiondef_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/framework/graph_to_functiondef_test.cc", "EqualFunctionDef");

  // TODO(phawkins) use a more sophisticated equality test.
  if (a.DebugString() != b.DebugString()) {
    if (diff) {
      *diff = strings::StrCat("Definition mismatch for function ",
                              a.signature().name(), ":\n", a.DebugString(),
                              "\n ---- vs. ----\n", b.DebugString());
    }
    return false;
  }
  return true;
}

TEST(GraphToFunctionDefTest, Basics) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  auto b = ops::_Arg(root.WithOpName("B"), DT_FLOAT, 1);
  auto c = ops::_Arg(root.WithOpName("C"), DT_FLOAT, 2);
  auto d = ops::Add(root.WithOpName("D"), a, b);
  auto e = ops::Add(root.WithOpName("b"), d, c);
  auto f = ops::Neg(root.WithOpName("h"), e);
  auto g = ops::AddN(root.WithOpName("G"), std::initializer_list<Output>{e, f});
  auto h = ops::_Retval(root.WithOpName("H"), g, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", &fdef));

  FunctionDef fdef_expected = FunctionDefHelper::Create(
      "test_fn",                             // function name
      {"a: float", "b: float", "c: float"},  // inputs
      {"h: float"},                          // outputs
      {},                                    // attrs
      {
          // nodes in the function body
          {{"D"}, "Add", {"a", "b"}, {{"T", DT_FLOAT}}},
          {{"b_0"}, "Add", {"D:z:0", "c"}, {{"T", DT_FLOAT}}},
          {{"h_0"}, "Neg", {"b_0:z:0"}, {{"T", DT_FLOAT}}},
          {{"G"}, "AddN", {"b_0:z:0", "h_0:y:0"}, {{"N", 2}, {"T", DT_FLOAT}}},
      },
      {{"h", "G:sum:0"}});  // return values

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);
  EXPECT_TRUE(fdefs_equal) << diff;
}

// Regression test for a crash if there was a control edge to a _Retval node.
TEST(GraphToFunctionDefTest, ControlDependencies) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("a"), DT_FLOAT, 0);
  auto b = ops::Neg(root.WithOpName("b").WithControlDependencies(a), a);
  auto c = ops::_Retval(root.WithOpName("c").WithControlDependencies(b), b, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", &fdef));

  FunctionDef fdef_expected = FunctionDefHelper::Create(
      "test_fn",     // function name
      {"a: float"},  // inputs
      {"c: float"},  // outputs
      {},            // attrs
      {
          // nodes in the function body
          {{"b"}, "Neg", {"a", "^a"}, {{"T", DT_FLOAT}}},
      },
      {{"c", "b:y:0"}});  // return values

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);
  EXPECT_TRUE(fdefs_equal) << diff;
}

TEST(GraphToFunctionDefTest, ControlOutputs) {
  Scope root = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(root.WithOpName("a"), DT_FLOAT, 0);
  auto b = ops::Neg(root.WithOpName("b"), a);
  auto c = ops::_Retval(root.WithOpName("c"), b, 0);

  GraphDef graph_def;
  TF_EXPECT_OK(root.ToGraphDef(&graph_def));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphConstructorOptions options;
  TF_EXPECT_OK(ConvertGraphDefToGraph(options, graph_def, graph.get()));

  // Add a 'b' node to the control return set.
  const auto control_ret = [](const Node* n) -> absl::optional<string> {
    if (n->name() == "b") return absl::make_optional<string>("must_execute");
    return absl::nullopt;
  };

  FunctionDef fdef;
  TF_EXPECT_OK(GraphToFunctionDef(*graph, "test_fn", control_ret, &fdef));

  FunctionDef fdef_expected =
      FunctionDefHelper::Create("test_fn",     // function name
                                {"a: float"},  // inputs
                                {"c: float"},  // outputs
                                {},            // attrs
                                {
                                    // nodes in the function body
                                    {{"b"}, "Neg", {"a"}, {{"T", DT_FLOAT}}},
                                },
                                {{"c", "b:y:0"}},          // return values
                                {{"must_execute", "b"}});  // control returns

  string diff;
  bool fdefs_equal =
      EqualFunctionDef(fdef_expected, RemoveDebugInfo(fdef), &diff);
  EXPECT_TRUE(fdefs_equal) << diff;
}

}  // namespace
}  // namespace tensorflow
