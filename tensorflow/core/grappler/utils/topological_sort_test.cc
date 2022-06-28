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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc() {
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

#include "tensorflow/core/grappler/utils/topological_sort.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace grappler {

class TopologicalSortTest : public ::testing::Test {
 protected:
  struct NodeConfig {
    NodeConfig(string name, std::vector<string> inputs)
        : name(std::move(name)), inputs(std::move(inputs)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/utils/topological_sort_test.cc", "NodeConfig");
}
    NodeConfig(string name, string op, std::vector<string> inputs)
        : name(std::move(name)), op(std::move(op)), inputs(std::move(inputs)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/grappler/utils/topological_sort_test.cc", "NodeConfig");
}

    string name;
    string op;
    std::vector<string> inputs;
  };

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/grappler/utils/topological_sort_test.cc", "CreateGraph");

    GraphDef graph;

    for (const NodeConfig& node : nodes) {
      NodeDef node_def;
      node_def.set_name(node.name);
      node_def.set_op(node.op);
      for (const string& input : node.inputs) {
        node_def.add_input(input);
      }
      *graph.add_node() = std::move(node_def);
    }

    return graph;
  }
};

TEST_F(TopologicalSortTest, NoLoop) {
  GraphDef graph = CreateGraph({
      {"2", {"5"}},       //
      {"0", {"5", "4"}},  //
      {"1", {"4", "3"}},  //
      {"3", {"2"}},       //
      {"5", {}},          //
      {"4", {}}           //
  });

  std::vector<const NodeDef*> topo_order;
  TF_EXPECT_OK(ComputeTopologicalOrder(graph, &topo_order));

  const std::vector<string> order = {"5", "4", "2", "0", "3", "1"};

  ASSERT_EQ(topo_order.size(), order.size());
  for (int i = 0; i < topo_order.size(); ++i) {
    const NodeDef* node = topo_order[i];
    EXPECT_EQ(node->name(), order[i]);
  }

  TF_EXPECT_OK(TopologicalSort(&graph));
  for (int i = 0; i < topo_order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, WithLoop) {
  GraphDef graph = CreateGraph({
      // Graph with a loop.
      {"2", "Merge", {"1", "5"}},     //
      {"3", "Switch", {"2"}},         //
      {"4", "Identity", {"3"}},       //
      {"5", "NextIteration", {"4"}},  //
      {"1", {}}                       //
  });

  std::vector<const NodeDef*> topo_order;
  TF_EXPECT_OK(ComputeTopologicalOrder(graph, &topo_order));

  const std::vector<string> order = {"1", "2", "3", "4", "5"};

  ASSERT_EQ(topo_order.size(), order.size());
  for (int i = 0; i < topo_order.size(); ++i) {
    const NodeDef* node = topo_order[i];
    EXPECT_EQ(node->name(), order[i]);
  }

  TF_EXPECT_OK(TopologicalSort(&graph));
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, WithIllegalLoop) {
  // A loop without Merge and NextIteration is illegal and the original node
  // order and graph will be preserved.
  GraphDef graph = CreateGraph({
      {"2", {"1", "3"}},  //
      {"3", {"2"}},       //
      {"1", {}}           //
  });

  EXPECT_FALSE(TopologicalSort(&graph).ok());
  std::vector<string> order = {"2", "3", "1"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, DuplicatedInputs) {
  GraphDef graph = CreateGraph({
      {"2", {"1", "1"}},  //
      {"1", {}}           //
  });

  TF_EXPECT_OK(TopologicalSort(&graph));
  std::vector<string> order = {"1", "2"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, Idempotent) {
  GraphDef graph = CreateGraph({
      {"1", {}},          //
      {"2", {}},          //
      {"3", {"1", "2"}},  //
      {"4", {"1", "3"}},  //
      {"5", {"2", "3"}}   //
  });

  TF_EXPECT_OK(TopologicalSort(&graph));
  std::vector<string> order = {"1", "2", "3", "4", "5"};
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }

  // Run topo sort again to verify that it is idempotent.
  TF_EXPECT_OK(TopologicalSort(&graph));
  for (int i = 0; i < order.size(); i++) {
    EXPECT_EQ(graph.node(i).name(), order[i]);
  }
}

TEST_F(TopologicalSortTest, ExtraDependencies) {
  GraphDef graph = CreateGraph({
      {"2", {"5"}},       //
      {"0", {"5", "4"}},  //
      {"1", {"4", "3"}},  //
      {"3", {"2"}},       //
      {"5", {}},          //
      {"4", {}}           //
  });

  // Add an edge from 4 to 5.
  std::vector<TopologicalDependency> extra_dependencies;
  extra_dependencies.push_back({&graph.node(5), &graph.node(4)});

  std::vector<const NodeDef*> topo_order;
  TF_EXPECT_OK(ComputeTopologicalOrder(graph, extra_dependencies, &topo_order));

  const std::vector<string> valid_order_1 = {"4", "5", "2", "0", "3", "1"};
  const std::vector<string> valid_order_2 = {"4", "5", "0", "2", "3", "1"};

  ASSERT_EQ(topo_order.size(), valid_order_1.size());

  std::vector<string> computed_order(6, "");
  for (int i = 0; i < topo_order.size(); ++i) {
    const NodeDef* node = topo_order[i];
    computed_order[i] = node->name();
  }
  EXPECT_TRUE(computed_order == valid_order_1 ||
              computed_order == valid_order_2);

  // Add an edge from `0` to `4`. This will create a loop.
  extra_dependencies.push_back({&graph.node(1), &graph.node(5)});
  EXPECT_FALSE(
      ComputeTopologicalOrder(graph, extra_dependencies, &topo_order).ok());
}

static void BM_ComputeTopologicalOrder(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStopological_sort_testDTcc mht_3(mht_3_v, 381, "", "./tensorflow/core/grappler/utils/topological_sort_test.cc", "BM_ComputeTopologicalOrder");

  const int size = state.range(0);

  GraphDef graph = test::CreateRandomGraph(size);

  std::vector<const NodeDef*> topo_order;
  for (auto s : state) {
    topo_order.clear();
    Status st = ComputeTopologicalOrder(graph, &topo_order);
    CHECK(st.ok()) << "Failed to compute topological order";
  }
}
BENCHMARK(BM_ComputeTopologicalOrder)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(25000)
    ->Arg(50000)
    ->Arg(100000);

}  // namespace grappler
}  // namespace tensorflow
