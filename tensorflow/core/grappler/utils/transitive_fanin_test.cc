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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStransitive_fanin_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStransitive_fanin_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStransitive_fanin_testDTcc() {
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

#include "tensorflow/core/grappler/utils/transitive_fanin.h"

#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class TransitiveFaninTest : public ::testing::Test {
 protected:
  struct NodeConfig {
    NodeConfig(string name, std::vector<string> inputs)
        : name(std::move(name)), inputs(std::move(inputs)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStransitive_fanin_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/grappler/utils/transitive_fanin_test.cc", "NodeConfig");
}
    NodeConfig(string name, string op, std::vector<string> inputs)
        : name(std::move(name)), op(std::move(op)), inputs(std::move(inputs)) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStransitive_fanin_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/grappler/utils/transitive_fanin_test.cc", "NodeConfig");
}

    string name;
    string op;
    std::vector<string> inputs;
  };

  static GraphDef CreateGraph(const std::vector<NodeConfig>& nodes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPStransitive_fanin_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/grappler/utils/transitive_fanin_test.cc", "CreateGraph");

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

TEST_F(TransitiveFaninTest, NoPruning) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}}      //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
}

TEST_F(TransitiveFaninTest, PruneNodesUnreachableFromSingleTerminalNode) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"1"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
  ASSERT_FALSE(node_map.NodeExists("5"));
}

TEST_F(TransitiveFaninTest, PruneNodesUnreachableFromMultipleTerminalNodes) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"2"}},  //
      {"6", {"1"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1", "5"};
  TF_EXPECT_OK(SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes));
  NodeMap node_map(&output_graph);
  ASSERT_TRUE(node_map.NodeExists("1"));
  ASSERT_TRUE(node_map.NodeExists("2"));
  ASSERT_TRUE(node_map.NodeExists("3"));
  ASSERT_TRUE(node_map.NodeExists("4"));
  ASSERT_TRUE(node_map.NodeExists("5"));
  ASSERT_FALSE(node_map.NodeExists("6"));
}

TEST_F(TransitiveFaninTest, InvalidGraphOrTerminalNodes) {
  GraphDef graph = CreateGraph({
      {"1", {"2"}},  //
      {"2", {"3"}},  //
      {"3", {"4"}},  //
      {"4", {}},     //
      {"5", {"6"}},  //
      {"7", {"8"}}   //
  });

  GraphDef output_graph;
  const std::vector<string> terminal_nodes = {"1", "5"};
  auto s = SetTransitiveFaninGraph(graph, &output_graph, terminal_nodes);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(), "Graph does not contain input 6 of node 5.");
  const std::vector<string> invalid_terminal_nodes = {"0", "1", "5"};
  s = SetTransitiveFaninGraph(graph, &output_graph, invalid_terminal_nodes);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.error_message(), "Graph does not contain terminal node 0.");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
