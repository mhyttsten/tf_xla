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
class MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc() {
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

#include "tensorflow/core/graph/subgraph.h"

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// TODO(josh11b): Test setting the "device" field of a NodeDef.
// TODO(josh11b): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

class SubgraphTest : public ::testing::Test {
 protected:
  SubgraphTest() : g_(new Graph(OpRegistry::Global())) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/graph/subgraph_test.cc", "SubgraphTest");

    device_info_.set_name("/job:a/replica:0/task:0/cpu:0");
    device_info_.set_device_type(DeviceType(DEVICE_CPU).type());
    device_info_.set_incarnation(0);
  }

  ~SubgraphTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/graph/subgraph_test.cc", "~SubgraphTest");
}

  void ExpectOK(const string& gdef_ascii) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("gdef_ascii: \"" + gdef_ascii + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/graph/subgraph_test.cc", "ExpectOK");

    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &gdef_));
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef_, g_.get()));
  }

  Node* FindNode(const string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/core/graph/subgraph_test.cc", "FindNode");

    for (Node* n : g_->nodes()) {
      if (n->name() == name) return n;
    }
    return nullptr;
  }

  bool HasNode(const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_4(mht_4_v, 249, "", "./tensorflow/core/graph/subgraph_test.cc", "HasNode");
 return FindNode(name) != nullptr; }

  void ExpectNodes(const string& nodes) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("nodes: \"" + nodes + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/graph/subgraph_test.cc", "ExpectNodes");

    int count = 0;
    std::vector<string> actual_nodes;
    for (Node* n : g_->nodes()) {
      if (n->IsOp()) {
        count++;
        actual_nodes.push_back(n->name());
      }
    }
    std::sort(actual_nodes.begin(), actual_nodes.end());

    LOG(INFO) << "Nodes present: " << absl::StrJoin(actual_nodes, " ");

    std::vector<string> expected_nodes = str_util::Split(nodes, ',');
    std::sort(expected_nodes.begin(), expected_nodes.end());
    for (const string& s : expected_nodes) {
      Node* n = FindNode(s);
      EXPECT_TRUE(n != nullptr) << s;
      if (n->type_string() == "_Send" || n->type_string() == "_Recv") {
        EXPECT_EQ(device_info_.name(), n->assigned_device_name()) << s;
      }
    }

    EXPECT_TRUE(actual_nodes.size() == expected_nodes.size())
        << "\nActual:   " << absl::StrJoin(actual_nodes, ",")
        << "\nExpected: " << absl::StrJoin(expected_nodes, ",");
  }

  bool HasEdge(const string& src, int src_out, const string& dst, int dst_in) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("src: \"" + src + "\"");
   mht_6_v.push_back("dst: \"" + dst + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_6(mht_6_v, 288, "", "./tensorflow/core/graph/subgraph_test.cc", "HasEdge");

    for (const Edge* e : g_->edges()) {
      if (e->src()->name() == src && e->src_output() == src_out &&
          e->dst()->name() == dst && e->dst_input() == dst_in)
        return true;
    }
    return false;
  }
  bool HasControlEdge(const string& src, const string& dst) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("src: \"" + src + "\"");
   mht_7_v.push_back("dst: \"" + dst + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_7(mht_7_v, 301, "", "./tensorflow/core/graph/subgraph_test.cc", "HasControlEdge");

    return HasEdge(src, Graph::kControlSlot, dst, Graph::kControlSlot);
  }

  string Subgraph(const string& fed_str, const string& fetch_str,
                  const string& targets_str,
                  bool use_function_convention = false) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("fed_str: \"" + fed_str + "\"");
   mht_8_v.push_back("fetch_str: \"" + fetch_str + "\"");
   mht_8_v.push_back("targets_str: \"" + targets_str + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_8(mht_8_v, 313, "", "./tensorflow/core/graph/subgraph_test.cc", "Subgraph");

    Graph* subgraph = new Graph(OpRegistry::Global());
    CopyGraph(*g_, subgraph);
    std::vector<string> fed =
        str_util::Split(fed_str, ',', str_util::SkipEmpty());
    std::vector<string> fetch =
        str_util::Split(fetch_str, ',', str_util::SkipEmpty());
    std::vector<string> targets =
        str_util::Split(targets_str, ',', str_util::SkipEmpty());

    subgraph::RewriteGraphMetadata metadata;
    Status s = subgraph::RewriteGraphForExecution(
        subgraph, fed, fetch, targets, device_info_, use_function_convention,
        &metadata);
    if (!s.ok()) {
      delete subgraph;
      return s.ToString();
    }

    EXPECT_EQ(fed.size(), metadata.feed_types.size());
    EXPECT_EQ(fetch.size(), metadata.fetch_types.size());

    // Replace the graph with the subgraph for the rest of the display program
    g_.reset(subgraph);
    return "OK";
  }

  Graph* graph() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_9(mht_9_v, 343, "", "./tensorflow/core/graph/subgraph_test.cc", "graph");
 return g_.get(); }

 private:
  GraphDef gdef_;
  std::unique_ptr<Graph> g_;
  DeviceAttributes device_info_;
};

REGISTER_OP("TestParams").Output("o: float");
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestRelu").Input("i: float").Output("o: float");
REGISTER_OP("TestMul").Input("a: float").Input("b: float").Output("o: float");

TEST_F(SubgraphTest, Targets1) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "", "t1"));
  ExpectNodes("W1,input,t1");
}

TEST_F(SubgraphTest, Targets2) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: 'W1' input: 'input:1' }"
      "node { name: 't2' op: 'TestMul' input: 'W2' input: 't1' }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "", "t2,t3_a"));
  ExpectNodes("W1,W2,input,t1,t2,t3_a");
}

TEST_F(SubgraphTest, FedOutputs1) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("input:1", "", "t2"));
  ExpectNodes("W1,W2,_recv_input_1,t1,t2");
}

TEST_F(SubgraphTest, FedOutputs1_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK",
            Subgraph("input:1", "", "t2", true /* use_function_convention */));
  ExpectNodes("W1,W2,_arg_input_1_0,t1,t2");
}

TEST_F(SubgraphTest, FedRefNode) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W2', 'W1' ] }");
  EXPECT_EQ("OK", Subgraph("W1:0", "", "t1"));
  ExpectNodes("_recv_W1_0,W2,t1");
  Node* n = FindNode("_recv_W1_0");
  EXPECT_FALSE(IsRefType(CHECK_NOTNULL(n)->output_type(0)));
}

TEST_F(SubgraphTest, FedRefNode_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W2', 'W1' ] }");
  EXPECT_EQ("OK",
            Subgraph("W1:0", "", "t1", true /* use_function_convention */));
  ExpectNodes("_arg_W1_0_0,W2,t1");
  Node* n = FindNode("_arg_W1_0_0");
  EXPECT_FALSE(IsRefType(CHECK_NOTNULL(n)->output_type(0)));
}

TEST_F(SubgraphTest, FedOutputs2_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  // We feed input:1, but nothing connects to it, so the _recv(input:1)
  // node also disappears.
  EXPECT_EQ("OK", Subgraph("input:1,t1,W2", "", "t2",
                           true /* use_function_convention */));
  ExpectNodes("_arg_t1_0_1,_arg_W2_0_2,t2");
}

TEST_F(SubgraphTest, FetchOutputs1) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "W2,input:1,t1,t2", "t2"));
  ExpectNodes(
      "W1,W2,input,t1,t2,_send_W2_0,_send_input_1,_send_t1_0,_send_t2_0");
}

TEST_F(SubgraphTest, FetchOutputs1_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "W2,input:1,t1,t2", "t2",
                           true /* use_function_convention */));
  ExpectNodes(
      "W1,W2,input,t1,t2,_retval_W2_0_0,_retval_input_1_1,_retval_t1_0_2,_"
      "retval_t2_0_3");
}

TEST_F(SubgraphTest, FetchOutputs2) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK", Subgraph("", "t3_a", "t2"));
  ExpectNodes("W1,W2,input,t1,t2,t3_a,_send_t3_a_0");
}

TEST_F(SubgraphTest, FetchOutputs2_FunctionConvention) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W2', 't1' ] }"
      "node { name: 't3_a' op: 'TestRelu' input: 't2' }"
      "node { name: 't3_b' op: 'TestRelu' input: 't2' }");
  EXPECT_EQ("OK",
            Subgraph("", "t3_a", "t2", true /* use_function_convention */));
  ExpectNodes("W1,W2,input,t1,t2,t3_a,_retval_t3_a_0_0");
}

TEST_F(SubgraphTest, ChainOfFools) {
  ExpectOK(
      "node { name: 'a' op: 'TestParams' }"
      "node { name: 'b' op: 'TestRelu' input: 'a'}"
      "node { name: 'c' op: 'TestRelu' input: 'b'}"
      "node { name: 'd' op: 'TestRelu' input: 'c'}"
      "node { name: 'e' op: 'TestRelu' input: 'd'}"
      "node { name: 'f' op: 'TestRelu' input: 'e'}");
  EXPECT_EQ("OK", Subgraph("c:0", "b:0,e:0", ""));
  ExpectNodes("a,b,_send_b_0,_recv_c_0,d,e,_send_e_0");
  EXPECT_TRUE(HasEdge("a", 0, "b", 0));
  EXPECT_TRUE(HasEdge("b", 0, "_send_b_0", 0));
  EXPECT_TRUE(HasEdge("_recv_c_0", 0, "d", 0));
  EXPECT_TRUE(HasEdge("d", 0, "e", 0));
  EXPECT_TRUE(HasEdge("e", 0, "_send_e_0", 0));
}

static bool HasSubstr(StringPiece base, StringPiece substr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_10(mht_10_v, 525, "", "./tensorflow/core/graph/subgraph_test.cc", "HasSubstr");

  bool ok = absl::StrContains(base, substr);
  EXPECT_TRUE(ok) << base << ", expected substring " << substr;
  return ok;
}

TEST_F(SubgraphTest, Errors) {
  ExpectOK(
      "node { name: 'a' op: 'TestParams' }"
      "node { name: 'b' op: 'TestRelu' input: 'a'}"
      "node { name: 'c' op: 'TestRelu' input: 'b'}"
      "node { name: 'd' op: 'TestRelu' input: 'c'}"
      "node { name: 'e' op: 'TestRelu' input: 'd'}"
      "node { name: 'f' op: 'TestRelu' input: 'e'}");
  // Duplicated feed and fetch
  EXPECT_TRUE(
      HasSubstr(Subgraph("c:0", "b:0,c:0", ""), "both fed and fetched"));
  // Feed not found.
  EXPECT_TRUE(HasSubstr(Subgraph("foo:0", "c:0", ""), "unable to find"));
  // Fetch not found.
  EXPECT_TRUE(HasSubstr(Subgraph("", "foo:0", ""), "not found"));
  // Target not found.
  EXPECT_TRUE(HasSubstr(Subgraph("", "", "foo"), "not found"));
  // No targets specified.
  EXPECT_TRUE(HasSubstr(Subgraph("", "", ""), "at least one target"));
}

REGISTER_OP("In").Output("o: float");
REGISTER_OP("Op").Input("i: float").Output("o: float");

void BM_SubgraphHelper(::testing::benchmark::State& state,
                       bool use_function_convention) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_11(mht_11_v, 559, "", "./tensorflow/core/graph/subgraph_test.cc", "BM_SubgraphHelper");

  const int num_nodes = state.range(0);
  DeviceAttributes device_info;
  device_info.set_name("/job:a/replica:0/task:0/cpu:0");
  device_info.set_device_type(DeviceType(DEVICE_CPU).type());
  device_info.set_incarnation(0);

  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* last_node = nullptr;
    for (int i = 0; i < num_nodes; i++) {
      string name = strings::StrCat("N", i);
      if (i > 0) {
        last_node = ops::UnaryOp("Op", last_node, b.opts().WithName(name));
      } else {
        last_node = ops::SourceOp("In", b.opts().WithName(name));
      }
    }
    TF_CHECK_OK(GraphDefBuilderToGraph(b, &g));
  }

  std::vector<string> fed;
  if (num_nodes > 1000) {
    fed.push_back(strings::StrCat("N", num_nodes - 1000));
  }
  std::vector<string> fetch;
  std::vector<string> targets = {strings::StrCat("N", num_nodes - 1)};

  for (auto s : state) {
    Graph* subgraph = new Graph(OpRegistry::Global());
    CopyGraph(g, subgraph);
    subgraph::RewriteGraphMetadata metadata;
    TF_CHECK_OK(subgraph::RewriteGraphForExecution(
        subgraph, fed, fetch, targets, device_info, use_function_convention,
        &metadata));
    delete subgraph;
  }
}

void BM_Subgraph(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_12(mht_12_v, 602, "", "./tensorflow/core/graph/subgraph_test.cc", "BM_Subgraph");

  BM_SubgraphHelper(state, false /* use_function_convention */);
}
void BM_SubgraphFunctionConvention(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraph_testDTcc mht_13(mht_13_v, 608, "", "./tensorflow/core/graph/subgraph_test.cc", "BM_SubgraphFunctionConvention");

  BM_SubgraphHelper(state, true /* use_function_convention */);
}
BENCHMARK(BM_Subgraph)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000);
BENCHMARK(BM_SubgraphFunctionConvention)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

}  // namespace
}  // namespace tensorflow
