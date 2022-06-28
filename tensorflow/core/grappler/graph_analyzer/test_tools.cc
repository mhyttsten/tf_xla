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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

//=== Helper methods to construct the nodes.

NodeDef MakeNodeConst(const string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "MakeNodeConst");

  NodeDef n;
  n.set_name(name);
  n.set_op("Const");
  return n;
}

NodeDef MakeNode2Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("opcode: \"" + opcode + "\"");
   mht_1_v.push_back("arg1: \"" + arg1 + "\"");
   mht_1_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "MakeNode2Arg");

  NodeDef n;
  n.set_name(name);
  n.set_op(opcode);
  n.add_input(arg1);
  n.add_input(arg2);
  return n;
}

NodeDef MakeNode4Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2, const string& arg3,
                     const string& arg4) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("opcode: \"" + opcode + "\"");
   mht_2_v.push_back("arg1: \"" + arg1 + "\"");
   mht_2_v.push_back("arg2: \"" + arg2 + "\"");
   mht_2_v.push_back("arg3: \"" + arg3 + "\"");
   mht_2_v.push_back("arg4: \"" + arg4 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_2(mht_2_v, 233, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "MakeNode4Arg");

  NodeDef n;
  n.set_name(name);
  n.set_op(opcode);
  n.add_input(arg1);
  n.add_input(arg2);
  n.add_input(arg3);
  n.add_input(arg4);
  return n;
}

// Not really a 2-argument but convenient to construct.
NodeDef MakeNodeShapeN(const string& name, const string& arg1,
                       const string& arg2) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   mht_3_v.push_back("arg1: \"" + arg1 + "\"");
   mht_3_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "MakeNodeShapeN");

  // This opcode is multi-input but not commutative.
  return MakeNode2Arg(name, "ShapeN", arg1, arg2);
}

// Not really a 2-argument but convenient to construct.
NodeDef MakeNodeIdentityN(const string& name, const string& arg1,
                          const string& arg2) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   mht_4_v.push_back("arg1: \"" + arg1 + "\"");
   mht_4_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_4(mht_4_v, 265, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "MakeNodeIdentityN");

  // The argument is of a list type.
  return MakeNode2Arg(name, "IdentityN", arg1, arg2);
}

NodeDef MakeNodeQuantizedConcat(const string& name, const string& arg1,
                                const string& arg2, const string& arg3,
                                const string& arg4) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   mht_5_v.push_back("arg1: \"" + arg1 + "\"");
   mht_5_v.push_back("arg2: \"" + arg2 + "\"");
   mht_5_v.push_back("arg3: \"" + arg3 + "\"");
   mht_5_v.push_back("arg4: \"" + arg4 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "MakeNodeQuantizedConcat");

  // This opcode has multiple multi-inputs.
  return MakeNode4Arg(name, "QuantizedConcat", arg1, arg2, arg3, arg4);
}

//=== Helper methods for analysing the structures.

std::vector<string> DumpLinkMap(const GenNode::LinkMap& link_map) {
  // This will order the entries first.
  std::map<string, string> ordered;
  for (const auto& link : link_map) {
    string key = string(link.first);

    // Order the other sides too. They may be repeating, so store them
    // in a multiset.
    std::multiset<string> others;
    for (const auto& other : link.second) {
      others.emplace(
          absl::StrFormat("%s[%s]", other.node->name(), string(other.port)));
    }
    ordered[key] = absl::StrJoin(others, ", ");
  }
  // Now dump the result in a predictable order.
  std::vector<string> result;
  result.reserve(ordered.size());
  for (const auto& link : ordered) {
    result.emplace_back(link.first + ": " + link.second);
  }
  return result;
}

std::vector<string> DumpLinkHashMap(const SigNode::LinkHashMap& link_hash_map) {
  // The entries in this map are ordered by hash value which might change
  // at any point. Re-order them by the link tag.
  std::map<SigNode::LinkTag, size_t> tags;
  for (const auto& entry : link_hash_map) {
    tags[entry.second.tag] = entry.first;
  }

  std::vector<string> result;
  for (const auto& id : tags) {
    // For predictability, the nodes need to be sorted.
    std::vector<string> nodes;
    for (const auto& peer : link_hash_map.at(id.second).peers) {
      nodes.emplace_back(peer->name());
    }
    std::sort(nodes.begin(), nodes.end());
    result.emplace_back(string(id.first.local) + ":" + string(id.first.remote) +
                        ": " + absl::StrJoin(nodes, ", "));
  }
  return result;
}

std::vector<string> DumpHashedPeerVector(
    const SigNode::HashedPeerVector& hashed_peers) {
  std::vector<string> result;

  // Each subset of nodes with the same hash has to be sorted by name.
  // Other than that, the vector is already ordered by full tags.
  size_t last_hash = 0;
  // Index, since iterators may get invalidated on append.
  size_t subset_start = 0;

  for (const auto& entry : hashed_peers) {
    if (entry.link_hash != last_hash) {
      std::sort(result.begin() + subset_start, result.end());
      subset_start = result.size();
    }
    result.emplace_back(entry.peer->name());
  }
  std::sort(result.begin() + subset_start, result.end());

  return result;
}

TestGraphs::TestGraphs() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTcc mht_6(mht_6_v, 358, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.cc", "TestGraphs::TestGraphs");

  {
    GraphDef& graph = graph_3n_self_control_;
    // The topology includes a loop and a link to self.
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeSub("node2", "node3:1", "node3:0");
    auto node3 = graph.add_node();
    *node3 = MakeNodeBroadcastGradientArgs("node3", "node1", "node2");
    node3->add_input("^node3");  // The control link goes back to self.
  }
  {
    GraphDef& graph = graph_multi_input_;
    // The topology includes a loop and a link to self.
    (*graph.add_node()) = MakeNodeConst("const1_1");
    (*graph.add_node()) = MakeNodeConst("const1_2");
    (*graph.add_node()) = MakeNodeAddN("add1", "const1_1", "const1_2");

    (*graph.add_node()) = MakeNodeConst("const2_1");
    (*graph.add_node()) = MakeNodeConst("const2_2");
    (*graph.add_node()) = MakeNodeConst("const2_3");

    auto add2 = graph.add_node();
    *add2 = MakeNodeAddN("add2", "const2_1", "const2_2");
    // The 3rd node is connected twice, to 4 links total.
    add2->add_input("const2_3");
    add2->add_input("const2_3");

    (*graph.add_node()) = MakeNodeSub("sub", "add1", "add2");
  }
  {
    GraphDef& graph = graph_all_or_none_;
    // The topology includes a loop and a link to self.
    (*graph.add_node()) = MakeNodeConst("const1_1");
    (*graph.add_node()) = MakeNodeConst("const1_2");
    auto pass1 = graph.add_node();
    *pass1 = MakeNodeIdentityN("pass1", "const1_1", "const1_2");

    (*graph.add_node()) = MakeNodeConst("const2_1");
    (*graph.add_node()) = MakeNodeConst("const2_2");
    (*graph.add_node()) = MakeNodeConst("const2_3");

    auto pass2 = graph.add_node();
    *pass2 = MakeNodeIdentityN("pass2", "const2_1", "const2_2");
    // The 3rd node is connected twice, to 4 links total.
    pass2->add_input("const2_3");
    pass2->add_input("const2_3");

    // Add the control links, they get handled separately than the normal
    // links.
    pass1->add_input("^const2_1");
    pass1->add_input("^const2_2");
    pass1->add_input("^const2_3");

    (*graph.add_node()) = MakeNodeSub("sub", "pass1", "pass2");
  }
  {
    GraphDef& graph = graph_circular_onedir_;
    (*graph.add_node()) = MakeNodeMul("node1", "node5", "node5");
    (*graph.add_node()) = MakeNodeMul("node2", "node1", "node1");
    (*graph.add_node()) = MakeNodeMul("node3", "node2", "node2");
    (*graph.add_node()) = MakeNodeMul("node4", "node3", "node3");
    (*graph.add_node()) = MakeNodeMul("node5", "node4", "node4");
  }
  {
    GraphDef& graph = graph_circular_bidir_;
    // The left and right links are intentionally mixed up.
    (*graph.add_node()) = MakeNodeMul("node1", "node5", "node2");
    (*graph.add_node()) = MakeNodeMul("node2", "node3", "node1");
    (*graph.add_node()) = MakeNodeMul("node3", "node2", "node4");
    (*graph.add_node()) = MakeNodeMul("node4", "node5", "node3");
    (*graph.add_node()) = MakeNodeMul("node5", "node4", "node1");
  }
  {
    GraphDef& graph = graph_linear_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeMul("node2", "node1", "node1");
    (*graph.add_node()) = MakeNodeMul("node3", "node2", "node2");
    (*graph.add_node()) = MakeNodeMul("node4", "node3", "node3");
    (*graph.add_node()) = MakeNodeMul("node5", "node4", "node4");
  }
  {
    GraphDef& graph = graph_cross_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeMul("node2", "node1", "node1");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeMul("node4", "node3", "node3");
    (*graph.add_node()) = MakeNodeConst("node5");
    (*graph.add_node()) = MakeNodeMul("node6", "node5", "node5");
    (*graph.add_node()) = MakeNodeConst("node7");
    (*graph.add_node()) = MakeNodeMul("node8", "node7", "node7");

    auto center = graph.add_node();
    *center = MakeNodeMul("node9", "node2", "node4");
    center->add_input("node6");
    center->add_input("node8");
  }
  {
    GraphDef& graph = graph_small_cross_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeConst("node2");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeConst("node4");

    auto center = graph.add_node();
    *center = MakeNodeMul("node5", "node1", "node2");
    center->add_input("node3");
    center->add_input("node4");
  }
  {
    GraphDef& graph = graph_for_link_order_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeConst("node2");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeConst("node4");

    // One group of equivalent links.
    auto center = graph.add_node();
    *center = MakeNodeMul("node5", "node1", "node2");
    center->add_input("node3");
    center->add_input("node4");

    // Multiple groups, separated by unique links.
    auto center2 = graph.add_node();
    *center2 = MakeNodeMul("node6", "node1", "node2");
    center2->add_input("node2:1");
    center2->add_input("node3:2");
    center2->add_input("node4:2");
    center2->add_input("node4:3");
  }
  {
    GraphDef& graph = graph_sun_;
    (*graph.add_node()) = MakeNodeConst("node1");
    (*graph.add_node()) = MakeNodeConst("node2");
    (*graph.add_node()) = MakeNodeConst("node3");
    (*graph.add_node()) = MakeNodeConst("node4");
    (*graph.add_node()) = MakeNodeConst("node5");
    (*graph.add_node()) = MakeNodeSub("node6", "node1", "node10");
    (*graph.add_node()) = MakeNodeSub("node7", "node2", "node6");
    (*graph.add_node()) = MakeNodeSub("node8", "node3", "node7");
    (*graph.add_node()) = MakeNodeSub("node9", "node4", "node8");
    (*graph.add_node()) = MakeNodeSub("node10", "node5", "node9");
  }
}

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
