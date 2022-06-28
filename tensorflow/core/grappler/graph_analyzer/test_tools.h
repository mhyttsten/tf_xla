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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_TEST_TOOLS_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_TEST_TOOLS_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

//=== Helper methods to construct the nodes.

NodeDef MakeNodeConst(const string& name);

NodeDef MakeNode2Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2);

NodeDef MakeNode4Arg(const string& name, const string& opcode,
                     const string& arg1, const string& arg2, const string& arg3,
                     const string& arg4);

inline NodeDef MakeNodeMul(const string& name, const string& arg1,
                           const string& arg2) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("arg1: \"" + arg1 + "\"");
   mht_0_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh mht_0(mht_0_v, 216, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.h", "MakeNodeMul");

  return MakeNode2Arg(name, "Mul", arg1, arg2);
}

// Not really a 2-argument but convenient to construct.
inline NodeDef MakeNodeAddN(const string& name, const string& arg1,
                            const string& arg2) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("arg1: \"" + arg1 + "\"");
   mht_1_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh mht_1(mht_1_v, 228, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.h", "MakeNodeAddN");

  return MakeNode2Arg(name, "AddN", arg1, arg2);
}

inline NodeDef MakeNodeSub(const string& name, const string& arg1,
                           const string& arg2) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("arg1: \"" + arg1 + "\"");
   mht_2_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh mht_2(mht_2_v, 239, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.h", "MakeNodeSub");

  return MakeNode2Arg(name, "Sub", arg1, arg2);
}

// Has 2 honest outputs.
inline NodeDef MakeNodeBroadcastGradientArgs(const string& name,
                                             const string& arg1,
                                             const string& arg2) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   mht_3_v.push_back("arg1: \"" + arg1 + "\"");
   mht_3_v.push_back("arg2: \"" + arg2 + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPStest_toolsDTh mht_3(mht_3_v, 252, "", "./tensorflow/core/grappler/graph_analyzer/test_tools.h", "MakeNodeBroadcastGradientArgs");

  return MakeNode2Arg(name, "BroadcastGradientArgs", arg1, arg2);
}

NodeDef MakeNodeShapeN(const string& name, const string& arg1,
                       const string& arg2);

NodeDef MakeNodeIdentityN(const string& name, const string& arg1,
                          const string& arg2);

NodeDef MakeNodeQuantizedConcat(const string& name, const string& arg1,
                                const string& arg2, const string& arg3,
                                const string& arg4);

//=== A container of pre-constructed graphs.

class TestGraphs {
 public:
  TestGraphs();

  // Graph with 3 nodes and a control link to self (which is not valid in
  // reality but adds excitement to the tests).
  GraphDef graph_3n_self_control_;
  // Graph that has the multi-input links.
  GraphDef graph_multi_input_;
  // Graph that has the all-or-none nodes.
  GraphDef graph_all_or_none_;
  // All the nodes are connected in a circle that goes in one direction.
  GraphDef graph_circular_onedir_;
  // All the nodes are connected in a circle that goes in both directions.
  GraphDef graph_circular_bidir_;
  // The nodes are connected in a line.
  GraphDef graph_linear_;
  // The nodes are connected in a cross shape.
  GraphDef graph_cross_;
  GraphDef graph_small_cross_;
  // For testing the ordering of links at the end of signature generation,
  // a variation of a cross.
  GraphDef graph_for_link_order_;
  // Sun-shaped, a ring with "rays".
  GraphDef graph_sun_;
};

//=== Helper methods for analysing the structures.

std::vector<string> DumpLinkMap(const GenNode::LinkMap& link_map);

// Also checks for the consistency of hash values.
std::vector<string> DumpLinkHashMap(const SigNode::LinkHashMap& link_hash_map);

std::vector<string> DumpHashedPeerVector(
    const SigNode::HashedPeerVector& hashed_peers);

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_TEST_TOOLS_H_
