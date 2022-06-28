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

#ifndef TENSORFLOW_CORE_GRAPH_BENCHMARK_TESTLIB_H_
#define TENSORFLOW_CORE_GRAPH_BENCHMARK_TESTLIB_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh() {
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


#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace test {

REGISTER_OP("Input").Output("y: float");
REGISTER_OP("Output")
    .Input("x: N * float")
    .Attr("N: int >= 1")
    .Output("y: float");
REGISTER_OP("In2Out1").Input("a: float").Input("b: float").Output("y: float");
REGISTER_OP("In4Out1")
    .Input("a: float")
    .Input("b: float")
    .Input("c: float")
    .Input("d: float")
    .Output("y: float");
REGISTER_OP("In8Out1")
    .Input("a: float")
    .Input("b: float")
    .Input("c: float")
    .Input("d: float")
    .Input("e: float")
    .Input("f: float")
    .Input("g: float")
    .Input("h: float")
    .Output("y: float");
REGISTER_OP("In16Out1")
    .Input("a: float")
    .Input("b: float")
    .Input("c: float")
    .Input("d: float")
    .Input("e: float")
    .Input("f: float")
    .Input("g: float")
    .Input("h: float")
    .Input("i: float")
    .Input("j: float")
    .Input("k: float")
    .Input("l: float")
    .Input("m: float")
    .Input("n: float")
    .Input("o: float")
    .Input("p: float")
    .Output("y: float");

GraphDef CreateGraphDef(int num_nodes, int num_edges_per_node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh mht_0(mht_0_v, 241, "", "./tensorflow/core/graph/benchmark_testlib.h", "CreateGraphDef");

  const int kNumInNodes = 10 * num_edges_per_node;
  GraphDef graph_def;

  auto create_node = [](const string& name, const string& op) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh mht_1(mht_1_v, 250, "", "./tensorflow/core/graph/benchmark_testlib.h", "lambda");

    NodeDef node;
    node.set_name(name);
    node.set_op(op);
    return node;
  };

  NodeDef node;
  for (int in = 0; in < kNumInNodes; ++in) {
    node = create_node(/*name=*/absl::StrFormat("in%04d", in), /*op=*/"Input");
    *graph_def.add_node() = std::move(node);
  }

  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < num_nodes; ++op) {
    node = create_node(/*name=*/absl::StrFormat("op%05d", op),
                       /*op=*/absl::StrFormat("In%dOut1", num_edges_per_node));
    for (int edge = 0; edge < num_edges_per_node; ++edge) {
      node.add_input(absl::StrFormat("in%04d", rnd.Uniform(kNumInNodes)));
    }
    *graph_def.add_node() = std::move(node);
  }

  // Add a single sink node. Otherwise a lot of time is spent in
  // FixupSourceAndSinkEdges().
  node = create_node(/*name=*/"out", /*op=*/"Output");
  for (int op = 0; op < num_nodes; ++op) {
    node.add_input(absl::StrFormat("op%05d", op));
  }
  AttrValue attr;
  attr.set_i(num_nodes);
  node.mutable_attr()->insert({"N", std::move(attr)});
  *graph_def.add_node() = std::move(node);

  return graph_def;
}

GraphDef CreateRandomGraph(int size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh mht_2(mht_2_v, 291, "", "./tensorflow/core/graph/benchmark_testlib.h", "CreateRandomGraph");

  random::PhiloxRandom philox(0x12345);
  random::SimplePhilox rnd(&philox);

  string prefix = "long_node_name_prefix_to_measure_string_copy_overhead";

  GraphDef graph;
  for (int i = 0; i < size; ++i) {
    const string name = absl::StrCat(prefix, i);
    const uint32 num_inputs = rnd.Uniform(std::min(i, 5));

    NodeDef node;
    node.set_name(name);
    for (int n = 0; n < num_inputs; ++n) {
      const uint32 input_node = rnd.Uniform(i);
      node.add_input(absl::StrCat(prefix, input_node));
    }

    *graph.add_node() = std::move(node);
  }

  return graph;
}

GraphDef CreateFaninFanoutNodeGraph(int num_regular_fanins,
                                    int num_regular_fanouts,
                                    int num_controlling_fanins,
                                    int num_controlled_fanouts,
                                    bool fanout_unique_index) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh mht_3(mht_3_v, 322, "", "./tensorflow/core/graph/benchmark_testlib.h", "CreateFaninFanoutNodeGraph");

  GraphDef graph;

  auto create_node = [](const string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSbenchmark_testlibDTh mht_4(mht_4_v, 329, "", "./tensorflow/core/graph/benchmark_testlib.h", "lambda");

    NodeDef node;
    node.set_name(name);
    return node;
  };

  NodeDef node = create_node(/*name=*/"node");

  for (int i = 0; i < num_regular_fanins; ++i) {
    const string input_node_name = absl::StrFormat("in%05d", i);
    NodeDef input_node = create_node(/*name=*/input_node_name);
    *graph.add_node() = std::move(input_node);
    node.add_input(input_node_name);
  }

  for (int i = 0; i < num_controlling_fanins; ++i) {
    const string input_node_name = absl::StrFormat("control_in%05d", i);
    NodeDef input_node = create_node(/*name=*/input_node_name);
    *graph.add_node() = std::move(input_node);
    node.add_input(absl::StrCat("^", input_node_name));
  }

  for (int i = 0; i < num_regular_fanouts; ++i) {
    NodeDef output_node = create_node(/*name=*/absl::StrFormat("out%05d", i));
    const string input_node_index =
        fanout_unique_index ? absl::StrCat(node.name(), ":", i) : node.name();
    output_node.add_input(input_node_index);
    *graph.add_node() = std::move(output_node);
  }

  const string controlled_fanout_input = absl::StrCat("^", node.name());
  for (int i = 0; i < num_controlled_fanouts; ++i) {
    NodeDef output_node =
        create_node(/*name=*/absl::StrFormat("control_out%05d", i));
    output_node.add_input(controlled_fanout_input);
    *graph.add_node() = std::move(output_node);
  }

  *graph.add_node() = std::move(node);

  return graph;
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_BENCHMARK_TESTLIB_H_
