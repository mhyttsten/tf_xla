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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc() {
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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)

#include "tensorflow/core/common_runtime/mkl_tfconversion_pass.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class MklToTfConversionPass : public ::testing::Test {
 public:
  MklToTfConversionPass() : graph_(OpRegistry::Global()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "MklToTfConversionPass");
}

  static void InitGraph(const string& s, Graph* graph) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "InitGraph");

    GraphDef graph_def;

    auto parser = protobuf::TextFormat::Parser();
    CHECK(parser.MergeFromString(s, &graph_def)) << s;
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
  }

  void InitGraph(const string& s) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "InitGraph");

    InitGraph(s, &graph_);
    original_ = CanonicalGraphString(&graph_);
  }

  static bool IncludeNode(const Node* n) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "IncludeNode");
 return n->IsOp(); }

  static string EdgeId(const Node* n, int index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "EdgeId");

    if (index == 0) {
      return n->name();
    } else if (index == Graph::kControlSlot) {
      return strings::StrCat(n->name(), ":control");
    } else {
      return strings::StrCat(n->name(), ":", index);
    }
  }

  string CanonicalGraphString(Graph* g) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_5(mht_5_v, 257, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "CanonicalGraphString");

    std::vector<string> nodes;
    std::vector<string> edges;
    for (const Node* n : g->nodes()) {
      if (IncludeNode(n)) {
        nodes.push_back(strings::StrCat(n->name(), "(", n->type_string(), ")"));
      }
    }
    for (const Edge* e : g->edges()) {
      if (IncludeNode(e->src()) && IncludeNode(e->dst())) {
        edges.push_back(strings::StrCat(EdgeId(e->src(), e->src_output()), "->",
                                        EdgeId(e->dst(), e->dst_input())));
      }
    }
    // Canonicalize
    std::sort(nodes.begin(), nodes.end());
    std::sort(edges.begin(), edges.end());
    return strings::StrCat(absl::StrJoin(nodes, ";"), "|",
                           absl::StrJoin(edges, ";"));
  }

  string DoRunMklToTfConversionPass() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_6(mht_6_v, 281, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "DoRunMklToTfConversionPass");

    string before = CanonicalGraphString(&graph_);
    LOG(ERROR) << "Before MklToTf conversion pass: " << before;

    std::unique_ptr<Graph>* ug = new std::unique_ptr<Graph>(&graph_);
    InsertMklToTfConversionNodes(ug);

    string result = CanonicalGraphString(&graph_);
    LOG(ERROR) << "After MklToTf conversion pass:  " << result;
    return result;
  }

  const string& OriginalGraph() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_7(mht_7_v, 296, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "OriginalGraph");
 return original_; }

  Graph graph_;
  string original_;
};

REGISTER_OP("Float_Input").Output("o: float").SetIsStateful();
REGISTER_OP("_Mkl_Input").Output("o: uint8").SetIsStateful();

TEST_F(MklToTfConversionPass, Basic) {
  InitGraph(
      "node { name: 'A' op: 'Float_Input'}"
      "node { name: 'B' op: 'Float_Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }");
  EXPECT_EQ(DoRunMklToTfConversionPass(),
            "A(Float_Input);B(Float_Input);C(Mul);D(Mul)|"
            "A->C;A->D;B->C:1;B->D:1");
}

// MklConv2D followed by Non-Mkl layer
// C=MklConv2D(A,M,B,N); E=Sub(C,D) (for interleaved ordering)
// C=MklConv2D(A,B,M,N); E=Sub(C,D) (for contiguous ordering)
TEST_F(MklToTfConversionPass, Positive) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    InitGraph(
        "node { name: 'A' op: 'Float_Input'}"
        "node { name: 'M' op: '_Mkl_Input'}"
        "node { name: 'B' op: 'Float_Input'}"
        "node { name: 'N' op: '_Mkl_Input'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'M', 'B', 'N']}"
        "node { name: 'D' op: 'Float_Input'}"
        "node { name: 'E' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['C', 'D']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Float_Input);B(Float_Input);C(_MklConv2D);D(Float_Input);E("
              "Sub);M(_Mkl_Input);"
              "Mkl2Tf/_0(_MklToTf);N(_Mkl_Input)|A->C;B->C:2;C->Mkl2Tf/_0;"
              "C:1->Mkl2Tf/_0:1;D->E:1;M->C:1;Mkl2Tf/_0->E;N->C:3");
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    InitGraph(
        "node { name: 'A' op: 'Float_Input'}"
        "node { name: 'B' op: 'Float_Input'}"
        "node { name: 'M' op: '_Mkl_Input'}"
        "node { name: 'N' op: '_Mkl_Input'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'B', 'M', 'N']}"
        "node { name: 'D' op: 'Float_Input'}"
        "node { name: 'E' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['C', 'D']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Float_Input);B(Float_Input);C(_MklConv2D);D(Float_Input);E("
              "Sub);M(_Mkl_Input);"
              "Mkl2Tf/_0(_MklToTf);N(_Mkl_Input)|A->C;B->C:1;C->Mkl2Tf/_0;"
              "C:2->Mkl2Tf/_0:1;D->E:1;M->C:2;Mkl2Tf/_0->E;N->C:3");
  }
}

// MklConv2D followed by MklToTf op followed by Non-Mkl layer.
// C=MklConv2D(A,M,B,N); D=MklToTf(C:0, C:1) F=Sub(D,E) (for interleaved)
// C=MklConv2D(A,B,M,N); D=MklToTf(C:0, C:2) F=Sub(D,E) (for contiguous)
// MklToTf node should not be inserted again.
TEST_F(MklToTfConversionPass, Negative_DoubleInsert) {
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    InitGraph(
        "node { name: 'A' op: 'Float_Input'}"
        "node { name: 'M' op: '_Mkl_Input'}"
        "node { name: 'B' op: 'Float_Input'}"
        "node { name: 'N' op: '_Mkl_Input'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'M', 'B', 'N']}"
        "node { name: 'D' op: '_MklToTf'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " input: ['C:0', 'C:1']}"
        "node { name: 'E' op: 'Float_Input'}"
        "node { name: 'F' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['D', 'E']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Float_Input);B(Float_Input);C(_MklConv2D);D(_MklToTf);E(Float_"
              "Input);"
              "F(Sub);M(_Mkl_Input);N(_Mkl_Input)|"
              "A->C;B->C:2;C->D;C:1->D:1;D->F;E->F:1;M->C:1;N->C:3");
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    InitGraph(
        "node { name: 'A' op: 'Float_Input'}"
        "node { name: 'B' op: 'Float_Input'}"
        "node { name: 'M' op: '_Mkl_Input'}"
        "node { name: 'N' op: '_Mkl_Input'}"
        "node { name: 'C' op: '_MklConv2D'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
        " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } "
        "}"
        " attr { key: 'padding'          value { s: 'SAME' } }"
        " input: ['A', 'B', 'M', 'N']}"
        "node { name: 'D' op: '_MklToTf'"
        " attr { key: 'T'                value { type: DT_FLOAT } }"
        " attr { key: 'data_format'      value { s: 'NCHW' } }"
        " input: ['C:0', 'C:2']}"
        "node { name: 'E' op: 'Float_Input'}"
        "node { name: 'F' op: 'Sub'"
        " attr {key: 'T'                 value { type: DT_FLOAT } }"
        " input: ['D', 'E']}");
    EXPECT_EQ(DoRunMklToTfConversionPass(),
              "A(Float_Input);B(Float_Input);C(_MklConv2D);D(_MklToTf);E(Float_"
              "Input);"
              "F(Sub);M(_Mkl_Input);N(_Mkl_Input)|"
              "A->C;B->C:1;C->D;C:2->D:1;D->F;E->F:1;M->C:2;N->C:3");
  }
}

// C=Conv2D(A,B); E=BiasAdd(C,D); Z=Sub(E,Y);
// There is no Mkl layer so no conversion op should be inserted.
TEST_F(MklToTfConversionPass, Negative_NoMklLayer) {
  InitGraph(
      "node { name: 'A' op: 'Float_Input'}"
      "node { name: 'B' op: 'Float_Input'}"
      "node { name: 'C' op: 'Conv2D'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " attr { key: 'use_cudnn_on_gpu' value { b: false } }"
      " attr { key: 'strides'          value { list: {i: 1, i:1, i:1, i:1} } }"
      " attr { key: 'padding'          value { s: 'SAME' } }"
      " input: ['A', 'B']}"
      "node { name: 'D' op: 'Float_Input'}"
      "node { name: 'E' op: 'BiasAdd'"
      " attr { key: 'T'                value { type: DT_FLOAT } }"
      " attr { key: 'data_format'      value { s: 'NCHW' } }"
      " input: ['C', 'D'] }"
      "node { name: 'Y' op: 'Float_Input'}"
      "node { name: 'Z' op: 'Sub'"
      " attr {key: 'T'                 value { type: DT_FLOAT } }"
      " input: ['E', 'Y']}");
  EXPECT_EQ(DoRunMklToTfConversionPass(),
            "A(Float_Input);B(Float_Input);C(Conv2D);D(Float_Input);E(BiasAdd);"
            "Y(Float_Input);Z(Sub)|"
            "A->C;B->C:1;C->E;D->E:1;E->Z;Y->Z:1");
}

static void BM_RunMklToTfConversionPass(int iters, int op_nodes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSmkl_tfconversion_pass_testDTcc mht_8(mht_8_v, 466, "", "./tensorflow/core/common_runtime/mkl_tfconversion_pass_test.cc", "BM_RunMklToTfConversionPass");

  testing::StopTiming();
  string s;
  for (int in = 0; in < 10; in++) {
    s += strings::Printf("node { name: 'in%04d' op: 'Float_Input'}", in);
  }
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  for (int op = 0; op < op_nodes; op++) {
    s += strings::Printf(
        "node { name: 'op%04d' op: 'Mul' attr { key: 'T' value { "
        "type: DT_FLOAT } } input: ['in%04d', 'in%04d' ] }",
        op, rnd.Uniform(10), rnd.Uniform(10));
  }

  bool first = true;
  while (iters > 0) {
    Graph* graph = new Graph(OpRegistry::Global());
    MklToTfConversionPass::InitGraph(s, graph);
    int N = graph->num_node_ids();
    if (first) {
      testing::SetLabel(strings::StrCat("Per graph node.  Nodes: ", N));
      first = false;
    }
    {
      testing::StartTiming();
      std::unique_ptr<Graph> ug(graph);
      InsertMklToTfConversionNodes(&ug);
      testing::StopTiming();
    }
    iters -= N;  // Our benchmark units are individual graph nodes,
                 // not whole graphs
    // delete graph;
  }
}
BENCHMARK(BM_RunMklToTfConversionPass)->Arg(1000)->Arg(10000);

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL && ENABLE_MKL
