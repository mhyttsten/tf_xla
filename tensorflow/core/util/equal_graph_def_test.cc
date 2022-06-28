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
class MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc() {
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

#include <utility>

#include "tensorflow/core/util/equal_graph_def.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

REGISTER_OP("Input").Output("o: float");
REGISTER_OP("Alternate").Output("o: float");
REGISTER_OP("Combine").Input("a: float").Input("b: float").Output("o: float");

Node* Input(const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/util/equal_graph_def_test.cc", "Input");

  return ops::SourceOp("Input", opts);
}

Node* Alternate(const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/util/equal_graph_def_test.cc", "Alternate");

  return ops::SourceOp("Alternate", opts);
}

Node* Combine(ops::NodeOut a, ops::NodeOut b,
              const GraphDefBuilder::Options& opts) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/util/equal_graph_def_test.cc", "Combine");

  return ops::BinaryOp("Combine", std::move(a), std::move(b), opts);
}

class EqualGraphDefTest : public ::testing::Test {
 protected:
  EqualGraphDefTest()
      : e_(GraphDefBuilder::kFailImmediately),
        a_(GraphDefBuilder::kFailImmediately) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/util/equal_graph_def_test.cc", "EqualGraphDefTest");
}

  bool Match() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSequal_graph_def_testDTcc mht_4(mht_4_v, 235, "", "./tensorflow/core/util/equal_graph_def_test.cc", "Match");

    GraphDef expected;
    TF_EXPECT_OK(e_.ToGraphDef(&expected));
    GraphDef actual;
    TF_EXPECT_OK(a_.ToGraphDef(&actual));
    bool match = EqualGraphDef(actual, expected, &diff_);
    if (match) {
      EXPECT_EQ(GraphDefHash(expected), GraphDefHash(actual));
    } else {
      // While, strictly speaking, this does not have to be the case,
      // we want to check that our hash is more than "return 0;".
      // If, in the extremely unlikely case, some different graphs
      // legitimately produce equal hash values in this test, we can always
      // tweak them a little to produce different hash values.
      EXPECT_NE(GraphDefHash(expected), GraphDefHash(actual));
    }
    return match;
  }

  GraphDefBuilder e_;
  GraphDefBuilder a_;
  string diff_;
};

TEST_F(EqualGraphDefTest, Match) {
  Input(e_.opts().WithName("A"));
  Input(a_.opts().WithName("A"));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, NoMatch) {
  Input(e_.opts().WithName("A"));
  Input(a_.opts().WithName("B"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Did not find expected node '{{node A}} = Input[]()'", diff_);
}

TEST_F(EqualGraphDefTest, MissingNode) {
  Input(e_.opts().WithName("A"));
  Input(e_.opts().WithName("B"));
  Input(a_.opts().WithName("A"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Did not find expected node '{{node B}} = Input[]()'", diff_);
}

TEST_F(EqualGraphDefTest, ExtraNode) {
  Input(e_.opts().WithName("A"));
  Input(a_.opts().WithName("A"));
  Input(a_.opts().WithName("B"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Found unexpected node '{{node B}} = Input[]()'", diff_);
}

TEST_F(EqualGraphDefTest, NodeOrder) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Combine(a, b, e_.opts().WithName("C"));

  b = Input(a_.opts().WithName("B"));
  a = Input(a_.opts().WithName("A"));
  Combine(a, b, a_.opts().WithName("C"));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, NameMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  // Have to call EqualNodeDef() directly here, since EqualGraphDef()
  // only calls EqualNodeDef() with nodes that have matching names.
  EXPECT_FALSE(EqualNodeDef(a->def(), b->def(), &diff_));
  EXPECT_EQ("Actual node name 'A' is not expected 'B'", diff_);
}

TEST_F(EqualGraphDefTest, OpMismatch) {
  Input(e_.opts().WithName("A"));
  Alternate(a_.opts().WithName("A"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'A' has op 'Alternate' that is not expected 'Input'",
            diff_);
}

TEST_F(EqualGraphDefTest, DeviceMatch) {
  Input(e_.opts().WithName("A").WithDevice("/cpu:0"));
  Input(a_.opts().WithName("A").WithDevice("/cpu:0"));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, DeviceMismatch) {
  Input(e_.opts().WithName("A").WithDevice("/cpu:0"));
  Input(a_.opts().WithName("A").WithDevice("/cpu:1"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'A' has device '/cpu:1' that is not expected '/cpu:0'",
            diff_);
}

TEST_F(EqualGraphDefTest, InputMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Combine(a, a, e_.opts().WithName("C"));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  Combine(b, b, a_.opts().WithName("C"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'C' has input 0 'B' that doesn't match expected 'A'",
            diff_);
}

TEST_F(EqualGraphDefTest, InputOrderMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Combine(a, b, e_.opts().WithName("C"));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  Combine(b, a, a_.opts().WithName("C"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'C' has input 0 'B' that doesn't match expected 'A'",
            diff_);
}

TEST_F(EqualGraphDefTest, ControlInputOrder) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Node* d = Input(e_.opts().WithName("D"));
  Combine(a, a,
          e_.opts()
              .WithName("E")
              .WithControlInput(b)
              .WithControlInput(c)
              .WithControlInput(d));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  d = Input(a_.opts().WithName("D"));
  Combine(a, a,
          a_.opts()
              .WithName("E")
              .WithControlInput(c)
              .WithControlInput(d)
              .WithControlInput(b));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, ControlInputMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Node* d = Input(e_.opts().WithName("D"));
  Combine(a, a,
          e_.opts().WithName("E").WithControlInput(b).WithControlInput(c));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  d = Input(a_.opts().WithName("D"));
  Combine(a, a,
          a_.opts().WithName("E").WithControlInput(b).WithControlInput(d));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'E' missing expected control input '^C'", diff_);
}

TEST_F(EqualGraphDefTest, ControlInputAdded) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Combine(a, a, e_.opts().WithName("D").WithControlInput(b));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  Combine(a, a,
          a_.opts().WithName("D").WithControlInput(b).WithControlInput(c));
  EXPECT_FALSE(Match());
  EXPECT_EQ(
      "Node named 'D' has inputs 'A, A, ^B, ^C' that don't match "
      "expected 'A, A, ^B'",
      diff_);
}

TEST_F(EqualGraphDefTest, ControlInputRemoved) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Combine(a, a,
          e_.opts().WithName("D").WithControlInput(b).WithControlInput(c));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  Combine(a, a, a_.opts().WithName("D").WithControlInput(b));
  EXPECT_FALSE(Match());
  EXPECT_EQ(
      "Node named 'D' has inputs 'A, A, ^B' that don't match "
      "expected 'A, A, ^B, ^C'",
      diff_);
}

TEST_F(EqualGraphDefTest, Attr) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef same(a->def());
  AddNodeAttr("foo", "bar", &same);
  EXPECT_TRUE(EqualNodeDef(same, same, &diff_)) << diff_;
}

TEST_F(EqualGraphDefTest, AttrAdded) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  EXPECT_FALSE(EqualNodeDef(actual, a->def(), &diff_));
  EXPECT_EQ("Node named 'A' has unexpected attr 'foo' with value: \"bar\"",
            diff_);
}

TEST_F(EqualGraphDefTest, AttrRemoved) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef expected(a->def());
  AddNodeAttr("foo", "bar", &expected);
  EXPECT_FALSE(EqualNodeDef(a->def(), expected, &diff_));
  EXPECT_EQ("Node named 'A' missing expected attr 'foo' with value: \"bar\"",
            diff_);
}

TEST_F(EqualGraphDefTest, AttrOrder) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  AddNodeAttr("baz", 42, &actual);

  NodeDef expected(a->def());
  AddNodeAttr("baz", 42, &expected);
  AddNodeAttr("foo", "bar", &expected);

  EXPECT_TRUE(EqualNodeDef(actual, expected, &diff_)) << diff_;
}

TEST_F(EqualGraphDefTest, AttrMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  AddNodeAttr("baz", 5, &actual);

  NodeDef expected(a->def());
  AddNodeAttr("baz", 42, &expected);
  AddNodeAttr("foo", "bar", &expected);

  EXPECT_FALSE(EqualNodeDef(actual, expected, &diff_));
  EXPECT_EQ(
      "Node named 'A' has attr 'baz' with value: 5 that does not match "
      "expected: 42",
      diff_);
}

TEST_F(EqualGraphDefTest, IgnoreInternalAttrs) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  // Internal attrs are ignored.
  AddNodeAttr("_class", 5, &actual);

  NodeDef expected(a->def());
  AddNodeAttr("foo", "bar", &expected);
  AddNodeAttr("_kernel", "eigen", &actual);
  EXPECT_TRUE(EqualNodeDef(actual, expected, &diff_));
}

}  // namespace
}  // namespace tensorflow
