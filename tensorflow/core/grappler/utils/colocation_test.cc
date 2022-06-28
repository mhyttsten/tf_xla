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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocation_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocation_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocation_testDTcc() {
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

#include "tensorflow/core/grappler/utils/colocation.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

class ColocationTest : public ::testing::Test {};

bool VerifyNodeHasColocation(const NodeDef& ndef, const string& coloc) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("coloc: \"" + coloc + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPScolocation_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/grappler/utils/colocation_test.cc", "VerifyNodeHasColocation");

  if (ndef.attr().empty()) {
    return false;
  }
  if (ndef.attr().find("_class") == ndef.attr().end()) {
    return false;
  }
  return ndef.attr().at("_class").list().s(0) == coloc;
}

TEST(ColocationTest, ReassignColocation_SingleNode) {
  // Node A colocates with B, but node B is not in the graph.
  //   A
  //   |
  //   |
  //  [B]

  NodeDef ndef;
  const Status status =
      NodeDefBuilder("A", "Const").Attr("_class", {"loc:@B"}).Finalize(&ndef);
  TF_EXPECT_OK(status);
  GraphDef gdef = test::function::GDef({ndef});

  EXPECT_EQ(1, gdef.node_size());
  EXPECT_EQ(1, gdef.node(0).attr_size());

  ReassignColocation(&gdef);

  // Validates that node A's colocation info is cleared.
  EXPECT_EQ(1, gdef.node_size());
  EXPECT_EQ(0, gdef.node(0).attr_size());
}

TEST(ColocationTest, ReassignColocation_MultiNode_SingleGroup) {
  // Node A, B, C colocate with X. D colocates with C. E colocates with D.
  // Node X is not in the graph.
  //  A   B   C---D---E
  //  |   |   |
  //  |   |   |
  //  +--[X]--+
  // After re-assign of colocation, A, B, C, D should colocate with E.
  // A   B   C   D
  // |   |   |   |
  // |   |   |   |
  // +---+-E-+---+

  NodeDef ndef_a, ndef_b, ndef_c, ndef_d, ndef_e;
  Status status =
      NodeDefBuilder("A", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_a);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("B", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_b);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("C", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_c);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("D", "Const").Attr("_class", {"loc:@C"}).Finalize(&ndef_d);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("E", "Const").Attr("_class", {"loc:@D"}).Finalize(&ndef_e);
  TF_EXPECT_OK(status);
  GraphDef gdef =
      test::function::GDef({ndef_a, ndef_b, ndef_c, ndef_d, ndef_e});

  EXPECT_EQ(5, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@X"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@X"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@X"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@C"));  // D
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(4), "loc:@D"));  // E

  ReassignColocation(&gdef);

  EXPECT_EQ(5, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@E"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@E"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@E"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@E"));  // D
  EXPECT_EQ(0, gdef.node(4).attr_size());                        // E
}

TEST(ColocationTest, ReassignColocation_MultiNode_MultiGroup) {
  // Before re-assign:
  // Node A, B, C colocate with X. D colocates with C. E colocates with D.
  // Node U, V colocates with W. Node X, W are not in the graph:
  //  A   B   C---D---E
  //  |   |   |
  //  |   |   |
  //  +--[X]--+
  //
  //  U       V
  //  |       |
  //  |       |
  //  +--[W]--+
  //
  // After re-assign:
  // A, B, C, D should colocate with E. U should colocate with V.
  // A   B   C   D
  // |   |   |   |
  // |   |   |   |
  // +---+-E-+---+
  //
  // U
  // |
  // |
  // V

  NodeDef ndef_a, ndef_b, ndef_c, ndef_d, ndef_e, ndef_u, ndef_v;
  Status status =
      NodeDefBuilder("A", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_a);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("B", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_b);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("C", "Const").Attr("_class", {"loc:@X"}).Finalize(&ndef_c);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("D", "Const").Attr("_class", {"loc:@C"}).Finalize(&ndef_d);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("E", "Const").Attr("_class", {"loc:@D"}).Finalize(&ndef_e);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("U", "Const").Attr("_class", {"loc:@W"}).Finalize(&ndef_u);
  TF_EXPECT_OK(status);
  status =
      NodeDefBuilder("V", "Const").Attr("_class", {"loc:@W"}).Finalize(&ndef_v);
  TF_EXPECT_OK(status);
  GraphDef gdef = test::function::GDef(
      {ndef_a, ndef_b, ndef_c, ndef_d, ndef_e, ndef_u, ndef_v});

  EXPECT_EQ(7, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@X"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@X"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@X"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@C"));  // D
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(4), "loc:@D"));  // E
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(5), "loc:@W"));  // U
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(6), "loc:@W"));  // V

  ReassignColocation(&gdef);

  EXPECT_EQ(7, gdef.node_size());
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(0), "loc:@E"));  // A
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(1), "loc:@E"));  // B
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(2), "loc:@E"));  // C
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(3), "loc:@E"));  // D
  EXPECT_EQ(0, gdef.node(4).attr_size());                        // E
  EXPECT_TRUE(VerifyNodeHasColocation(gdef.node(5), "loc:@V"));  // U
  EXPECT_EQ(0, gdef.node(6).attr_size());                        // V
}

}  // namespace grappler
}  // namespace tensorflow
