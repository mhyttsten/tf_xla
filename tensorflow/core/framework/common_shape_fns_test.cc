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
class MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/common_shape_fns.h"

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace shape_inference {

namespace {

PartialTensorShape S(std::initializer_list<int64_t> dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "S");

  return PartialTensorShape(dims);
}

PartialTensorShape Unknown() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "Unknown");
 return PartialTensorShape(); }

OpDef MakeOpDef(int num_inputs, int num_outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "MakeOpDef");

  OpRegistrationData op_reg_data;
  OpDefBuilder b("dummy");
  for (int i = 0; i < num_inputs; ++i) {
    b.Input(strings::StrCat("i", i, ": float"));
  }
  for (int i = 0; i < num_outputs; ++i) {
    b.Output(strings::StrCat("o", i, ": float"));
  }
  CHECK(b.Attr("foo:string").Finalize(&op_reg_data).ok());
  return op_reg_data.op_def;
}

}  // namespace

TEST(CommonShapeFnsTest, NoOutputShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("Assert")
                  .Input("condition: bool")
                  .Input("data: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "Assert")
                  .Input("condition", 0, DT_BOOL)
                  .Input({{"data", 0, DT_FLOAT}})
                  .Finalize(&def));

  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({}), S({10})}, {},
                     {}, {});
  TF_EXPECT_OK(NoOutputs(&c));
  EXPECT_EQ(0, c.num_outputs());
}

TEST(CommonShapeFnsTest, ScalarShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("L2Loss")
                  .Input("t: float")
                  .Output("t: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(
      NodeDefBuilder("test", "L2Loss").Input("t", 0, DT_FLOAT).Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({})}, {}, {}, {});
    TF_EXPECT_OK(ScalarShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(0, c.Rank(output));
  }

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({1, 23, 4, 4, 2})},
                       {}, {}, {});
    TF_EXPECT_OK(ScalarShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(0, c.Rank(output));
  }
}

TEST(CommonShapeFnsTest, MatMulShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("MatMul")
                  .Input("a: float")
                  .Input("b: float")
                  .Output("c: float")
                  .Attr("transpose_a:bool=false")
                  .Attr("transpose_b:bool=false")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                  .Input("a", 0, DT_FLOAT)
                  .Input("b", 0, DT_FLOAT)
                  .Attr("transpose_a", false)
                  .Attr("transpose_b", false)
                  .Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3}), S({3, 4})}, {}, {}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // Unknown inner dimension for one
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, -1}), S({3, 4})}, {}, {}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // Invalid rank.
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2}), S({3, 4})},
                       {}, {}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "Shape must be rank 2 but is rank 1"));
  }

  {
    // Unknown outer dimension
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3}), S({3, -1})}, {}, {}, {});
    TF_EXPECT_OK(MatMulShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_FALSE(c.ValueKnown(c.Dim(output, 1)));
  }

  {
    // Inner shapes not compatible
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 5}), S({3, 4})}, {}, {}, {});
    auto s = MatMulShape(&c);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "Dimensions must be equal, but are 5 and 3"));
  }

  {
    // Inner shapes not compatible
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 5, 3}), S({3, 5, 4})}, {}, {}, {});
    auto s = MatMulShape(&c);
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "Shape must be rank 2 but is rank 3"));
  }

  {
    // transpose_a
    TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("transpose_a", true)
                    .Attr("transpose_b", false)
                    .Attr("type", DT_FLOAT)
                    .Finalize(&def));

    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({3, 2}), S({3, 4})}, {}, {}, {});
    auto s = MatMulShape(&c);
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }

  {
    // transpose_b
    TF_CHECK_OK(NodeDefBuilder("test", "MatMul")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("transpose_a", false)
                    .Attr("transpose_b", true)
                    .Attr("type", DT_FLOAT)
                    .Finalize(&def));

    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3}), S({4, 3})}, {}, {}, {});
    auto s = MatMulShape(&c);
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(4, c.Value(c.Dim(output, 1)));
  }
}

TEST(CommonShapeFnsTest, Einsum_ShapeFn) {
  ShapeInferenceTestOp op("Einsum");
  auto set_equation = [&op](int n, string equation) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("equation: \"" + equation + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_3(mht_3_v, 399, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    std::vector<NodeDefBuilder::NodeOut> input_list;
    input_list.reserve(n);
    for (int i = 0; i < n; ++i) {
      input_list.emplace_back("a", 0, DT_FLOAT);
    }
    TF_ASSERT_OK(NodeDefBuilder("test", "Einsum")
                     .Input(input_list)
                     .Attr("equation", equation)
                     .Finalize(&op.node_def));
  };

  // Unary cases.
  set_equation(1, "abc->c");
  INFER_OK(op, "[?,?,?]", "[d0_2]");
  set_equation(1, "abc->aabbcc");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_0,d0_1,d0_1,d0_2,d0_2]");
  set_equation(1, "abc->");
  INFER_OK(op, "[?,?,?]", "[]");
  set_equation(1, "->");
  INFER_OK(op, "[]", "[]");

  // Binary cases.
  set_equation(2, "ij,jk->ik");
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  set_equation(2, "ij,jk->ik");
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  set_equation(2, "ab,ab->");
  INFER_OK(op, "[?,?];[?,?]", "[]");
  set_equation(2, "ab,->b");
  INFER_OK(op, "[?,?];[]", "[d0_1]");
  set_equation(2, ",->");
  INFER_OK(op, "[];[]", "[]");
  set_equation(2, "aaa,b->abbb");
  INFER_OK(op, "[?,?,?];[?]", "[d0_0,d1_0,d1_0,d1_0]");
  set_equation(2, ",abcd->badc");
  INFER_OK(op, "[];[?,?,?,?]", "[d1_1,d1_0,d1_3,d1_2]");

  // Ellipsis cases.
  set_equation(1, "a...bc->c...");
  INFER_OK(op, "[?,?,?,?,?]", "[d0_4,d0_1,d0_2]");
  set_equation(2, "...ij,...jk->...ik");
  INFER_OK(op, "[?,?,?,?,?];[1,?,?]", "[d0_0,d0_1,d0_2,d0_3,d1_2]");
  INFER_OK(op, "[1,?,?];[?,?,?,?,?]", "[d1_0,d1_1,d1_2,d0_1,d1_4]");

  // Unknown rank.
  set_equation(1, "abc->c");
  INFER_OK(op, "?", "[?]");
  set_equation(1, "a...bc->c");
  INFER_OK(op, "?", "[?]");
  set_equation(1, "a...bc->c...");
  INFER_OK(op, "?", "?");

  set_equation(2, "...ij,...jk->...ik");
  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[?,?,?];?", "?");
  INFER_OK(op, "?;[?,?,?]", "?");
  set_equation(2, "...ij,...jk->ik");
  INFER_OK(op, "?;?", "[?,?]");
  set_equation(2, "abd,b...c->...cad");
  INFER_OK(op, "[?,?,?];[?,?,?,?]", "[d1_1,d1_2,d1_3,d0_0,d0_2]");
  set_equation(2, "...ab,b...c->ac...");
  INFER_OK(op, "[?,1,?,?];[?,?,?]", "[d0_2,d1_2,d0_0,d1_1]");

  // Wrong number of inputs.
  set_equation(2, "ab->b");
  INFER_ERROR("got: 2", op, "[?,?];[?,?]");
  set_equation(1, "ab,a->b");
  INFER_ERROR("got: 1", op, "[?,?]");

  // Invalid format. Implicit form is not supported.
  set_equation(1, "a");
  INFER_ERROR("equation", op, "[2]");
  set_equation(2, "ab,bc");
  INFER_ERROR("equation", op, "[2,2];[2,2]");

  // Wrong number of ellipsis or periods outside of ellipsis.
  set_equation(1, "..a.->a...");
  INFER_ERROR("ellipsis", op, "[1,1,2,1]");
  set_equation(1, "...a->.a..");
  INFER_ERROR("ellipsis", op, "[1,1,1,2]");
  set_equation(1, "...a...->...a");
  INFER_ERROR("ellipsis", op, "[1,1,1,2]");
  set_equation(1, "..a..b..->...ab");
  INFER_ERROR("ellipsis", op, "[1,1,2,1]");
  set_equation(2, "...a...,ab->a");
  INFER_ERROR("ellipsis", op, "[1,2,1];[2,1]");
  set_equation(2, "a,...ab...->a");
  INFER_ERROR("ellipsis", op, "[2];[1,2,1,1]");
  set_equation(2, "a,ab->a......");
  INFER_ERROR("ellipsis", op, "[2];[2,1]");

  // Output label doesn't appear in input.
  set_equation(1, "abc->d");
  INFER_ERROR("'d'", op, "[?,?,?]");

  // Mismatch in input rank.
  set_equation(1, "abc->c");
  INFER_ERROR("4", op, "[?,?,?,?]");
  INFER_ERROR("2", op, "[?,?]");
  set_equation(1, "...abc->...c");
  INFER_ERROR("2", op, "[?,?]");

  // Input dimensions are not consistent.
  set_equation(2, "ab,ab->a");
  INFER_ERROR("are 1 and 2", op, "[1,2];[2,1]");
  set_equation(2, "aa,bb->a");
  INFER_ERROR("are 1 and 2", op, "[1,2];[2,2]");

  // Invalid broadcasting dimensions.
  set_equation(2, "...ij,...jk->...ik");
  INFER_ERROR("are 2 and 3", op, "[2,?,?];[3,?,?]");
  set_equation(2, "i...j,jk...->...ik");
  INFER_ERROR("are 2 and 3", op, "[?,2,?];[?,?,3]");
  set_equation(2, "...ij,...jk->ik");
  set_equation(2, "i...j,jk...->ik");
  INFER_ERROR("non-empty broadcasting", op, "[?,2,?];[?,?]");
  set_equation(2, "...ab,b...c->ac...");
  INFER_OK(op, "?;[4,5,3]", "?");
}

TEST(CommonShapeFnsTest, BatchMatMulV2_ShapeFn) {
  ShapeInferenceTestOp op("BatchMatMulV2");
  auto set_adj = [&op](bool adj_x, bool adj_y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_4(mht_4_v, 525, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "BatchMatMulV2")
                     .Input({"a", 0, DT_FLOAT})
                     .Input({"b", 0, DT_FLOAT})
                     .Attr("adj_x", adj_x)
                     .Attr("adj_y", adj_y)
                     .Finalize(&op.node_def));
  };

  set_adj(false, false);

  // Rank checks.
  INFER_ERROR("at least rank 2", op, "[];?");
  INFER_ERROR("at least rank 2", op, "[1];?");
  INFER_ERROR("at least rank 2", op, "?;[]");
  INFER_ERROR("at least rank 2", op, "?;[2]");

  INFER_OK(op, "?;?", "?");

  // 0 batch dims.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");

  // 1 batch dims.
  INFER_OK(op, "[3,?,?];[3,?,?]", "[d0_0,d0_1,d1_2]");
  INFER_OK(op, "[?,?,?];[1,?,?]", "[d0_0,d0_1,d1_2]");
  INFER_OK(op, "[?,?,?];[2,?,?]", "[d1_0,d0_1,d1_2]");
  INFER_OK(op, "[1,?,?];[?,?,?]", "[d1_0,d0_1,d1_2]");
  INFER_OK(op, "[2,?,?];[?,?,?]", "[d0_0,d0_1,d1_2]");
  INFER_OK(op, "[?,?,?];[?,?,?]", "[?,d0_1,d1_2]");

  // Empty batch dim with broadcasting.
  INFER_OK(op, "[?,?];[?,?,?]", "[d1_0,d0_0,d1_2]");
  INFER_OK(op, "[?,?,?];[?,?]", "[d0_0,d0_1,d1_1]");
  INFER_OK(op, "[?,?];[?,?,?,?]", "[d1_0,d1_1,d0_0,d1_3]");
  INFER_OK(op, "[?,?,?,?];[?,?]", "[d0_0,d0_1,d0_2,d1_1]");

  // Unknown number of batch dims.
  INFER_OK(op, "[?,?];?", "?");
  INFER_OK(op, "?;[?,?]", "?");
  INFER_OK(op, "[?,?,?,?];?", "?");

  // Large number of batch dims.
  INFER_OK(op, "[?,?,?,?,?];[1,?,?]", "[d0_0,d0_1,d0_2,d0_3,d1_2]");
  INFER_OK(op, "[1,?,?];[?,?,?,?,?]", "[d1_0,d1_1,d1_2,d0_1,d1_4]");

  // Batch dim mismatch.
  INFER_ERROR("are 2 and 3", op, "[?,?,2,?,?];[3,?,?]");
  INFER_ERROR("are 2 and 3", op, "[2,?,?];[?,?,3,?,?]");

  // Test adj_a, testing output and that inner dims are compared.
  set_adj(false, false);
  INFER_OK(op, "[2,2,3,4];[2,2,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,3,1]");  // inner dim mismatch
  set_adj(true, false);
  INFER_OK(op, "[2,2,3,4];[2,2,?,?]", "[d0_0,d0_1,d0_3,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,3,1]");  // inner dim mismatch

  // Test adj_b=true.
  set_adj(false, true);
  INFER_OK(op, "[2,2,?,?];[2,2,3,4]", "[d0_0,d0_1,d0_2,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,1,3]");  // inner dim mismatch
  set_adj(true, true);
  INFER_OK(op, "[2,2,?,?];[2,2,3,4]", "[d0_0,d0_1,d0_3,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,1,3]");  // inner dim mismatch
}

TEST(CommonShapeFnsTest, BiasAddShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("BiasAdd")
                  .Input("a: float")
                  .Input("b: float")
                  .Output("c: float")
                  .Finalize(&op_reg_data));

  OpDef op_def = op_reg_data.op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                  .Input("a", 0, DT_FLOAT)
                  .Input("b", 0, DT_FLOAT)
                  .Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 10}), S({10})},
                       {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(2, c.Value(c.Dim(output, 0)));
    EXPECT_EQ(10, c.Value(c.Dim(output, 1)));
  }

  {
    // Unknown ranks.
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {Unknown(), Unknown()}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_FALSE(c.RankKnown(output));
  }

  {
    // Rank > 2
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({4, 3, 4, 2, 15}), S({15})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ("[4,3,4,2,15]", c.DebugString(output));
  }

  {
    // NCHW format
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({2, 3, 4, 5}), S({3})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ("[2,3,4,5]", c.DebugString(output));
  }

  {
    // NCHW format with high input rank
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({8, 6, 4, 2, 3, 4, 5}), S({3})}, {}, {}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }

  {
    // NCHW format with input rank 3
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({10, 11, 12}), S({11})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ("[10,11,12]", c.DebugString(output));
  }

  {
    // Input rank not high enough
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({3}), S({3})}, {},
                       {}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }

  {
    // NCHW rank not high enough
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAdd")
                    .Input("a", 0, DT_FLOAT)
                    .Input("b", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    // NCHW format
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 3}), S({3})},
                       {}, {}, {});
    EXPECT_FALSE(BiasAddShape(&c).ok());
  }
}

TEST(CommonShapeFnsTest, FusedBatchNormExTest) {
  ShapeInferenceTestOp op("_FusedBatchNormEx");

  std::vector<NodeDefBuilder::NodeOut> no_side_inputs;
  TF_CHECK_OK(NodeDefBuilder("test", "_FusedBatchNormEx")
                  .Input("x", 0, DT_HALF)
                  .Input("scale", 0, DT_FLOAT)
                  .Input("offset", 0, DT_FLOAT)
                  .Input("mean", 0, DT_FLOAT)
                  .Input("variance", 0, DT_FLOAT)
                  .Input(no_side_inputs)
                  .Attr("T", DT_HALF)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Attr("data_format", "NHWC")
                  .Attr("activation_mode", "Relu")
                  .Attr("num_side_inputs", 0)
                  .Attr("is_training", true)
                  .Finalize(&op.node_def));

  // Channels are not multiple of 4.
  INFER_ERROR("must be divisible by 4", op, "[2,2,2,2];[2];[2];[2];[2]");

  INFER_OK(op, "[2,2,2,4];[4];[4];[4];[4]",
           "[d0_0,d0_1,d0_2,d0_3];[d0_3];[d0_3];[d0_3];[d0_3];?");
}

TEST(CommonShapeFnsTest, BiasAddGradShapeTest) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("BiasAddGrad")
                  .Input("a: float")
                  .Output("b: float")
                  .Finalize(&op_reg_data));

  OpDef op_def = op_reg_data.op_def;
  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                  .Input("a", 0, DT_FLOAT)
                  .Finalize(&def));

  {
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 10})}, {}, {},
                       {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(10, c.Value(c.Dim(output, 0)));
  }

  {
    // Rank > 2
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({5, 7, 2, 10})},
                       {}, {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(10, c.Value(c.Dim(output, 0)));
  }

  {
    // NCHW format
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 3, 4, 5})}, {},
                       {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(3, c.Value(c.Dim(output, 0)));
  }

  {
    // NCHW format with high input rank
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                       {S({8, 6, 4, 2, 3, 4, 5})}, {}, {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(6, c.Value(c.Dim(output, 0)));
  }

  {
    // NCHW format with input rank 3
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({10, 11, 12})}, {},
                       {}, {});
    TF_EXPECT_OK(BiasAddGradShape(&c));
    ShapeHandle output = c.output(0);
    EXPECT_EQ(11, c.Value(c.Dim(output, 0)));
  }

  {
    // Input rank not high enough
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({3})}, {}, {}, {});
    EXPECT_FALSE(BiasAddGradShape(&c).ok());
  }

  {
    // NCHW rank not high enough
    TF_CHECK_OK(NodeDefBuilder("test", "BiasAddGrad")
                    .Input("a", 0, DT_FLOAT)
                    .Attr("data_format", "NCHW")
                    .Finalize(&def));
    // NCHW format
    InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def, {S({2, 3})}, {}, {},
                       {});
    EXPECT_FALSE(BiasAddGradShape(&c).ok());
  }
}

TEST(CommonShapeFnsTest, Conv2DShapeTest) {
  ShapeInferenceTestOp op("Conv2D");
  auto set_op = [&op](const std::vector<int32>& strides, const string& padding,
                      const string& data_format, const string& filter_format,
                      const std::vector<int32>& explicit_paddings = {}) {
    TF_CHECK_OK(NodeDefBuilder("test", "Conv2D")
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Attr("explicit_paddings", explicit_paddings)
                    .Attr("data_format", data_format)
                    .Attr("filter_format", filter_format)
                    .Finalize(&op.node_def));
  };

  // Invalid rank for input
  INFER_ERROR("must be rank 4", op, "[4,4];[2,1,1,1]");
  // Invalid rank for filter
  INFER_ERROR("must be rank 4", op, "[1,4,4,1];[2,1,1]");

  // Invalid value for strides
  set_op({{1, 1, 0, 1}}, "VALID", "NHWC", "HWIO");
  INFER_ERROR("must be > 0", op, "[1,2,2,1];[1,1,1,1]");

  // 1x1 filter
  set_op({{1, 1, 1, 1}}, "VALID", "NHWC", "HWIO");
  INFER_OK(op, "[1,2,2,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 2x2 filter
  set_op({{1, 1, 1, 1}}, "VALID", "NHWC", "HWIO");
  INFER_OK(op, "[1,2,2,1];[2,2,1,1]", "[d0_0,1,1,d1_3]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op({{1, 2, 2, 1}}, "VALID", "NHWC", "HWIO");
  INFER_OK(op, "[1,3,3,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 3x3 input, 1x1 filter, 2x1 stride
  set_op({{1, 2, 1, 1}}, "VALID", "NHWC", "HWIO");
  INFER_OK(op, "[1,3,3,1];[1,1,1,1]", "[d0_0,2,3,d1_3]");

  // 4x4 input, 2x1 filter, 1x2 stride
  set_op({{1, 1, 2, 1}}, "VALID", "NHWC", "HWIO");
  INFER_OK(op, "[1,4,4,1];[2,1,1,1]", "[d0_0,3,2,d1_3]");

  // Unknown dims in the critical fields lead to partial inference.
  INFER_OK(op, "[1,4,4,1];[2,1,1,1]", "[d0_0,3,2,d1_3]");
  INFER_OK(op, "[1,?,4,1];[2,1,1,1]", "[d0_0,?,2,d1_3]");
  INFER_OK(op, "[1,4,?,1];[2,1,1,1]", "[d0_0,3,?,d1_3]");
  INFER_OK(op, "[1,4,4,?];[2,1,1,1]", "[d0_0,3,2,d1_3]");
  INFER_OK(op, "[1,4,4,1];[?,1,1,1]", "[d0_0,?,2,d1_3]");
  INFER_OK(op, "[1,4,4,1];[2,?,1,1]", "[d0_0,3,?,d1_3]");

  // input depths must be multiple of filter.
  INFER_ERROR(
      "Depth of input (10) is not a multiple of input depth of filter (10000)",
      op, "[1,2,2,10];[1,1,10000,20]");

  // Tests for NCHW
  // 1x1 filter
  set_op({{1, 1, 1, 1}}, "VALID", "NCHW", "HWIO");
  INFER_OK(op, "[1,1,2,2];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 2x2 filter
  set_op({{1, 1, 1, 1}}, "VALID", "NCHW", "HWIO");
  INFER_OK(op, "[1,1,2,2];[2,2,1,1]", "[d0_0,d1_3,1,1]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op({{1, 1, 2, 2}}, "VALID", "NCHW", "HWIO");
  INFER_OK(op, "[1,1,3,3];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 3x3 input, 1x1 filter, 2x1 stride
  set_op({{1, 1, 2, 1}}, "VALID", "NCHW", "HWIO");
  INFER_OK(op, "[1,1,3,3];[1,1,1,1]", "[d0_0,d1_3,2,3]");

  // 4x4 input, 2x1 filter, 1x2 stride
  set_op({{1, 1, 1, 2}}, "VALID", "NCHW", "HWIO");
  INFER_OK(op, "[1,1,4,4];[2,1,1,1]", "[d0_0,d1_3,3,2]");

  // Tests for NCHW_VECT_C
  // 1x1 filter
  set_op({{1, 1, 1, 1}}, "VALID", "NCHW_VECT_C", "OIHW_VECT_I");
  INFER_OK(op, "[1,1,2,2,4];[4,1,1,1,4]", "[d0_0,1,2,2,d0_4]");

  // 2x2 filter
  set_op({{1, 1, 1, 1}}, "VALID", "NCHW_VECT_C", "OIHW_VECT_I");
  INFER_OK(op, "[1,1,2,2,4];[4,1,2,2,4]", "[d0_0,1,1,1,d0_4]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op({{1, 1, 2, 2}}, "VALID", "NCHW_VECT_C", "OIHW_VECT_I");
  INFER_OK(op, "[1,1,3,3,4];[8,1,1,1,4]", "[d0_0,2,2,2,d0_4]");

  // 3x3 input, 1x1 filter, 2x1 stride
  set_op({{1, 1, 2, 1}}, "VALID", "NCHW_VECT_C", "OIHW_VECT_I");
  INFER_OK(op, "[1,1,3,3,4];[4,1,1,1,4]", "[d0_0,1,2,3,d0_4]");

  // 4x4 input, 2x1 filter, 1x2 stride
  set_op({{1, 1, 1, 2}}, "VALID", "NCHW_VECT_C", "OIHW_VECT_I");
  INFER_OK(op, "[1,1,4,4,4];[4,1,2,1,4]", "[d0_0,1,3,2,d0_4]");

  // int8x32 input.
  set_op({{1, 1, 1, 2}}, "VALID", "NCHW_VECT_C", "OIHW_VECT_I");
  INFER_OK(op, "[1,1,4,4,32];[32,1,2,1,32]", "[d0_0,1,3,2,d0_4]");

  // Some tests for "SAME" padding

  // 4x4 input, 1x1 filter, 1x1 stride
  set_op({{1, 1, 1, 1}}, "SAME", "NHWC", "HWIO");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 3x3 input, 2x2 filter, 1x1 stride
  set_op({{1, 1, 1, 1}}, "SAME", "NHWC", "HWIO");
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 stride
  set_op({{1, 2, 2, 1}}, "SAME", "NHWC", "HWIO");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,2,2,d1_3]");

  // 4x4 input, 2x2 filter, 1x1 stride
  set_op({{1, 1, 1, 1}}, "SAME", "NHWC", "HWIO");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // With stride 1x1 and SAME, unknown dims don't matter - filter dims except
  // for output channels are ignored for output, so all inputs are carried
  // through to output.
  set_op({{1, 1, 1, 1}}, "SAME", "NHWC", "HWIO");
  INFER_OK(op, "[1,4,4,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[1,?,4,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[1,4,?,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[1,4,4,?];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_OK(op, "[?,4,4,1];[?,?,?,?]", "[d0_0,d0_1,d0_2,d1_3]");

  // With stride != 1, the input HW dims are divided to produce output dims.
  set_op({{1, 2, 2, 1}}, "SAME", "NHWC", "HWIO");
  INFER_OK(op, "[?,4,4,1];[?,?,?,?]", "[d0_0,2,2,d1_3]");
  INFER_OK(op, "[1,?,4,1];[?,?,?,?]", "[d0_0,?,2,d1_3]");
  INFER_OK(op, "[1,4,?,1];[?,?,?,?]", "[d0_0,2,?,d1_3]");
  INFER_OK(op, "[1,4,4,?];[?,?,?,?]", "[d0_0,2,2,d1_3]");

  // Some tests for "EXPLICIT" padding

  // 4x4 input, 1x1 filter, 1x1 stride, [0, 2, 1, 4] padding
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 0, 2, 1, 4, 0, 0});
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,6,9,d1_3]");

  // 3x3 input, 2x2 filter, 1x1 stride, [1, 0, 1, 2] padding
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 1, 0, 1, 2, 0, 0});
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,3,5,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 stride, [3, 2, 1, 0] padding
  set_op({{1, 2, 2, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 3, 2, 1, 0, 0, 0});
  INFER_OK(op, "[1,4,4,2];[2,2,2,3]", "[d0_0,4,2,d1_3]");

  // 2x2 input, 2x1 filter, 1x2 stride, [1, 1, 2, 2] padding
  set_op({{1, 1, 2, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 1, 1, 2, 2, 0, 0});
  INFER_OK(op, "[1,2,2,1];[2,1,1,1]", "[d0_0,3,3,d1_3]");

  // Unknown dims in the critical fields lead to partial inference.
  INFER_OK(op, "[1,4,4,1];[2,1,1,1]", "[d0_0,5,4,d1_3]");
  INFER_OK(op, "[1,?,4,1];[2,1,1,1]", "[d0_0,?,4,d1_3]");
  INFER_OK(op, "[1,4,?,1];[2,1,1,1]", "[d0_0,5,?,d1_3]");
  INFER_OK(op, "[1,4,4,?];[2,1,1,1]", "[d0_0,5,4,d1_3]");
  INFER_OK(op, "[1,4,4,1];[?,1,1,1]", "[d0_0,?,4,d1_3]");
  INFER_OK(op, "[1,4,4,1];[2,?,1,1]", "[d0_0,5,?,d1_3]");

  // Explicit padding errors
  // Negative padding
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 0, -1, 0, 0, 0, 0});
  INFER_ERROR("must be nonnegative", op, "[1,2,2,1];[1,1,1,1]");

  // Too little padding (7 explicit paddings instead of 8)
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("must contain 8 values", op, "[1,2,2,1];[1,1,1,1]");

  // Too much padding (9 explicit paddings instead of 8)
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO",
         {0, 0, 0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("must contain 8 values", op, "[1,2,2,1];[1,1,1,1]");

  // Padding in batch dimension
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO", {1, 0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("batch or depth dimensions", op, "[1,2,2,1];[1,1,1,1]");

  // Padding in depth dimension
  set_op({{1, 1, 1, 1}}, "EXPLICIT", "NHWC", "HWIO", {0, 0, 0, 0, 0, 0, 1, 0});
  INFER_ERROR("batch or depth dimensions", op, "[1,2,2,1];[1,1,1,1]");

  // Padding explicit_paddings when padding is not EXPLICIT
  set_op({{1, 1, 1, 1}}, "VALID", "NHWC", "HWIO", {0, 0, 0, 0, 0, 0, 0, 0});
  INFER_ERROR("must be empty", op, "[1,2,2,1];[1,1,1,1]");
}

TEST(CommonShapeFnsTest, Conv2DDilatedShapeTest) {
  ShapeInferenceTestOp op("Conv2D");
  auto set_op = [&op](const std::vector<int32>& dilations,
                      const std::vector<int32>& strides, const string& padding,
                      const string& data_format,
                      const std::vector<int32>& explicit_paddings = {}) {
    TF_CHECK_OK(NodeDefBuilder("test", "Conv2D")
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("dilations", dilations)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Attr("explicit_paddings", explicit_paddings)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
  };

  // Invalid rank for dilation
  set_op({{1, 2, 1}}, {{1, 1, 1, 1}}, "VALID", "NHWC");
  INFER_ERROR("contain 4 values", op, "[1,2,2,1];[1,1,1,1]");

  // Invalid value for dilation
  set_op({{1, 0, 1, 1}}, {{1, 1, 1, 1}}, "VALID", "NHWC");
  INFER_ERROR("must be >= 1", op, "[1,2,2,1];[1,1,1,1]");

  // Tests for NHWC
  // 1x1 filter, 2x1 dilations, 1x1 strides
  set_op({{1, 2, 1, 1}}, {{1, 1, 1, 1}}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 1x1 filter, 2x1 dilations, 2x1 strides
  set_op({{1, 2, 1, 1}}, {{1, 2, 1, 1}}, "VALID", "NHWC");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,2,4,d1_3]");

  // 1x1 filter, 2x1 dilations, 2x2 strides
  set_op({{1, 2, 1, 1}}, {{1, 2, 2, 1}}, "VALID", "NHWC");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,2,2,d1_3]");

  // 3x3 filter, 2x1 dilations, 1x1 strides
  set_op({{1, 2, 1, 1}}, {{1, 1, 1, 1}}, "VALID", "NHWC");
  INFER_OK(op, "[1,5,5,1];[3,3,1,1]", "[d0_0,1,3,d1_3]");

  // 3x3 filter, 2x1 dilations, 2x1 strides
  set_op({{1, 2, 1, 1}}, {{1, 2, 1, 1}}, "VALID", "NHWC");
  INFER_OK(op, "[1,5,5,1];[3,3,1,1]", "[d0_0,1,3,d1_3]");

  // 3x3 filter, 1x2 dilations, 2x2 strides
  set_op({{1, 1, 2, 1}}, {{1, 2, 2, 1}}, "VALID", "NHWC");
  INFER_OK(op, "[1,5,5,1];[3,3,1,1]", "[d0_0,2,1,d1_3]");

  // Tests for NCHW
  // 1x1 filter, 2x1 dilations, 1x1 strides
  set_op({{1, 1, 2, 1}}, {{1, 1, 1, 1}}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,2,2];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 1x1 filter, 2x1 dilations, 2x1 strides
  set_op({{1, 1, 2, 1}}, {{1, 1, 2, 1}}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,4,4];[1,1,1,1]", "[d0_0,d1_3,2,4]");

  // 1x1 filter, 2x1 dilations, 2x2 strides
  set_op({{1, 1, 2, 1}}, {{1, 1, 2, 2}}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,4,4];[1,1,1,1]", "[d0_0,d1_3,2,2]");

  // 3x3 filter, 2x1 dilations, 1x1 strides
  set_op({{1, 1, 2, 1}}, {{1, 1, 1, 1}}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,5,5];[3,3,1,1]", "[d0_0,d1_3,1,3]");

  // 3x3 filter, 2x1 dilations, 2x1 strides
  set_op({{1, 1, 2, 1}}, {{1, 1, 2, 1}}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,5,5];[3,3,1,1]", "[d0_0,d1_3,1,3]");

  // 3x3 filter, 1x2 dilations, 2x2 strides
  set_op({{1, 1, 1, 2}}, {{1, 1, 2, 2}}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,5,5];[3,3,1,1]", "[d0_0,d1_3,2,1]");

  // Some tests for "SAME" padding

  // 4x4 input, 1x1 filter, 2x1 dilations, 1x1 stride
  set_op({{1, 2, 1, 1}}, {{1, 1, 1, 1}}, "SAME", "NHWC");
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 3x3 input, 2x2 filter, 2x2 dilations, 1x1 stride
  set_op({{1, 2, 2, 1}}, {{1, 1, 1, 1}}, "SAME", "NHWC");
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // 4x4 input, 2x2 filter, 1x2 dilations, 2x2 stride
  set_op({{1, 1, 2, 1}}, {{1, 2, 2, 1}}, "SAME", "NHWC");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,2,2,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 dilations, 1x1 stride
  set_op({{1, 2, 2, 1}}, {{1, 1, 1, 1}}, "SAME", "NHWC");
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,d0_1,d0_2,d1_3]");

  // Some tests for "EXPLICIT" padding

  // 4x4 input, 1x1 filter, 2x1 dilations, 1x1 stride, [0, 2, 1, 4] padding
  set_op({{1, 2, 1, 1}}, {{1, 1, 1, 1}}, "EXPLICIT", "NHWC",
         {0, 0, 0, 2, 1, 4, 0, 0});
  INFER_OK(op, "[1,4,4,1];[1,1,1,1]", "[d0_0,6,9,d1_3]");

  // 3x3 input, 2x2 filter, 2x2 dilations, 1x1 stride, [1, 0, 1, 2] padding
  set_op({{1, 2, 2, 1}}, {{1, 1, 1, 1}}, "EXPLICIT", "NHWC",
         {0, 0, 1, 0, 1, 2, 0, 0});
  INFER_OK(op, "[1,3,3,1];[2,2,1,1]", "[d0_0,2,4,d1_3]");

  // 4x4 input, 2x2 filter, 1x2 dilations, 2x2 stride, [3, 2, 1, 0] padding
  set_op({{1, 1, 2, 1}}, {{1, 2, 2, 1}}, "EXPLICIT", "NHWC",
         {0, 0, 3, 2, 1, 0, 0, 0});
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,4,2,d1_3]");

  // 4x4 input, 2x2 filter, 2x2 dilations, 1x1 stride, [1, 1, 2, 2] padding
  set_op({{1, 2, 2, 1}}, {{1, 1, 1, 1}}, "EXPLICIT", "NHWC",
         {0, 0, 1, 1, 2, 2, 0, 0});
  INFER_OK(op, "[1,4,4,1];[2,2,1,1]", "[d0_0,4,6,d1_3]");
}

TEST(CommonShapeFnsTest, Conv3DShapeTest) {
  ShapeInferenceTestOp op("Conv3D");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const string& padding) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("padding: \"" + padding + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_5(mht_5_v, 1124, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_CHECK_OK(NodeDefBuilder("test", "Conv3D")
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  // Invalid rank for input
  INFER_ERROR("must be rank 5", op, "[4,4];[2,1,1,1]");
  // Invalid rank for filter
  INFER_ERROR("must be rank 5", op, "[1,4,4,1];[2,1,1]");

  // Invalid value for strides
  set_op({{1, 1, 1, 0, 1}}, "VALID");
  INFER_ERROR("must be > 0", op, "[1,2,2,2,1];[1,1,1,1,1]");

  // 1x1x1 filter
  set_op({{1, 1, 1, 1, 1}}, "VALID");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // unknown dims in the critical fields give partial inference.
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");
  INFER_OK(op, "[1,?,2,2,1];[1,1,1,1,1]", "[d0_0,?,2,2,d1_4]");
  INFER_OK(op, "[1,2,?,2,1];[1,1,1,1,1]", "[d0_0,2,?,2,d1_4]");
  INFER_OK(op, "[1,2,2,?,1];[1,1,1,1,1]", "[d0_0,2,2,?,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[?,1,1,1,1]", "[d0_0,?,2,2,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,?,1,1,1]", "[d0_0,2,?,2,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,1,?,1,1]", "[d0_0,2,2,?,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,?,1]", "[d0_0,2,2,2,d1_4]");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,?]", "[d0_0,2,2,2,d1_4]");

  // input depth must be multiple of filter depth for group convolutions
  INFER_ERROR(
      "Depth of input (10) is not a multiple of input depth of filter (6)", op,
      "[1,2,2,2,10];[1,1,1,6,20]");

  // Output dimensions must be multiple of group number
  INFER_ERROR(
      "Depth of output (1) is not a multiple of the number of groups (2)", op,
      "[1,2,2,2,10];[1,1,1,5,1]");

  // 2x2x2 filter
  set_op({{1, 1, 1, 1, 1}}, "VALID");
  INFER_OK(op, "[1,2,2,2,1];[2,2,2,1,1]", "[d0_0,1,1,1,d1_4]");

  // 3x3 input, 1x1 filter, 2x2 stride
  set_op({{1, 2, 2, 2, 1}}, "VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // 3x3 input, 1x1 filter, 2x1x1 stride
  set_op({{1, 2, 1, 1, 1}}, "VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,3,3,d1_4]");

  // 4x4 input, 2x2 filter, 1x1 stride
  set_op({{1, 1, 1, 1, 1}}, "SAME");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // 4x4 input of depth 10, 2x2 filter with depth 5, 1x1 stride
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,5,2]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // test output multiple of group size is ok
  // 4x4 input of depth 10, 2x2 filter with depth 5, 1x1 stride
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,5,2]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // Depthwise convolution first step
  // 4x4 input of depth 10, 2x2 filter with depth 1, 1x1 stride
  INFER_OK(op, "[1,4,4,4,10];[2,2,2,1,10]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // with SAME, filter doesn't matter except for last dim.
  set_op({{1, 1, 1, 1, 1}}, "SAME");
  INFER_OK(op, "[?,4,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,?,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,?,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,?,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,?];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[?,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,?,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,2,?,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,?,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,1,?]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");

  // with SAME, and stride != 1, division happens to produce output.
  set_op({{1, 2, 3, 4, 1}}, "SAME");
  INFER_OK(op, "[1,4,9,4,1];[2,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[?,4,9,4,1];[2,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,?,9,4,1];[2,2,2,1,1]", "[d0_0,?,3,1,d1_4]");
  INFER_OK(op, "[1,4,?,4,1];[2,2,2,1,1]", "[d0_0,2,?,1,d1_4]");
  INFER_OK(op, "[1,4,9,?,1];[2,2,2,1,1]", "[d0_0,2,3,?,d1_4]");
  INFER_OK(op, "[1,4,9,4,?];[2,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[?,2,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,?,2,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,2,?,1,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,2,2,?,1]", "[d0_0,2,3,1,d1_4]");
  INFER_OK(op, "[1,4,9,4,1];[2,2,2,1,?]", "[d0_0,2,3,1,d1_4]");
}

TEST(CommonShapeFnsTest, Conv3DDilatedShapeTest) {
  ShapeInferenceTestOp op("Conv3D");
  auto set_op = [&op](const std::vector<int32>& dilations,
                      const std::vector<int32>& strides,
                      const string& padding) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("padding: \"" + padding + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_6(mht_6_v, 1230, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_CHECK_OK(NodeDefBuilder("test", "Conv3D")
                    .Input("input", 0, DT_FLOAT)
                    .Input("filter", 0, DT_FLOAT)
                    .Attr("dilations", dilations)
                    .Attr("strides", strides)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  // Invalid rank for dilation
  set_op({{1, 2, 1, 1}}, {{1, 1, 1, 1, 1}}, "VALID");
  INFER_ERROR("contain 5 values", op, "[1,2,2,2,1];[1,1,1,1,1]");

  // Invalid value for dilation
  set_op({{1, 2, 0, 1, 1}}, {{1, 1, 1, 1, 1}}, "VALID");
  INFER_ERROR("must be >= 1", op, "[1,2,2,2,1];[1,1,1,1,1]");

  // 2x1x1 dilation 1x1x1 filter
  set_op({{1, 2, 1, 1, 1}}, {{1, 1, 1, 1, 1}}, "VALID");
  INFER_OK(op, "[1,2,2,2,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // 2x1x1 dilation 2x2x2 filter
  set_op({{1, 2, 1, 1, 1}}, {{1, 1, 1, 1, 1}}, "VALID");
  INFER_OK(op, "[1,3,2,2,1];[2,2,2,1,1]", "[d0_0,1,1,1,d1_4]");

  // 2x1x1 dilation 3x3x3 input, 1x1x1 filter, 2x2x2 stride
  set_op({{1, 2, 1, 1, 1}}, {{1, 2, 2, 2, 1}}, "VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,2,2,d1_4]");

  // 2x1x1 dilation 3x3x3 input, 1x1x1 filter, 2x1x1 stride
  set_op({{1, 2, 1, 1, 1}}, {{1, 2, 1, 1, 1}}, "VALID");
  INFER_OK(op, "[1,3,3,3,1];[1,1,1,1,1]", "[d0_0,2,3,3,d1_4]");

  // 2x1x1 dilation 4x4x4 input, 2x2x2 filter, 1x1x1 stride
  set_op({{1, 2, 1, 1, 1}}, {{1, 1, 1, 1, 1}}, "SAME");
  INFER_OK(op, "[1,4,4,4,1];[2,2,2,1,1]", "[d0_0,d0_1,d0_2,d0_3,d1_4]");
}

TEST(CommonShapeFnsTest, DepthwiseConv2DShapeTest) {
  ShapeInferenceTestOp op("DepthwiseConv2dNative");
  std::vector<int32> strides = {{1, 1, 1, 1}};
  TF_CHECK_OK(NodeDefBuilder("test", "DepthwiseConv2dNative")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter", 0, DT_FLOAT)
                  .Attr("strides", strides)
                  .Attr("padding", "VALID")
                  .Attr("data_format", "NHWC")
                  .Finalize(&op.node_def));

  // Most of DepthwiseConv2D is implicitly tested by Conv2D, so
  // we test only the very-specific differences here.

  // 1x1 filter, depth multiplication
  INFER_OK(op, "[1,2,2,3];[1,1,3,4]", "[d0_0,2,2,12]");

  // Input depths not compatible
  INFER_ERROR("Dimensions must be equal, but are 3 and 12", op,
              "[1,2,2,3];[1,1,12,4]");

  // No unknown dims in the critical fields.
  INFER_OK(op, "[1,2,2,3];[1,1,3,4]", "[d0_0,2,2,12]");
  INFER_OK(op, "[1,?,2,3];[1,1,3,4]", "[d0_0,?,2,12]");
  INFER_OK(op, "[1,2,?,3];[1,1,3,4]", "[d0_0,2,?,12]");
  INFER_OK(op, "[1,2,2,3];[?,1,3,4]", "[d0_0,?,2,12]");
  INFER_OK(op, "[1,2,2,3];[1,?,3,4]", "[d0_0,2,?,12]");
  INFER_OK(op, "[1,2,2,3];[1,1,?,4]", "[d0_0,2,2,12]");
  INFER_OK(op, "[1,2,2,?];[1,1,?,4]", "[d0_0,2,2,?]");
  INFER_OK(op, "[1,2,2,3];[1,1,3,?]", "[d0_0,2,2,?]");

  // Test for NCHW format.
  TF_CHECK_OK(NodeDefBuilder("test", "DepthwiseConv2dNative")
                  .Input("input", 0, DT_FLOAT)
                  .Input("filter", 0, DT_FLOAT)
                  .Attr("strides", strides)
                  .Attr("padding", "VALID")
                  .Attr("data_format", "NCHW")
                  .Finalize(&op.node_def));

  // 1x1 filter, depth multiplication
  INFER_OK(op, "[1,3,2,2];[1,1,3,4]", "[d0_0,12,2,2]");
}

TEST(CommonShapeFnsTest, AvgPool2DShapeTest) {
  ShapeInferenceTestOp op("AvgPool");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& ksizes, const string& padding,
                      const string& data_format) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("padding: \"" + padding + "\"");
   mht_7_v.push_back("data_format: \"" + data_format + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_7(mht_7_v, 1322, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_CHECK_OK(NodeDefBuilder("test", "AvgPool")
                    .Input("input", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("ksize", ksizes)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check the very-specific avgpooling features here.

  // 1x1 filter, 1x1 stride
  set_op({1, 1, 1, 1}, {1, 1, 1, 1}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,1]", "[d0_0,2,2,d0_3]");

  // 4x4 input, 2x1 ksize, 1x2 stride
  set_op({1, 1, 2, 1}, {1, 2, 1, 1}, "VALID", "NHWC");
  INFER_OK(op, "[1,4,4,1]", "[d0_0,3,2,d0_3]");

  // 4x4 input, 2x1 ksize, 1x2 stride
  // unknown dims in the critical fields lead to partial inference.
  // Assumes NHWC format.
  INFER_OK(op, "[1,?,4,1]", "[d0_0,?,2,d0_3]");
  INFER_OK(op, "[1,4,?,1]", "[d0_0,3,?,d0_3]");

  // 4x4 input, 2x1 ksize, 1x2 stride, NCHW format
  set_op({{1, 1, 1, 2}}, {1, 1, 2, 1}, "VALID", "NCHW");
  INFER_OK(op, "[1,1,4,4]", "[d0_0,d0_1,3,2]");

  // 5x7 input, 2x2 ksize, 1x1 stride, NCHW_VECT_C test
  set_op({{1, 1, 1, 1}}, {1, 1, 2, 2}, "VALID", "NCHW_VECT_C");
  INFER_OK(op, "[2,3,5,7,4]", "[d0_0,d0_1,4,6,4]");
  INFER_OK(op, "[5,7,?,?,4]", "[d0_0,d0_1,?,?,4]");
  INFER_OK(op, "[?,?,?,?,4]", "[d0_0,d0_1,?,?,4]");
  INFER_ERROR("must be 4 or 32, but is 3", op, "[2,5,7,11,3]");

  // Invalid rank for input
  INFER_ERROR("Shape must be rank", op, "[4,4]");
}

TEST(CommonShapeFnsTest, MaxPool2DShapeTest) {
  ShapeInferenceTestOp op("MaxPool");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& ksizes, const string& padding,
                      const string& data_format) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("padding: \"" + padding + "\"");
   mht_8_v.push_back("data_format: \"" + data_format + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_8(mht_8_v, 1373, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_CHECK_OK(NodeDefBuilder("test", "MaxPool")
                    .Input("input", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("ksize", ksizes)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check the very-specific maxpooling features here,
  // namely depthwise kernel and striding.

  // all 1 strides, depth 2 filter
  set_op({1, 1, 1, 1}, {1, 1, 1, 2}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,2]", "[d0_0,2,2,1]");

  // depth 3 stride, 1x1x1 filter, NCHW
  set_op({1, 3, 1, 1}, {1, 1, 1, 1}, "VALID", "NCHW");
  INFER_OK(op, "[1,7,5,5]", "[d0_0,3,5,5]");

  // 5x7 input, 2x2 ksize, 1x1 stride, NCHW_VECT_C tests
  set_op({{1, 1, 1, 1}}, {1, 1, 2, 2}, "SAME", "NCHW_VECT_C");
  INFER_OK(op, "[2,3,5,7,4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[5,7,?,?,4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[?,?,?,?,4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_ERROR("must be 4 or 32, but is 8", op, "[2,3,5,7,8]");
}

TEST(CommonShapeFnsTest, MaxPoolV22DShapeTest) {
  ShapeInferenceTestOp op("MaxPoolV2");
  Tensor ksizes_tensor, strides_tensor;
  auto set_op = [&op, &ksizes_tensor, &strides_tensor](
                    const std::vector<int32>& strides,
                    const std::vector<int32>& ksizes, const string& padding,
                    const string& data_format) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("padding: \"" + padding + "\"");
   mht_9_v.push_back("data_format: \"" + data_format + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_9(mht_9_v, 1414, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_CHECK_OK(NodeDefBuilder("test", "MaxPoolV2")
                    .Input("input", 0, DT_FLOAT)
                    .Input("ksize", 1, DT_INT32)
                    .Input("strides", 2, DT_INT32)
                    .Attr("padding", padding)
                    .Attr("data_format", data_format)
                    .Finalize(&op.node_def));
    ksizes_tensor = test::AsTensor<int32>(ksizes);
    op.input_tensors.resize(3);
    op.input_tensors[0] = nullptr;
    op.input_tensors[1] = &ksizes_tensor;
    strides_tensor = test::AsTensor<int32>(strides);
    op.input_tensors[2] = &strides_tensor;
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check the very-specific maxpooling features here,
  // namely depthwise kernel and striding.

  // all 1 strides, depth 2 filter
  set_op({1, 1, 1, 1}, {1, 1, 1, 2}, "VALID", "NHWC");
  INFER_OK(op, "[1,2,2,2];[4];[4]", "[d0_0,2,2,1]");

  // depth 3 stride, 1x1x1 filter, NCHW
  set_op({1, 3, 1, 1}, {1, 1, 1, 1}, "VALID", "NCHW");
  INFER_OK(op, "[1,7,5,5];[4];[4]", "[d0_0,3,5,5]");

  // 5x7 input, 2x2 ksize, 1x1 stride, NCHW_VECT_C tests
  set_op({{1, 1, 1, 1}}, {1, 1, 2, 2}, "SAME", "NCHW_VECT_C");
  INFER_OK(op, "[2,3,5,7,4];[4];[4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[5,7,?,?,4];[4];[4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_OK(op, "[?,?,?,?,4];[4];[4]", "[d0_0,d0_1,d0_2,d0_3,4]");
  INFER_ERROR("must be 4 or 32, but is 8", op, "[2,3,5,7,8];[4];[4]");
}

TEST(CommonShapeFnsTest, Pool3DShapeTest) {
  ShapeInferenceTestOp op("MaxPool3D");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& ksizes, const string& padding) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("padding: \"" + padding + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPScommon_shape_fns_testDTcc mht_10(mht_10_v, 1457, "", "./tensorflow/core/framework/common_shape_fns_test.cc", "lambda");

    TF_CHECK_OK(NodeDefBuilder("test", "MaxPool3D")
                    .Input("input", 0, DT_FLOAT)
                    .Attr("strides", strides)
                    .Attr("ksize", ksizes)
                    .Attr("padding", padding)
                    .Finalize(&op.node_def));
  };

  // Most of the functionality is tested by conv-like shapes,
  // so we check that we handle the extra dimension properly.

  // 2x3x4 stride, 1x1x1 filter.
  set_op({1, 2, 3, 4, 1}, {1, 1, 1, 1, 1}, "VALID");
  INFER_OK(op, "[1,24,24,24,1]", "[d0_0,12,8,6,d0_4]");

  // Test partially known dimensions
  set_op({1, 1, 3, 4, 1}, {1, 1, 1, 1, 1}, "VALID");
  INFER_OK(op, "[1,?,24,24,1]", "[d0_0,?,8,6,d0_4]");
}

TEST(CommonShapeFnsTest, UnknownShapeTest) {
  {
    // Single output
    ShapeInferenceTestOp op("QueueDequeue");
    TF_CHECK_OK(NodeDefBuilder("test", "QueueDequeue")
                    .Input("handle", 0, DT_STRING_REF)
                    .Attr("component_types", {DT_FLOAT})
                    .Finalize(&op.node_def));
    INFER_OK(op, "[1]", "?");
  }

  {
    // Multiple outputs
    ShapeInferenceTestOp op("QueueDequeue");
    TF_CHECK_OK(NodeDefBuilder("test", "QueueDequeue")
                    .Input("handle", 0, DT_STRING_REF)
                    .Attr("component_types", {DT_FLOAT, DT_FLOAT, DT_STRING})
                    .Finalize(&op.node_def));
    INFER_OK(op, "[1]", "?;?;?");
  }
}

TEST(CommonShapeFnsTest, Reduce_ShapeFn) {
  ShapeInferenceTestOp op("Sum");
  op.input_tensors.resize(2);

  TF_ASSERT_OK(NodeDefBuilder("test", "Sum")
                   .Input("input", 0, DT_FLOAT)
                   .Input("reduction_indices", 1, DT_INT32)
                   .Attr("keep_dims", false)
                   .Finalize(&op.node_def));

  // Reduction indices not available, so output is unknown.
  INFER_OK(op, "[2,4,5];[2]", "?");
  INFER_OK(op, "?;[2]", "?");

  Tensor indices = test::AsTensor<int32>({1, 2});
  op.input_tensors[1] = &indices;

  // Reduction indices available
  INFER_OK(op, "[2,4,5];[2]", "[d0_0]");

  // Wrapped indices
  indices = test::AsTensor<int32>({-1, -2});
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[2]", "[d0_0]");

  // Scalar
  indices = test::AsScalar<int32>(0);
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[]", "[d0_1,d0_2]");

  indices = test::AsScalar<int32>(-4);
  op.input_tensors[1] = &indices;
  INFER_ERROR("Invalid reduction dimension", op, "[2,4,5];[]");

  // Empty reduction indices
  indices = test::AsTensor<int32>({});
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[0]", "[d0_0,d0_1,d0_2]");

  // Keep dims = true
  TF_ASSERT_OK(NodeDefBuilder("test", "Sum")
                   .Input("input", 0, DT_FLOAT)
                   .Input("reduction_indices", 1, DT_INT32)
                   .Attr("keep_dims", true)
                   .Finalize(&op.node_def));
  indices = test::AsTensor<int32>({-1, -2});
  op.input_tensors[1] = &indices;
  INFER_OK(op, "[2,4,5];[2]", "[d0_0, 1, 1]");

  // input rank is known, but reduction indices are not (with keep_dim=true).
  // The output rank matches input rank (because of keep_dims=true).
  op.input_tensors[1] = nullptr;
  INFER_OK(op, "[?,?,?];?", "[?,?,?]");
  INFER_OK(op, "[?,?,?];[2]", "[?,?,?]");

  // Reduction indices with too many dimensions.
  INFER_ERROR("must be at most rank 1 but is rank 2", op, "[?,?,?];[?,?]");
  // With older graph-def version, this is allowed.
  op.graph_def_version = 20;
  INFER_OK(op, "[?,?,?];[?,?]", "[?,?,?]");
  // And when the tensor is specified, it's still allowed.
  op.input_tensors[1] = &indices;
  indices = test::AsTensor<int32>({-1, -2}, TensorShape({2, 1}));
  INFER_OK(op, "[2,4,5];[2,1]", "[d0_0, 1, 1]");
  indices = test::AsTensor<int32>({-1, -2}, TensorShape({1, 2}));
  INFER_OK(op, "[2,4,5];[1,2]", "[d0_0, 1, 1]");
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownShapes) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {Unknown(), Unknown(), Unknown()}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownDims) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({-1, -1}), S({-1}), S({-1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_InvalidIndicesRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({-1}), S({-1}), S({-1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  EXPECT_EQ(error::INVALID_ARGUMENT,
            ValidateSparseTensor(&c, indices, values, shape).code());
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_InvalidNumElements) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({4}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  EXPECT_EQ(error::INVALID_ARGUMENT,
            ValidateSparseTensor(&c, indices, values, shape).code());
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_InvalidRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({5}), S({4})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  EXPECT_EQ(error::INVALID_ARGUMENT,
            ValidateSparseTensor(&c, indices, values, shape).code());
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownNumIndexElements) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({-1, 3}), S({5}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownNumValueElements) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({-1}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownIndexRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, -1}), S({5}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor_UnknownShapeRank) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({5}), S({-1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ValidateSparseTensor) {
  NodeDef def;
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, MakeOpDef(3, 1),
                     {S({5, 3}), S({5}), S({3})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  auto indices = c.input(0);
  auto values = c.input(1);
  auto shape = c.input(2);
  TF_EXPECT_OK(ValidateSparseTensor(&c, indices, values, shape));
}

TEST(CommonShapeFnsTest, ReduceScatterSuccess) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  const Tensor scatter_dimension = Tensor(0);
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({2, 2}), S({1, 2}), S({1})},
                     {nullptr, nullptr, &scatter_dimension}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  TF_EXPECT_OK(ReduceScatterShape(&c));
  ShapeHandle output = c.output(0);
  EXPECT_EQ(1, c.Value(c.Dim(output, 0)));
  EXPECT_EQ(2, c.Value(c.Dim(output, 1)));
}

TEST(CommonShapeFnsTest, ReduceScatter_MissingScatterDimension) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({2, 2}), S({1, 2}), S({1})}, {}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());

  TF_EXPECT_OK(ReduceScatterShape(&c));
  ShapeHandle output = c.output(0);
  EXPECT_FALSE(c.ValueKnown(c.Dim(output, 0)));
  EXPECT_FALSE(c.ValueKnown(c.Dim(output, 1)));
}

TEST(CommonShapeFnsTest, ReduceScatter_NotEvenlyDivisible) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  const Tensor scatter_dimension = Tensor(0);
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({3, 3}), S({1, 2}), S({1})},
                     {nullptr, nullptr, &scatter_dimension}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());
  EXPECT_THAT(ReduceScatterShape(&c),
              tensorflow::testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  "Dimension size must be evenly divisible by 2 but is 3"));
}

TEST(CommonShapeFnsTest, ReduceScatter_INVALID_GROUP_ASSIGNMENT) {
  OpRegistrationData op_reg_data;
  TF_CHECK_OK(OpDefBuilder("XlaReduceScatter")
                  .Input("input: float")
                  .Input("group_assignment: int32")
                  .Input("scatter_dimension: int32")
                  .Output("output: float")
                  .Finalize(&op_reg_data));
  OpDef op_def = op_reg_data.op_def;

  NodeDef def;
  TF_CHECK_OK(NodeDefBuilder("test", "XlaReduceScatter")
                  .Input("input", 0, DT_FLOAT)
                  .Input("group_assignment", 0, DT_INT32)
                  .Input("scatter_dimension", 0, DT_INT32)
                  .Finalize(&def));
  const Tensor scatter_dimension = Tensor(0);
  InferenceContext c(TF_GRAPH_DEF_VERSION, def, op_def,
                     {S({3, 3}), S({2}), S({1})},
                     {nullptr, nullptr, &scatter_dimension}, {}, {});
  EXPECT_EQ(3, c.num_inputs());
  EXPECT_EQ(1, c.num_outputs());
  EXPECT_THAT(ReduceScatterShape(&c),
              tensorflow::testing::StatusIs(
                  error::INVALID_ARGUMENT,
                  "ReduceScatter group_assignment should be rank 2"));
}

}  // namespace shape_inference
}  // namespace tensorflow
