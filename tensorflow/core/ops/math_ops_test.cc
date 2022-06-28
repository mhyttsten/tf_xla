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
class MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc() {
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

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(MathOpsTest, AddN_ShapeFn) {
  ShapeInferenceTestOp op("AddN");
  auto set_n = [&op](int n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/ops/math_ops_test.cc", "lambda");

    std::vector<NodeDefBuilder::NodeOut> src_list;
    src_list.reserve(n);
    for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_FLOAT);
    TF_ASSERT_OK(NodeDefBuilder("test", "AddN")
                     .Input(src_list)
                     .Attr("N", n)
                     .Finalize(&op.node_def));
  };

  set_n(2);
  // Adding two unknowns returns either input.
  INFER_OK(op, "?;?", "in0|in1");

  // known+unknown returns the known input.
  INFER_OK(op, "[1];[?]", "in0");
  INFER_OK(op, "[1];?", "in0");
  INFER_OK(op, "[?];[1]", "in1");
  INFER_OK(op, "?;[1]", "in1");

  set_n(2);
  INFER_OK(op, "[1,2];[?,2]", "in0");
  INFER_OK(op, "[1,2];[1,2]", "in0|in1");
  INFER_OK(op, "[?,2];[1,2]", "in1");

  set_n(3);
  INFER_OK(op, "[1,?];[?,2];[1,2]", "in2");
  INFER_OK(op, "[1,2];[?,2];[1,?]", "in0");
  INFER_OK(op, "?;?;[1,2]", "in2");

  set_n(2);
  INFER_OK(op, "?;[1,2]", "in1");
  INFER_OK(op, "[1,?];[?,2]", "[d0_0,d1_1]");
  INFER_OK(op, "[?,2,?];[?,?,3]", "[d0_0|d1_0,d0_1,d1_2]");
  INFER_OK(op, "[?,2];[1,?]", "[d1_0,d0_1]");

  set_n(3);
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 2 and 4", op,
              "[1,2];?;[1,4]");
  INFER_ERROR("From merging shape 0 with other shapes.", op, "[1,2];?;[1,4]");
  set_n(4);
  INFER_ERROR("Shapes must be equal rank, but are 2 and 3", op,
              "?;[1,2];?;[1,2,3]");
  INFER_ERROR("From merging shape 1 with other shapes.", op,
              "?;[1,2];?;[1,2,3]");
}

TEST(MathOpsTest, UnchangedShape_ShapeFn) {
  ShapeInferenceTestOp op("Cast");
  INFER_OK(op, "?", "in0");
  INFER_OK(op, "[?]", "in0");
  INFER_OK(op, "[1,?,3,4]", "in0");
}

TEST(MathOpsTest, Segment_ShapeFn) {
  // Tests SegmentReductionShapeFn.
  for (const auto* op_name : {"SegmentMax", "SegmentMean", "SegmentMin",
                              "SegmentProd", "SegmentSum"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "?;[100]", "?");

    // Data shape with single dimension.
    INFER_OK(op, "[?];?", "[?]");
    INFER_OK(op, "[?];[100]", "[?]");
    INFER_OK(op, "[1];?", "[?]");
    INFER_OK(op, "[1];[100]", "[?]");

    // Data shape with multiple dimensions.
    INFER_OK(op, "[?,?];?", "[?,d0_1]");
    INFER_OK(op, "[?,2];[100]", "[?,d0_1]");
    INFER_OK(op, "[?,2,?,4];[100]", "[?,d0_1,d0_2,d0_3]");
    INFER_OK(op, "[1,?];?", "[?,d0_1]");
    INFER_OK(op, "[1,2];[100]", "[?,d0_1]");
    INFER_OK(op, "[1,2,?,4];[100]", "[?,d0_1,d0_2,d0_3]");

    // Error cases.
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[1,2]");
    INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[1]");
  }
}

TEST(MathOpsTest, BroadcastBinaryOps_ShapeFn) {
  auto test_shapes = [&](ShapeInferenceTestOp& op,
                         bool incompatible_shape_error) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc mht_1(mht_1_v, 286, "", "./tensorflow/core/ops/math_ops_test.cc", "lambda");

    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "[1,2];?", "?");
    INFER_OK(op, "?;[1,2]", "?");

    INFER_OK(op, "[?];[1]", "[d0_0]");
    INFER_OK(op, "[1];[?]", "[d1_0]");
    INFER_OK(op, "[?];[2]", incompatible_shape_error ? "[d1_0]" : "?");
    INFER_OK(op, "[2];[?]", incompatible_shape_error ? "[d0_0]" : "?");
    INFER_OK(op, "[?];[?]", "[?]");
    INFER_OK(op, "[];[?]", "[d1_0]");
    INFER_OK(op, "[?];[]", "[d0_0]");

    INFER_OK(op, "[1];[1]", "[d0_0|d1_0]");
    INFER_OK(op, "[];[1]", "[d1_0]");
    INFER_OK(op, "[1];[]", "[d0_0]");

    INFER_OK(op, "[2];[2]", "[d0_0|d1_0]");
    INFER_OK(op, "[];[2]", "[d1_0]");
    INFER_OK(op, "[1];[2]", "[d1_0]");
    INFER_OK(op, "[2];[1]", "[d0_0]");
    INFER_OK(op, "[2];[]", "[d0_0]");
    INFER_OK(op, "[2];[?]", incompatible_shape_error ? "[d0_0]" : "?");

    INFER_OK(op, "[0];[0]", "[d0_0|d1_0]");
    INFER_OK(op, "[];[0]", "[d1_0]");
    INFER_OK(op, "[1];[0]", "[d1_0]");
    INFER_OK(op, "[0];[1]", "[d0_0]");
    INFER_OK(op, "[0];[]", "[d0_0]");

    INFER_OK(op, "[2];[?,?]", incompatible_shape_error ? "[d1_0,d0_0]" : "?");
    INFER_OK(op, "[2,2];[?,?,?]",
             incompatible_shape_error ? "[d1_0,d0_0,d0_1]" : "?");

    // Multiple dimension cases (same test cases, switching x and y).
    INFER_OK(op, "[?,1,2,3,4,5];[3,1,?]",
             incompatible_shape_error ? "[d0_0,d0_1,d0_2,d0_3|d1_0,d0_4,d0_5]"
                                      : "?");
    INFER_OK(op, "[3,1,?];[?,1,2,3,4,5]",
             incompatible_shape_error ? "[d1_0,d1_1,d1_2,d1_3|d0_0,d1_4,d1_5]"
                                      : "?");

    if (incompatible_shape_error) {
      INFER_ERROR("Dimensions must be equal", op, "[2];[3]");
    } else {
      INFER_OK(op, "[2];[3]", "[]");
    }
  };

  for (string op_name : {"Add",        "Complex",
                         "Div",        "Equal",
                         "Greater",    "GreaterEqual",
                         "Igamma",     "Igammac",
                         "Zeta",       "Polygamma",
                         "Less",       "LessEqual",
                         "LogicalAnd", "LogicalOr",
                         "Maximum",    "Minimum",
                         "Mod",        "Mul",
                         "NotEqual",   "Pow",
                         "Sub",        "SquaredDifference",
                         "DivNoNan"}) {
    ShapeInferenceTestOp op(op_name);
    AddNodeAttr("incompatible_shape_error", true, &op.node_def);
    test_shapes(op, true);

    if ((op_name == "Equal") || (op_name == "NotEqual")) {
      ShapeInferenceTestOp op(op_name);
      AddNodeAttr("incompatible_shape_error", false, &op.node_def);
      test_shapes(op, false);
    }
  }
}

TEST(MathOpsTest, Select_ShapeFn) {
  ShapeInferenceTestOp op("Select");
  INFER_OK(op, "?;?;?", "in1|in2");

  // scalar case
  INFER_OK(op, "[];[1];?", "in1");
  INFER_OK(op, "[];?;?", "in1|in2");

  INFER_OK(op, "[1];?;?",
           "in1|in2");  // When cond is vector, t/e may not match it.
  INFER_OK(op, "[1,2];?;?", "in1|in2?");

  INFER_OK(op, "?;[];?", "in1");
  INFER_OK(op, "?;?;[]", "in2");
  INFER_OK(op, "?;[1];?", "in1");
  INFER_OK(op, "?;?;[1]", "in2");
  INFER_OK(op, "?;[1,2];?", "in1");
  INFER_OK(op, "?;?;[1,2]", "in2");

  INFER_ERROR("Shapes must be equal rank, but are 0 and 1", op, "[1];[];?");
  INFER_ERROR("Shapes must be equal rank, but are 1 and 2", op, "[];[1];[1,2]");
  INFER_ERROR("Shapes must be equal rank, but are 1 and 2", op, "[1,2];[1];?");
  INFER_OK(op, "[2];[?];[?]", "in1|in2");

  INFER_OK(op, "[?];[?,?,3];[1,2,?]", "[d2_0,d2_1,d1_2]");
  INFER_OK(op, "[2];[?,?,3];[?,2,?]", "[d1_0|d2_0,d2_1,d1_2]");
  INFER_ERROR("must be equal", op, "[1];[2,?,3];[?,2,?]");
  INFER_ERROR("Shapes must be equal rank, but are 3 and 2", op,
              "[2,?];[?,?,3];[?,2,?]");
  INFER_OK(op, "[2,?,?];[?,?,3];[?,2,?]", "[d0_0,d2_1,d1_2]");
  INFER_ERROR("Dimension 2 in both shapes must be equal, but are 3 and 5", op,
              "[2,?,5];[?,?,3];[?,2,?]");

  // Test that handles were merged.
  //
  // Tests below will modify handle_data and call run_inference_for_handles to
  // rerun shape inference, updating the context <c>.
  const OpRegistrationData* op_reg_data;
  TF_ASSERT_OK(OpRegistry::Global()->LookUp(op.name, &op_reg_data));
  typedef std::vector<std::pair<PartialTensorShape, DataType>> ShapeDtypeV;
  std::vector<std::unique_ptr<ShapeDtypeV>> handle_data;
  std::unique_ptr<shape_inference::InferenceContext> c;
  auto run_inference_for_handles = [&]() -> Status {
    CHECK(op_reg_data->shape_inference_fn != nullptr);
    c.reset(new shape_inference::InferenceContext(
        TF_GRAPH_DEF_VERSION, op.node_def, op_reg_data->op_def,
        {PartialTensorShape(), PartialTensorShape(), PartialTensorShape()}, {},
        {}, handle_data));
    TF_CHECK_OK(c->construction_status());
    Status s = c->Run(op_reg_data->shape_inference_fn);
    LOG(INFO) << "Inference got " << s;
    return s;
  };
  auto shape_proto = [](std::initializer_list<int64_t> dim_sizes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc mht_2(mht_2_v, 415, "", "./tensorflow/core/ops/math_ops_test.cc", "lambda");

    TensorShapeProto p;
    for (auto i : dim_sizes) p.add_dim()->set_size(i);
    return p;
  };

  auto i0 = PartialTensorShape({1, -1});
  auto i1 = PartialTensorShape({-1, 2});
  PartialTensorShape unknown_shape;
  auto scalar = PartialTensorShape({});

  handle_data.emplace_back(
      new ShapeDtypeV{{scalar, DT_FLOAT}, {unknown_shape, DT_INT32}});
  handle_data.emplace_back(new ShapeDtypeV{{i0, DT_FLOAT}, {i1, DT_INT32}});
  handle_data.emplace_back(
      new ShapeDtypeV{{i1, DT_FLOAT}, {unknown_shape, DT_INT32}});

  TF_ASSERT_OK(run_inference_for_handles());
  auto* out = c->output_handle_shapes_and_types(0);
  ASSERT_EQ(2, out->size());
  EXPECT_EQ("[1,2]", c->DebugString(out->at(0).shape));
  EXPECT_EQ(DT_FLOAT, out->at(0).dtype);
  EXPECT_EQ("[?,2]", c->DebugString(out->at(1).shape));
  EXPECT_EQ(DT_INT32, out->at(1).dtype);

  // Expect an error when the shapes can't be merged.
  handle_data[2]->at(0).first = shape_proto({2, 2});
  EXPECT_TRUE(absl::StrContains(run_inference_for_handles().error_message(),
                                "must be equal, but are 1 and 2"));
  handle_data[2]->at(0).first = i1;  // restore to valid

  // Expect an error when the types can't be merged.
  handle_data[2]->at(1).second = DT_INT64;
  EXPECT_TRUE(absl::StrContains(run_inference_for_handles().error_message(),
                                "pointing to different dtypes"));
  handle_data[2]->at(1).second = DT_INT32;  // restore to valid

  // Expect an error when different numbers of tensors are merged.
  handle_data[2]->push_back({i1, DT_FLOAT});
  EXPECT_TRUE(absl::StrContains(run_inference_for_handles().error_message(),
                                "pointing to different numbers of tensors"));
  handle_data[2]->pop_back();  // restore to valid.
}

TEST(MathOpsTest, Range_ShapeFn) {
  ShapeInferenceTestOp op("Range");

  TF_ASSERT_OK(NodeDefBuilder("test", "Range")
                   .Input({"start", {}, DT_INT32})
                   .Input({"limit", {}, DT_INT32})
                   .Input({"delta", {}, DT_INT32})
                   .Attr("Tidx", DT_INT32)
                   .Finalize(&op.node_def));

  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "[?]");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "[1,2];?;?");
  INFER_ERROR("for 'start'", op, "[1,2];?;?");

  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "?;[1,2];?");
  INFER_ERROR("for 'limit'", op, "?;[1,2];?");

  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "?;?;[1,2]");
  INFER_ERROR("for 'delta'", op, "?;?;[1,2]");

  Tensor start_t = test::AsScalar(1);
  op.input_tensors[0] = &start_t;
  INFER_OK(op, "?;?;?", "[?]");
  Tensor limit_t = test::AsScalar(1);
  op.input_tensors[1] = &limit_t;
  INFER_OK(op, "?;?;?", "[?]");

  Tensor delta_t = test::AsScalar(1);
  op.input_tensors[2] = &delta_t;
  INFER_OK(op, "?;?;?", "[0]");

  delta_t = test::AsScalar(0);
  INFER_ERROR("Requires delta != 0", op, "?;?;?");
  delta_t = test::AsScalar(3);

  limit_t = test::AsScalar(-1);
  INFER_ERROR("Requires start <= limit when delta > 0: 1/-1", op, "?;?;?");

  delta_t = test::AsScalar(-1);
  INFER_OK(op, "?;?;?", "[2]");

  limit_t = test::AsScalar(4);
  INFER_ERROR("Requires start >= limit when delta < 0: 1/4", op, "?;?;?");

  limit_t = test::AsScalar(100);
  start_t = test::AsScalar(2);
  delta_t = test::AsScalar(3);
  INFER_OK(op, "?;?;?", "[33]");
}

TEST(MathOpsTest, LinSpace_ShapeFn) {
  ShapeInferenceTestOp op("LinSpace");
  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "[?]");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "[1,2];?;?");
  INFER_ERROR("for 'start'", op, "[1,2];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "?;[1,2];?");
  INFER_ERROR("for 'stop'", op, "?;[1,2];?");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "?;?;[1,2]");
  INFER_ERROR("for 'num'", op, "?;?;[1,2]");

  Tensor num_t = test::AsScalar(1);
  op.input_tensors[2] = &num_t;
  INFER_OK(op, "?;?;?", "[1]");
  num_t = test::AsScalar(2);
  INFER_OK(op, "?;?;?", "[2]");
  num_t = test::AsScalar(-1);
  INFER_ERROR("Requires num > 0: -1", op, "?;?;?");
}

TEST(MathOpsTest, UnsortedSegmentSum_ShapeFn) {
  ShapeInferenceTestOp op("UnsortedSegmentSum");
  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "?;[?];?", "?");
  INFER_ERROR("Shape must be rank 0 but is rank 2", op, "?;?;[1,2]");
  INFER_ERROR("Dimensions must be equal, but are 2 and 3", op,
              "[1,?,2];[1,?,3];?");
  INFER_OK(op, "?;[3];?", "?");
  INFER_ERROR("Shape must be at least rank 3 but is rank 2", op,
              "[1,2];[1,2,3];?");

  Tensor num_segments_t = test::AsScalar(100);
  op.input_tensors[2] = &num_segments_t;
  INFER_OK(op, "[?,2,3,?,5];[1,2,?];[]", "[100,d0_3,d0_4]");

  num_segments_t = test::AsScalar(-1);
  INFER_ERROR(("Dimension size, given by scalar input 2, must be "
               "non-negative but is -1"),
              op, "[3];[3];?");
}

TEST(MathOpsTest, SparseSegment_ShapeFn) {
  ShapeInferenceTestOp op("SparseSegmentSum");
  op.input_tensors.resize(3);
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "[2,4,3];[3];[3]", "[?,d0_1,d0_2]");

  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[2,4,3];[];[3]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[2,4,3];[3];[3,4]");

  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 3 and 4", op,
              "[2,4,3];[3];[4]");
}

TEST(MathOpsTest, SparseSegmentGrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseSegmentMeanGrad");
  op.input_tensors.resize(4);
  INFER_OK(op, "?;?;?;?", "?");
  INFER_OK(op, "[2,4,3];[3];[3];[]", "[?,d0_1,d0_2]");

  Tensor num_segments_t = test::AsScalar(100);
  op.input_tensors[3] = &num_segments_t;
  INFER_OK(op, "[2,4,3];[3];[3];[]", "[100,d0_1,d0_2]");

  INFER_ERROR("Shape must be rank 0 but is rank 2", op,
              "[2,4,3];[3];[3];[1,1]");

  // Negative value is not allowed
  num_segments_t = test::AsScalar(-100);
  op.input_tensors[3] = &num_segments_t;
  INFER_ERROR("Cannot specify a negative value", op, "[2,4,3];[3];[3];[]");
}

TEST(MathOpsTest, BatchMatMul_ShapeFn) {
  ShapeInferenceTestOp op("BatchMatMul");
  auto set_adj = [&op](bool adj_x, bool adj_y) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSmath_ops_testDTcc mht_3(mht_3_v, 589, "", "./tensorflow/core/ops/math_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "BatchMatMul")
                     .Input({"a", 0, DT_FLOAT})
                     .Input({"b", 0, DT_FLOAT})
                     .Attr("adj_x", adj_x)
                     .Attr("adj_y", adj_y)
                     .Finalize(&op.node_def));
  };

  set_adj(false, false);

  // Rank checks.
  INFER_ERROR("at least rank 2", op, "[1];?");
  INFER_ERROR("at least rank 2", op, "?;[2]");

  INFER_OK(op, "?;?", "?");

  // 0 batch dims.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");

  // 2 batch dims.
  INFER_OK(op, "[?,?,?,?];?", "[d0_0,d0_1,d0_2,?]");

  // Test adj_a, testing output and that inner dims are compared.
  set_adj(false, false);
  INFER_OK(op, "[1,2,3,4];[1,2,?,?]", "[d0_0,d0_1,d0_2,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,3,1]");  // inner dim mismatch
  set_adj(true, false);
  INFER_OK(op, "[1,2,3,4];[1,2,?,?]", "[d0_0,d0_1,d0_3,d1_3]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,3,1]");  // inner dim mismatch

  // Test adj_b=true.
  set_adj(false, true);
  INFER_OK(op, "[1,2,?,?];[1,2,3,4]", "[d0_0,d0_1,d0_2,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,1,2];[?,1,3]");  // inner dim mismatch
  set_adj(true, true);
  INFER_OK(op, "[1,2,?,?];[1,2,3,4]", "[d0_0,d0_1,d0_3,d1_2]");
  INFER_ERROR("are 2 and 3", op, "[?,2,1];[?,1,3]");  // inner dim mismatch
}

TEST(MathOpsTest, ArgOps_ShapeFn) {
  ShapeInferenceTestOp op("ArgMax");
  op.input_tensors.resize(2);

  INFER_OK(op, "?;?", "?");

  // input rank <= 1 produces scalar
  INFER_OK(op, "[2];?", "[]");
  INFER_OK(op, "[];?", "[]");

  // Incorrect rank for dimension
  INFER_ERROR("must be rank 0", op, "[2];[1]");

  // dimension not available, but input rank is.  Output is unknown
  // shape with rank one less than input rank.
  INFER_OK(op, "[2,3,4];?", "[?,?]");
  INFER_OK(op, "[2,3,4,5,6];?", "[?,?,?,?]");

  // Dimension values known
  Tensor dimension = test::AsScalar(0);
  op.input_tensors[1] = &dimension;
  INFER_OK(op, "[2,3,4];[]", "[d0_1,d0_2]");

  dimension = test::AsScalar(1);
  op.input_tensors[1] = &dimension;
  INFER_OK(op, "[2,3,4];[]", "[d0_0,d0_2]");

  dimension = test::AsScalar(2);
  op.input_tensors[1] = &dimension;
  INFER_OK(op, "[2,3,4];[]", "[d0_0,d0_1]");

  // Dimension value out of bounds
  dimension = test::AsScalar(10);
  op.input_tensors[1] = &dimension;
  INFER_ERROR("must be in the range [-3, 3)", op, "[2,3,4];[]");

  dimension = test::AsScalar(-10);
  op.input_tensors[1] = &dimension;
  INFER_ERROR("must be in the range [-3, 3)", op, "[2,3,4];[]");

  dimension = test::AsScalar(-1);
  op.input_tensors[1] = &dimension;
  INFER_OK(op, "[2,3,4];[]", "[d0_0,d0_1]");
}

TEST(MathOpsTest, Betainc_ShapeFn) {
  ShapeInferenceTestOp op("Betainc");

  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "[?,?];?;?", "in0");
  INFER_OK(op, "[?,2];?;[1,?]", "[d2_0,d0_1]");
  INFER_OK(op, "[?,2,?];[1,?,?];[?,?,3]", "[d1_0,d0_1,d2_2]");

  INFER_OK(op, "[?,2,?];[];[?,?,3]", "[d0_0|d2_0,d0_1,d2_2]");
  INFER_OK(op, "[];[];[?,?,3]", "in2");

  // All but one is a scalar, so use it.
  INFER_OK(op, "[];[];?", "in2");
  INFER_OK(op, "[];[];[1,2,3,4]", "in2");

  // All scalar input; implementation picks in0.
  INFER_OK(op, "[];[];[]", "in0");

  // Non-scalars must match shape.
  INFER_ERROR("must be equal", op, "[1,2];[];[1,4]");
  INFER_ERROR("must be equal", op, "[1,2];[];[1,2,3]");
}

TEST(MathOpsTest, Requantize_ShapeFn) {
  ShapeInferenceTestOp op("Requantize");

  INFER_OK(op, "?;?;?;?;?", "in0;[];[]");
  INFER_OK(op, "?;[];[];[];[]", "in0;[];[]");

  // Rank checks on input scalars.
  INFER_ERROR("must be rank 0", op, "?;[1];?;?;?");
  INFER_ERROR("must be rank 0", op, "?;?;[2];?;?");
  INFER_ERROR("must be rank 0", op, "?;?;?;[3];?");
  INFER_ERROR("must be rank 0", op, "?;?;?;?;[4]");
}

TEST(MathOpstest, RequantizationRange_ShapeFn) {
  ShapeInferenceTestOp op("RequantizationRange");

  INFER_OK(op, "?;?;?", "[];[]");
  INFER_OK(op, "?;[];[]", "[];[]");

  // Rank checks on input scalars.
  INFER_ERROR("must be rank 0", op, "?;[1];?");
  INFER_ERROR("must be rank 0", op, "?;?;[2]");
}

TEST(MathOpsTest, Cross_ShapeFn) {
  ShapeInferenceTestOp op("Cross");

  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but", op, "[3];[5]");
  INFER_ERROR("Dimension must be 3 but", op, "[3,5];[3,5]");

  INFER_OK(op, "?;?", "in0");
  INFER_OK(op, "[?];[?]", "in0");
  INFER_OK(op, "[1,?,3];[?,?,?]", "in0");
}

TEST(MathOpsTest, HistogramFixedWidth_ShapeFn) {
  ShapeInferenceTestOp op("HistogramFixedWidth");

  // value_range should be vector.
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];[];[]");
  // value_range should have 2 elements.
  INFER_ERROR("Dimension must be 2 but is 3", op, "[];[3];[]");
  // nbins should be scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[];[2];[2]");

  INFER_OK(op, "?;?;?", "[?]");
  INFER_OK(op, "[?];[2];[]", "[?]");
  INFER_OK(op, "[?];[2];?", "[?]");
}

TEST(MathOpsTest, QuantizedAdd_ShapeFn) {
  ShapeInferenceTestOp op("QuantizedAdd");

  INFER_OK(op, "?;?;?;?;?;?", "?;[];[]");
  INFER_OK(op, "?;?;[];[];[];[]", "?;[];[]");
  INFER_OK(op, "[1,2];?;[];[];[];[]", "?;[];[]");
  INFER_OK(op, "[];[2];[];[];[];[]", "[d1_0];[];[]");

  // Rank checks on input scalars.
  INFER_ERROR("must be rank 0", op, "?;?;[1];?;?;?");
  INFER_ERROR("must be rank 0", op, "?;?;?;[2];?;?");
  INFER_ERROR("must be rank 0", op, "?;?;?;?;[3];?");
  INFER_ERROR("must be rank 0", op, "?;?;?;?;?;[4]");
}

TEST(MathOpsTest, Bincount_ShapeFn) {
  ShapeInferenceTestOp op("Bincount");

  // size should be scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[1];?");

  INFER_OK(op, "?;?;?", "[?]");
  INFER_OK(op, "?;[];?", "[?]");
  INFER_OK(op, "[?];[];?", "[?]");
  INFER_OK(op, "[?];[];[?]", "[?]");
}

TEST(MathOpsTest, SobolSample) {
  ShapeInferenceTestOp op("SobolSample");

  // All inputs should be scalar.
  INFER_ERROR("must be rank 0", op, "[1];?;?");
  INFER_ERROR("must be rank 0", op, "?;[1];?");
  INFER_ERROR("must be rank 0", op, "?;?;[1]");

  INFER_OK(op, "[];[];[]", "[?,?]");
}

TEST(MathOpsTest, EqualOp) {
  ShapeInferenceTestOp op("Equal");
  AddNodeAttr("incompatible_shape_error", true, &op.node_def);

  INFER_OK(op, "?;?", "?");
  INFER_OK(op, "[1,2];?", "?");
  INFER_OK(op, "?;[1,2]", "?");

  INFER_OK(op, "[1,2,3];[1]", "[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,2,1];[1,3]", "[d0_0,d0_1,d1_1]");
  INFER_OK(op, "[1,?,3];[3,1]", "[d0_0,d1_0,d0_2]");
  INFER_OK(op, "[1,2,3];[2,1,3]", "[d1_0,d0_1,d0_2]");

  // Note: Test case for GitHub issue 40471
  INFER_OK(op, "[?,10,1];[?,1,4]", "[?,d0_1,d1_2]");
  INFER_OK(op, "[10,?,1];[1,?,4]", "[d0_0,?,d1_2]");
}
}  // end namespace tensorflow
