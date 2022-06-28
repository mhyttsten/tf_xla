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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc() {
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

#include "tensorflow/core/common_runtime/shape_refiner.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class ShapeRefinerTest : public ::testing::Test {
 protected:
  // These give access to private functions of DimensionHandle and ShapeHandle.
  bool SameHandle(shape_inference::DimensionHandle a,
                  shape_inference::DimensionHandle b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "SameHandle");

    return a.SameHandle(b);
  }

  bool SameHandle(shape_inference::ShapeHandle a,
                  shape_inference::ShapeHandle b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "SameHandle");

    return a.SameHandle(b);
  }

  // These give access to private functions of ShapeRefiner.
  bool SameDefinedShape(shape_inference::InferenceContext* c,
                        shape_inference::ShapeHandle s0,
                        shape_inference::ShapeHandle s1) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "SameDefinedShape");

    return ShapeRefiner::SameDefinedShape(c, s0, s1);
  }

  bool IsUpdatedShapesOrTypes(
      shape_inference::InferenceContext* c,
      const std::vector<shape_inference::ShapeAndType>& existing,
      const std::vector<shape_inference::ShapeAndType>& updated) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "IsUpdatedShapesOrTypes");

    return ShapeRefiner::IsUpdatedShapesOrTypes(c, existing, updated);
  }

  static constexpr int64_t kMaxTensorSize = ShapeRefiner::kMaxTensorSize;

  void TestStridedSlice(const PartialTensorShape& input_shape, int begin,
                        int end, int stride, const char* expected,
                        int begin_mask = 0, int end_mask = 0,
                        int ellipsis_mask = 0, int shrink_axis_mask = 0,
                        StringPiece test_op = "TensorAsShapeInt32") {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("expected: \"" + (expected == nullptr ? std::string("nullptr") : std::string((char*)expected)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_4(mht_4_v, 250, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "TestStridedSlice");

    Scope root = Scope::DisabledShapeInferenceScope();
    auto placeholder =
        ops::Placeholder(root, DT_INT32, ops::Placeholder::Shape(input_shape));
    auto input = ops::Shape(root, placeholder);
    auto begin_op = ops::Const(root, {begin});
    auto end_op = ops::Const(root, {end});
    auto stride_op = ops::Const(root, {stride});
    auto slice = ops::StridedSlice(root, input, begin_op, end_op, stride_op,
                                   ops::StridedSlice::BeginMask(begin_mask)
                                       .EndMask(end_mask)
                                       .EllipsisMask(ellipsis_mask)
                                       .ShrinkAxisMask(shrink_axis_mask));
    Node* result;
    TF_ASSERT_OK(NodeBuilder("test", test_op)
                     .Input(slice.node())
                     .Finalize(root.graph(), &result));

    ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(placeholder.node()));
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(begin_op.node()));
    TF_ASSERT_OK(m.AddNode(end_op.node()));
    TF_ASSERT_OK(m.AddNode(stride_op.node()));
    TF_ASSERT_OK(m.AddNode(slice.node()));
    TF_ASSERT_OK(m.AddNode(result));

    shape_inference::InferenceContext* ctx = m.GetContext(result);
    EXPECT_EQ(ctx->DebugString(ctx->output(0)), expected);
  }
};

namespace {

#define EXPECT_SHAPE(EXPECTED, M, OP, IDX)                            \
  do {                                                                \
    shape_inference::InferenceContext* ctx = M.GetContext(OP.node()); \
    EXPECT_EQ(EXPECTED, ctx->DebugString(ctx->output(IDX)));          \
  } while (0);

#define EXPECT_RESOURCE_SINGLE_SHAPE(EXPECTED, M, OP, IDX)            \
  do {                                                                \
    shape_inference::InferenceContext* ctx = M.GetContext(OP.node()); \
    auto* v = ctx->output_handle_shapes_and_types(IDX);               \
    EXPECT_NE(v, nullptr);                                            \
    EXPECT_EQ(v->size(), 1);                                          \
    EXPECT_EQ(EXPECTED, ctx->DebugString((*v)[0].shape));             \
  } while (0);

#define EXPECT_RESOURCE_SINGLE_TYPE(EXPECTED, M, OP, IDX)             \
  do {                                                                \
    shape_inference::InferenceContext* ctx = M.GetContext(OP.node()); \
    auto* v = ctx->output_handle_shapes_and_types(IDX);               \
    EXPECT_NE(v, nullptr);                                            \
    EXPECT_EQ(v->size(), 1);                                          \
    EXPECT_EQ(EXPECTED, (*v)[0].dtype);                               \
  } while (0);

TEST_F(ShapeRefinerTest, Constant) {
  // Create a constant node and validate that adding it is successful
  // and that its shape is correct.
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(c.node()));

  EXPECT_SHAPE("[]", m, c, 0);
}

TEST_F(ShapeRefinerTest, MatMul) {
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());

  Scope root = Scope::NewRootScope();
  auto a = ops::Const(root, {{1.0f}, {2.0f}});
  auto b = ops::Const(root, {{1.0f, 2.0f}});
  auto mm = ops::MatMul(root, a, b);

  TF_ASSERT_OK(m.AddNode(a.node()));
  TF_ASSERT_OK(m.AddNode(b.node()));
  TF_ASSERT_OK(m.AddNode(mm.node()));

  EXPECT_SHAPE("[2,1]", m, a, 0);
  EXPECT_SHAPE("[1,2]", m, b, 0);
  EXPECT_SHAPE("[2,2]", m, mm, 0);
}

TEST_F(ShapeRefinerTest, BadShapes) {
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  Scope root = Scope::NewRootScope();
  auto a = ops::Const(root, {{1.0f}, {2.0f}});
  auto b = ops::Const(root, {{1.0f}, {2.0f}});
  auto mm = ops::MatMul(root, a, b);

  TF_ASSERT_OK(m.AddNode(a.node()));
  TF_ASSERT_OK(m.AddNode(b.node()));
  // The shape of the inputs are not compatible, so we should expect
  // an error.
  Status s = m.AddNode(mm.node());
  ASSERT_FALSE(s.ok());
  ASSERT_TRUE(absl::StrContains(s.error_message(),
                                "Dimensions must be equal, but are 1 and 2"));
}

TEST_F(ShapeRefinerTest, SetShape) {
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());

  Scope root = Scope::NewRootScope();
  auto a = ops::Placeholder(root, DT_FLOAT);

  TF_ASSERT_OK(m.AddNode(a.node()));

  auto ic = m.GetContext(a.node());
  ASSERT_NE(nullptr, ic);
  shape_inference::ShapeHandle h = ic->MakeShape({2, ic->UnknownDim()});
  TF_ASSERT_OK(m.SetShape(a.node(), 0, h));
  EXPECT_SHAPE("[2,?]", m, a, 0);

  // Check that shapes are merged with the existing shape.
  shape_inference::ShapeHandle h2 = ic->MakeShape({ic->UnknownDim(), 2});
  TF_ASSERT_OK(m.SetShape(a.node(), 0, h2));
  EXPECT_SHAPE("[2,2]", m, a, 0);

  // Out of range.
  ASSERT_FALSE(m.SetShape(a.node(), 1, h).ok());
  ASSERT_FALSE(m.SetShape(a.node(), -1, h).ok());

  auto b = ops::Const(root, {{1.0f}, {2.0f}});
  // Forget to add node first.
  ASSERT_FALSE(m.SetShape(b.node(), 0, h).ok());

  // Set an incompatible shape (3 vs 2)
  h = ic->MakeShape({3, ic->UnknownDim()});
  ASSERT_FALSE(m.SetShape(a.node(), 0, h).ok());
}

namespace {

// An op with no shape function.
REGISTER_OP("TestOpWithNoShapeFn").Input("a: int32").Output("o: int32");

}  // namespace

TEST_F(ShapeRefinerTest, MissingShapeInferenceFns) {
  Scope root = Scope::NewRootScope();
  auto a = ops::Const(root, 42);
  Node* b;
  TF_ASSERT_OK(NodeBuilder("b", "TestOpWithNoShapeFn")
                   .Input(a.node())
                   .Finalize(root.graph(), &b));
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(a.node()));
  EXPECT_FALSE(m.AddNode(b).ok());
  m.set_require_shape_inference_fns(false);
  TF_EXPECT_OK(m.AddNode(b));
}

TEST_F(ShapeRefinerTest, PropagateConstants) {
  // Reduction dimension is a variable, so we don't know its value.
  // So the output shape value is unknown (though its rank is known).
  {
    Scope root = Scope::NewRootScope();
    // 3x2 input
    auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    // Reduce along unspecified dimension
    auto dim = ops::Variable(root, {}, DT_INT32);

    auto am = ops::ArgMax(root, input, dim);
    ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(dim.node()));
    TF_ASSERT_OK(m.AddNode(am.node()));
    EXPECT_SHAPE("[?]", m, am, 0);
  }

  // Constant is used as dimension, which can be materialized,
  // so the shape function can be more precise about the output.
  {
    Scope root = Scope::NewRootScope();
    // 3x2 input
    auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    // Reduce along 2nd dimension
    auto dim = ops::Const(root, 1);

    auto am = ops::ArgMax(root, input, dim);
    ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(dim.node()));
    TF_ASSERT_OK(m.AddNode(am.node()));
    EXPECT_SHAPE("[3]", m, am, 0);
  }

  // Reduce along known first dimension.
  {
    Scope root = Scope::NewRootScope();
    // 3x2 input
    auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    // Reduce along 1st dimension
    auto dim = ops::Const(root, 0);

    auto am = ops::ArgMax(root, input, dim);
    ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input.node()));
    TF_ASSERT_OK(m.AddNode(dim.node()));
    TF_ASSERT_OK(m.AddNode(am.node()));
    EXPECT_SHAPE("[2]", m, am, 0);
  }
}

TEST_F(ShapeRefinerTest, ExtractConstantSubgraphMultiOutput) {
  // Test when a node yields two outputs, one of which has a constant
  // value that is small enough to be cached, and one which does not.
  //
  // ShapeVectorForAllElements nodes are used in here to call
  // input_tensor from the shape function.
  {
    Scope root = Scope::NewRootScope();
    auto small = ops::Const(root, {static_cast<int32>(1), TensorShape({1, 1})});
    auto large = ops::Const(
        root, {static_cast<int32>(2), TensorShape({4, kMaxTensorSize / 2})});
    Node* multi;
    TF_ASSERT_OK(NodeBuilder("MI", "MultiIdentity")
                     .Input(std::vector<NodeBuilder::NodeOut>{small.node(),
                                                              large.node()})
                     .Attr("N", 2)
                     .Finalize(root.graph(), &multi));

    Node* shape_v;
    TF_ASSERT_OK(NodeBuilder("Test", "ShapeVectorForAllElements")
                     .Input(multi, 0)
                     .Finalize(root.graph(), &shape_v));

    auto add = ops::Add(root, Output(multi, 0), Output(multi, 1));
    Node* shape_v2;
    TF_ASSERT_OK(NodeBuilder("Test", "ShapeVectorForAllElements")
                     .Input(add.node())
                     .Finalize(root.graph(), &shape_v2));
    ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(small.node()));
    TF_ASSERT_OK(m.AddNode(large.node()));
    TF_ASSERT_OK(m.AddNode(multi));
    TF_ASSERT_OK(m.AddNode(shape_v));
    TF_ASSERT_OK(m.AddNode(add.node()));
    TF_ASSERT_OK(m.AddNode(shape_v2));

    // The output shape is a vector of length equal to the result of the add.
    // The add adds 1 and 2 together, and its output has kMaxTensorSize*2
    // elements.
    shape_inference::InferenceContext* ctx = m.GetContext(shape_v2);
    EXPECT_EQ(strings::StrCat("[", kMaxTensorSize * 2 * 3, "]"),
              ctx->DebugString(ctx->output(0)));
  }
}

namespace {

// An op with a shape function whose outputs depend in a complex
// way on whether input tensors are available.
REGISTER_OP("TestOp")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->input_tensor(0)) {
        if (c->input_tensor(1)) {
          c->set_output(0, c->Matrix(10, 10));
          return Status::OK();
        }
        return shape_inference::ScalarShape(c);
      }
      return shape_inference::UnknownShape(c);
    });

}  // namespace

TEST_F(ShapeRefinerTest, InputTensorDependencies) {
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  Graph graph(OpRegistry::Global());
  Node* node;

  Tensor a(DT_FLOAT, TensorShape({}));
  a.scalar<float>()() = 1.0;

  Tensor b(DT_FLOAT, TensorShape({}));
  b.scalar<float>()() = 2.0;

  Node* input_a = test::graph::Constant(&graph, a);
  Node* input_b = test::graph::Constant(&graph, b);
  TF_ASSERT_OK(NodeBuilder("Test", "TestOp")
                   .Input(input_a)
                   .Input(input_b)
                   .Finalize(&graph, &node));

  TF_ASSERT_OK(m.AddNode(input_a));
  TF_ASSERT_OK(m.AddNode(input_b));
  TF_ASSERT_OK(m.AddNode(node));
  shape_inference::InferenceContext* ctx = m.GetContext(node);
  EXPECT_EQ("[10,10]", ctx->DebugString(ctx->output(0)));
}

namespace {

// An op with a shape function that looks at its input tensor
// data and makes a Shape out of it.
REGISTER_OP("ShapeData")
    .Input("a: int32")
    .Output("o: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      const Tensor* shape_data = c->input_tensor(0);
      if (shape_data == nullptr) {
        return shape_inference::UnknownShape(c);
      }

      std::vector<shape_inference::DimensionHandle> dims;
      dims.reserve(shape_data->NumElements());
      for (int i = 0; i < shape_data->NumElements(); ++i) {
        dims.emplace_back(c->MakeDim(shape_data->flat<int32>()(i)));
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    });

REGISTER_OP("ShapeDataInt64")
    .Input("a: int64")
    .Output("o: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      const Tensor* shape_data = c->input_tensor(0);
      if (shape_data == nullptr) {
        return shape_inference::UnknownShape(c);
      }

      std::vector<shape_inference::DimensionHandle> dims;
      dims.reserve(shape_data->NumElements());
      for (int i = 0; i < shape_data->NumElements(); ++i) {
        dims.emplace_back(c->MakeDim(shape_data->flat<int64_t>()(i)));
      }

      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    });

// An op with a shape function that looks at its input tensor
// data and makes a rank 1 shape out of the sum of all input values.
REGISTER_OP("ShapeVectorForAllElements")
    .Input("a: int32")
    .Output("o: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      const Tensor* shape_data = c->input_tensor(0);
      if (shape_data == nullptr) {
        return shape_inference::UnknownShape(c);
      }
      int64_t total = 0;
      for (int i = 0; i < shape_data->NumElements(); ++i) {
        total += shape_data->flat<int32>()(i);
      }

      c->set_output(0, c->Vector(total));
      return Status::OK();
    });

REGISTER_OP("MultiIdentity")
    .Input("a: N * int32")
    .Output("o: N * int32")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = 0; i < c->num_inputs(); ++i) {
        c->set_output(i, c->input(i));
      }
      return Status::OK();
    });

class MultiIdentity : public OpKernel {
 public:
  explicit MultiIdentity(OpKernelConstruction* c) : OpKernel(c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_5(mht_5_v, 626, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "MultiIdentity");
}

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_6(mht_6_v, 631, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "Compute");

    for (int i = 0; i < c->num_inputs(); ++i) {
      c->set_output(i, c->input(i));
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("MultiIdentity").Device(DEVICE_CPU),
                        MultiIdentity);

}  // namespace

TEST_F(ShapeRefinerTest, PropagateShapeAcrossTensorContent) {
  Scope root = Scope::NewRootScope();

  // Create variable 2x4 tensor.
  auto input = ops::Variable(root, {2, 4}, DT_INT32);

  // Shape is a vector of 2 elements (2,4)
  auto shape = ops::Shape(root, input);

  // Ones for indices of the slice. (get the 4).
  auto ones = ops::Const(root, {1});

  // Slice an element of the shape (4).
  auto sliced = ops::Slice(root, shape, ones, ones);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(sliced.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(ones.node()));
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(shape.node()));
  TF_ASSERT_OK(m.AddNode(sliced.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[4]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateShapeAcrossTensorContentInt64) {
  Scope root = Scope::NewRootScope();

  // Create variable 2x4 tensor.
  auto input = ops::Variable(
      root, {2, 4, static_cast<int64_t>(std::numeric_limits<int32>::max()) * 2},
      DT_INT64);

  // Shape is a vector of 2 elements (2,4)
  auto attrs = ops::Shape::OutType(DT_INT64);
  auto shape = ops::Shape(root, input, attrs);

  // Ones for indices of the slice. (get the 4).
  auto ones = ops::Const(root, {1});

  // Slice an element of the shape (4).
  auto sliced = ops::Slice(root, shape, ones, ones);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeDataInt64")
                   .Input(sliced.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(ones.node()));
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(shape.node()));
  TF_ASSERT_OK(m.AddNode(sliced.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[4]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateShapeAcrossTensorContentInt32Overflow) {
  Scope root = Scope::NewRootScope();

  // Create variable 2x4 tensor.
  auto input = ops::Variable(
      root, {2, 4, static_cast<int64_t>(std::numeric_limits<int32>::max()) * 2},
      DT_INT32);

  // Shape is a vector of 2 elements (2,4)
  auto shape = ops::Shape(root, input);

  // Ones for indices of the slice. (get the 4).
  auto ones = ops::Const(root, {1});

  // Slice an element of the shape (4).
  auto sliced = ops::Slice(root, shape, ones, ones);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(sliced.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(ones.node()));
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(shape.node()));
  TF_ASSERT_OK(m.AddNode(sliced.node()));

  // Expect an error since there's an overflow.
  EXPECT_FALSE(m.AddNode(shape_data).ok());
}

TEST_F(ShapeRefinerTest, PropagateRankAcrossTensorContent) {
  Scope root = Scope::NewRootScope();

  // Create variable 2x4x3 tensor.
  auto input = ops::Variable(root, {2, 4, 3}, DT_INT32);

  // Rank 3.
  auto rank = ops::Rank(root, input);

  auto identity = ops::Identity(root, rank);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(identity.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(rank.node()));
  TF_ASSERT_OK(m.AddNode(identity.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[3]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateSizeAcrossTensorContent) {
  Scope root = Scope::NewRootScope();

  // Create variable.
  auto input = ops::Variable(root, {1, 2, 3, 4, 5}, DT_INT32);

  // 5!.
  auto size = ops::Size(root, input);

  auto identity = ops::Identity(root, size);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(identity.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(size.node()));
  TF_ASSERT_OK(m.AddNode(identity.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[120]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateSizeAcrossTensorContentInt64) {
  Scope root = Scope::NewRootScope();

  // Create variable.
  auto input = ops::Variable(
      root,
      {1, 2, 3, 4, 5,
       static_cast<int64_t>(std::numeric_limits<int32>::max()) * 2},
      DT_INT64);

  // 5! * int32_max_value * 2.
  auto attrs = ops::Size::OutType(DT_INT64);
  auto size = ops::Size(root, input, attrs);

  auto identity = ops::Identity(root, size);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeDataInt64")
                   .Input(identity.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(size.node()));
  TF_ASSERT_OK(m.AddNode(identity.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[515396075280]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateSizeAcrossTensorContentInt32Overflow) {
  Scope root = Scope::NewRootScope();

  // Create variable.
  auto input = ops::Variable(
      root,
      {1, 2, 3, 4, 5,
       static_cast<int64_t>(std::numeric_limits<int32>::max()) * 2},
      DT_INT32);

  // 5!.
  auto size = ops::Size(root, input);

  auto identity = ops::Identity(root, size);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(identity.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(size.node()));
  TF_ASSERT_OK(m.AddNode(identity.node()));
  EXPECT_FALSE(m.AddNode(shape_data).ok());
}

TEST_F(ShapeRefinerTest, PropagateShape) {
  Scope root = Scope::NewRootScope();
  // 3x2 input
  auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

  // Shape is a vector of 2 elements (3,2)
  auto shape = ops::Shape(root, input);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(shape.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(shape.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[3,2]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateSize) {
  Scope root = Scope::NewRootScope();
  // 3x2 input
  auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

  auto size = ops::Size(root, input);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(size.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(size.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[6]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateRank) {
  Scope root = Scope::NewRootScope();
  // 3x2 input
  auto input = ops::Const(root, {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

  auto rank = ops::Rank(root, input);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(rank.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(rank.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[2]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, PropagateRange) {
  Scope root = Scope::NewRootScope();
  auto begin = ops::Const(root, 1);
  auto limit = ops::Const(root, 11);
  auto delta = ops::Const(root, 3);
  auto range = ops::Range(root, begin, limit, delta);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(range.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(begin.node()));
  TF_ASSERT_OK(m.AddNode(limit.node()));
  TF_ASSERT_OK(m.AddNode(delta.node()));
  TF_ASSERT_OK(m.AddNode(range.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[1,4,7,10]", ctx->DebugString(ctx->output(0)));
}

// Make sure PlaceholderWithDefaults aren't treated as constants.
TEST_F(ShapeRefinerTest, NoPropagatePlaceholderWithDefault) {
  Scope root = Scope::NewRootScope();
  auto constant = ops::Const<int>(root, 2);
  auto placeholder =
      ops::PlaceholderWithDefault(root, constant, PartialTensorShape());
  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(placeholder.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(constant.node()));
  TF_ASSERT_OK(m.AddNode(placeholder.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));
  shape_inference::InferenceContext* ic = m.GetContext(shape_data);
  EXPECT_EQ(ic->DebugString(ic->output(0)), "?");
}

TEST_F(ShapeRefinerTest, ConstantValueTwoInputsToSameNode) {
  Scope root = Scope::NewRootScope();
  // This node is used as two inputs to 'range'.
  auto begin_and_delta = ops::Const(root, 1);
  auto limit = ops::Const(root, 4);
  auto range = ops::Range(root, begin_and_delta, limit, begin_and_delta);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(range.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(begin_and_delta.node()));
  TF_ASSERT_OK(m.AddNode(limit.node()));
  TF_ASSERT_OK(m.AddNode(range.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[1,2,3]", ctx->DebugString(ctx->output(0)));
}

// Creates a graph where 'begin' is attempted to be visited during
// constant value evaluation after having been processed once.
TEST_F(ShapeRefinerTest, ConstantValueVisitNodeTwice) {
  Scope root = Scope::NewRootScope();
  auto begin = ops::Const(root, 1);
  auto limit = ops::Const(root, 8);
  auto delta = ops::Const(root, 3);

  auto d1 = ops::Add(root, begin, limit);  // 9
  auto d2 = ops::Add(root, begin, delta);  // 4
  // Visiting flimit's children will visit 'begin' before 'd1'.
  // It will then visit d1, whose child is 'begin'.  That edge still
  // must be visited.
  auto flimit = ops::Sub(root, begin, d1);  // 1-9=-8
  auto fdelta = ops::Sub(root, begin, d2);  // 1-4=-3
  auto nl = ops::Abs(root, flimit);         // 8
  auto nd = ops::Abs(root, fdelta);         // 3

  auto range = ops::Range(root, begin, nl, nd);

  Node* shape_data;
  TF_ASSERT_OK(NodeBuilder("Test", "ShapeData")
                   .Input(range.node())
                   .Finalize(root.graph(), &shape_data));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(begin.node()));
  TF_ASSERT_OK(m.AddNode(limit.node()));
  TF_ASSERT_OK(m.AddNode(delta.node()));
  TF_ASSERT_OK(m.AddNode(d1.node()));
  TF_ASSERT_OK(m.AddNode(d2.node()));
  TF_ASSERT_OK(m.AddNode(flimit.node()));
  TF_ASSERT_OK(m.AddNode(fdelta.node()));
  TF_ASSERT_OK(m.AddNode(nl.node()));
  TF_ASSERT_OK(m.AddNode(nd.node()));
  TF_ASSERT_OK(m.AddNode(range.node()));
  TF_ASSERT_OK(m.AddNode(shape_data));

  shape_inference::InferenceContext* ctx = m.GetContext(shape_data);
  EXPECT_EQ("[1,4,7]", ctx->DebugString(ctx->output(0)));
}

namespace {

Status TensorAsShapeShapeFn(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_7(mht_7_v, 1024, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "TensorAsShapeShapeFn");

  shape_inference::ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0 /* input_idx */, &out));
  c->set_output(0, out);
  return Status::OK();
}

Status PartialTensorAsShapeShapeFn(shape_inference::InferenceContext* c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_8(mht_8_v, 1034, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "PartialTensorAsShapeShapeFn");

  shape_inference::ShapeHandle out;
  const Tensor* t = c->input_tensor(0);
  if (t == nullptr || t->NumElements() != 1) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      c->MakeShapeFromTensorShape(TensorShape({t->flat<int32>()(0)}), &out));
  c->set_output(0, out);
  return Status::OK();
}

// Register ops used by the ConstantValueAsShape* tests.
REGISTER_OP("PartialTensorAsShapeInt32")
    .Input("a: int32")
    .Output("o: int32")
    .SetShapeFn(PartialTensorAsShapeShapeFn);

REGISTER_OP("TensorAsShapeInt32")
    .Input("a: int32")
    .Output("o: int32")
    .SetShapeFn(TensorAsShapeShapeFn);

REGISTER_OP("TensorAsShapeInt64")
    .Input("a: int64")
    .Output("o: int64")
    .SetShapeFn(TensorAsShapeShapeFn);

REGISTER_OP("NonConstScalarInt32")
    .Output("o: int32")
    .SetDoNotOptimize()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("NonConstScalarInt64")
    .Output("o: int64")
    .SetDoNotOptimize()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("WithEmptyVectorShape")
    .Output("o: int32")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(0));
      return Status::OK();
    });

REGISTER_OP("WithPartialShape")
    .Output("o: int32")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(
          0, c->MakeShape({1, shape_inference::InferenceContext::kUnknownDim, 3,
                           shape_inference::InferenceContext::kUnknownDim, 5}));
      return Status::OK();
    });

REGISTER_OP("WithPartialShape2")
    .Output("o: int32")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(
          0,
          c->MakeShape({6, shape_inference::InferenceContext::kUnknownDim, 8}));
      return Status::OK();
    });

REGISTER_OP("WithUnknownShape")
    .Output("o: int32")
    .SetDoNotOptimize()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return Status::OK();
    });

}  // namespace

TEST_F(ShapeRefinerTest, ConstantValueAsShape_EmptyVector) {
  Scope root = Scope::NewRootScope();
  Node* input;
  TF_ASSERT_OK(
      NodeBuilder("in", "WithEmptyVectorShape").Finalize(root.graph(), &input));
  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(input)
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_Shape) {
  for (int pass = 0; pass < 2; ++pass) {
    Scope root = Scope::NewRootScope();
    Node* input;
    TF_ASSERT_OK(
        NodeBuilder("in", pass == 0 ? "WithPartialShape" : "WithUnknownShape")
            .Finalize(root.graph(), &input));
    auto shape = ops::Shape(root, Output(input));
    Node* result;
    TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                     .Input(shape.node())
                     .Finalize(root.graph(), &result));

    ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
    TF_ASSERT_OK(m.AddNode(input));
    TF_ASSERT_OK(m.AddNode(shape.node()));
    TF_ASSERT_OK(m.AddNode(result));

    shape_inference::InferenceContext* ctx = m.GetContext(result);
    if (pass == 0) {
      EXPECT_EQ("[1,?,3,?,5]", ctx->DebugString(ctx->output(0)));
    } else {
      EXPECT_EQ("?", ctx->DebugString(ctx->output(0)));
    }
  }
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_PackInt32) {
  Scope root = Scope::DisabledShapeInferenceScope();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt32")
                   .Finalize(root.graph(), &scalar_non_const));

  InputList inputs{
      // clang-format off
      Input(ops::Const<int32>(root, 10)),
      Input(ops::Const<int32>(root, 20)),
      Input(Output(scalar_non_const)),
      Input(ops::Const<int32>(root, 40)),
  };  // clang-format on
  auto pack = ops::Stack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[10,20,?,40]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_PackInt64) {
  Scope root = Scope::DisabledShapeInferenceScope();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt64")
                   .Finalize(root.graph(), &scalar_non_const));

  InputList inputs{
      // clang-format off
      Input(ops::Const<int64_t>(root, int64_t{10})),
      Input(ops::Const<int64_t>(root, int64_t{20})),
      Input(Output(scalar_non_const)),
      Input(ops::Const<int64_t>(root, int64_t{1} << 40)),
  };  // clang-format on
  auto pack = ops::Stack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt64")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[10,20,?,1099511627776]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_PackUnknownDim) {
  Scope root = Scope::NewRootScope();

  InputList inputs{
      Input(ops::Const<int64_t>(root, int64_t{10})),
      Input(ops::Const<int64_t>(root, int64_t{-1})),
  };
  auto pack = ops::Stack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt64")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[10,?]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_PackInvalidInput) {
  Scope root = Scope::NewRootScope();

  // Inputs are length 2 vectors instead of scalars.
  InputList inputs{
      Input(ops::Const<int64_t>(root, {int64_t{10}, int64_t{20}})),
      Input(ops::Const<int64_t>(root, {int64_t{10}, int64_t{21}})),
  };
  auto pack = ops::Stack(root, inputs);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt64")
                   .Input(pack.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  for (const auto& input : inputs) {
    TF_ASSERT_OK(m.AddNode(input.node()));
  }
  TF_ASSERT_OK(m.AddNode(pack.node()));
  EXPECT_TRUE(
      absl::StrContains(m.AddNode(result).error_message(), "but is rank 2"));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_Concat) {
  Scope root = Scope::DisabledShapeInferenceScope();
  Graph* g = root.graph();
  Node* partial_1;
  Node* partial_2;
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape").Finalize(g, &partial_1));
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape2").Finalize(g, &partial_2));
  auto const_input = ops::Const(root, {9, 10, 11});
  OutputList concat_inputs{
      // clang-format off
      ops::Shape(root, Output(partial_1)),
      ops::Shape(root, Output(partial_2)),
      const_input,
  };  // clang-format on
  auto concat_dim = ops::Const(root, 0);
  auto concat = ops::Concat(root, concat_inputs, concat_dim);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(concat.node())
                   .Finalize(g, &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(partial_1));
  TF_ASSERT_OK(m.AddNode(partial_2));
  for (const auto& o : concat_inputs) {
    TF_ASSERT_OK(m.AddNode(o.node()));
  }
  TF_ASSERT_OK(m.AddNode(concat_dim.node()));
  TF_ASSERT_OK(m.AddNode(concat.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("[1,?,3,?,5,6,?,8,9,10,11]", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_ConcatWithUnknown) {
  Scope root = Scope::DisabledShapeInferenceScope();
  Graph* g = root.graph();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt32")
                   .Finalize(root.graph(), &scalar_non_const));

  Node* partial_1;
  Node* partial_2;
  Node* unknown;
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape").Finalize(g, &partial_1));
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape2").Finalize(g, &partial_2));
  TF_ASSERT_OK(NodeBuilder("in", "WithUnknownShape").Finalize(g, &unknown));
  OutputList concat_inputs{
      // clang-format off
      ops::Shape(root, Output(partial_1)),
      ops::Shape(root, Output(partial_2)),
      ops::Shape(root, Output(unknown)),
  };  // clang-format on
  auto concat_dim = ops::Const(root, 0);
  auto concat = ops::Concat(root, concat_inputs, concat_dim);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(concat.node())
                   .Finalize(g, &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(partial_1));
  TF_ASSERT_OK(m.AddNode(partial_2));
  TF_ASSERT_OK(m.AddNode(unknown));
  for (const auto& o : concat_inputs) {
    TF_ASSERT_OK(m.AddNode(o.node()));
  }
  TF_ASSERT_OK(m.AddNode(concat_dim.node()));
  TF_ASSERT_OK(m.AddNode(concat.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ("?", ctx->DebugString(ctx->output(0)));
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_ConcatInvalidDimValue) {
  Scope root = Scope::DisabledShapeInferenceScope();
  Graph* g = root.graph();
  Node* scalar_non_const;
  TF_ASSERT_OK(NodeBuilder("in", "NonConstScalarInt32")
                   .Finalize(root.graph(), &scalar_non_const));

  Node* partial_1;
  Node* partial_2;
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape").Finalize(g, &partial_1));
  TF_ASSERT_OK(NodeBuilder("in", "WithPartialShape2").Finalize(g, &partial_2));
  auto const_input = ops::Const(root, {9, -2, 11});
  OutputList concat_inputs{
      // clang-format off
      ops::Shape(root, Output(partial_1)),
      ops::Shape(root, Output(partial_2)),
      const_input,
  };  // clang-format on
  auto concat_dim = ops::Const(root, 0);
  auto concat = ops::Concat(root, concat_inputs, concat_dim);
  TF_ASSERT_OK(root.status());

  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(concat.node())
                   .Finalize(g, &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(partial_1));
  TF_ASSERT_OK(m.AddNode(partial_2));
  for (const auto& o : concat_inputs) {
    TF_ASSERT_OK(m.AddNode(o.node()));
  }
  TF_ASSERT_OK(m.AddNode(concat_dim.node()));
  TF_ASSERT_OK(m.AddNode(concat.node()));
  EXPECT_EQ("Invalid value in tensor used for shape: -2",
            m.AddNode(result).error_message());
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_StridedSlice) {
  TestStridedSlice(
      /*input_shape=*/{1, -1, 3, -1, 5},
      /*begin=*/2,
      /*end=*/5,
      /*stride=*/1,
      /*expected=*/"[3,?,5]");
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_StridedSliceNegativeStride) {
  // clang-format off
  TestStridedSlice(
      /*input_shape=*/{1, -1, 3, -1, 5},
      /*begin=*/10,
      /*end=*/0,
      /*stride=*/-1,
      /*expected=*/"[5,?,3,?]");
  // clang-format on
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_StridedSliceMasks) {
  TestStridedSlice(
      /*input_shape=*/{1, -1, 3, -1, 5},
      /*begin=*/3,
      /*end=*/4,
      /*stride=*/1,
      /*expected=*/"[1,?,3,?,5]",
      /*begin_mask=*/1,
      /*end_mask=*/1);
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_StridedSliceInvalidMask) {
  TestStridedSlice(
      /*input_shape=*/{1, -1, 3},
      /*begin=*/2,
      /*end=*/3,
      /*stride=*/1,
      /*expected=*/"[?,?,?]",
      /*begin_mask=*/0,
      /*end_mask=*/0,
      /*ellipsis_mask=*/1);
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_StridedSliceWithShrinkAxis) {
  TestStridedSlice(
      /*input_shape=*/{1, -1, 3, -1, 5},
      /*begin=*/2,
      /*end=*/3,
      /*stride=*/1,
      /*expected=*/"[3]",
      /*begin_mask=*/0,
      /*end_mask=*/0,
      /*ellipsis_mask=*/0,
      /*shrink_axis_mask=*/1,
      /*test_op=*/"PartialTensorAsShapeInt32");
}

TEST_F(ShapeRefinerTest,
       ConstantValueAsShape_StridedSliceWithShrinkAxisOnUnknownDim) {
  TestStridedSlice(
      /*input_shape=*/{1, -1, 3, -1, 5},
      /*begin=*/1,
      /*end=*/2,
      /*stride=*/1,
      /*expected=*/"?",
      /*begin_mask=*/0,
      /*end_mask=*/0,
      /*ellipsis_mask=*/0,
      /*shrink_axis_mask=*/1,
      /*test_op=*/"PartialTensorAsShapeInt32");
}

TEST_F(ShapeRefinerTest, ConstantValueAsShape_StridedSliceMulti) {
  Scope root = Scope::DisabledShapeInferenceScope();
  auto input = ops::Placeholder(root, DT_INT32);
  auto begin = ops::Const(root, {0, 0});
  auto end = ops::Const(root, {2, 2});
  auto stride = ops::Const(root, {1, 1});
  auto slice = ops::StridedSlice(root, input, begin, end, stride);
  Node* result;
  TF_ASSERT_OK(NodeBuilder("test", "TensorAsShapeInt32")
                   .Input(slice.node())
                   .Finalize(root.graph(), &result));

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(input.node()));
  TF_ASSERT_OK(m.AddNode(begin.node()));
  TF_ASSERT_OK(m.AddNode(end.node()));
  TF_ASSERT_OK(m.AddNode(stride.node()));
  TF_ASSERT_OK(m.AddNode(slice.node()));
  TF_ASSERT_OK(m.AddNode(result));

  shape_inference::InferenceContext* ctx = m.GetContext(result);
  EXPECT_EQ(ctx->DebugString(ctx->output(0)), "?");
}

namespace {

// Dummy op to test ShapeRefiner util functions
REGISTER_OP("Dummy");

}  // namespace

TEST_F(ShapeRefinerTest, SameDefinedShape) {
  Scope root = Scope::NewRootScope();
  Graph* g = root.graph();
  Node* test;
  TF_CHECK_OK(NodeBuilder("test", "Dummy").Finalize(g, &test));
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  m.set_require_shape_inference_fns(false);
  TF_ASSERT_OK(m.AddNode(test));
  shape_inference::InferenceContext* ctx = m.GetContext(test);

  auto unknown = ctx->UnknownShape();
  auto unknown_b = ctx->UnknownShape();
  auto s_1_2 = ctx->MakeShape({1, 2});
  auto s_1_2_b = ctx->MakeShape({1, 2});
  auto s_2_2 = ctx->MakeShape({2, 2});
  auto s_unknown_2 = ctx->MakeShape({-1, 2});
  auto s_unknown_2_b = ctx->MakeShape({-1, 2});

  EXPECT_TRUE(SameDefinedShape(ctx, unknown, unknown));
  EXPECT_FALSE(SameDefinedShape(ctx, unknown, unknown_b));
  EXPECT_FALSE(SameDefinedShape(ctx, unknown, s_1_2));
  EXPECT_TRUE(SameDefinedShape(ctx, s_1_2, s_1_2_b));
  EXPECT_FALSE(SameDefinedShape(ctx, s_1_2, s_2_2));
  EXPECT_TRUE(SameDefinedShape(ctx, s_unknown_2, s_unknown_2));
  EXPECT_FALSE(SameDefinedShape(ctx, s_unknown_2, s_unknown_2_b));
}

TEST_F(ShapeRefinerTest, IsUpdatedShapesOrTypes) {
  Scope root = Scope::NewRootScope();
  Graph* g = root.graph();
  Node* test;
  TF_CHECK_OK(NodeBuilder("test", "Dummy").Finalize(g, &test));
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  m.set_require_shape_inference_fns(false);
  TF_ASSERT_OK(m.AddNode(test));
  shape_inference::InferenceContext* ctx = m.GetContext(test);

  shape_inference::ShapeHandle unknown = ctx->UnknownShape();
  std::vector<shape_inference::ShapeAndType> t0{
      {ctx->MakeShape({1, 2, 3}), DT_FLOAT},
      {unknown, DT_INVALID},
      {ctx->MakeShape({4, 3, 2, 1}), DT_INT32}};

  std::vector<shape_inference::ShapeAndType> t1{
      {ctx->MakeShape({1, 2, 3}), DT_FLOAT},
      {unknown, DT_INVALID},
      {ctx->MakeShape({4, 3, 2, 1}), DT_INT32}};

  std::vector<shape_inference::ShapeAndType> t2{
      {ctx->MakeShape({1, 2, 4}), DT_FLOAT},
      {ctx->UnknownShape(), DT_INVALID},
      {ctx->MakeShape({4, 3, 2, 1}), DT_INT32}};

  std::vector<shape_inference::ShapeAndType> t3{
      {ctx->MakeShape({1, 2, 3}), DT_INT32},
      {ctx->UnknownShape(), DT_INVALID},
      {ctx->MakeShape({4, 3, 2, 1}), DT_INT32}};

  EXPECT_FALSE(IsUpdatedShapesOrTypes(ctx, t0, t1));

  // A shape has been modified
  EXPECT_TRUE(IsUpdatedShapesOrTypes(ctx, t0, t2));

  // A type has been modified
  EXPECT_TRUE(IsUpdatedShapesOrTypes(ctx, t0, t3));
}

TEST_F(ShapeRefinerTest, IncrementalUpdates) {
  Scope root = Scope::NewRootScope();
  Graph* g = root.graph();
  Node* queue;
  TF_CHECK_OK(NodeBuilder("queue", "FIFOQueueV2")
                  .Attr("component_types", {DT_FLOAT})
                  .Finalize(g, &queue));
  Node* dequeue;
  TF_CHECK_OK(NodeBuilder("dequeue", "QueueDequeueV2")
                  .Attr("component_types", {DT_FLOAT})
                  .Input(queue)
                  .Finalize(g, &dequeue));
  ShapeRefiner m(TF_GRAPH_DEF_VERSION, OpRegistry::Global());
  TF_ASSERT_OK(m.AddNode(queue));
  TF_ASSERT_OK(m.AddNode(dequeue));

  // At this point, the shapes of the dequeued tensor are unknown.
  shape_inference::InferenceContext* ctx = m.GetContext(dequeue);
  EXPECT_EQ("?", ctx->DebugString(ctx->output(0)));

  // Inject a shape, and incrementally propagate it to the dequeue op.
  ctx = m.GetContext(queue);
  shape_inference::ShapeHandle shp = ctx->MakeShape({3, 7});
  ctx->set_output_handle_shapes_and_types(
      0, std::vector<shape_inference::ShapeAndType>{{shp, DT_FLOAT}});
  bool refined = false;
  TF_ASSERT_OK(m.UpdateNode(dequeue, false /* relax */, &refined));
  EXPECT_TRUE(refined);
  ctx = m.GetContext(dequeue);
  EXPECT_EQ("[3,7]", ctx->DebugString(ctx->output(0)));

  // Inject another shape, but relax instead of merge.
  ctx = m.GetContext(queue);
  shp = ctx->MakeShape({2, 7});
  ctx->set_output_handle_shapes_and_types(
      0, std::vector<shape_inference::ShapeAndType>{{shp, DT_FLOAT}});
  refined = false;
  TF_ASSERT_OK(m.UpdateNode(dequeue, true /* relax */, &refined));
  EXPECT_TRUE(refined);
  ctx = m.GetContext(dequeue);
  EXPECT_EQ("[?,7]", ctx->DebugString(ctx->output(0)));

  // Inject another partially unknown shape and attempt to relax it.
  ctx = m.GetContext(queue);
  shp = ctx->MakeShape({shape_inference::InferenceContext::kUnknownDim, 7});
  ctx->set_output_handle_shapes_and_types(
      0, std::vector<shape_inference::ShapeAndType>{{shp, DT_FLOAT}});
  refined = false;
  TF_ASSERT_OK(m.UpdateNode(dequeue, true /* relax */, &refined));
  EXPECT_TRUE(refined);
  ctx = m.GetContext(dequeue);
  EXPECT_EQ("[?,7]", ctx->DebugString(ctx->output(0)));
  EXPECT_TRUE(SameHandle(ctx->Dim(ctx->output(0), 0), ctx->Dim(shp, 0)));

  // Inject a shape of the same handle and expect refined to not change.
  ctx = m.GetContext(queue);
  shape_inference::ShapeHandle shp2 = shp;
  ctx->set_output_handle_shapes_and_types(
      0, std::vector<shape_inference::ShapeAndType>{{shp2, DT_FLOAT}});
  refined = false;
  TF_ASSERT_OK(m.UpdateNode(dequeue, /*relax=*/false, &refined));
  EXPECT_FALSE(refined);
  EXPECT_TRUE(SameHandle(ctx->Dim(shp, 0), ctx->Dim(shp2, 0)));
}

void TestSimpleFunctionInference(bool enable_function_inference) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSshape_refiner_testDTcc mht_9(mht_9_v, 1628, "", "./tensorflow/core/common_runtime/shape_refiner_test.cc", "TestSimpleFunctionInference");

  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();
  FunctionLibraryDefinition f_lib(OpRegistry::Global(), f_lib_proto);

  Scope root = Scope::NewRootScope();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto x = ops::Const(root, {{1.0f, 2.0f}});
  auto x2 = test::function::Call(&root, "x2", "XTimesTwo", {x});

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, &f_lib);
  if (enable_function_inference) {
    m.set_function_library_for_shape_inference(&f_lib);
  }

  TF_ASSERT_OK(m.AddNode(x.node()));
  TF_ASSERT_OK(m.AddNode(x2.node()));

  EXPECT_SHAPE("[1,2]", m, x, 0);

  if (enable_function_inference) {
    EXPECT_SHAPE("[1,2]", m, x2, 0);
  } else {
    // Default inference behavior: functions output shapes are unknown.
    EXPECT_SHAPE("?", m, x2, 0);
  }
}

TEST_F(ShapeRefinerTest, SimpleFunctionShapeInference_Disabled) {
  // Nesting flag doesn't matter, when function inference is disabled.
  TestSimpleFunctionInference(false /* enable_function_inference */);
}

TEST_F(ShapeRefinerTest, SimpleFunctionShapeInference) {
  TestSimpleFunctionInference(true /* enable_function_inference */);
}

TEST_F(ShapeRefinerTest, FunctionShapeInferenceFallback) {
  // Test that function inference falls back to returning unknown shapes,
  // if the function lookup fails.

  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();
  FunctionLibraryDefinition f_lib(OpRegistry::Global(), f_lib_proto);

  Scope root = Scope::NewRootScope();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto x = ops::Const(root, {{.0f, .0f}});
  auto x2 = test::function::Call(&root, "x2", "XTimesTwo", {x});

  FunctionDefLibrary empty_f_lib_proto;
  FunctionLibraryDefinition empty_f_lib(OpRegistry::Global(),
                                        empty_f_lib_proto);

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, &f_lib);
  m.set_function_library_for_shape_inference(&empty_f_lib);

  TF_ASSERT_OK(m.AddNode(x.node()));
  TF_ASSERT_OK(m.AddNode(x2.node()));

  EXPECT_SHAPE("[1,2]", m, x, 0);

  // Default inference behavior: functions output shapes are unknown.
  EXPECT_SHAPE("?", m, x2, 0);
}

TEST_F(ShapeRefinerTest, ChainedFunctionShapeInferenceWithMultipleInputs) {
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::XTimesTwo();
  *(f_lib_proto.add_function()) = test::function::XTimesFour();
  *(f_lib_proto.add_function()) = test::function::XTimes16();
  *(f_lib_proto.add_function()) = test::function::WXPlusB();
  FunctionLibraryDefinition f_lib(OpRegistry::Global(), f_lib_proto);

  Scope root = Scope::NewRootScope();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto w = ops::Const(root, {{.0f}, {.0f}, {.0f}});
  auto x = ops::Const(root, {{.0f, .0f, .0f}});
  auto b = ops::Const(root, {{.0f}});

  auto wxplusb = test::function::Call(&root, "wxplusb", "WXPlusB", {w, x, b});
  auto wxplusb16 =
      test::function::Call(&root, "wxplusb16", "XTimes16", {wxplusb});

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, &f_lib);
  m.set_function_library_for_shape_inference(&f_lib);

  TF_ASSERT_OK(m.AddNode(w.node()));
  TF_ASSERT_OK(m.AddNode(x.node()));
  TF_ASSERT_OK(m.AddNode(b.node()));
  TF_ASSERT_OK(m.AddNode(wxplusb.node()));
  TF_ASSERT_OK(m.AddNode(wxplusb16.node()));

  EXPECT_SHAPE("[3,1]", m, w, 0);
  EXPECT_SHAPE("[1,3]", m, x, 0);
  EXPECT_SHAPE("[1,1]", m, b, 0);
  EXPECT_SHAPE("[3,3]", m, wxplusb, 0);
  EXPECT_SHAPE("[3,3]", m, wxplusb16, 0);
}

TEST_F(ShapeRefinerTest, FunctionShapeInferenceWorksForResourceHandles) {
  FunctionDefLibrary f_lib_proto;
  *(f_lib_proto.add_function()) = test::function::Swap();

  FunctionLibraryDefinition f_lib(OpRegistry::Global(), f_lib_proto);

  Scope root = Scope::NewRootScope().ExitOnError();
  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));

  auto x1 = ops::VarHandleOp(root, DataType::DT_FLOAT, TensorShape({128, 256}));
  auto x2 = ops::VarHandleOp(root, DataType::DT_DOUBLE, TensorShape({1024}));
  auto swap = test::function::Call(&root, "swap", "Swap", {x1, x2});

  EXPECT_EQ(swap.node()->num_outputs(), 2);

  ShapeRefiner m(TF_GRAPH_DEF_VERSION, &f_lib);
  m.set_function_library_for_shape_inference(&f_lib);

  TF_ASSERT_OK(m.AddNode(x1.node()));
  TF_ASSERT_OK(m.AddNode(x2.node()));
  TF_ASSERT_OK(m.AddNode(swap.node()));

  EXPECT_EQ(m.GetContext(swap.node())->num_outputs(), 2);

  EXPECT_RESOURCE_SINGLE_SHAPE("[128,256]", m, x1, 0);
  EXPECT_RESOURCE_SINGLE_SHAPE("[1024]", m, x2, 0);
  EXPECT_RESOURCE_SINGLE_SHAPE("[1024]", m, swap, 0);
  EXPECT_RESOURCE_SINGLE_SHAPE("[128,256]", m, swap, 1);
  EXPECT_RESOURCE_SINGLE_TYPE(DataType::DT_DOUBLE, m, swap, 0);
  EXPECT_RESOURCE_SINGLE_TYPE(DataType::DT_FLOAT, m, swap, 1);
}

}  // namespace
}  // namespace tensorflow
