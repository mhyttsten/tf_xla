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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc() {
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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declaring this here so it doesn't need to be in the public header.
Status ReplaceSendRecvs(const GraphDef& original_graph_def,
                        const GraphDef& rewritten_graph_def,
                        const std::vector<string>& inputs,
                        const std::vector<string>& outputs,
                        GraphDef* output_graph_def);

class ConstantFoldingTest : public ::testing::Test {
 protected:
  void TestSimpleAdd() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestSimpleAdd");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const =
        Const(root.WithOpName("a_expect_removed"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const =
        Const(root.WithOpName("b_expect_removed"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add_expect_removed"), a_const, b_const);

    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);

    Output mul =
        Mul(root.WithOpName("output_expect_remains"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&placeholder_tensor, 1.0f);
    TestConstantFolding(graph_def,
                        {{"placeholder_expect_remains", placeholder_tensor}},
                        {}, {"output_expect_remains"}, {});
    TestConstantFolding(graph_def,
                        {{"placeholder_expect_remains:0", placeholder_tensor}},
                        {}, {"output_expect_remains:0"}, {});
  }

  void TestOpExclusionAdd() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_1(mht_1_v, 255, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestOpExclusionAdd");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const =
        Const(root.WithOpName("a_expect_remains"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const =
        Const(root.WithOpName("b_expect_remains"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add_expect_remains"), a_const, b_const);

    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);

    Output mul =
        Mul(root.WithOpName("output_expect_remains"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&placeholder_tensor, 1.0f);
    TestConstantFolding(graph_def,
                        {{"placeholder_expect_remains", placeholder_tensor}},
                        {"Add"}, {"output_expect_remains"}, {});
  }

  void TestShapePropagation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_2(mht_2_v, 292, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestShapePropagation");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);
    Output a_const =
        Const(root.WithOpName("a_expect_removed"),
              Input::Initializer({1, 1, 1}, TensorShape({1, 1, 3})));
    Output shape = Shape(root.WithOpName("shape_expect_removed"), a_const);
    Output cast = Cast(root.WithOpName("cast_expect_removed"), shape, DT_FLOAT);
    Output mul =
        Mul(root.WithOpName("output_expect_remains"), cast, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({3}));
    test::FillIota<float>(&placeholder_tensor, 1.0);
    TestConstantFolding(graph_def,
                        {{"placeholder_expect_remains", placeholder_tensor}},
                        {}, {"output_expect_remains"}, {});
  }

  void TestPreserveOutputShapes() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_3(mht_3_v, 319, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestPreserveOutputShapes");

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    tensorflow::AttrValue shape_attr;
    auto* shape_proto = shape_attr.mutable_list()->add_shape();
    shape_proto->add_dim()->set_size(1);
    shape_proto->add_dim()->set_size(1);
    shape_proto->add_dim()->set_size(3);

    Output placeholder =
        Placeholder(root.WithOpName("placeholder_expect_remains"), DT_FLOAT);
    placeholder.node()->AddAttr("_output_shapes", shape_attr);

    Output shape = Shape(root.WithOpName("shape_expect_removed"), placeholder);
    Output cast = Cast(root.WithOpName("cast_expect_removed"), shape, DT_FLOAT);
    Output mul =
        Mul(root.WithOpName("output_expect_remains"), cast, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({1, 1, 3}));
    test::FillIota<float>(&placeholder_tensor, 1.0);

    graph_transforms::TransformFuncContext context;
    context.params["clear_output_shapes"] = {"false"};
    TestConstantFolding(graph_def,
                        {{"placeholder_expect_remains", placeholder_tensor}},
                        {}, {"output_expect_remains"}, context);
  }

  void TestConstantFolding(const GraphDef& graph_def,
                           std::vector<std::pair<string, Tensor> > inputs,
                           std::vector<string> excluded_ops,
                           const std::vector<string>& outputs,
                           graph_transforms::TransformFuncContext context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_4(mht_4_v, 358, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestConstantFolding");

    std::unique_ptr<tensorflow::Session> unfolded_session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(unfolded_session->Create(graph_def));
    std::vector<Tensor> unfolded_tensors;
    TF_ASSERT_OK(unfolded_session->Run(inputs, outputs, {}, &unfolded_tensors));

    GraphDef folded_graph_def;
    for (const std::pair<string, Tensor>& input : inputs) {
      context.input_names.push_back(input.first);
    }
    context.output_names = outputs;
    context.params["exclude_op"] = std::move(excluded_ops);
    TF_ASSERT_OK(
        graph_transforms::FoldConstants(graph_def, context, &folded_graph_def));

    std::unique_ptr<tensorflow::Session> folded_session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(folded_session->Create(folded_graph_def));
    std::vector<Tensor> folded_tensors;
    TF_ASSERT_OK(folded_session->Run(inputs, outputs, {}, &folded_tensors));

    EXPECT_EQ(unfolded_tensors.size(), folded_tensors.size());
    for (int i = 0; i < unfolded_tensors.size(); ++i) {
      test::ExpectTensorNear<float>(unfolded_tensors[i], folded_tensors[i],
                                    1e-5);
    }

    std::map<string, const NodeDef*> folded_node_map;
    for (const NodeDef& node : folded_graph_def.node()) {
      folded_node_map.insert({node.name(), &node});
    }

    for (const NodeDef& node : graph_def.node()) {
      const StringPiece name(node.name());
      const int occurrence_count = folded_node_map.count(node.name());
      if (str_util::EndsWith(name, "expect_removed")) {
        EXPECT_EQ(0, occurrence_count) << "node.name()=" << node.name();
      }
      if (str_util::EndsWith(name, "expect_remains")) {
        EXPECT_EQ(1, occurrence_count) << "node.name()=" << node.name();
      }
    }
  }

  void TestReplaceSendRecvs() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_5(mht_5_v, 406, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestReplaceSendRecvs");

    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    const int width = 100;
    Tensor a_const_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_const_data, 1.0f);

    auto o_root = tensorflow::Scope::NewRootScope();
    _Recv(o_root.WithOpName("original_recv"), DT_FLOAT, "", "", 0, "");
    Output o_a_const =
        Const(o_root.WithOpName("a_const"), Input::Initializer(a_const_data));
    Placeholder(o_root.WithOpName("placeholder"), DT_FLOAT);
    _Send(o_root.WithOpName("original_send"), o_a_const, "", "", 0, "");
    GraphDef o_graph_def;
    TF_ASSERT_OK(o_root.ToGraphDef(&o_graph_def));

    auto n_root = tensorflow::Scope::NewRootScope();
    _Recv(n_root.WithOpName("original_recv"), DT_FLOAT, "", "", 0, "");
    Output n_a_const =
        Const(n_root.WithOpName("a_const"), Input::Initializer(a_const_data));
    _Recv(n_root.WithOpName("_recv_placeholder_0"), DT_FLOAT, "", "", 0, "");
    _Send(n_root.WithOpName("original_send"), n_a_const, "", "", 0, "");
    _Send(n_root.WithOpName("new_send"), n_a_const, "", "", 0, "");
    GraphDef n_graph_def;
    TF_ASSERT_OK(n_root.ToGraphDef(&n_graph_def));

    GraphDef result_graph_def;
    TF_ASSERT_OK(graph_transforms::ReplaceSendRecvs(
        o_graph_def, n_graph_def, {"placeholder"}, {"a_const"},
        &result_graph_def));

    std::map<string, const NodeDef*> node_map;
    graph_transforms::MapNamesToNodes(result_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("original_recv"));
    EXPECT_EQ(1, node_map.count("a_const"));
    EXPECT_EQ(1, node_map.count("placeholder"));
    EXPECT_EQ(1, node_map.count("original_send"));
    EXPECT_EQ(0, node_map.count("_recv_placeholder_0"));
    EXPECT_EQ(0, node_map.count("new_send"));
  }

  void TestReplaceSendRecvsPrefixNames() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_6(mht_6_v, 450, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestReplaceSendRecvsPrefixNames");

    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    auto o_root = tensorflow::Scope::NewRootScope();
    auto a = Placeholder(o_root.WithOpName("placeholder"), DT_FLOAT);
    auto b = Placeholder(o_root.WithOpName("placeholder_1"), DT_FLOAT);
    auto add_o = Add(o_root.WithOpName("add"), a, b);
    GraphDef o_graph_def;
    TF_ASSERT_OK(o_root.ToGraphDef(&o_graph_def));

    auto n_root = tensorflow::Scope::NewRootScope();
    auto c = _Recv(n_root.WithOpName("_recv_placeholder_0"), DT_FLOAT, "", "",
                   0, "");
    auto d = _Recv(n_root.WithOpName("_recv_placeholder_1_0"), DT_FLOAT, "", "",
                   0, "");
    auto add_n = Add(n_root.WithOpName("add"), c, d);
    GraphDef n_graph_def;
    TF_ASSERT_OK(n_root.ToGraphDef(&n_graph_def));

    GraphDef result_graph_def;
    TF_ASSERT_OK(graph_transforms::ReplaceSendRecvs(
        o_graph_def, n_graph_def, {"placeholder", "placeholder_1"}, {"add"},
        &result_graph_def));

    std::map<string, const NodeDef*> node_map;
    graph_transforms::MapNamesToNodes(result_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("placeholder"));
    EXPECT_EQ(1, node_map.count("placeholder_1"));
    EXPECT_EQ(1, node_map.count("add"));
  }

  void TestRemoveUnusedNodes() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_7(mht_7_v, 484, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestRemoveUnusedNodes");

    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    auto root = tensorflow::Scope::NewRootScope();

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = Const(root.WithOpName("a"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = Const(root.WithOpName("b"), Input::Initializer(b_data));

    Output add = Add(root.WithOpName("add"), a_const, b_const);
    Output placeholder = Placeholder(root.WithOpName("placeholder"), DT_FLOAT);
    Output mul = Mul(root.WithOpName("output"), add, placeholder);

    Tensor unused_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&unused_data, 1.0f);
    Output unused_const =
        Const(root.WithOpName("unused"), Input::Initializer(unused_data));

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    GraphDef result_graph_def;
    TF_ASSERT_OK(graph_transforms::RemoveUnusedNodes(
        graph_def, {{"placeholder"}, {"output"}}, &result_graph_def));

    std::map<string, const NodeDef*> node_map;
    graph_transforms::MapNamesToNodes(result_graph_def, &node_map);
    EXPECT_EQ(1, node_map.count("a"));
    EXPECT_EQ(1, node_map.count("b"));
    EXPECT_EQ(1, node_map.count("add"));
    EXPECT_EQ(1, node_map.count("placeholder"));
    EXPECT_EQ(1, node_map.count("output"));
    EXPECT_EQ(0, node_map.count("unused"));
  }

  void TestMaxConstantSizeInBytes() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfold_constants_testDTcc mht_8(mht_8_v, 526, "", "./tensorflow/tools/graph_transforms/fold_constants_test.cc", "TestMaxConstantSizeInBytes");

    auto root = tensorflow::Scope::NewRootScope();

    const int width = 100;

    Tensor a_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&a_data, 1.0f);
    Output a_const = ::tensorflow::ops::Const(
        root.WithOpName("a_expect_remains"), Input::Initializer(a_data));

    Tensor b_data(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&b_data, 1.0f);
    Output b_const = ::tensorflow::ops::Const(
        root.WithOpName("b_expect_remains"), Input::Initializer(b_data));

    Output add = ::tensorflow::ops::Add(root.WithOpName("add_expect_remains"),
                                        a_const, b_const);

    Output placeholder = ::tensorflow::ops::Placeholder(
        root.WithOpName("placeholder_expect_remains"), DT_FLOAT);

    Output mul = ::tensorflow::ops::Mul(
        root.WithOpName("output_expect_remains"), add, placeholder);

    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));

    Tensor placeholder_tensor(DT_FLOAT, TensorShape({width}));
    test::FillIota<float>(&placeholder_tensor, 1.0f);

    // Setting the maximum constant size to 10 bytes should stop the constant
    // folding at add(a, b) that would have yielded a constant of
    // 100*sizeof(float) bytes.
    graph_transforms::TransformFuncContext context;
    context.params["max_constant_size_in_bytes"] = {"10"};
    TestConstantFolding(graph_def,
                        {{"placeholder_expect_remains", placeholder_tensor}},
                        {}, {"output_expect_remains"}, context);
  }
};

TEST_F(ConstantFoldingTest, TestSimpleAdd) { TestSimpleAdd(); }

TEST_F(ConstantFoldingTest, TestOpExclusionAdd) { TestOpExclusionAdd(); }

TEST_F(ConstantFoldingTest, TestShapePropagation) { TestShapePropagation(); }

TEST_F(ConstantFoldingTest, TestPreserveOutputShapes) {
  TestPreserveOutputShapes();
}

TEST_F(ConstantFoldingTest, TestReplaceSendRecvs) { TestReplaceSendRecvs(); }

TEST_F(ConstantFoldingTest, TestReplaceSendRecvsPrefixNames) {
  TestReplaceSendRecvsPrefixNames();
}

TEST_F(ConstantFoldingTest, TestRemoveUnusedNodes) { TestRemoveUnusedNodes(); }

TEST_F(ConstantFoldingTest, TestMaxConstantSizeInBytes) {
  TestMaxConstantSizeInBytes();
}

}  // namespace graph_transforms
}  // namespace tensorflow
