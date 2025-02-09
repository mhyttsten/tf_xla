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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc() {
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
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declarations so we don't need a public header.
Status SparsifyGather(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def);
Status ReadTensorFromCheckpoint(
    const string& tensor_name, const std::unique_ptr<BundleReader>& ckpt_reader,
    const string& shape_and_slice, Tensor* tensor);

class SparsifyGatherTest : public ::testing::Test {
 protected:
  NodeDef* CreateNode(const StringPiece name, const StringPiece op,
                      const std::vector<NodeDef*>& inputs, GraphDef* graph_def,
                      bool control_dep = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/tools/graph_transforms/sparsify_gather_test.cc", "CreateNode");

    NodeDef* node_def = graph_def->add_node();
    node_def->set_name(string(name));
    node_def->set_op(string(op));
    if (!control_dep) {
      std::for_each(inputs.begin(), inputs.end(), [&node_def](NodeDef* input) {
        node_def->add_input(input->name());
      });
    } else {
      std::for_each(inputs.begin(), inputs.end(), [&node_def](NodeDef* input) {
        node_def->add_input(strings::StrCat("^", input->name()));
      });
    }
    return node_def;
  }

  void MakeGather(StringPiece name, bool gather_v2, NodeDef* params,
                  NodeDef* indices, GraphDef* graph_def) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/tools/graph_transforms/sparsify_gather_test.cc", "MakeGather");

    if (gather_v2) {
      NodeDef* axis_node =
          CreateNode(strings::StrCat(name, "_axis"), "Const", {}, graph_def);
      Tensor axis_t(DT_INT32, TensorShape({}));
      axis_t.scalar<int32>()() = 0;
      SetNodeTensorAttr<int32>("value", axis_t, axis_node);
      CreateNode(name, "GatherV2", {params, indices, axis_node}, graph_def);
    } else {
      CreateNode(name, "Gather", {params, indices}, graph_def);
    }
  }

  void TestSinglePartition(bool gather_v2, bool include_shared_init,
                           bool test_variable, bool test_kept_concat,
                           const string& shared_init_name = "group_deps") {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/tools/graph_transforms/sparsify_gather_test.cc", "TestSinglePartition");

    GraphDef graph_def;

    const auto checkpoint_path =
        io::JoinPath(testing::TmpDir(), "checkpoint_single");
    // Build the graph.
    NodeDef* input_node = CreateNode("ids", "Const", {}, &graph_def);
    NodeDef* w_node;
    NodeDef* zeros_const;
    NodeDef* zeros_shape;
    NodeDef* zeros_node;
    NodeDef* assign_node;

    Tensor weights(DT_FLOAT, TensorShape({4, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001, 1.2, 0.001});

    if (!test_variable) {
      w_node = CreateNode("w/part_1", "Const", {}, &graph_def);
      SetNodeTensorAttr<float>("value", weights, w_node);
    } else {
      w_node = CreateNode("w/part_1", "VariableV2", {}, &graph_def);

      zeros_shape = CreateNode("w/part_1/Initializer/zeros/shape_as_tensor",
                               "Const", {}, &graph_def);
      zeros_const = CreateNode("w/part_1/Initializer/zeros/Const", "Const", {},
                               &graph_def);
      zeros_node = CreateNode("w/part_1/Initializer/zeros", "Fill",
                              {zeros_shape, zeros_const}, &graph_def);
      assign_node = CreateNode("w/part_1/Assign", "Assign",
                               {w_node, zeros_node}, &graph_def);

      NodeDef* save_const_node =
          CreateNode("save/Const", "Const", {}, &graph_def);

      Tensor tensor_names_values(DT_STRING, TensorShape({1}));
      test::FillValues<tstring>(&tensor_names_values, {"w"});
      NodeDef* tensor_names_node =
          CreateNode("save/RestoreV2/tensor_names", "Const", {}, &graph_def);
      SetNodeTensorAttr<string>("value", tensor_names_values,
                                tensor_names_node);

      NodeDef* tensor_shapes_slices_node = CreateNode(
          "save/RestoreV2/shape_and_slices", "Const", {}, &graph_def);
      Tensor shapes_slices_val(DT_STRING, TensorShape({1}));
      shapes_slices_val.flat<tstring>()(0) = "4 1 0,4:0,1";
      SetNodeTensorAttr<string>("value", shapes_slices_val,
                                tensor_shapes_slices_node);

      NodeDef* restore_node = CreateNode(
          "save/RestoreV2", "RestoreV2",
          {save_const_node, tensor_names_node, tensor_shapes_slices_node},
          &graph_def);
      CreateNode("save/Assign", "Assign", {w_node, restore_node}, &graph_def);

      BundleWriter writer(Env::Default(), checkpoint_path);
      TF_ASSERT_OK(writer.Add("w", weights));
      TF_ASSERT_OK(writer.Finish());
    }
    SetNodeAttr("dtype", DT_FLOAT, w_node);

    NodeDef* identity_node =
        CreateNode("w/read", "Identity", {w_node}, &graph_def);
    MakeGather("gather", gather_v2, identity_node, input_node, &graph_def);
    if (include_shared_init) {
      if (!test_variable) {
        CreateNode(shared_init_name, "NoOp", {}, &graph_def);
      } else {
        CreateNode(shared_init_name, "NoOp", {assign_node}, &graph_def, true);
      }
    }

    NodeDef* concat_axis_node =
        CreateNode("linear/concat/axis", "Const", {}, &graph_def);
    NodeDef* concat_input_node =
        CreateNode("concat/input/node", "Const", {}, &graph_def);
    NodeDef* concat_node = nullptr;
    if (!test_kept_concat) {
      concat_node = CreateNode(
          "concat/node", "ConcatV2",
          {identity_node, concat_input_node, concat_axis_node}, &graph_def);
      SetNodeAttr("N", 2, concat_node);
    } else {
      NodeDef* concat_input_node_2 =
          CreateNode("concat/input/node_2", "Const", {}, &graph_def);
      concat_node = CreateNode("concat/node", "ConcatV2",
                               {identity_node, concat_input_node,
                                concat_input_node_2, concat_axis_node},
                               &graph_def);
      SetNodeAttr("N", 3, concat_node);
    }

    // Run the op.
    GraphDef result;
    TransformFuncContext context;
    context.input_names = {"ids"};
    context.output_names = {"gather"};
    if (test_variable) {
      context.params["input_checkpoint"] = {checkpoint_path};
    }
    if (shared_init_name != "group_deps") {
      context.params["group_init_node"] = {shared_init_name};
    }
    TF_ASSERT_OK(SparsifyGather(graph_def, context, &result));

    // Validation begins.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    // Check nodes.
    EXPECT_EQ(0,
              node_lookup.count("w/part_1/Initializer/zeros/shape_as_tensor"));
    EXPECT_EQ(0, node_lookup.count("w/part_1/Initializer/zeros/Const"));
    EXPECT_EQ(0, node_lookup.count("w/part_1/Initializer/zeros"));
    EXPECT_EQ(0, node_lookup.count("w/part_1/Assign"));

    EXPECT_EQ(1, node_lookup.count("ids"));
    EXPECT_EQ("Const", node_lookup.at("ids")->op());

    EXPECT_EQ(1, node_lookup.count("concat/node"));

    if (!test_kept_concat) {
      EXPECT_EQ(0, node_lookup.count("linear/concat/axis"));
      EXPECT_EQ("Identity", node_lookup.at("concat/node")->op());
      EXPECT_EQ(1, node_lookup.at("concat/node")->input_size());
      EXPECT_EQ("concat/input/node", node_lookup.at("concat/node")->input(0));
    } else {
      EXPECT_EQ(1, node_lookup.count("linear/concat/axis"));
      EXPECT_EQ("ConcatV2", node_lookup.at("concat/node")->op());
      EXPECT_EQ(3, node_lookup.at("concat/node")->input_size());
      EXPECT_EQ("concat/input/node", node_lookup.at("concat/node")->input(0));
      EXPECT_EQ("concat/input/node_2", node_lookup.at("concat/node")->input(1));
      EXPECT_EQ("linear/concat/axis", node_lookup.at("concat/node")->input(2));
      EXPECT_EQ(2, node_lookup.at("concat/node")->attr().at("N").i());
    }

    EXPECT_EQ(1, node_lookup.count("w/part_1/indices"));
    EXPECT_EQ("Const", node_lookup.at("w/part_1/indices")->op());
    Tensor expected_indices_tensor(DT_INT64, TensorShape({3}));
    test::FillValues<int64_t>(&expected_indices_tensor, {0, 2, 3});
    test::ExpectTensorEqual<int64_t>(
        expected_indices_tensor,
        GetNodeTensorAttr(*(node_lookup.at("w/part_1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("w/part_1/values"));
    EXPECT_EQ("Const", node_lookup.at("w/part_1/values")->op());
    Tensor expected_values_tensor(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor,
        GetNodeTensorAttr(*(node_lookup.at("w/part_1/values")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("w/part_1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("w/part_1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("w/part_1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("w/part_1/InitializeTable")->op());

    // Nodes in "gather" scope.
    EXPECT_EQ(1, node_lookup.count("gather/LookupTableFind"));
    EXPECT_EQ("LookupTableFind",
              node_lookup.at("gather/LookupTableFind")->op());

    EXPECT_EQ(1, node_lookup.count("gather/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather/Const")->op());
    Tensor expected_gather_default_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected_gather_default_tensor, {0.0});
    test::ExpectTensorNear<float>(
        expected_gather_default_tensor,
        GetNodeTensorAttr(*(node_lookup.at("gather/Const")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("gather/ExpandDims/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather/ExpandDims/Const")->op());
    Tensor expected_expand_dims_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected_expand_dims_tensor, {-1});
    test::ExpectTensorEqual<int32>(
        expected_expand_dims_tensor,
        GetNodeTensorAttr(*(node_lookup.at("gather/ExpandDims/Const")),
                          "value"));

    EXPECT_EQ(1, node_lookup.count("gather"));
    EXPECT_EQ("ExpandDims", node_lookup.at("gather")->op());

    EXPECT_EQ(1, node_lookup.count(shared_init_name));
    EXPECT_EQ("NoOp", node_lookup.at(shared_init_name)->op());

    // Check connections
    EXPECT_EQ("w/part_1/HashTable",
              node_lookup.at("w/part_1/InitializeTable")->input(0));
    EXPECT_EQ("w/part_1/indices",
              node_lookup.at("w/part_1/InitializeTable")->input(1));
    EXPECT_EQ("w/part_1/values",
              node_lookup.at("w/part_1/InitializeTable")->input(2));

    EXPECT_EQ("w/part_1/HashTable",
              node_lookup.at("gather/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather/LookupTableFind")->input(1));
    EXPECT_EQ("gather/Const",
              node_lookup.at("gather/LookupTableFind")->input(2));

    EXPECT_EQ("gather/LookupTableFind", node_lookup.at("gather")->input(0));

    // Check control dependency.
    EXPECT_NE(std::find(node_lookup.at(shared_init_name)->input().begin(),
                        node_lookup.at(shared_init_name)->input().end(),
                        "^w/part_1/InitializeTable"),
              node_lookup.at(shared_init_name)->input().end());
    EXPECT_EQ(1, node_lookup.at(shared_init_name)->input().size());
  }

  void TestMultiPartition(bool gather_v2, bool include_shared_init,
                          bool test_variable,
                          const string& shared_init_name = "group_deps") {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc mht_3(mht_3_v, 465, "", "./tensorflow/tools/graph_transforms/sparsify_gather_test.cc", "TestMultiPartition");

    // The 'ids' node is served input for two 'Gather's.
    GraphDef graph_def;

    const auto checkpoint_path =
        io::JoinPath(testing::TmpDir(), "checkpoint_multiple");
    // Build Graph:
    // Shared input node
    NodeDef* input_node = CreateNode("ids", "Const", {}, &graph_def);

    // Two partitions
    NodeDef* w_node1;
    NodeDef* w_node2;
    NodeDef* zeros_const1;
    NodeDef* zeros_shape1;
    NodeDef* zeros_node1;
    NodeDef* zeros_const2;
    NodeDef* zeros_shape2;
    NodeDef* zeros_node2;
    NodeDef* assign_node1;
    NodeDef* assign_node2;

    Tensor weights(DT_FLOAT, TensorShape({4, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001, 1.2, 0.001});
    if (!test_variable) {
      w_node1 = CreateNode("w1/part_1", "Const", {}, &graph_def);
      w_node2 = CreateNode("w2/part_1", "Const", {}, &graph_def);
      SetNodeTensorAttr<float>("value", weights, w_node1);
      SetNodeTensorAttr<float>("value", weights, w_node2);
    } else {
      NodeDef* save_const_node =
          CreateNode("save/Const", "Const", {}, &graph_def);

      NodeDef* tensor_names_node =
          CreateNode("save/RestoreV2/tensor_names", "Const", {}, &graph_def);
      Tensor tensor_names_values(DT_STRING, TensorShape({2}));
      test::FillValues<tstring>(&tensor_names_values, {"w1", "w2"});
      SetNodeTensorAttr<string>("value", tensor_names_values,
                                tensor_names_node);

      NodeDef* tensor_shapes_slices_node = CreateNode(
          "save/RestoreV2/shape_and_slices", "Const", {}, &graph_def);
      Tensor shapes_slices_val(DT_STRING, TensorShape({2}));
      shapes_slices_val.flat<tstring>()(0) = "4 1 0,4:0,1";
      shapes_slices_val.flat<tstring>()(1) = "4 1 0,4:0,1";
      SetNodeTensorAttr<string>("value", shapes_slices_val,
                                tensor_shapes_slices_node);

      NodeDef* restore_node = CreateNode(
          "save/RestoreV2", "RestoreV2",
          {save_const_node, tensor_names_node, tensor_shapes_slices_node},
          &graph_def);

      w_node1 = CreateNode("w1/part_1", "VariableV2", {}, &graph_def);

      zeros_shape1 = CreateNode("w1/part_1/Initializer/zeros/shape_as_tensor",
                                "Const", {}, &graph_def);
      zeros_const1 = CreateNode("w1/part_1/Initializer/zeros/Const", "Const",
                                {}, &graph_def);
      zeros_node1 = CreateNode("w1/part_1/Initializer/zeros", "Fill",
                               {zeros_shape1, zeros_const1}, &graph_def);
      assign_node1 = CreateNode("w1/part_1/Assign", "Assign",
                                {w_node1, zeros_node1}, &graph_def);

      CreateNode("save/Assign", "Assign", {w_node1, restore_node}, &graph_def);

      w_node2 = CreateNode("w2/part_1", "VariableV2", {}, &graph_def);
      zeros_shape2 = CreateNode("w2/part_1/Initializer/zeros/shape_as_tensor",
                                "Const", {}, &graph_def);
      zeros_const2 = CreateNode("w2/part_1/Initializer/zeros/Const", "Const",
                                {}, &graph_def);
      zeros_node2 = CreateNode("w2/part_1/Initializer/zeros", "Fill",
                               {zeros_shape2, zeros_const2}, &graph_def);
      assign_node2 = CreateNode("w2/part_1/Assign", "Assign",
                                {w_node2, zeros_node2}, &graph_def);

      CreateNode("save/Assign_1", "Assign", {w_node2, restore_node},
                 &graph_def);

      BundleWriter writer(Env::Default(), checkpoint_path);
      TF_ASSERT_OK(writer.Add("w1", weights));
      TF_ASSERT_OK(writer.Add("w2", weights));
      TF_ASSERT_OK(writer.Finish());
    }
    SetNodeAttr("dtype", DT_FLOAT, w_node1);
    SetNodeAttr("dtype", DT_FLOAT, w_node2);

    NodeDef* identity_node1 =
        CreateNode("w1/part_1/read", "Identity", {w_node1}, &graph_def);
    NodeDef* identity_node2 =
        CreateNode("w2/part_1/read", "Identity", {w_node2}, &graph_def);
    MakeGather("gather1", gather_v2, identity_node1, input_node, &graph_def);
    MakeGather("gather2", gather_v2, identity_node2, input_node, &graph_def);

    NodeDef* concat_axis_node =
        CreateNode("linear/concat/axis", "Const", {}, &graph_def);
    NodeDef* concat_node = CreateNode(
        "concat/node", "ConcatV2",
        {identity_node1, identity_node2, concat_axis_node}, &graph_def);
    SetNodeAttr("N", 2, concat_node);

    // Shared init node
    if (include_shared_init) {
      if (!test_variable) {
        CreateNode(shared_init_name, "NoOp", {}, &graph_def);
      } else {
        CreateNode(shared_init_name, "NoOp", {assign_node1, assign_node2},
                   &graph_def, true);
      }
    }

    // Run the op.
    GraphDef result;
    TransformFuncContext context;
    context.input_names = {"ids"};
    context.output_names = {"gather1", "gather2"};
    if (test_variable) {
      context.params["input_checkpoint"] = {checkpoint_path};
    }
    if (shared_init_name != "group_deps") {
      context.params["group_init_node"] = {shared_init_name};
    }
    TF_ASSERT_OK(SparsifyGather(graph_def, context, &result));

    // Validation begins.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    // Check nodes.
    EXPECT_EQ(0,
              node_lookup.count("w1/part_1/Initializer/zeros/shape_as_tensor"));
    EXPECT_EQ(0, node_lookup.count("w1/part_1/Initializer/zeros/Const"));
    EXPECT_EQ(0, node_lookup.count("w1/part_1/Initializer/zeros"));
    EXPECT_EQ(0, node_lookup.count("w1/part_1/Assign"));
    EXPECT_EQ(0,
              node_lookup.count("w2/part_1/Initializer/zeros/shape_as_tensor"));
    EXPECT_EQ(0, node_lookup.count("w2/part_1/Initializer/zeros/Const"));
    EXPECT_EQ(0, node_lookup.count("w2/part_1/Initializer/zeros"));
    EXPECT_EQ(0, node_lookup.count("w2/part_1/Assign"));
    EXPECT_EQ(1, node_lookup.count("ids"));
    EXPECT_EQ("Const", node_lookup.at("ids")->op());

    EXPECT_EQ(1, node_lookup.count(shared_init_name));
    EXPECT_EQ("NoOp", node_lookup.at(shared_init_name)->op());

    EXPECT_EQ(1, node_lookup.count("w1/part_1/indices"));
    EXPECT_EQ("Const", node_lookup.at("w1/part_1/indices")->op());
    Tensor expected_indices_tensor1(DT_INT64, TensorShape({3}));
    test::FillValues<int64_t>(&expected_indices_tensor1, {0, 2, 3});
    test::ExpectTensorEqual<int64_t>(
        expected_indices_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("w1/part_1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("w1/part_1/values"));
    EXPECT_EQ("Const", node_lookup.at("w1/part_1/values")->op());
    Tensor expected_values_tensor1(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor1, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("w1/part_1/values")), "value"),
        1e-5);

    EXPECT_EQ(1, node_lookup.count("w1/part_1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("w1/part_1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("w1/part_1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("w1/part_1/InitializeTable")->op());

    // Nodes in "gather1" scope.
    EXPECT_EQ(1, node_lookup.count("gather1/LookupTableFind"));
    EXPECT_EQ("LookupTableFind",
              node_lookup.at("gather1/LookupTableFind")->op());

    EXPECT_EQ(1, node_lookup.count("gather1/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather1/Const")->op());
    Tensor expected_gather_default_tensor1(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected_gather_default_tensor1, {0.0});
    test::ExpectTensorNear<float>(
        expected_gather_default_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("gather1/Const")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("gather1/ExpandDims/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather1/ExpandDims/Const")->op());
    Tensor expected_expand_dims_tensor1(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected_expand_dims_tensor1, {-1});
    test::ExpectTensorEqual<int32>(
        expected_expand_dims_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("gather1/ExpandDims/Const")),
                          "value"));

    EXPECT_EQ(1, node_lookup.count("gather1"));
    EXPECT_EQ("ExpandDims", node_lookup.at("gather1")->op());

    EXPECT_EQ(1, node_lookup.count("w2/part_1/indices"));
    EXPECT_EQ("Const", node_lookup.at("w2/part_1/indices")->op());
    Tensor expected_indices_tensor2(DT_INT64, TensorShape({3}));
    test::FillValues<int64_t>(&expected_indices_tensor2, {0, 2, 3});
    test::ExpectTensorEqual<int64_t>(
        expected_indices_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("w2/part_1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("w2/part_1/values"));
    EXPECT_EQ("Const", node_lookup.at("w2/part_1/values")->op());
    Tensor expected_values_tensor2(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor2, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("w2/part_1/values")), "value"),
        1e-5);

    EXPECT_EQ(1, node_lookup.count("w2/part_1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("w2/part_1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("w2/part_1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("w2/part_1/InitializeTable")->op());

    // Nodes in "gather2" scope.
    EXPECT_EQ(1, node_lookup.count("gather2/LookupTableFind"));
    EXPECT_EQ("LookupTableFind",
              node_lookup.at("gather2/LookupTableFind")->op());

    EXPECT_EQ(1, node_lookup.count("gather2/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather2/Const")->op());
    Tensor expected_gather_default_tensor2(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected_gather_default_tensor2, {0.0});
    test::ExpectTensorNear<float>(
        expected_gather_default_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("gather2/Const")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("gather2/ExpandDims/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather2/ExpandDims/Const")->op());
    Tensor expected_expand_dims_tensor2(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected_expand_dims_tensor2, {-1});
    test::ExpectTensorEqual<int32>(
        expected_expand_dims_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("gather2/ExpandDims/Const")),
                          "value"));

    EXPECT_EQ(1, node_lookup.count("gather2"));
    EXPECT_EQ("ExpandDims", node_lookup.at("gather2")->op());

    // Check connections
    EXPECT_EQ("w1/part_1/HashTable",
              node_lookup.at("w1/part_1/InitializeTable")->input(0));
    EXPECT_EQ("w1/part_1/indices",
              node_lookup.at("w1/part_1/InitializeTable")->input(1));
    EXPECT_EQ("w1/part_1/values",
              node_lookup.at("w1/part_1/InitializeTable")->input(2));

    EXPECT_EQ("w2/part_1/HashTable",
              node_lookup.at("w2/part_1/InitializeTable")->input(0));
    EXPECT_EQ("w2/part_1/indices",
              node_lookup.at("w2/part_1/InitializeTable")->input(1));
    EXPECT_EQ("w2/part_1/values",
              node_lookup.at("w2/part_1/InitializeTable")->input(2));

    EXPECT_EQ("w1/part_1/HashTable",
              node_lookup.at("gather1/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather1/LookupTableFind")->input(1));
    EXPECT_EQ("gather1/Const",
              node_lookup.at("gather1/LookupTableFind")->input(2));
    EXPECT_EQ("gather1/LookupTableFind", node_lookup.at("gather1")->input(0));

    EXPECT_EQ("w2/part_1/HashTable",
              node_lookup.at("gather2/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather2/LookupTableFind")->input(1));
    EXPECT_EQ("gather2/Const",
              node_lookup.at("gather2/LookupTableFind")->input(2));
    EXPECT_EQ("gather2/LookupTableFind", node_lookup.at("gather2")->input(0));

    EXPECT_EQ(0, node_lookup.count("linear/concat/axis"));
    EXPECT_EQ(0, node_lookup.count("concat/node"));

    // Check control deps.
    EXPECT_EQ(2, node_lookup.at(shared_init_name)->input_size());
    EXPECT_NE(std::find(node_lookup.at(shared_init_name)->input().begin(),
                        node_lookup.at(shared_init_name)->input().end(),
                        "^w1/part_1/InitializeTable"),
              node_lookup.at(shared_init_name)->input().end());

    EXPECT_NE(std::find(node_lookup.at(shared_init_name)->input().begin(),
                        node_lookup.at(shared_init_name)->input().end(),
                        "^w2/part_1/InitializeTable"),
              node_lookup.at(shared_init_name)->input().end());
  }
  void TestReadTensorSlice() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSsparsify_gather_testDTcc mht_4(mht_4_v, 755, "", "./tensorflow/tools/graph_transforms/sparsify_gather_test.cc", "TestReadTensorSlice");

    const auto checkpoint_path =
        io::JoinPath(testing::TmpDir(), "checkpoint_slice");

    Tensor weights(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001});
    BundleWriter writer(Env::Default(), checkpoint_path);
    TF_ASSERT_OK(writer.AddSlice("w", TensorShape({4, 1}),
                                 TensorSlice::ParseOrDie("0,2:0,1"), weights));
    TF_ASSERT_OK(writer.Finish());

    std::unique_ptr<BundleReader> reader(
        new BundleReader(Env::Default(), checkpoint_path));

    Tensor results;
    TF_ASSERT_OK(
        ReadTensorFromCheckpoint("w/part_0", reader, "4 1 0,2:0,1", &results));

    test::ExpectTensorEqual<float>(weights, results);
  }
};

TEST_F(SparsifyGatherTest, TestSinglePartition) {
  TestSinglePartition(false, false, false, false);
  TestSinglePartition(false, true, false, false);
  TestSinglePartition(true, false, false, false);
  TestSinglePartition(true, true, false, false);
  TestSinglePartition(false, false, true, false);
  TestSinglePartition(false, true, true, false);
  TestSinglePartition(true, false, true, false);
  TestSinglePartition(true, true, true, false);
  TestSinglePartition(false, true, false, false, "shared_inits");
  TestSinglePartition(true, true, false, false, "shared_inits");
  TestSinglePartition(false, true, true, false, "shared_inits");
  TestSinglePartition(true, true, true, false, "shared_inits");

  TestSinglePartition(false, false, false, true);
  TestSinglePartition(false, true, false, true);
  TestSinglePartition(true, false, false, true);
  TestSinglePartition(true, true, false, true);
  TestSinglePartition(false, false, true, true);
  TestSinglePartition(false, true, true, true);
  TestSinglePartition(true, false, true, true);
  TestSinglePartition(true, true, true, true);
  TestSinglePartition(false, true, false, true, "shared_inits");
  TestSinglePartition(true, true, false, true, "shared_inits");
  TestSinglePartition(false, true, true, true, "shared_inits");
  TestSinglePartition(true, true, true, true, "shared_inits");
}

TEST_F(SparsifyGatherTest, TestMultiPartition) {
  TestMultiPartition(false, false, false);
  TestMultiPartition(false, true, false);
  TestMultiPartition(true, false, false);
  TestMultiPartition(true, true, false);
  TestMultiPartition(false, false, true);
  TestMultiPartition(false, true, true);
  TestMultiPartition(true, false, true);
  TestMultiPartition(true, true, true);
  TestMultiPartition(false, true, false, "shared_inits");
  TestMultiPartition(true, true, false, "shared_inits");
  TestMultiPartition(false, true, true, "shared_inits");
  TestMultiPartition(true, true, true, "shared_inits");
}

TEST_F(SparsifyGatherTest, TestTensorSlice) { TestReadTensorSlice(); }

}  // namespace graph_transforms
}  // namespace tensorflow
