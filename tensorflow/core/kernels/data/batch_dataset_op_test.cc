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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/batch_dataset_op.h"

#include <string>

#include "tensorflow/core/common_runtime/forward_type_inference.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "batch_dataset";

class BatchDatasetOpTest : public DatasetOpsTestBase {};

// Test Case 1: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDatasetParams1() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams1");

  return BatchDatasetParams(RangeDatasetParams(0, 12, 1),
                            /*batch_size=*/4,
                            /*drop_remainder=*/false,
                            /*parallel_copy=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({4})},
                            /*node_name=*/kNodeName);
}

// Test Case 2: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDatasetParams2() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams2");

  return BatchDatasetParams(RangeDatasetParams(0, 12, 1),
                            /*batch_size=*/4,
                            /*drop_remainder=*/true,
                            /*parallel_copy=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({4})},
                            /*node_name=*/kNodeName);
}

// Test Case 3: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDatasetParams3() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams3");

  return BatchDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*batch_size=*/3,
                            /*drop_remainder=*/false,
                            /*parallel_copy=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({-1})},
                            /*node_name=*/kNodeName);
}

// Test Case 4: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDatasetParams4() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams4");

  return BatchDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*batch_size=*/3,
                            /*drop_remainder=*/true,
                            /*parallel_copy=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({3})},
                            /*node_name=*/kNodeName);
}

// Test Case 5: test BatchDatasetV2 with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDatasetParams5() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams5");

  return BatchDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*batch_size=*/12,
                            /*drop_remainder=*/true,
                            /*parallel_copy=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({12})},
                            /*node_name=*/kNodeName);
}

// Test Case 6: test BatchDatasetV2 with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDatasetParams6() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams6");

  return BatchDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*batch_size=*/12,
                            /*drop_remainder=*/false,
                            /*parallel_copy=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({-1})},
                            /*node_name=*/kNodeName);
}

// Test Case 7: test BatchDatasetV2 with `drop_remainder` = false and
// the output of the input dataset is empty.
BatchDatasetParams BatchDatasetParams7() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_6(mht_6_v, 290, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "BatchDatasetParams7");

  return BatchDatasetParams(RangeDatasetParams(0, 0, 1),
                            /*batch_size=*/4,
                            /*drop_remainder=*/false,
                            /*parallel_copy=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({4})},
                            /*node_name=*/kNodeName);
}

// Test Case 8: test BatchDatasetV2 with an invalid batch size
BatchDatasetParams InvalidBatchSizeBatchDatasetParams() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_7(mht_7_v, 304, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "InvalidBatchSizeBatchDatasetParams");

  return BatchDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*batch_size=*/-1,
                            /*drop_remainder=*/false,
                            /*parallel_copy=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({3})},
                            /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<BatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({4}), {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({4}), {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64_t>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64_t>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64_t>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({3}),
                                  {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({10}),
                                  {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})},

          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                         GetNextTestCases())

TEST_F(BatchDatasetOpTest, DatasetNodeName) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(batch_dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(batch_dataset_params.node_name()));
}

TEST_F(BatchDatasetOpTest, DatasetTypeString) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(batch_dataset_params));
  name_utils::OpNameParams params;
  params.op_version = batch_dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(BatchDatasetOp::kDatasetType, params)));
}

TEST_F(BatchDatasetOpTest, DatasetOutputDtypes) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(batch_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<BatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<BatchDatasetParams>> CardinalityTestCases() {
  return {
      {/*dataset_params=*/BatchDatasetParams1(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams2(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams3(), /*expected_cardinality=*/4},
      {/*dataset_params=*/BatchDatasetParams4(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams5(), /*expected_cardinality=*/0},
      {/*dataset_params=*/BatchDatasetParams6(), /*expected_cardinality=*/1},
      {/*dataset_params=*/BatchDatasetParams7(), /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                           CardinalityTestCases())

TEST_F(BatchDatasetOpTest, IteratorOutputDtypes) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(batch_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<BatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(BatchDatasetOpTest, IteratorOutputPrefix) {
  auto batch_dataset_params = BatchDatasetParams1();
  TF_ASSERT_OK(Initialize(batch_dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = batch_dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      BatchDatasetOp::kDatasetType, batch_dataset_params.iterator_prefix(),
      params)));
}

std::vector<IteratorSaveAndRestoreTestCase<BatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({4}), {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({4}), {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64_t>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64_t>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64_t>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64_t>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64_t>(TensorShape({3}), {6, 7, 8})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape({10}),
                                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(BatchDatasetOpTest, InvalidBatchSize) {
  auto batch_dataset_params = InvalidBatchSizeBatchDatasetParams();
  EXPECT_EQ(Initialize(batch_dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

// TODO(b/222556529) when Const has type constructor, remove the following
REGISTER_OP("BatchDatasetOpTest>ConstTypeCtor")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type")
    .SetTypeConstructor(full_type::Unary(TFT_TENSOR, "dtype"));

// Adds identity notes to all outputs of this node
static void add_identity_nodes(Node* node, Graph& graph,
                               std::vector<Node*>& identity_nodes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_8(mht_8_v, 499, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "add_identity_nodes");

  for (int i = 0; i < node->num_outputs(); i++) {
    Node* new_node;
    std::string name = absl::StrCat("Identity", i);
    TF_EXPECT_OK(NodeBuilder(name, "Identity")
                     .Attr("T", node->output_type(i))
                     .Input(node, i)
                     .Finalize(&graph, &new_node));
    identity_nodes.push_back(new_node);
  }
}

// Runs type inference pass on graph
static Status type_inference(Graph& graph) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSbatch_dataset_op_testDTcc mht_9(mht_9_v, 515, "", "./tensorflow/core/kernels/data/batch_dataset_op_test.cc", "type_inference");

  GraphOptimizationPassOptions opt_options;
  std::unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
  graph_ptr->Copy(graph);
  opt_options.graph = &graph_ptr;
  opt_options.flib_def = graph.mutable_flib_def();
  ForwardTypeInferencePass pass;
  return pass.Run(opt_options);
}

TEST(BatchDatsetOpTest, TypeInference) {
  Graph graph(OpRegistry::Global());
  Node* input_dataset;
  Node* batch_size;
  Node* drop_remainder;
  Node* batch_dataset_v2;
  FullTypeDef input_dataset_t;
  protobuf::TextFormat::Parser parser;
  CHECK(parser.ParseFromString(
      R"pb(type_id: TFT_PRODUCT
           args {
             type_id: TFT_DATASET
             args {
               type_id: TFT_PRODUCT
               args {
                 type_id: TFT_RAGGED
                 args { type_id: TFT_STRING }
               }
             }
           })pb",
      &input_dataset_t));
  TensorProto tensor_proto;
  TF_EXPECT_OK(NodeBuilder("input_dataset", "Const")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_VARIANT)
                   .Finalize(&graph, &input_dataset));
  (*input_dataset->mutable_def()->mutable_experimental_type()) =
      input_dataset_t;
  // TODO(b/222556529) when Const has type constructor, use Const
  TF_EXPECT_OK(NodeBuilder("batch_size", "BatchDatasetOpTest>ConstTypeCtor")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_INT64)
                   .Finalize(&graph, &batch_size));
  // TODO(b/222556529) when Const has type constructor, use Const
  TF_EXPECT_OK(NodeBuilder("drop_remainder", "BatchDatasetOpTest>ConstTypeCtor")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_BOOL)
                   .Finalize(&graph, &drop_remainder));
  TF_EXPECT_OK(NodeBuilder("BatchDatasetV2", "BatchDatasetV2")
                   .Attr("output_types", {DT_VARIANT})
                   .Attr("output_shapes", {TensorShape({1})})
                   .Input(input_dataset)
                   .Input(batch_size)
                   .Input(drop_remainder)
                   .Finalize(&graph, &batch_dataset_v2));

  std::vector<Node*> identity_nodes;
  add_identity_nodes(batch_dataset_v2, graph, identity_nodes);
  TF_EXPECT_OK(type_inference(graph));
  EXPECT_TRUE(full_type::IsEqual(identity_nodes[0]->def().experimental_type(),
                                 input_dataset_t))
      << "fulltype is\n"
      << identity_nodes[0]->def().experimental_type().DebugString()
      << "\nexpected\n"
      << input_dataset_t.DebugString();
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
