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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/auto_shard_dataset_op.h"

#include <string>

#include "tensorflow/core/common_runtime/forward_type_inference.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/data/shard_dataset_op.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "auto_shard_dataset";

class AutoShardDatasetParams : public DatasetParams {
 public:
  template <typename T>
  AutoShardDatasetParams(T input_dataset_params, int64_t num_workers,
                         int64_t index, int auto_shard_policy,
                         int64_t num_replicas, DataTypeVector output_dtypes,
                         std::vector<PartialTensorShape> output_shapes,
                         string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_workers_(num_workers),
        num_replicas_(num_replicas),
        index_(index),
        auto_shard_policy_(auto_shard_policy) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return CreateTensors<int64_t>(TensorShape({}), {{num_workers_}, {index_}});
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(AutoShardDatasetOp::kInputDataset);
    input_names->emplace_back(AutoShardDatasetOp::kNumWorkers);
    input_names->emplace_back(AutoShardDatasetOp::kIndex);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back(AutoShardDatasetOp::kAutoShardPolicy,
                              auto_shard_policy_);
    attr_vector->emplace_back(AutoShardDatasetOp::kNumReplicas, num_replicas_);
    attr_vector->emplace_back(AutoShardDatasetOp::kOutputTypes, output_dtypes_);
    attr_vector->emplace_back(AutoShardDatasetOp::kOutputShapes,
                              output_shapes_);
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "dataset_type");

    return AutoShardDatasetOp::kDatasetType;
  }

 private:
  int64_t num_workers_;
  int64_t num_replicas_;
  int64_t index_;
  int auto_shard_policy_;
};

class AutoShardDatasetOpTest : public DatasetOpsTestBase {};

// Test Case 1: simple case.
AutoShardDatasetParams AutoShardDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams1");

  return AutoShardDatasetParams(RangeDatasetParams(0, 10, 1),
                                /*num_workers=*/5,
                                /*index=*/2,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/5,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

// Test Case 2: the index is larger than the available elements.
AutoShardDatasetParams AutoShardDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams2");

  return AutoShardDatasetParams(RangeDatasetParams(0, 1, 1),
                                /*num_workers=*/5,
                                /*index=*/2,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/5,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

// Test Case 3: the number of outputs could not be evenly divided by
// num_workers.
AutoShardDatasetParams AutoShardDatasetParams3() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_6(mht_6_v, 299, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams3");

  return AutoShardDatasetParams(RangeDatasetParams(0, 10, 1),
                                /*num_workers=*/4,
                                /*index=*/3,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/4,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

// TODO(feihugis): add more test cases that have ReaderDatasets (e.g. a
// CSVDataset or a TFRecordDataset) in the pipeline.

// Test case 4: the index is greater than the number of workers.
AutoShardDatasetParams AutoShardDatasetParams4() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_7(mht_7_v, 317, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams4");

  return AutoShardDatasetParams(RangeDatasetParams(0, 10, 1),
                                /*num_workers=*/5,
                                /*index=*/7,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/5,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

// Test case 5: the index is negative.
AutoShardDatasetParams AutoShardDatasetParams5() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_8(mht_8_v, 332, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams5");

  return AutoShardDatasetParams(RangeDatasetParams(0, 10, 1),
                                /*num_workers=*/5,
                                /*index=*/-3,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/5,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

// Test case 6: num_workers is negative.
AutoShardDatasetParams AutoShardDatasetParams6() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_9(mht_9_v, 347, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams6");

  return AutoShardDatasetParams(RangeDatasetParams(0, 10, 1),
                                /*num_workers=*/-3,
                                /*index=*/1,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/5,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

// Test case 7: num_workers is zero.
AutoShardDatasetParams AutoShardDatasetParams7() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_10(mht_10_v, 362, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "AutoShardDatasetParams7");

  return AutoShardDatasetParams(RangeDatasetParams(0, 10, 1),
                                /*num_workers=*/0,
                                /*index=*/1,
                                /*auto_shard_policy=*/0,
                                /*num_replicas=*/5,
                                /*output_dtypes=*/{DT_INT64},
                                /*output_shapes=*/{PartialTensorShape({})},
                                /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<AutoShardDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/AutoShardDatasetParams1(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{2}, {7}})},
      {/*dataset_params=*/AutoShardDatasetParams2(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/AutoShardDatasetParams3(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{3}, {7}})}};
}

ITERATOR_GET_NEXT_TEST_P(AutoShardDatasetOpTest, AutoShardDatasetParams,
                         GetNextTestCases())

TEST_F(AutoShardDatasetOpTest, InvalidArguments) {
  std::vector<AutoShardDatasetParams> invalid_dataset_params = {
      AutoShardDatasetParams4(), AutoShardDatasetParams5(),
      AutoShardDatasetParams6(), AutoShardDatasetParams7()};
  for (const auto& dataset_params : invalid_dataset_params) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

// TODO(b/222556529) when Const has type constructor, remove the following
REGISTER_OP("AutoShardDatasetOpTest>ConstTypeCtor")
    .Output("output: dtype")
    .Attr("value: tensor")
    .Attr("dtype: type")
    .SetTypeConstructor(full_type::Unary(TFT_TENSOR, "dtype"));

// Adds identity notes to all outputs of this node
static void add_identity_nodes(Node* node, Graph& graph,
                               std::vector<Node*>& identity_nodes) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_11(mht_11_v, 408, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "add_identity_nodes");

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
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSauto_shard_dataset_op_testDTcc mht_12(mht_12_v, 424, "", "./tensorflow/core/kernels/data/experimental/auto_shard_dataset_op_test.cc", "type_inference");

  GraphOptimizationPassOptions opt_options;
  std::unique_ptr<Graph> graph_ptr(new Graph(OpRegistry::Global()));
  graph_ptr->Copy(graph);
  opt_options.graph = &graph_ptr;
  opt_options.flib_def = graph.mutable_flib_def();
  ForwardTypeInferencePass pass;
  return pass.Run(opt_options);
}

TEST_F(AutoShardDatasetOpTest, AutoShardDatasetTypeInference) {
  Graph graph(OpRegistry::Global());
  Node* input_dataset;
  Node* num_workers;
  Node* index;
  Node* auto_shard_dataset;
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
  TF_EXPECT_OK(
      NodeBuilder("num_workers", "AutoShardDatasetOpTest>ConstTypeCtor")
          .Attr("value", tensor_proto)
          .Attr("dtype", DT_INT64)
          .Finalize(&graph, &num_workers));
  // TODO(b/222556529) when Const has type constructor, use Const
  TF_EXPECT_OK(NodeBuilder("index", "AutoShardDatasetOpTest>ConstTypeCtor")
                   .Attr("value", tensor_proto)
                   .Attr("dtype", DT_INT64)
                   .Finalize(&graph, &index));
  TF_EXPECT_OK(NodeBuilder("AutoShardDataset", "AutoShardDataset")
                   .Attr("output_types", {DT_VARIANT})
                   .Attr("output_shapes", {TensorShape({1})})
                   .Input(input_dataset)
                   .Input(num_workers)
                   .Input(index)
                   .Finalize(&graph, &auto_shard_dataset));
  std::vector<Node*> identity_nodes;
  add_identity_nodes(auto_shard_dataset, graph, identity_nodes);
  TF_EXPECT_OK(type_inference(graph));
  EXPECT_TRUE(full_type::IsEqual(identity_nodes[0]->def().experimental_type(),
                                 input_dataset_t))
      << "fulltype is\n"
      << identity_nodes[0]->def().experimental_type().DebugString()
      << "\nexpected\n"
      << input_dataset_t.DebugString();
}

TEST_F(AutoShardDatasetOpTest, RebatchDatasetTypeInference) {
  Graph graph(OpRegistry::Global());
  Node* input_dataset;
  Node* num_replicas;
  Node* rebatch_dataset;
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
  TF_EXPECT_OK(
      NodeBuilder("num_replicas", "AutoShardDatasetOpTest>ConstTypeCtor")
          .Attr("value", tensor_proto)
          .Attr("dtype", DT_INT64)
          .Finalize(&graph, &num_replicas));
  TF_EXPECT_OK(NodeBuilder("RebatchDataset", "RebatchDataset")
                   .Attr("output_types", {DT_VARIANT})
                   .Attr("output_shapes", {TensorShape({1})})
                   .Input(input_dataset)
                   .Input(num_replicas)
                   .Finalize(&graph, &rebatch_dataset));
  std::vector<Node*> identity_nodes;
  add_identity_nodes(rebatch_dataset, graph, identity_nodes);
  TF_EXPECT_OK(type_inference(graph));
  EXPECT_TRUE(full_type::IsEqual(identity_nodes[0]->def().experimental_type(),
                                 input_dataset_t))
      << "fulltype is\n"
      << identity_nodes[0]->def().experimental_type().DebugString()
      << "\nexpected\n"
      << input_dataset_t.DebugString();
}

TEST_F(AutoShardDatasetOpTest, RebatchDatasetV2TypeInference) {
  Graph graph(OpRegistry::Global());
  Node* input_dataset;
  Node* batch_sizes;
  Node* drop_remainder;
  Node* rebatch_dataset_v2;
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
  TF_EXPECT_OK(
      NodeBuilder("num_replicas", "AutoShardDatasetOpTest>ConstTypeCtor")
          .Attr("value", tensor_proto)
          .Attr("dtype", DT_INT64)
          .Finalize(&graph, &batch_sizes));
  // TODO(b/222556529) when Const has type constructor, use Const
  TF_EXPECT_OK(
      NodeBuilder("drop_remainder", "AutoShardDatasetOpTest>ConstTypeCtor")
          .Attr("value", tensor_proto)
          .Attr("dtype", DT_BOOL)
          .Finalize(&graph, &drop_remainder));
  TF_EXPECT_OK(NodeBuilder("RebatchDatasetV2", "RebatchDatasetV2")
                   .Attr("output_types", {DT_VARIANT})
                   .Attr("output_shapes", {TensorShape({1})})
                   .Input(input_dataset)
                   .Input(batch_sizes)
                   .Input(drop_remainder)
                   .Finalize(&graph, &rebatch_dataset_v2));
  std::vector<Node*> identity_nodes;
  add_identity_nodes(rebatch_dataset_v2, graph, identity_nodes);
  TF_EXPECT_OK(type_inference(graph));
  EXPECT_TRUE(full_type::IsEqual(identity_nodes[0]->def().experimental_type(),
                                 input_dataset_t))
      << "fulltype is\n"
      << identity_nodes[0]->def().experimental_type().DebugString()
      << "\nexpected\n"
      << input_dataset_t.DebugString();
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
