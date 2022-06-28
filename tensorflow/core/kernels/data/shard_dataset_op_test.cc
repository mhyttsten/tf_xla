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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/shard_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "shard_dataset";

class ShardDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ShardDatasetParams(T input_dataset_params, int64_t num_shards, int64_t index,
                     bool require_non_empty, DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_shards_(num_shards),
        index_(index),
        require_non_empty_(require_non_empty) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return CreateTensors<int64_t>(TensorShape({}), {{num_shards_}, {index_}});
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(ShardDatasetOp::kInputDataset);
    input_names->emplace_back(ShardDatasetOp::kNumShards);
    input_names->emplace_back(ShardDatasetOp::kIndex);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("require_non_empty", require_non_empty_);
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "dataset_type");
 return ShardDatasetOp::kDatasetType; }

 private:
  int64_t num_shards_;
  int64_t index_;
  bool require_non_empty_;
};

class ShardDatasetOpTest : public DatasetOpsTestBase {};

// Test Case 1: simple case.
ShardDatasetParams ShardDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_4(mht_4_v, 254, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams1");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/2,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 2: zero offset.
ShardDatasetParams ShardDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams2");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/0,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 3: iterator ends before first element.
ShardDatasetParams ShardDatasetParams3() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_6(mht_6_v, 282, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams3");

  return ShardDatasetParams(RangeDatasetParams(0, 1, 1),
                            /*num_shards=*/5,
                            /*index=*/2,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 4: larger num_shards.
ShardDatasetParams ShardDatasetParams4() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_7(mht_7_v, 296, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams4");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/7,
                            /*index=*/5,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 5: index == num_shards.
ShardDatasetParams ShardDatasetParams5() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_8(mht_8_v, 310, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams5");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/4,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 6: similar with test_case_5 but the number of outputs could not be
// divided evenly by num_shards.
ShardDatasetParams ShardDatasetParams6() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_9(mht_9_v, 325, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams6");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/4,
                            /*index=*/3,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 7: num_shard is larger than the cardinality of input dataset;
// require_non_empty = false.
ShardDatasetParams ShardDatasetParams7() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_10(mht_10_v, 340, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "ShardDatasetParams7");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/20,
                            /*index=*/5,
                            /*require_non_empty=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 8: similar with test_case_7 but require_non_empty = true.
ShardDatasetParams InvalidShardDatasetParamsWithNoElemForEachShard() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_11(mht_11_v, 354, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "InvalidShardDatasetParamsWithNoElemForEachShard");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/20,
                            /*index=*/5,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 9: index is greater than the number of shards.
ShardDatasetParams InvalidShardDatasetParams1() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_12(mht_12_v, 368, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "InvalidShardDatasetParams1");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/7,
                            /*require_non_empty=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 10: negative index.
ShardDatasetParams InvalidShardDatasetParams2() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_13(mht_13_v, 382, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "InvalidShardDatasetParams2");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/-3,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 11: negative number of shards.
ShardDatasetParams InvalidShardDatasetParams3() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_14(mht_14_v, 396, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "InvalidShardDatasetParams3");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/-3,
                            /*index=*/1,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 12: zero number of shards.
ShardDatasetParams InvalidShardDatasetParams4() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_op_testDTcc mht_15(mht_15_v, 410, "", "./tensorflow/core/kernels/data/shard_dataset_op_test.cc", "InvalidShardDatasetParams4");

  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/0,
                            /*index=*/1,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ShardDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/ShardDatasetParams1(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{2}, {7}})},
      {/*dataset_params=*/ShardDatasetParams2(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{0}, {5}})},
      {/*dataset_params=*/ShardDatasetParams3(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ShardDatasetParams4(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{5}})},
      {/*dataset_params=*/ShardDatasetParams5(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{4}, {9}})},
      {/*dataset_params=*/ShardDatasetParams6(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{3}, {7}})},
      {/*dataset_params=*/ShardDatasetParams7(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{5}})}};
}

ITERATOR_GET_NEXT_TEST_P(ShardDatasetOpTest, ShardDatasetParams,
                         GetNextTestCases())

TEST_F(ShardDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ShardDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(ShardDatasetOp::kDatasetType)));
}

TEST_F(ShardDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ShardDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<ShardDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/ShardDatasetParams1(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams2(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams3(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/ShardDatasetParams4(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/ShardDatasetParams5(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams6(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams7(),
           /*expected_cardinality=*/1}};
}

DATASET_CARDINALITY_TEST_P(ShardDatasetOpTest, ShardDatasetParams,
                           CardinalityTestCases())

TEST_F(ShardDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ShardDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(ShardDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ShardDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ShardDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/ShardDatasetParams1(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{2}, {7}})},
      {/*dataset_params=*/ShardDatasetParams2(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{0}, {5}})},
      {/*dataset_params=*/ShardDatasetParams3(),
       /*breakpoints=*/{0, 1},
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ShardDatasetParams4(),
       /*breakpoints=*/{0, 5},
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{5}})},
      {/*dataset_params=*/ShardDatasetParams5(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{4}, {9}})},
      {/*dataset_params=*/ShardDatasetParams6(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{3}, {7}})},
      {/*dataset_params=*/ShardDatasetParams7(),
       /*breakpoints=*/{0, 5},
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape{}, {{5}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ShardDatasetOpTest, ShardDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(ShardDatasetOpTest, NoElemForEachShard) {
  auto dataset_params = InvalidShardDatasetParamsWithNoElemForEachShard();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(ShardDatasetOpTest, InvalidArguments) {
  std::vector<ShardDatasetParams> invalid_dataset_params = {
      InvalidShardDatasetParams1(), InvalidShardDatasetParams2(),
      InvalidShardDatasetParams3(), InvalidShardDatasetParams4()};
  for (const auto& dataset_params : invalid_dataset_params) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
