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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc() {
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

#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "prefetch_dataset";

class PrefetchDatasetOpTest : public DatasetOpsTestBase {};

class PrefetchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  PrefetchDatasetParams(T input_dataset_params, int64_t buffer_size,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        int64_t slack_period, bool legacy_autotune,
                        int64_t buffer_size_min, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        buffer_size_(buffer_size),
        slack_period_(slack_period),
        legacy_autotune_(legacy_autotune),
        buffer_size_min_(buffer_size_min) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {buffer_size_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(PrefetchDatasetOp::kInputDataset);
    input_names->emplace_back(PrefetchDatasetOp::kBufferSize);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("slack_period", slack_period_);
    attr_vector->emplace_back("legacy_autotune", legacy_autotune_);
    attr_vector->emplace_back("buffer_size_min", buffer_size_min_);
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "dataset_type");

    return PrefetchDatasetOp::kDatasetType;
  }

 private:
  int64_t buffer_size_;
  int64_t slack_period_;
  bool legacy_autotune_;
  int64_t buffer_size_min_;
};

// Test case 1: positive buffer size.
PrefetchDatasetParams PrefetchDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_4(mht_4_v, 261, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/5,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 2: zero buffer size.
PrefetchDatasetParams PrefetchDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_5(mht_5_v, 281, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams2");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/0,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 3: autotune buffer size.
PrefetchDatasetParams PrefetchDatasetParams3() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_6(mht_6_v, 301, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams3");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 4: slack_period > 0.
PrefetchDatasetParams PrefetchDatasetParams4() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_7(mht_7_v, 321, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams4");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/5,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 5: legacy_autotune = false.
PrefetchDatasetParams PrefetchDatasetParams5() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_8(mht_8_v, 341, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams5");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/5,
      /*legacy_autotune=*/false,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

// Test case 6: buffer_size_min > 0.
PrefetchDatasetParams PrefetchDatasetParams6() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_9(mht_9_v, 361, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "PrefetchDatasetParams6");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-1,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/3,
      /*node_name=*/kNodeName);
}

PrefetchDatasetParams InvalidBufferSizePrefetchDatasetParams() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSprefetch_dataset_op_testDTcc mht_10(mht_10_v, 380, "", "./tensorflow/core/kernels/data/prefetch_dataset_op_test.cc", "InvalidBufferSizePrefetchDatasetParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{10, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})},
      /*node_name=*/"tensor_slice");
  return PrefetchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*buffer_size=*/-2,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*slack_period=*/0,
      /*legacy_autotune=*/true,
      /*buffer_size_min=*/0,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<PrefetchDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/PrefetchDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/PrefetchDatasetParams2(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams3(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams4(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams5(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams6(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1},
           {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})}};
}

ITERATOR_GET_NEXT_TEST_P(PrefetchDatasetOpTest, PrefetchDatasetParams,
                         GetNextTestCases())

TEST_F(PrefetchDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(PrefetchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(PrefetchDatasetOp::kDatasetType)));
}

TEST_F(PrefetchDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(PrefetchDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<PrefetchDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/PrefetchDatasetParams1(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams2(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams3(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams4(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/PrefetchDatasetParams5(),
           /*expected_cardinality=*/10}};
}

DATASET_CARDINALITY_TEST_P(PrefetchDatasetOpTest, PrefetchDatasetParams,
                           CardinalityTestCases())

TEST_F(PrefetchDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(PrefetchDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(PrefetchDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = PrefetchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      PrefetchDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<PrefetchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/PrefetchDatasetParams1(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/PrefetchDatasetParams2(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams3(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams4(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/
       PrefetchDatasetParams5(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(
           TensorShape{1},
           {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(PrefetchDatasetOpTest, PrefetchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(PrefetchDatasetOpTest, InvalidBufferSize) {
  auto dataset_params = InvalidBufferSizePrefetchDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(), error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
