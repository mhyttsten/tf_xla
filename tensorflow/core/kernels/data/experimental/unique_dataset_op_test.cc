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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/unique_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "unique_dataset";

class UniqueDatasetParams : public DatasetParams {
 public:
  template <typename T>
  UniqueDatasetParams(T input_dataset_params, DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      kNodeName) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "UniqueDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(UniqueDatasetOp::kInputDataset);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attributes) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "GetAttributes");

    *attributes = {{"output_types", output_dtypes_},
                   {"output_shapes", output_shapes_},
                   {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_3(mht_3_v, 231, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "dataset_type");
 return UniqueDatasetOp::kDatasetType; }
};

class UniqueDatasetOpTest : public DatasetOpsTestBase {};

UniqueDatasetParams NormalCaseParams() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "NormalCaseParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{12, 1},
                             {1, 1, 2, 3, 5, 8, 13, 3, 21, 8, 8, 34})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(tensor_slice_dataset_params,
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams LastRecordIsDuplicateParams() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_5(mht_5_v, 253, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "LastRecordIsDuplicateParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{11, 1},
                             {1, 1, 2, 3, 5, 8, 13, 3, 21, 8, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams AllRecordsTheSameParams() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_6(mht_6_v, 267, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "AllRecordsTheSameParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{5, 1}, {1, 1, 1, 1, 1})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams EmptyInputParams() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_7(mht_7_v, 280, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "EmptyInputParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{0, 1}, {})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

UniqueDatasetParams StringParams() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_8(mht_8_v, 293, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "StringParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(
          TensorShape{11, 1},
          {"one", "One", "two", "three", "five", "eight", "thirteen",
           "twenty-one", "eight", "eight", "thirty-four"})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_STRING},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

// Two components in input dataset --> Should result in error during dataset
// construction
UniqueDatasetParams TwoComponentsParams() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_9(mht_9_v, 311, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "TwoComponentsParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {
          CreateTensor<int64_t>(TensorShape{1, 1}, {1}),
          CreateTensor<int64_t>(TensorShape{1, 1}, {42}),
      },
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1}), PartialTensorShape({1})});
}

// Zero components in input dataset --> Should result in error during dataset
// construction
UniqueDatasetParams NoInputParams() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_10(mht_10_v, 330, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "NoInputParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_INT64},
                             /*output_shapes=*/{PartialTensorShape({})});
}

// Floating-point --> Should result in error during dataset construction
UniqueDatasetParams FP32Params() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunique_dataset_op_testDTcc mht_11(mht_11_v, 343, "", "./tensorflow/core/kernels/data/experimental/unique_dataset_op_test.cc", "FP32Params");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<float>(TensorShape{1, 1}, {3.14})},
      /*node_name=*/"tensor_slice_dataset");
  return UniqueDatasetParams(std::move(tensor_slice_dataset_params),
                             /*output_dtypes=*/{DT_FLOAT},
                             /*output_shapes=*/{PartialTensorShape({1})});
}

std::vector<GetNextTestCase<UniqueDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}, {34}})},
          {/*dataset_params=*/LastRecordIsDuplicateParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}})},
          {/*dataset_params=*/AllRecordsTheSameParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{1}})},
          {/*dataset_params=*/EmptyInputParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {})},
          {/*dataset_params=*/StringParams(),
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"one"},
                                                     {"One"},
                                                     {"two"},
                                                     {"three"},
                                                     {"five"},
                                                     {"eight"},
                                                     {"thirteen"},
                                                     {"twenty-one"},
                                                     {"thirty-four"}})}};
}

ITERATOR_GET_NEXT_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                         GetNextTestCases())

TEST_F(UniqueDatasetOpTest, DatasetNodeName) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(UniqueDatasetOpTest, DatasetTypeString) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(UniqueDatasetOp::kDatasetType)));
}

TEST_F(UniqueDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(UniqueDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({1})}));
}

std::vector<CardinalityTestCase<UniqueDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_cardinality=*/kUnknownCardinality},
          // Current implementation doesn't propagate input cardinality of zero
          // to its output cardinality.
          {/*dataset_params=*/EmptyInputParams(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<UniqueDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/StringParams(),
           /*expected_output_dtypes=*/{DT_STRING}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<UniqueDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/StringParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(UniqueDatasetOpTest, IteratorPrefix) {
  auto dataset_params = NormalCaseParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      UniqueDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<UniqueDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/NormalCaseParams(),
           /*breakpoints=*/{0, 2, 6, 8},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}, {34}})},
          {/*dataset_params=*/LastRecordIsDuplicateParams(),
           /*breakpoints=*/{0, 2, 6, 8},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}),
                                  {{1}, {2}, {3}, {5}, {8}, {13}, {21}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(UniqueDatasetOpTest, UniqueDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

class ParameterizedInvalidInputTest
    : public UniqueDatasetOpTest,
      public ::testing::WithParamInterface<UniqueDatasetParams> {};

TEST_P(ParameterizedInvalidInputTest, InvalidInput) {
  auto dataset_params = GetParam();
  auto result = Initialize(dataset_params);
  EXPECT_FALSE(result.ok());
}

INSTANTIATE_TEST_SUITE_P(FilterDatasetOpTest, ParameterizedInvalidInputTest,
                         ::testing::ValuesIn({TwoComponentsParams(),
                                              NoInputParams(), FP32Params()}));

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
