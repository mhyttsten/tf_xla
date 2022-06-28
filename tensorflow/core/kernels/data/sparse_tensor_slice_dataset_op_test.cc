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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc() {
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

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "sparse_tensor_slice_dataset";
constexpr char kDatasetType[] = "SparseTensorSlice";

class SparseTensorSliceDatasetParams : public DatasetParams {
 public:
  SparseTensorSliceDatasetParams(Tensor indices, Tensor values,
                                 Tensor dense_shape, DataType tvalues,
                                 string node_name)
      : DatasetParams({tvalues}, {PartialTensorShape({})},
                      std::move(node_name)),
        indices_(std::move(indices)),
        values_(std::move(values)),
        dense_shape_(std::move(dense_shape)),
        tvalues_(tvalues) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "SparseTensorSliceDatasetParams");

    iterator_prefix_ = "Iterator";
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {indices_, values_, dense_shape_};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back("indices");
    input_names->emplace_back("values");
    input_names->emplace_back("dense_shape");
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("Tvalues", tvalues_);
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "dataset_type");
 return kDatasetType; }

 private:
  Tensor indices_;
  Tensor values_;
  Tensor dense_shape_;
  DataType tvalues_;
};

class SparseTensorSliceDatasetOpTest : public DatasetOpsTestBase {};

SparseTensorSliceDatasetParams TwoDimsSparseTensorSliceDatasetParams() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "TwoDimsSparseTensorSliceDatasetParams");

  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64_t>({2, 2}, {0, 0, 1, 1}),
      /*values=*/CreateTensor<int32>({2}, {888, 999}),
      /*dense_shape=*/CreateTensor<int64_t>({2}, {2, 2}),
      /*tvalues=*/DT_INT32,
      /*node_name=*/kNodeName);
}

SparseTensorSliceDatasetParams ThreeDimsSparseTensorSliceDatasetParams() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_5(mht_5_v, 267, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "ThreeDimsSparseTensorSliceDatasetParams");

  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64_t>({2, 3}, {0, 0, 0, 1, 1, 1}),
      /*values=*/CreateTensor<double>({2}, {888.0, 999.0}),
      /*dense_shape=*/CreateTensor<int64_t>({3}, {2, 2, 2}),
      /*tvalues=*/DT_DOUBLE,
      /*node_name=*/kNodeName);
}

SparseTensorSliceDatasetParams FourDimsSparseTensorSliceDatasetParams() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "FourDimsSparseTensorSliceDatasetParams");

  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64_t>({2, 4}, {0, 0, 0, 0, 1, 1, 1, 1}),
      /*values=*/CreateTensor<tstring>({2}, {"a", "b"}),
      /*dense_shape=*/CreateTensor<int64_t>({4}, {3, 2, 2, 2}),
      /*tvalues=*/DT_STRING,
      /*node_name=*/kNodeName);
}

SparseTensorSliceDatasetParams FiveDimsSparseTensorSliceDatasetParams() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSsparse_tensor_slice_dataset_op_testDTcc mht_7(mht_7_v, 291, "", "./tensorflow/core/kernels/data/sparse_tensor_slice_dataset_op_test.cc", "FiveDimsSparseTensorSliceDatasetParams");

  return SparseTensorSliceDatasetParams(
      /*indices=*/CreateTensor<int64_t>({2, 5}, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}),
      /*values=*/CreateTensor<int32>({2}, {888, 999}),
      /*dense_shape=*/CreateTensor<int64_t>({5}, {3, 2, 2, 2, 2}),
      /*tvalues=*/DT_INT32,
      /*node_name=*/kNodeName);
}

template <typename T>
struct GetNextTestCase {
  T dataset_params;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<GetNextTestCase<SparseTensorSliceDatasetParams>>
GetNextTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 1}, {0}),
             /*values*/ CreateTensor<int32>({1}, {888}),
             /*dense_shape*/ CreateTensor<int64_t>({1}, {2})},
            {/*indices*/ CreateTensor<int64_t>({1, 1}, {1}),
             /*values*/ CreateTensor<int32>({1}, {999}),
             /*dense_shape*/ CreateTensor<int64_t>({1}, {2})}}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 2}, {0, 0}),
             /*values*/ CreateTensor<double>({1}, {888.0}),
             /*dense_shape*/ CreateTensor<int64_t>({2}, {2, 2})},
            {{/*indices*/ CreateTensor<int64_t>({1, 2}, {1, 1})},
             {/*values*/ CreateTensor<double>({1}, {999.0})},
             {/*dense_shape*/ CreateTensor<int64_t>({2}, {2, 2})}}}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 3}, {0, 0, 0}),
             /*values*/ CreateTensor<tstring>({1}, {"a"}),
             /*dense_shape*/
             CreateTensor<int64_t>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64_t>({1, 3}, {1, 1, 1}),
             /*values*/ CreateTensor<tstring>({1}, {"b"}),
             /*dense_shape*/
             CreateTensor<int64_t>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64_t>({0, 3}, {}),
             /*values*/ CreateTensor<tstring>({0}, {}),
             /*dense_shape*/
             CreateTensor<int64_t>({3}, {2, 2, 2})}}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_outputs=*/{
               {/*indices*/ CreateTensor<int64_t>({1, 4}, {0, 0, 0, 0}),
                /*values*/ CreateTensor<int32>({1}, {888}),
                /*dense_shape*/
                CreateTensor<int64_t>({4}, {2, 2, 2, 2})},
               {/*indices*/ CreateTensor<int64_t>({1, 4}, {1, 1, 1, 1}),
                /*values*/ CreateTensor<int32>({1}, {999}),
                /*dense_shape*/
                CreateTensor<int64_t>({4}, {2, 2, 2, 2})},
               {/*indices*/ CreateTensor<int64_t>({0, 4}, {}),
                /*values*/ CreateTensor<int32>({0}, {}),
                /*dense_shape*/
                CreateTensor<int64_t>({4}, {2, 2, 2, 2})}}}};
}

class ParameterizedGetNextTest
    : public SparseTensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<
          GetNextTestCase<SparseTensorSliceDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
    if (!end_of_sequence) {
      TF_EXPECT_OK(ExpectEqual(out_tensors[0], expected_outputs_it->at(0)));
      TF_EXPECT_OK(ExpectEqual(out_tensors[1], expected_outputs_it->at(1)));
      TF_EXPECT_OK(ExpectEqual(out_tensors[2], expected_outputs_it->at(2)));
      expected_outputs_it++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(SparseTensorSliceDatasetOpTest,
                        ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

TEST_F(SparseTensorSliceDatasetOpTest, DatasetTypeString) {
  auto dataset_params = TwoDimsSparseTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(name_utils::OpName(kDatasetType)));
}

TEST_F(SparseTensorSliceDatasetOpTest, DatasetNodeName) {
  auto dataset_params = TwoDimsSparseTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

std::vector<DatasetOutputDtypesTestCase<SparseTensorSliceDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_DOUBLE, DT_INT64}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING, DT_INT64}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SparseTensorSliceDatasetOpTest,
                             SparseTensorSliceDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<SparseTensorSliceDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 1}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 2}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({2})}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 3}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({3})}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 4}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({4})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SparseTensorSliceDatasetOpTest,
                             SparseTensorSliceDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<SparseTensorSliceDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(SparseTensorSliceDatasetOpTest,
                           SparseTensorSliceDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<SparseTensorSliceDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_DOUBLE, DT_INT64}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_STRING, DT_INT64}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT32, DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(SparseTensorSliceDatasetOpTest,
                              SparseTensorSliceDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<SparseTensorSliceDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 1}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({1})}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 2}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({2})}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 3}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({3})}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*expected_output_shapes=*/{PartialTensorShape({1, 4}),
                                       PartialTensorShape({1}),
                                       PartialTensorShape({4})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(SparseTensorSliceDatasetOpTest,
                              SparseTensorSliceDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(SparseTensorSliceDatasetOpTest, IteratorPrefix) {
  auto dataset_params = TwoDimsSparseTensorSliceDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      kDatasetType, dataset_params.iterator_prefix())));
}

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<IteratorSaveAndRestoreTestCase<SparseTensorSliceDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/TwoDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 1}, {0}),
             /*values*/ CreateTensor<int32>({1}, {888}),
             /*dense_shape*/ CreateTensor<int64_t>({1}, {2})},
            {/*indices*/ CreateTensor<int64_t>({1, 1}, {1}),
             /*values*/ CreateTensor<int32>({1}, {999}),
             /*dense_shape*/ CreateTensor<int64_t>({1}, {2})}}},
          {/*dataset_params=*/ThreeDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 2}, {0, 0}),
             /*values*/ CreateTensor<double>({1}, {888.0}),
             /*dense_shape*/ CreateTensor<int64_t>({2}, {2, 2})},
            {{/*indices*/ CreateTensor<int64_t>({1, 2}, {1, 1})},
             {/*values*/ CreateTensor<double>({1}, {999.0})},
             {/*dense_shape*/ CreateTensor<int64_t>({2}, {2, 2})}}}},
          {/*dataset_params=*/FourDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 3},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 3}, {0, 0, 0}),
             /*values*/ CreateTensor<tstring>({1}, {"a"}),
             /*dense_shape*/
             CreateTensor<int64_t>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64_t>({1, 3}, {1, 1, 1}),
             /*values*/ CreateTensor<tstring>({1}, {"b"}),
             /*dense_shape*/
             CreateTensor<int64_t>({3}, {2, 2, 2})},
            {/*indices*/ CreateTensor<int64_t>({0, 3}, {}),
             /*values*/ CreateTensor<tstring>({0}, {}),
             /*dense_shape*/
             CreateTensor<int64_t>({3}, {2, 2, 2})}}},
          {/*dataset_params=*/FiveDimsSparseTensorSliceDatasetParams(),
           /*breakpoints=*/{0, 1, 2},
           /*expected_outputs=*/
           {{/*indices*/ CreateTensor<int64_t>({1, 4}, {0, 0, 0, 0}),
             /*values*/ CreateTensor<int32>({1}, {888}),
             /*dense_shape*/
             CreateTensor<int64_t>({4}, {2, 2, 2, 2})},
            {/*indices*/ CreateTensor<int64_t>({1, 4}, {1, 1, 1, 1}),
             /*values*/ CreateTensor<int32>({1}, {999}),
             /*dense_shape*/
             CreateTensor<int64_t>({4}, {2, 2, 2, 2})},
            {/*indices*/ CreateTensor<int64_t>({0, 4}, {}),
             /*values*/ CreateTensor<int32>({0}, {}),
             /*dense_shape*/
             CreateTensor<int64_t>({4}, {2, 2, 2, 2})}}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public SparseTensorSliceDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<SparseTensorSliceDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  int cur_iteration = 0;
  bool end_of_sequence = false;
  int64_t num_slices = dataset_->Cardinality();
  std::vector<Tensor> out_tensors;

  for (int breakpoint : test_case.breakpoints) {
    while (cur_iteration < breakpoint) {
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      cur_iteration++;
    }

    if (breakpoint == 0) {
      EXPECT_FALSE(end_of_sequence);
    } else if (breakpoint <= num_slices) {
      for (int i = 0; i < out_tensors.size(); ++i) {
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[0], test_case.expected_outputs[cur_iteration - 1][0]));
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[1], test_case.expected_outputs[cur_iteration - 1][1]));
        TF_EXPECT_OK(ExpectEqual(
            out_tensors[2], test_case.expected_outputs[cur_iteration - 1][2]));
      }
    } else {
      EXPECT_TRUE(end_of_sequence);
    }

    VariantTensorDataWriter writer;
    TF_ASSERT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));
  }
}

INSTANTIATE_TEST_CASE_P(SparseTensorSliceDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
