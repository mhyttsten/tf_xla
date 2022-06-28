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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/filter_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "filter_dataset";

class FilterDatasetParams : public DatasetParams {
 public:
  template <typename T>
  FilterDatasetParams(T input_dataset_params,
                      std::vector<Tensor> other_arguments,
                      FunctionDefHelper::AttrValueWrapper pred_func,
                      std::vector<FunctionDef> func_lib,
                      DataTypeVector type_arguments,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        pred_func_(std::move(pred_func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "FilterDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return other_arguments_;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->reserve(input_dataset_params_.size() +
                         other_arguments_.size());
    input_names->emplace_back(FilterDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(FilterDatasetOp::kOtherArguments, "_", i));
    }

    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{"predicate", pred_func_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"metadata", ""}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "dataset_type");
 return FilterDatasetOp::kDatasetType; }

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper pred_func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class FilterDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: norm case.
FilterDatasetParams FilterDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_4(mht_4_v, 267, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "FilterDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9, 1}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// Test case 2: the input dataset has no outputs.
FilterDatasetParams FilterDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_5(mht_5_v, 287, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "FilterDatasetParams2");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{0}, {})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib*/ {test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

// Test case 3: the filter function returns two outputs.
FilterDatasetParams InvalidPredFuncFilterDatasetParams1() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_6(mht_6_v, 307, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "InvalidPredFuncFilterDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("GetUnique",
                                     {{"T", DT_INT64}, {"out_idx", DT_INT32}}),
      /*func_lib*/ {test::function::Unique()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})},
      /*node_name=*/kNodeName);
}

// Test case 4: the filter function returns a 1-D bool tensor.
FilterDatasetParams InvalidPredFuncFilterDatasetParams2() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_7(mht_7_v, 329, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "InvalidPredFuncFilterDatasetParams2");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::IsZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})},
      /*node_name=*/kNodeName);
}

// Test case 5: the filter function returns a scalar int64 tensor.
FilterDatasetParams InvalidPredFuncFilterDatasetParams3() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_op_testDTcc mht_8(mht_8_v, 351, "", "./tensorflow/core/kernels/data/filter_dataset_op_test.cc", "InvalidPredFuncFilterDatasetParams3");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{9}, {0, 0, 0, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return FilterDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*pred_func=*/
      FunctionDefHelper::FunctionRef("NonZero", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::NonZero()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<FilterDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                         GetNextTestCases())

TEST_F(FilterDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(FilterDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(FilterDatasetOp::kDatasetType)));
}

TEST_F(FilterDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<FilterDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<FilterDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                           CardinalityTestCases())

TEST_F(FilterDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<FilterDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(FilterDatasetOpTest, IteratorPrefix) {
  auto dataset_params = FilterDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      FilterDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<FilterDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/FilterDatasetParams1(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}, {0}, {0}})},
          {/*dataset_params=*/FilterDatasetParams2(),
           /*breakpoints=*/{0, 2, 6},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(FilterDatasetOpTest, FilterDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

class ParameterizedInvalidPredicateFuncTest
    : public FilterDatasetOpTest,
      public ::testing::WithParamInterface<FilterDatasetParams> {};

TEST_P(ParameterizedInvalidPredicateFuncTest, InvalidPredicateFunc) {
  auto dataset_params = GetParam();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
  EXPECT_TRUE(out_tensors.empty());
}

INSTANTIATE_TEST_SUITE_P(
    FilterDatasetOpTest, ParameterizedInvalidPredicateFuncTest,
    ::testing::ValuesIn({InvalidPredFuncFilterDatasetParams1(),
                         InvalidPredFuncFilterDatasetParams2(),
                         InvalidPredFuncFilterDatasetParams3()}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
