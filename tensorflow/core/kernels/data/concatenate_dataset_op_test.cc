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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSconcatenate_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSconcatenate_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSconcatenate_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/concatenate_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "concatenate_dataset";

// Test case 1: same shape.
ConcatenateDatasetParams SameShapeConcatenateDatasetParams() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSconcatenate_dataset_op_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/data/concatenate_dataset_op_test.cc", "SameShapeConcatenateDatasetParams");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{2, 2},
                                            {{1, 2, 3, 4}, {5, 6, 7, 8}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(
          TensorShape{2, 2}, {{11, 12, 13, 14}, {15, 16, 17, 18}}),
      /*node_name=*/"tensor_slice_1");
  return ConcatenateDatasetParams(
      std::move(tensor_slice_dataset_params_0),
      std::move(tensor_slice_dataset_params_1),
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2}), PartialTensorShape({2})},
      /*node_name=*/kNodeName);
}

// Test case 2: different shape.
ConcatenateDatasetParams DifferentShapeConcatenateDatasetParams() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSconcatenate_dataset_op_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/data/concatenate_dataset_op_test.cc", "DifferentShapeConcatenateDatasetParams");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{2, 3}, {1, 2, 3, 4, 5, 6}),
       CreateTensor<int64_t>(TensorShape{2, 2}, {7, 8, 9, 10})},
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{2, 2}, {11, 12, 13, 14}),
       CreateTensor<int64_t>(TensorShape{2, 1}, {15, 16})},
      /*node_name=*/"tensor_slice_1");
  return ConcatenateDatasetParams(
      std::move(tensor_slice_dataset_params_0),
      std::move(tensor_slice_dataset_params_1),
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1}), PartialTensorShape({-1})},
      /*node_name=*/kNodeName);
}

// Test case 3: different dtypes
ConcatenateDatasetParams DifferentDtypeConcatenateDatasetParams() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSconcatenate_dataset_op_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/kernels/data/concatenate_dataset_op_test.cc", "DifferentDtypeConcatenateDatasetParams");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{2, 2}, {{1, 2, 3, 4}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/
      CreateTensors<double>(TensorShape{2, 2}, {{1.0, 2.0, 3.0, 4.0}}),
      /*node_name=*/"tensor_slice_1");
  return ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                                  std::move(tensor_slice_dataset_params_1),
                                  /*output_dtypes=*/{DT_INT64},
                                  /*output_shapes=*/{PartialTensorShape({2})},
                                  /*node_name=*/kNodeName);
}

class ConcatenateDatasetOpTest : public DatasetOpsTestBase {};

std::vector<GetNextTestCase<ConcatenateDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({2}), {{1, 2},
                                                     {5, 6},
                                                     {3, 4},
                                                     {7, 8},
                                                     {11, 12},
                                                     {15, 16},
                                                     {13, 14},
                                                     {17, 18}})},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{3}, {1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{3}, {4, 5, 6}),
            CreateTensor<int64_t>(TensorShape{2}, {9, 10}),
            CreateTensor<int64_t>(TensorShape{2}, {11, 12}),
            CreateTensor<int64_t>(TensorShape{1}, {15}),
            CreateTensor<int64_t>(TensorShape{2}, {13, 14}),
            CreateTensor<int64_t>(TensorShape{1}, {16})}}};
}

ITERATOR_GET_NEXT_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                         GetNextTestCases())

TEST_F(ConcatenateDatasetOpTest, DifferentDtypes) {
  auto dataset_params = DifferentDtypeConcatenateDatasetParams();

  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(ConcatenateDatasetOpTest, DatasetNodeName) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ConcatenateDatasetOpTest, DatasetTypeString) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ConcatenateDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<ConcatenateDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           SameShapeConcatenateDatasetParams().output_dtypes()},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           DifferentShapeConcatenateDatasetParams().output_dtypes()}};
}

DATASET_OUTPUT_DTYPES_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<ConcatenateDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_shapes*/
           SameShapeConcatenateDatasetParams().output_shapes()},
          {/*dataset_params=*/
           DifferentShapeConcatenateDatasetParams(),
           /*expected_output_shapes*/
           DifferentShapeConcatenateDatasetParams().output_shapes()}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ConcatenateDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_cardinality=*/4}};
}

DATASET_CARDINALITY_TEST_P(ConcatenateDatasetOpTest, ConcatenateDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<ConcatenateDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           SameShapeConcatenateDatasetParams().output_dtypes()},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_output_dtypes=*/
           DifferentShapeConcatenateDatasetParams().output_dtypes()}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(ConcatenateDatasetOpTest,
                              ConcatenateDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<ConcatenateDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*expected_output_shapes=*/
           SameShapeConcatenateDatasetParams().output_shapes()},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*expected_output_shapes=*/
           DifferentShapeConcatenateDatasetParams().output_shapes()}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ConcatenateDatasetOpTest,
                              ConcatenateDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ConcatenateDatasetOpTest, IteratorPrefix) {
  auto dataset_params = SameShapeConcatenateDatasetParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ConcatenateDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ConcatenateDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/SameShapeConcatenateDatasetParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({2}), {{1, 2},
                                                     {5, 6},
                                                     {3, 4},
                                                     {7, 8},
                                                     {11, 12},
                                                     {15, 16},
                                                     {13, 14},
                                                     {17, 18}})},
          {/*dataset_params=*/DifferentShapeConcatenateDatasetParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{3}, {1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{3}, {4, 5, 6}),
            CreateTensor<int64_t>(TensorShape{2}, {9, 10}),
            CreateTensor<int64_t>(TensorShape{2}, {11, 12}),
            CreateTensor<int64_t>(TensorShape{1}, {15}),
            CreateTensor<int64_t>(TensorShape{2}, {13, 14}),
            CreateTensor<int64_t>(TensorShape{1}, {16})}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ConcatenateDatasetOpTest,
                                 ConcatenateDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
