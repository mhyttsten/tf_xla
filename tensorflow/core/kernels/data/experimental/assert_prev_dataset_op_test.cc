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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "assert_prev_dataset";

// Returns a stringified `NameAttrList`, the input to `AssertPrevDatasetOp`.
std::string GetTransformation(
    absl::string_view name,
    std::initializer_list<std::pair<std::string, bool>> attrs = {}) {
  NameAttrList message;
  message.set_name(absl::StrCat(name, "Dataset"));
  for (const auto& attr : attrs) {
    AttrValue value;
    value.set_b(attr.second);
    message.mutable_attr()->insert({attr.first, value});
  }
  std::string output;
  protobuf::TextFormat::PrintToString(message, &output);
  return output;
}

class AssertPrevDatasetParams : public DatasetParams {
 public:
  template <typename T>
  AssertPrevDatasetParams(T input_dataset_params,
                          const std::vector<tstring>& transformations,
                          DataTypeVector output_dtypes,
                          std::vector<PartialTensorShape> output_shapes,
                          string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        transformations_(transformations) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "AssertPrevDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    int num_transformations = transformations_.size();
    return {CreateTensor<tstring>(TensorShape({num_transformations}),
                                  transformations_)};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_1(mht_1_v, 243, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "GetInputNames");

    input_names->reserve(input_dataset_params_.size() + 1);
    input_names->emplace_back(AssertPrevDatasetOp::kInputDataset);
    input_names->emplace_back(AssertPrevDatasetOp::kTransformations);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{AssertPrevDatasetOp::kOutputShapes, output_shapes_},
                    {AssertPrevDatasetOp::kOutputTypes, output_dtypes_}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "dataset_type");

    return AssertPrevDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> transformations_;
};

class AssertPrevDatasetOpTest : public DatasetOpsTestBase {};

AssertPrevDatasetParams AssertPrevDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_4(mht_4_v, 275, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "AssertPrevDatasetParams1");

  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType)},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams AssertPrevDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_5(mht_5_v, 294, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "AssertPrevDatasetParams2");

  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType),
       GetTransformation(RangeDatasetOp::kDatasetType)},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams AssertPrevDatasetParams2WithAttrs() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_6(mht_6_v, 314, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "AssertPrevDatasetParams2WithAttrs");

  TakeDatasetParams take_dataset_params = TakeDatasetParams(
      TensorSliceDatasetParams(
          /*components=*/
          {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                 {0, 1, 2, 3, 4, 5, 6, 7, 8})},
          /*node_name=*/"tensor_slice_dataset"),
      /*count=*/3,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType),
       GetTransformation(TensorSliceDatasetOp::kDatasetType,
                         {{"is_files", false}})},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams InvalidAssertPrevDatasetParams() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_7(mht_7_v, 339, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "InvalidAssertPrevDatasetParams");

  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/{GetTransformation("Whoops")},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

AssertPrevDatasetParams ShortAssertPrevDatasetParams() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_op_testDTcc mht_8(mht_8_v, 357, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op_test.cc", "ShortAssertPrevDatasetParams");

  TakeDatasetParams take_dataset_params =
      TakeDatasetParams(RangeDatasetParams(0, 10, 1),
                        /*count=*/3,
                        /*output_dtypes=*/{DT_INT64},
                        /*output_shapes=*/{PartialTensorShape({})},
                        /*node_name=*/"take_dataset");
  return AssertPrevDatasetParams(
      std::move(take_dataset_params),
      /*transformations=*/
      {GetTransformation(TakeDatasetOp::kDatasetType),
       GetTransformation(RangeDatasetOp::kDatasetType),
       GetTransformation("Whoops")},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<AssertPrevDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/AssertPrevDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/AssertPrevDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})}};
}

ITERATOR_GET_NEXT_TEST_P(AssertPrevDatasetOpTest, AssertPrevDatasetParams,
                         GetNextTestCases())

TEST_F(AssertPrevDatasetOpTest, DatasetNodeName) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(AssertPrevDatasetOpTest, DatasetAttrs) {
  auto dataset_params = AssertPrevDatasetParams2WithAttrs();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(AssertPrevDatasetOpTest, DatasetTypeString) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(AssertPrevDatasetOp::kDatasetType)));
}

TEST_F(AssertPrevDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(AssertPrevDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(AssertPrevDatasetOpTest, Cardinality) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(/*expected_cardinality=*/3));
}

TEST_F(AssertPrevDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(AssertPrevDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(AssertPrevDatasetOpTest, IteratorPrefix) {
  auto dataset_params = AssertPrevDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      AssertPrevDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<AssertPrevDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/AssertPrevDatasetParams1(),
           /*breakpoints*/ {0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/AssertPrevDatasetParams2(),
           /*breakpoints*/ {0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(AssertPrevDatasetOpTest,
                                 AssertPrevDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(AssertPrevDatasetOpTest, InvalidArguments) {
  auto dataset_params = InvalidAssertPrevDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(AssertPrevDatasetOpTest, ShortAssertPrev) {
  auto dataset_params = ShortAssertPrevDatasetParams();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
