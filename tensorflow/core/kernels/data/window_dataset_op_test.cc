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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/window_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "window_dataset";

class WindowDatasetParams : public DatasetParams {
 public:
  template <typename T>
  WindowDatasetParams(T input_dataset_params, int64_t size, int64_t shift,
                      int64_t stride, bool drop_remainder,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        size_(size),
        shift_(shift),
        stride_(stride),
        drop_remainder_(drop_remainder) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {size_}),
            CreateTensor<int64_t>(TensorShape({}), {shift_}),
            CreateTensor<int64_t>(TensorShape({}), {stride_}),
            CreateTensor<bool>(TensorShape({}), {drop_remainder_})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(WindowDatasetOp::kInputDataset);
    input_names->emplace_back(WindowDatasetOp::kSize);
    input_names->emplace_back(WindowDatasetOp::kShift);
    input_names->emplace_back(WindowDatasetOp::kStride);
    input_names->emplace_back(WindowDatasetOp::kDropRemainder);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_2(mht_2_v, 240, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "dataset_type");
 return WindowDatasetOp::kDatasetType; }

 private:
  int64_t size_;
  int64_t shift_;
  int64_t stride_;
  bool drop_remainder_;
};

class WindowDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: size=2, shift=2, stride=1, drop_remainder=false.
WindowDatasetParams WindowDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams1");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/1,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 2: size=2, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_5(mht_5_v, 281, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams2");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 3: size=8, shift=3, stride=1, drop_remainder=false.
WindowDatasetParams WindowDatasetParams3() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_6(mht_6_v, 296, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams3");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/8,
                             /*shift=*/3,
                             /*stride=*/1,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 4: size=8, shift=3, stride=1, drop_remainder=true.
WindowDatasetParams WindowDatasetParams4() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_7(mht_7_v, 311, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams4");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/8,
                             /*shift=*/3,
                             /*stride=*/1,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 5: size=2, shift=8, stride=1, drop_remainder=false.
WindowDatasetParams WindowDatasetParams5() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_8(mht_8_v, 326, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams5");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/8,
                             /*stride=*/1,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 6: size=2, shift=8, stride=1, drop_remainder=true.
WindowDatasetParams WindowDatasetParams6() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_9(mht_9_v, 341, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams6");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/8,
                             /*stride=*/1,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 7: size=2, shift=2, stride=8, drop_remainder=false.
WindowDatasetParams WindowDatasetParams7() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_10(mht_10_v, 356, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams7");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/8,
                             /*drop_remainder=*/false,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 8: size=2, shift=2, stride=8, drop_remainder=true.
WindowDatasetParams WindowDatasetParams8() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_11(mht_11_v, 371, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams8");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/8,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 9: size=4, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParams9() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_12(mht_12_v, 386, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams9");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/4,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 10: size=5, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParams10() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_13(mht_13_v, 401, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParams10");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/5,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 11: size=0, shift=2, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParamsWithInvalidWindowSize() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_14(mht_14_v, 416, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParamsWithInvalidWindowSize");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/0,
                             /*shift=*/2,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 12: size=2, shift=0, stride=2, drop_remainder=true.
WindowDatasetParams WindowDatasetParamswithInvalidWindowShift() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_15(mht_15_v, 431, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParamswithInvalidWindowShift");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/0,
                             /*stride=*/2,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

// Test case 13: size=2, shift=2, stride=0, drop_remainder=true.
WindowDatasetParams WindowDatasetParamsWithInvalidWindowStride() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_op_testDTcc mht_16(mht_16_v, 446, "", "./tensorflow/core/kernels/data/window_dataset_op_test.cc", "WindowDatasetParamsWithInvalidWindowStride");

  return WindowDatasetParams(RangeDatasetParams(0, 7, 1),
                             /*size=*/2,
                             /*shift=*/2,
                             /*stride=*/0,
                             /*drop_remainder=*/true,
                             /*output_dtypes=*/{DT_VARIANT},
                             /*output_shapes=*/{PartialTensorShape({})},
                             /*node_name=*/kNodeName);
}

template <typename T>
struct GetNextTestCase {
  T dataset_params;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<GetNextTestCase<WindowDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/WindowDatasetParams1(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {1}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {3}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {5}}),
            CreateTensors<int64_t>(TensorShape{}, {{6}})}},
          {/*dataset_params=*/WindowDatasetParams2(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {2}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {4}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}),
                                   {{0}, {1}, {2}, {3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams4(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams5(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams6(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams7(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}}),
            CreateTensors<int64_t>(TensorShape({}), {{2}}),
            CreateTensors<int64_t>(TensorShape({}), {{4}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams8(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams9(),
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams10(),
           /*expected_outputs=*/{}}};
}

class ParameterizedGetNextTest : public WindowDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<WindowDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  while (!end_of_sequence) {
    // Owns the window_datasets, which are stored as the variant tensors in the
    // vector.
    std::vector<Tensor> out_tensors;
    TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                    &end_of_sequence));
    if (!end_of_sequence) {
      for (const auto& window_dataset_tensor : out_tensors) {
        // Not owned.
        DatasetBase* window_dataset;
        TF_ASSERT_OK(GetDatasetFromVariantTensor(window_dataset_tensor,
                                                 &window_dataset));
        std::unique_ptr<IteratorBase> window_dataset_iterator;
        TF_ASSERT_OK(window_dataset->MakeIterator(
            iterator_ctx_.get(), /*parent=*/nullptr,
            test_case.dataset_params.iterator_prefix(),
            &window_dataset_iterator));
        bool end_of_window_dataset = false;
        std::vector<Tensor> window_elements;
        // Fetches all the elements in window_dataset.
        while (!end_of_window_dataset) {
          std::vector<Tensor> next_element;
          TF_EXPECT_OK(window_dataset_iterator->GetNext(
              iterator_ctx_.get(), &next_element, &end_of_window_dataset));
          window_elements.insert(window_elements.end(), next_element.begin(),
                                 next_element.end());
        }
        EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(window_elements, *expected_outputs_it, false));
        expected_outputs_it++;
      }
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(WindowDatasetOpTest, ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

TEST_F(WindowDatasetOpTest, DatasetTypeString) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(WindowDatasetOp::kDatasetType)));
}

TEST_F(WindowDatasetOpTest, DatasetNodeName) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(WindowDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(WindowDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<WindowDatasetParams>>
DatasetCardinalityTestCases() {
  return {{/*dataset_params=*/WindowDatasetParams1(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/WindowDatasetParams2(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/WindowDatasetParams3(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/WindowDatasetParams4(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/WindowDatasetParams5(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/WindowDatasetParams6(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/WindowDatasetParams7(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/WindowDatasetParams8(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/WindowDatasetParams9(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/WindowDatasetParams10(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(WindowDatasetOpTest, WindowDatasetParams,
                           DatasetCardinalityTestCases())

TEST_F(WindowDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes(dataset_params.output_dtypes()));
}

TEST_F(WindowDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(WindowDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = WindowDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      WindowDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<std::vector<Tensor>> expected_outputs;
};

std::vector<IteratorSaveAndRestoreTestCase<WindowDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/WindowDatasetParams1(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {1}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {3}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {5}}),
            CreateTensors<int64_t>(TensorShape{}, {{6}})}},
          {/*dataset_params=*/WindowDatasetParams2(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape{}, {{0}, {2}}),
            CreateTensors<int64_t>(TensorShape{}, {{2}, {4}}),
            CreateTensors<int64_t>(TensorShape{}, {{4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams3(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}),
                                   {{0}, {1}, {2}, {3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{3}, {4}, {5}, {6}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams4(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams5(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams6(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}})}},
          {/*dataset_params=*/WindowDatasetParams7(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}}),
            CreateTensors<int64_t>(TensorShape({}), {{2}}),
            CreateTensors<int64_t>(TensorShape({}), {{4}}),
            CreateTensors<int64_t>(TensorShape({}), {{6}})}},
          {/*dataset_params=*/WindowDatasetParams8(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/WindowDatasetParams9(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/
           {CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}})}},
          {/*dataset_params=*/WindowDatasetParams10(),
           /*breakpoints=*/{0, 1, 9},
           /*expected_outputs=*/{}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public WindowDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<WindowDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  int cur_iteration = 0;
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));
    while (cur_iteration <= breakpoint) {
      while (!end_of_sequence) {
        // Owns the datasets, which are stored as the variant tensors in the
        // vector.
        std::vector<Tensor> out_tensors;
        TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                        &end_of_sequence));
        if (!end_of_sequence) {
          for (const auto& window_dataset_tensor : out_tensors) {
            // Not owned.
            DatasetBase* window_dataset;
            TF_ASSERT_OK(GetDatasetFromVariantTensor(window_dataset_tensor,
                                                     &window_dataset));
            std::unique_ptr<IteratorBase> window_dataset_iterator;
            TF_ASSERT_OK(window_dataset->MakeIterator(
                iterator_ctx_.get(), /*parent=*/nullptr,
                test_case.dataset_params.iterator_prefix(),
                &window_dataset_iterator));
            bool end_of_window_dataset = false;
            std::vector<Tensor> window_elements;
            while (!end_of_window_dataset) {
              std::vector<Tensor> next_element;
              TF_EXPECT_OK(window_dataset_iterator->GetNext(
                  iterator_ctx_.get(), &next_element, &end_of_window_dataset));
              window_elements.insert(window_elements.end(),
                                     next_element.begin(), next_element.end());
            }
            EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
            TF_EXPECT_OK(
                ExpectEqual(window_elements, *expected_outputs_it, false));
            expected_outputs_it++;
          }
        }
      }
      cur_iteration++;
    }
  }
  EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
}

INSTANTIATE_TEST_CASE_P(WindowDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(WindowDatasetOpTest, InvalidArguments) {
  std::vector<WindowDatasetParams> dataset_params_vec(
      {WindowDatasetParamsWithInvalidWindowSize(),
       WindowDatasetParamswithInvalidWindowShift(),
       WindowDatasetParamsWithInvalidWindowStride()});
  for (const auto& dataset_params : dataset_params_vec) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
