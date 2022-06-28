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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/padded_batch_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "padded_batch_dataset";
constexpr int kOpVersion = 2;

class PaddedBatchDatasetOpTest : public DatasetOpsTestBase {};

class PaddedBatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  PaddedBatchDatasetParams(T input_dataset_params, int64_t batch_size,
                           std::vector<Tensor> padded_shapes,
                           std::vector<Tensor> padded_values,
                           bool drop_remainder, bool parallel_copy,
                           DataTypeVector output_dtypes,
                           std::vector<PartialTensorShape> output_shapes,
                           int num_padded_shapes, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        batch_size_(batch_size),
        padded_shapes_(std::move(padded_shapes)),
        padded_values_(std::move(padded_values)),
        drop_remainder_(drop_remainder),
        parallel_copy_(parallel_copy),
        num_padded_shapes_(num_padded_shapes) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    op_version_ = kOpVersion;
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {batch_size_}));
    for (auto& padded_shape : padded_shapes_) {
      input_tensors.emplace_back(padded_shape);
    }
    for (auto& padded_value : padded_values_) {
      input_tensors.emplace_back(padded_value);
    }
    input_tensors.emplace_back(
        CreateTensor<bool>(TensorShape({}), {drop_remainder_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "GetInputNames");

    *input_names = {PaddedBatchDatasetOp::kInputDataset,
                    PaddedBatchDatasetOp::kBatchSize};
    // Create the input names for the input padded_shapes.
    for (int i = 0; i < num_padded_shapes_; ++i) {
      input_names->emplace_back(
          strings::StrCat(PaddedBatchDatasetOp::kPaddedShapes, "_", i));
    }
    // Create the input names for the input padding_values.
    for (int j = 0; j < padded_values_.size(); ++j) {
      input_names->emplace_back(
          strings::StrCat(PaddedBatchDatasetOp::kPaddingValues, "_", j));
    }
    input_names->push_back(PaddedBatchDatasetOp::kDropRemainder);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_2(mht_2_v, 258, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{"parallel_copy", parallel_copy_},
                    {"Toutput_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"N", num_padded_shapes_},
                    {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "dataset_type");

    return PaddedBatchDatasetOp::kDatasetType;
  }

 private:
  int64_t batch_size_;
  std::vector<Tensor> padded_shapes_;
  std::vector<Tensor> padded_values_;
  bool drop_remainder_;
  bool parallel_copy_;
  int num_padded_shapes_;
};

// Test case 1: input elements with same shapes.
PaddedBatchDatasetParams PaddedBatchDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_4(mht_4_v, 287, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(
          TensorShape{7, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})},
      /*node_name=*/"tensor_slice");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/tensor_slice_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/true,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 2: input elements with different shapes.
PaddedBatchDatasetParams PaddedBatchDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_5(mht_5_v, 309, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams2");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/true,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 3: similar with the test case 2 but drop_remainder = false.
PaddedBatchDatasetParams PaddedBatchDatasetParams3() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_6(mht_6_v, 340, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams3");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({2, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 4: similar with the test case 3 but the input elements can be
// divided by the batch size evenly. As drop_remainder = false, the output
// shape is still {-1, 3} instead of {2, 3}.
PaddedBatchDatasetParams PaddedBatchDatasetParams4() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_7(mht_7_v, 373, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams4");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 1}, {{6, 7, 8}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, 3})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 5: similar with the test case 3 but padded_shapes = {-1}.
PaddedBatchDatasetParams PaddedBatchDatasetParams5() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_8(mht_8_v, 404, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams5");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {-1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 6: similar with the test case 5 but parallel_copy = true.
PaddedBatchDatasetParams PaddedBatchDatasetParams6() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_9(mht_9_v, 435, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams6");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{4, 1}, {{6, 7, 8, 9}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({-1})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {-1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 7: empty input elements.
PaddedBatchDatasetParams PaddedBatchDatasetParams7() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_10(mht_10_v, 466, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParams7");

  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/RangeDatasetParams(0, 0, 1),
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {-1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

// Test case 8: short padding shape.
PaddedBatchDatasetParams PaddedBatchDatasetParamsWithShortPaddingShape() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_11(mht_11_v, 484, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithShortPaddingShape");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {1})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams PaddedBatchDatasetParamsWithInvalidPaddingShape() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_12(mht_12_v, 515, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithInvalidPaddingShape");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{2}, {1, 2})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams PaddedBatchDatasetParamsWithInvalidBatchSize() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_13(mht_13_v, 546, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithInvalidBatchSize");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/-1,
      /*padded_shapes=*/{CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingShapesSize() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_14(mht_14_v, 578, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithInvalidPaddingShapesSize");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3}),
       CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/{CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/2,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingValuesSize() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_15(mht_15_v, 612, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithInvalidPaddingValuesSize");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/
      {CreateTensor<int64_t>(TensorShape{}, {1}),
       CreateTensor<int64_t>(TensorShape{}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/2,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingValuesDType() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_16(mht_16_v, 647, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithInvalidPaddingValuesDType");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/
      {CreateTensor<tstring>(TensorShape{}, {"a"})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

PaddedBatchDatasetParams
PaddedBatchDatasetParamsWithInvalidPaddingValuesShape() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSpadded_batch_dataset_op_testDTcc mht_17(mht_17_v, 681, "", "./tensorflow/core/kernels/data/padded_batch_dataset_op_test.cc", "PaddedBatchDatasetParamsWithInvalidPaddingValuesShape");

  auto tensor_slice_dataset_params_0 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{0, 1, 2, 3, 4, 5}}),
      /*node_name=*/"tensor_slice_0");
  auto tensor_slice_dataset_params_1 = TensorSliceDatasetParams(
      /*components=*/CreateTensors<int64_t>(TensorShape{3, 2},
                                            {{6, 7, 8, 9, 10, 11}}),
      /*node_name=*/"tensor_slice_1");
  auto concatenate_dataset_params =
      ConcatenateDatasetParams(std::move(tensor_slice_dataset_params_0),
                               std::move(tensor_slice_dataset_params_1),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({2})},
                               /*node_name=*/"concatenate");
  return PaddedBatchDatasetParams(
      /*input_dataset_params=*/concatenate_dataset_params,
      /*batch_size=*/2,
      /*padded_shapes=*/
      {CreateTensor<int64_t>(TensorShape{1}, {3})},
      /*padded_values=*/
      {CreateTensor<int64_t>(TensorShape{1}, {1})},
      /*drop_remainder=*/false,
      /*parallel_copy=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1, -1})},
      /*num_padded_shapes=*/1,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<PaddedBatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 7, 1}, {8, 9, 1, 10, 11, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 3}, {0, 1, 1, 2, 3, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {4, 5, 1, 6, 1, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {7, 1, 1, 8, 1, 1}),
            CreateTensor<int64_t>(TensorShape{1, 3}, {9, 1, 1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                         GetNextTestCases())

TEST_F(PaddedBatchDatasetOpTest, DatasetNodeName) {
  auto dataset_params = PaddedBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(PaddedBatchDatasetOpTest, DatasetTypeString) {
  auto dataset_params = PaddedBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(PaddedBatchDatasetOp::kDatasetType, params)));
}

std::vector<DatasetOutputDtypesTestCase<PaddedBatchDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<PaddedBatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<PaddedBatchDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(PaddedBatchDatasetOpTest, PaddedBatchDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<PaddedBatchDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(PaddedBatchDatasetOpTest,
                              PaddedBatchDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<PaddedBatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({2, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, 3})}},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({-1, -1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(PaddedBatchDatasetOpTest,
                              PaddedBatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(PaddedBatchDatasetOpTest, IteratorPrefix) {
  auto dataset_params = PaddedBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(PaddedBatchDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix(), params)));
}

std::vector<IteratorSaveAndRestoreTestCase<PaddedBatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PaddedBatchDatasetParams1(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 7, 1}, {8, 9, 1, 10, 11, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams2(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams3(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 3}, {0, 1, 1, 2, 3, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {4, 5, 1, 6, 1, 1}),
            CreateTensor<int64_t>(TensorShape{2, 3}, {7, 1, 1, 8, 1, 1}),
            CreateTensor<int64_t>(TensorShape{1, 3}, {9, 1, 1})}},
          {/*dataset_params=*/PaddedBatchDatasetParams4(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{2, 3},
               {{0, 1, 1, 2, 3, 1}, {4, 5, 1, 6, 1, 1}, {7, 1, 1, 8, 1, 1}})},
          {/*dataset_params=*/PaddedBatchDatasetParams5(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams6(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           {CreateTensor<int64_t>(TensorShape{2, 2}, {0, 1, 2, 3}),
            CreateTensor<int64_t>(TensorShape{2, 2}, {4, 5, 6, 1}),
            CreateTensor<int64_t>(TensorShape{2, 1}, {7, 8}),
            CreateTensor<int64_t>(TensorShape{1, 1}, {9})}},
          {/*dataset_params=*/PaddedBatchDatasetParams7(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(PaddedBatchDatasetOpTest,
                                 PaddedBatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(PaddedBatchDatasetOpTest, ShortPadding) {
  auto dataset_params = PaddedBatchDatasetParamsWithShortPaddingShape();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::DATA_LOSS);
}

TEST_F(PaddedBatchDatasetOpTest, InvalidPaddedShapes) {
  auto dataset_params = PaddedBatchDatasetParamsWithInvalidPaddingShape();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

class ParameterizedInvalidArgumentTest
    : public PaddedBatchDatasetOpTest,
      public ::testing::WithParamInterface<PaddedBatchDatasetParams> {};

TEST_P(ParameterizedInvalidArgumentTest, InvalidPredicateFunc) {
  auto dataset_params = GetParam();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

INSTANTIATE_TEST_SUITE_P(
    PaddedBatchDatasetOpTest, ParameterizedInvalidArgumentTest,
    ::testing::ValuesIn(
        {PaddedBatchDatasetParamsWithInvalidBatchSize(),
         PaddedBatchDatasetParamsWithInvalidPaddingShapesSize(),
         PaddedBatchDatasetParamsWithInvalidPaddingValuesSize(),
         PaddedBatchDatasetParamsWithInvalidPaddingValuesDType(),
         PaddedBatchDatasetParamsWithInvalidPaddingValuesShape()}));

}  // namespace
}  // namespace data
}  // namespace tensorflow
