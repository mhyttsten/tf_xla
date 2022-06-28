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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "directed_interleave_dataset";

class DirectedInterleaveDatasetParams : public DatasetParams {
 public:
  template <typename S, typename T>
  DirectedInterleaveDatasetParams(S selector_input_dataset_params,
                                  std::vector<T> input_dataset_params_vec,
                                  bool stop_on_empty_dataset,
                                  DataTypeVector output_dtypes,
                                  std::vector<PartialTensorShape> output_shapes,
                                  int num_input_datasets, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        stop_on_empty_dataset_(stop_on_empty_dataset),
        num_input_datasets_(input_dataset_params_vec.size()) {
    input_dataset_params_.push_back(
        absl::make_unique<S>(selector_input_dataset_params));
    for (auto input_dataset_params : input_dataset_params_vec) {
      input_dataset_params_.push_back(
          absl::make_unique<T>(input_dataset_params));
    }

    if (!input_dataset_params_vec.empty()) {
      iterator_prefix_ = name_utils::IteratorPrefix(
          input_dataset_params_vec[0].dataset_type(),
          input_dataset_params_vec[0].iterator_prefix());
    }
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(
        DirectedInterleaveDatasetOp::kSelectorInputDataset);
    for (int i = 0; i < num_input_datasets_; ++i) {
      input_names->emplace_back(absl::StrCat(
          DirectedInterleaveDatasetOp::kDataInputDatasets, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back(DirectedInterleaveDatasetOp::kOutputTypes,
                              output_dtypes_);
    attr_vector->emplace_back(DirectedInterleaveDatasetOp::kOutputShapes,
                              output_shapes_);
    attr_vector->emplace_back(DirectedInterleaveDatasetOp::kNumInputDatasets,
                              num_input_datasets_);
    attr_vector->emplace_back(DirectedInterleaveDatasetOp::kStopOnEmptyDataset,
                              stop_on_empty_dataset_);
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "dataset_type");

    return DirectedInterleaveDatasetOp::kDatasetType;
  }

 private:
  bool stop_on_empty_dataset_;
  int32 num_input_datasets_;
};

class DirectedInterleaveDatasetOpTest : public DatasetOpsTestBase {};

DirectedInterleaveDatasetParams AlternateInputsParams() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_3(mht_3_v, 265, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "AlternateInputsParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams SelectExhaustedInputParams() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "SelectExhaustedInputParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 2, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams OneInputDatasetParams() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_5(mht_5_v, 305, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "OneInputDatasetParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 0, 0, 0, 0, 0})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 6, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*num_input_datasets=*/1,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams ZeroInputDatasetParams() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_6(mht_6_v, 324, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "ZeroInputDatasetParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 0, 0, 0, 0, 0})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/std::vector<RangeDatasetParams>{},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*num_input_datasets=*/0,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams StopOnEmptyDatasetParams() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_7(mht_7_v, 342, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "StopOnEmptyDatasetParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 0, 0, 0, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 50, 1)},
      /*stop_on_empty_dataset=*/true,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams SkipEmptyDatasetParams() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_8(mht_8_v, 362, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "SkipEmptyDatasetParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 0, 0, 0, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 50, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams EmptyInputDatasetParams() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_9(mht_9_v, 382, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "EmptyInputDatasetParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 0, 0, 0, 0, 0})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 0, 1),
                                      RangeDatasetParams(10, 50, 1)},
      /*stop_on_empty_dataset=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*num_input_datasets=*/0,
      /*node_name=*/kNodeName);
}

// Test case: `num_input_datasets` is larger than the size of
// `input_dataset_params_vec`.
DirectedInterleaveDatasetParams LargeNumInputDatasetsParams() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_10(mht_10_v, 404, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "LargeNumInputDatasetsParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/5,
      /*node_name=*/kNodeName);
}

// Test case: `num_input_datasets` is smaller than the size of
// `input_dataset_params_vec`.
DirectedInterleaveDatasetParams SmallNumInputDatasetsParams() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_11(mht_11_v, 426, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "SmallNumInputDatasetsParams");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/1,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams InvalidSelectorOuputDataType() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_12(mht_12_v, 446, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "InvalidSelectorOuputDataType");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int32>(TensorShape{6}, {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams InvalidSelectorOuputShape() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_13(mht_13_v, 465, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "InvalidSelectorOuputShape");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6, 1},
                                            {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams InvalidSelectorValues() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_14(mht_14_v, 485, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "InvalidSelectorValues");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {2, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

DirectedInterleaveDatasetParams InvalidInputDatasetsDataType() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_op_testDTcc mht_15(mht_15_v, 505, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op_test.cc", "InvalidInputDatasetsDataType");

  auto selector_input_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{6},
                                            {0, 1, 0, 1, 0, 1})},
      /*node_name=*/"tensor_slice");
  return DirectedInterleaveDatasetParams(
      selector_input_dataset_params,
      /*input_dataset_params_vec=*/
      std::vector<RangeDatasetParams>{
          RangeDatasetParams(0, 3, 1, {DT_INT32}),
          RangeDatasetParams(10, 13, 1, {DT_INT64})},
      /*stop_on_empty_dataset=*/false,
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<DirectedInterleaveDatasetParams>>
GetNextTestCases() {
  return {{/*dataset_params=*/AlternateInputsParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(
               TensorShape({}), {{0}, {10}, {1}, {11}, {2}, {12}})}},
          {/*dataset_params=*/SelectExhaustedInputParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(
               TensorShape({}), {{0}, {10}, {1}, {11}, {12}})}},
          {/*dataset_params=*/OneInputDatasetParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(
               TensorShape({}), {{0}, {1}, {2}, {3}, {4}, {5}})}},
          {/*dataset_params=*/StopOnEmptyDatasetParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(TensorShape({}),
                                                        {{0}, {1}, {2}})}},
          {/*dataset_params=*/SkipEmptyDatasetParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(
               TensorShape({}), {{0}, {1}, {2}, {10}})}},
          {/*dataset_params=*/EmptyInputDatasetParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(TensorShape({}), {})}},
          {/*dataset_params=*/LargeNumInputDatasetsParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(
               TensorShape({}), {{0}, {10}, {1}, {11}, {2}, {12}})}},
          {/*dataset_params=*/SmallNumInputDatasetsParams(),
           /*expected_outputs=*/{CreateTensors<int64_t>(
               TensorShape({}), {{0}, {10}, {1}, {11}, {2}, {12}})}}};
}

ITERATOR_GET_NEXT_TEST_P(DirectedInterleaveDatasetOpTest,
                         DirectedInterleaveDatasetParams, GetNextTestCases())

TEST_F(DirectedInterleaveDatasetOpTest, DatasetNodeName) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(DirectedInterleaveDatasetOpTest, DatasetTypeString) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(DirectedInterleaveDatasetOp::kDatasetType)));
}

TEST_F(DirectedInterleaveDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(DirectedInterleaveDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(DirectedInterleaveDatasetOpTest, Cardinality) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(DirectedInterleaveDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(DirectedInterleaveDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(DirectedInterleaveDatasetOpTest, IteratorPrefix) {
  auto dataset_params = AlternateInputsParams();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(DirectedInterleaveDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<DirectedInterleaveDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/AlternateInputsParams(),
       /*breakpoints=*/{0, 5, 8},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape{}, {{0}, {10}, {1}, {11}, {2}, {12}}),
       /*compare_order=*/true},
      {/*dataset_params=*/SelectExhaustedInputParams(),
       /*breakpoints=*/{0, 4, 8},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape{}, {{0}, {10}, {1}, {11}, {12}}),
       /*compare_order=*/true},
      {/*dataset_params=*/OneInputDatasetParams(),
       /*breakpoints=*/{0, 5, 8},
       /*expected_outputs=*/
       {CreateTensors<int64_t>(TensorShape({}),
                               {{0}, {1}, {2}, {3}, {4}, {5}})}},
      {/*dataset_params=*/StopOnEmptyDatasetParams(),
       /*breakpoints=*/{0, 2, 4},
       /*expected_outputs=*/
       {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})}},
      {/*dataset_params=*/SkipEmptyDatasetParams(),
       /*breakpoints=*/{0, 2, 4},
       /*expected_outputs=*/
       {CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {10}})}},
      {/*dataset_params=*/EmptyInputDatasetParams(),
       /*breakpoints=*/{0, 2, 4},
       /*expected_outputs=*/
       {CreateTensors<int64_t>(TensorShape({}), {})}},
      {/*dataset_params=*/LargeNumInputDatasetsParams(),
       /*breakpoints=*/{0, 5, 8},
       /*expected_outputs=*/
       {CreateTensors<int64_t>(TensorShape({}),
                               {{0}, {10}, {1}, {11}, {2}, {12}})}},
      {/*dataset_params=*/SmallNumInputDatasetsParams(),
       /*breakpoints=*/{0, 5, 8},
       /*expected_outputs=*/
       {CreateTensors<int64_t>(TensorShape({}),
                               {{0}, {10}, {1}, {11}, {2}, {12}})}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(DirectedInterleaveDatasetOpTest,
                                 DirectedInterleaveDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(DirectedInterleaveDatasetOpTest, InvalidArguments) {
  std::vector<DirectedInterleaveDatasetParams> invalid_params_vec = {
      InvalidSelectorOuputDataType(), InvalidSelectorOuputShape(),
      InvalidInputDatasetsDataType(), ZeroInputDatasetParams()};
  for (auto& dataset_params : invalid_params_vec) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

TEST_F(DirectedInterleaveDatasetOpTest, InvalidSelectorValues) {
  auto dataset_params = InvalidSelectorValues();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> next;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence).code(),
      tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
