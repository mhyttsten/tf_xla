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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/interleave_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "interleave_dataset";

class InterleaveDatasetParams : public DatasetParams {
 public:
  template <typename T>
  InterleaveDatasetParams(T input_dataset_params,
                          std::vector<Tensor> other_arguments,
                          int64_t cycle_length, int64_t block_length,
                          FunctionDefHelper::AttrValueWrapper func,
                          std::vector<FunctionDef> func_lib,
                          DataTypeVector type_arguments,
                          DataTypeVector output_dtypes,
                          std::vector<PartialTensorShape> output_shapes,
                          string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {cycle_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {block_length_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->reserve(input_dataset_params_.size() +
                         other_arguments_.size() + 2);
    input_names->emplace_back(InterleaveDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(InterleaveDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(InterleaveDatasetOp::kCycleLength);
    input_names->emplace_back(InterleaveDatasetOp::kBlockLength);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_2(mht_2_v, 247, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{"f", func_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"metadata", ""}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "dataset_type");

    return InterleaveDatasetOp::kDatasetType;
  }

 private:
  std::vector<Tensor> other_arguments_;
  int64_t cycle_length_;
  int64_t block_length_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
};

class InterleaveDatasetOpTest : public DatasetOpsTestBase {};

FunctionDefHelper::AttrValueWrapper MakeTensorSliceDatasetFunc(
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "MakeTensorSliceDatasetFunc");

  return FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{"Toutput_types", output_types},
                 {"output_shapes", output_shapes}});
}

// test case 1: cycle_length = 1, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams1() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_5(mht_5_v, 292, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 2: cycle_length = 2, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams2() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_6(mht_6_v, 318, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams2");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 3: cycle_length = 3, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams3() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_7(mht_7_v, 344, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams3");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/3,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 4: cycle_length = 5, block_length = 1.
InterleaveDatasetParams InterleaveDatasetParams4() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_8(mht_8_v, 370, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams4");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/5,
      /*block_length=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 5: cycle_length = 2, block_length = 2.
InterleaveDatasetParams InterleaveDatasetParams5() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_9(mht_9_v, 396, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams5");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(TensorShape{3, 3, 1},
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 6: cycle_length = 2, block_length = 3.
InterleaveDatasetParams InterleaveDatasetParams6() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_10(mht_10_v, 422, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams6");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(TensorShape{3, 3, 1},
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/3,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 7: cycle_length = 2, block_length = 5.
InterleaveDatasetParams InterleaveDatasetParams7() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_11(mht_11_v, 448, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParams7");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<tstring>(TensorShape{3, 3, 1},
                             {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/5,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 8: cycle_length = 0, block_length = 5.
InterleaveDatasetParams InterleaveDatasetParamsWithInvalidCycleLength() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_12(mht_12_v, 474, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParamsWithInvalidCycleLength");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/0,
      /*block_length=*/5,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

// test case 9: cycle_length = 1, block_length = -1.
InterleaveDatasetParams InterleaveDatasetParamsWithInvalidBlockLength() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSinterleave_dataset_op_testDTcc mht_13(mht_13_v, 500, "", "./tensorflow/core/kernels/data/interleave_dataset_op_test.cc", "InterleaveDatasetParamsWithInvalidBlockLength");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/
      {CreateTensor<int64_t>(TensorShape{3, 3, 1},
                             {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice_dataset");
  return InterleaveDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/-1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<InterleaveDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/InterleaveDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams2(),
       /*expected_outputs=*/CreateTensors<int64_t>(
           TensorShape({1}), {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams3(),
       /*expected_outputs=*/CreateTensors<int64_t>(
           TensorShape({1}), {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams4(),
       /*expected_outputs=*/CreateTensors<int64_t>(
           TensorShape({1}), {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams5(),
       /*expected_outputs=*/CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams6(),
       /*expected_outputs=*/CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams7(),
       /*expected_outputs=*/CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})}};
}

ITERATOR_GET_NEXT_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                         GetNextTestCases())

std::vector<SkipTestCase<InterleaveDatasetParams>> SkipTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 0, /*expected_num_skipped*/ 0, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{0}})},
          {/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{5}})},
          {/*dataset_params=*/InterleaveDatasetParams1(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{5}})},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{7}})},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*num_to_skip*/ 5, /*expected_num_skipped*/ 5, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({1}), {{7}})},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*num_to_skip*/ 3, /*expected_num_skipped*/ 3, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"e"}})},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*num_to_skip*/ 3, /*expected_num_skipped*/ 3, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"d"}})},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*num_to_skip*/ 3, /*expected_num_skipped*/ 3, /*get_next*/ true,
           /*expected_outputs=*/
           CreateTensors<tstring>(TensorShape({1}), {{"d"}})},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*num_to_skip*/ 10, /*expected_num_skipped*/ 9}};
}

ITERATOR_SKIP_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                     SkipTestCases())

TEST_F(InterleaveDatasetOpTest, DatasetNodeName) {
  auto dataset_params = InterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(InterleaveDatasetOpTest, DatasetTypeString) {
  auto dataset_params = InterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(InterleaveDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<InterleaveDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_dtypes=*/{DT_STRING}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<InterleaveDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<InterleaveDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<InterleaveDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_dtypes=*/{DT_STRING}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_dtypes=*/{DT_STRING}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<InterleaveDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/InterleaveDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}},
          {/*dataset_params=*/InterleaveDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({1})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(InterleaveDatasetOpTest, InterleaveDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(InterleaveDatasetOpTest, IteratorPrefix) {
  auto dataset_params = InterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      InterleaveDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<InterleaveDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/InterleaveDatasetParams1(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams2(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams3(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams4(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({1}),
                              {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}})},
      {/*dataset_params=*/InterleaveDatasetParams5(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams6(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})},
      {/*dataset_params=*/InterleaveDatasetParams7(),
       /*breakpoints=*/{0, 4, 11},
       /*expected_outputs=*/
       CreateTensors<tstring>(
           TensorShape({1}),
           {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(InterleaveDatasetOpTest,
                                 InterleaveDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(InterleaveDatasetOpTest, InvalidCycleLength) {
  auto dataset_params = InterleaveDatasetParamsWithInvalidCycleLength();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(InterleaveDatasetOpTest, InvalidLength) {
  auto dataset_params = InterleaveDatasetParamsWithInvalidBlockLength();
  EXPECT_EQ(Initialize(dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
