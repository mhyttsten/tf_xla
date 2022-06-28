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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/parallel_interleave_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "parallel_interleave_dataset";
constexpr int kOpVersion = 4;

class ParallelInterleaveDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ParallelInterleaveDatasetParams(
      T input_dataset_params, std::vector<Tensor> other_arguments,
      int64_t cycle_length, int64_t block_length,
      int64_t buffer_output_elements, int64_t prefetch_input_elements,
      int64_t num_parallel_calls, FunctionDefHelper::AttrValueWrapper func,
      std::vector<FunctionDef> func_lib, DataTypeVector type_arguments,
      const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes,
      const std::string& deterministic, const std::string& node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        other_arguments_(std::move(other_arguments)),
        cycle_length_(cycle_length),
        block_length_(block_length),
        buffer_output_elements_(buffer_output_elements),
        prefetch_input_elements_(prefetch_input_elements),
        num_parallel_calls_(num_parallel_calls),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)),
        deterministic_(deterministic) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("deterministic: \"" + deterministic + "\"");
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    op_version_ = kOpVersion;
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    iterator_prefix_ = name_utils::IteratorPrefix(
        input_dataset_params.dataset_type(),
        input_dataset_params.iterator_prefix(), params);
  }

  std::vector<Tensor> GetInputTensors() const override {
    auto input_tensors = other_arguments_;
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {cycle_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {block_length_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {buffer_output_elements_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {prefetch_input_elements_}));
    input_tensors.emplace_back(
        CreateTensor<int64_t>(TensorShape({}), {num_parallel_calls_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_1(mht_1_v, 245, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "GetInputNames");

    input_names->emplace_back(ParallelInterleaveDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_names->emplace_back(
          absl::StrCat(ParallelInterleaveDatasetOp::kOtherArguments, "_", i));
    }
    input_names->emplace_back(ParallelInterleaveDatasetOp::kCycleLength);
    input_names->emplace_back(ParallelInterleaveDatasetOp::kBlockLength);
    input_names->emplace_back(
        ParallelInterleaveDatasetOp::kBufferOutputElements);
    input_names->emplace_back(
        ParallelInterleaveDatasetOp::kPrefetchInputElements);
    input_names->emplace_back(ParallelInterleaveDatasetOp::kNumParallelCalls);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_2(mht_2_v, 264, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{"f", func_},
                    {"deterministic", deterministic_},
                    {"Targuments", type_arguments_},
                    {"output_shapes", output_shapes_},
                    {"output_types", output_dtypes_},
                    {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "dataset_type");

    return ParallelInterleaveDatasetOp::kDatasetType;
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::vector<Tensor> other_arguments_;
  int64_t cycle_length_;
  int64_t block_length_;
  int64_t buffer_output_elements_;
  int64_t prefetch_input_elements_;
  int64_t num_parallel_calls_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
  std::string deterministic_;
};

class ParallelInterleaveDatasetOpTest : public DatasetOpsTestBase {};

FunctionDefHelper::AttrValueWrapper MakeTensorSliceDatasetFunc(
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_4(mht_4_v, 303, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "MakeTensorSliceDatasetFunc");

  return FunctionDefHelper::FunctionRef(
      /*name=*/"MakeTensorSliceDataset",
      /*attrs=*/{{"Toutput_types", output_types},
                 {"output_shapes", output_shapes}});
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams1() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_5(mht_5_v, 313, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams2() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_6(mht_6_v, 341, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams2");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/1,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/0,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams3() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_7(mht_7_v, 369, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams3");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/3,
      /*block_length=*/1,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/1,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams4() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_8(mht_8_v, 397, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams4");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/5,
      /*block_length=*/1,
      /*buffer_output_elements=*/1,
      /*prefetch_input_elements=*/0,
      /*num_parallel_calls=*/4,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams5() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_9(mht_9_v, 425, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams5");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/2,
      /*buffer_output_elements=*/2,
      /*prefetch_input_elements=*/2,
      /*num_parallel_calls=*/1,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams6() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_10(mht_10_v, 453, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams6");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/2,
      /*block_length=*/3,
      /*buffer_output_elements=*/100,
      /*prefetch_input_elements=*/100,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams7() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_11(mht_11_v, 481, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams7");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/3,
      /*block_length=*/2,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams8() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_12(mht_12_v, 509, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams8");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/3,
      /*block_length=*/3,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/3,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams9() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_13(mht_13_v, 537, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams9");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/4,
      /*block_length=*/4,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/4,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams ParallelInterleaveDatasetParams10() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_14(mht_14_v, 565, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParams10");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/4,
      /*block_length=*/4,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/model::kAutotune,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams LongCycleDeterministicParams() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_15(mht_15_v, 593, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "LongCycleDeterministicParams");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<tstring>(
          TensorShape{3, 3, 1}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/11,
      /*block_length=*/1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_STRING}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_STRING},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams
ParallelInterleaveDatasetParamsWithInvalidCycleLength() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_16(mht_16_v, 622, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParamsWithInvalidCycleLength");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/0,
      /*block_length=*/1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams
ParallelInterleaveDatasetParamsWithInvalidBlockLength() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_17(mht_17_v, 651, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParamsWithInvalidBlockLength");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/-1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/2,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams
ParallelInterleaveDatasetParamsWithInvalidNumParallelCalls() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_18(mht_18_v, 680, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParamsWithInvalidNumParallelCalls");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/-5,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams
ParallelInterleaveDatasetParamsWithInvalidBufferOutputElements() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_19(mht_19_v, 709, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParamsWithInvalidBufferOutputElements");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/-5,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

ParallelInterleaveDatasetParams
ParallelInterleaveDatasetParamsWithInvalidPrefetchInputElements() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_interleave_dataset_op_testDTcc mht_20(mht_20_v, 738, "", "./tensorflow/core/kernels/data/parallel_interleave_dataset_op_test.cc", "ParallelInterleaveDatasetParamsWithInvalidPrefetchInputElements");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return ParallelInterleaveDatasetParams(
      tensor_slice_dataset_params,
      /*other_arguments=*/{},
      /*cycle_length=*/1,
      /*block_length=*/1,
      /*buffer_output_elements=*/model::kAutotune,
      /*prefetch_input_elements=*/model::kAutotune,
      /*num_parallel_calls=*/-5,
      /*func=*/
      MakeTensorSliceDatasetFunc(
          DataTypeVector({DT_INT64}),
          std::vector<PartialTensorShape>({PartialTensorShape({1})})),
      /*func_lib=*/{test::function::MakeTensorSliceDataset()},
      /*type_arguments=*/{},
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({1})},
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ParallelInterleaveDatasetParams>>
GetNextTestCases() {
  return {{/*dataset_params=*/ParallelInterleaveDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams5(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams6(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams7(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"d"}, {"e"}, {"g"}, {"h"}, {"c"}, {"f"}, {"i"}}),
           /*compare_order=*/true},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams8(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams9(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams10(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           LongCycleDeterministicParams(),
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"d"}, {"g"}, {"b"}, {"e"}, {"h"}, {"c"}, {"f"}, {"i"}}),
           /*compare_order=*/true}};
}

ITERATOR_GET_NEXT_TEST_P(ParallelInterleaveDatasetOpTest,
                         ParallelInterleaveDatasetParams, GetNextTestCases())

TEST_F(ParallelInterleaveDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ParallelInterleaveDatasetOp::kDatasetType, params)));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ParallelInterleaveDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({1})}));
}

std::vector<CardinalityTestCase<ParallelInterleaveDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/ParallelInterleaveDatasetParams1(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams2(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams3(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams4(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams5(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams6(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams7(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams8(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams9(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ParallelInterleaveDatasetParams10(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(ParallelInterleaveDatasetOpTest,
                           ParallelInterleaveDatasetParams,
                           CardinalityTestCases())

TEST_F(ParallelInterleaveDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ParallelInterleaveDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({1})}));
}

TEST_F(ParallelInterleaveDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ParallelInterleaveDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(
      name_utils::IteratorPrefix(ParallelInterleaveDatasetOp::kDatasetType,
                                 dataset_params.iterator_prefix(), params)));
}

std::vector<IteratorSaveAndRestoreTestCase<ParallelInterleaveDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ParallelInterleaveDatasetParams1(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams2(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {1}, {4}, {2}, {5}, {6}, {7}, {8}}),
           /*compare_order=*/true},
          {/*dataset_params=*/ParallelInterleaveDatasetParams3(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams4(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(
               TensorShape{1}, {{0}, {3}, {6}, {1}, {4}, {7}, {2}, {5}, {8}}),
           /*compare_order=*/false},
          {/*dataset_params=*/ParallelInterleaveDatasetParams5(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"d"}, {"e"}, {"c"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams6(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams7(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"d"}, {"e"}, {"g"}, {"h"}, {"c"}, {"f"}, {"i"}}),
           /*compare_order=*/true},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams8(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams9(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false},
          {/*dataset_params=*/
           ParallelInterleaveDatasetParams10(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_outputs=*/
           CreateTensors<tstring>(
               TensorShape{1},
               {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}, {"h"}, {"i"}}),
           /*compare_order=*/false}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ParallelInterleaveDatasetOpTest,
                                 ParallelInterleaveDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(ParallelInterleaveDatasetOpTest, InvalidArguments) {
  std::vector<ParallelInterleaveDatasetParams> invalid_params = {
      ParallelInterleaveDatasetParamsWithInvalidCycleLength(),
      ParallelInterleaveDatasetParamsWithInvalidBlockLength(),
      ParallelInterleaveDatasetParamsWithInvalidNumParallelCalls(),
      ParallelInterleaveDatasetParamsWithInvalidBufferOutputElements(),
      ParallelInterleaveDatasetParamsWithInvalidPrefetchInputElements(),
  };
  for (auto& dataset_params : invalid_params) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
