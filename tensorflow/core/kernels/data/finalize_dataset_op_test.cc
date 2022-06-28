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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/finalize_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/options_dataset_op.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

class FinalizeDatasetParams : public DatasetParams {
 public:
  template <typename T>
  FinalizeDatasetParams(T input_dataset_params, DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        has_captured_ref_(false) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "FinalizeDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "GetInputNames");

    input_names->emplace_back(FinalizeDatasetOp::kInputDataset);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{FinalizeDatasetOp::kHasCapturedRef, has_captured_ref_},
                    {FinalizeDatasetOp::kOutputTypes, output_dtypes_},
                    {FinalizeDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "dataset_type");
 return "Finalize"; }

 private:
  bool has_captured_ref_;
};

class FinalizeDatasetOpTest : public DatasetOpsTestBase {
 public:
  void CheckDatasetPipelineTypeStrings(
      const std::vector<std::string>& type_strings) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "CheckDatasetPipelineTypeStrings");

    CheckDatasetPipelineTypeString(dataset_, type_strings, 0);
  }

  void CheckDatasetPipelineTypeString(
      const DatasetBase* dataset, const std::vector<std::string>& type_strings,
      int index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "CheckDatasetPipelineTypeString");

    EXPECT_GT(type_strings.size(), index);
    EXPECT_EQ(dataset->type_string(), type_strings[index]);
    std::vector<const DatasetBase*> input_datasets;
    TF_ASSERT_OK(dataset->InputDatasets(&input_datasets));
    if (input_datasets.empty()) {
      return;
    }
    EXPECT_EQ(1, input_datasets.size());
    CheckDatasetPipelineTypeString(input_datasets[0], type_strings, index + 1);
  }
};

constexpr char kNoOptimizationOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: false }
)pb";
constexpr char kMaxIntraOpParallelismOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: false }
  threading_options { max_intra_op_parallelism: 10 }
)pb";
constexpr char kPrivateThreadPoolOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: false }
  threading_options { private_threadpool_size: 10 }
)pb";
constexpr char kModelOptions[] = R"proto(
  optimization_options { apply_default_optimizations: false }
)proto";
constexpr char kOptimizationsDefaultOptions[] = R"pb(
  autotune_options { enabled: false }
  optimization_options { apply_default_optimizations: true }
)pb";
constexpr char kAllChainedDatasetsOptions[] = R"pb(
  autotune_options { enabled: true }
  optimization_options { apply_default_optimizations: true }
  threading_options { max_intra_op_parallelism: 10 private_threadpool_size: 10 }
)pb";

OptionsDatasetParams NoOptimizationOptionsParams() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "NoOptimizationOptionsParams");

  Options options;
  protobuf::TextFormat::ParseFromString(kNoOptimizationOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams MaxIntraOpParallelismOptionsParams() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "MaxIntraOpParallelismOptionsParams");

  Options options;
  protobuf::TextFormat::ParseFromString(kMaxIntraOpParallelismOptions,
                                        &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams PrivateThreadPoolOptionsParams() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_8(mht_8_v, 321, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "PrivateThreadPoolOptionsParams");

  Options options;
  protobuf::TextFormat::ParseFromString(kPrivateThreadPoolOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams ModelOptionsParams() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_9(mht_9_v, 334, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "ModelOptionsParams");

  Options options;
  protobuf::TextFormat::ParseFromString(kModelOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams OptimizationsDefaultOptionsParams() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_10(mht_10_v, 347, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "OptimizationsDefaultOptionsParams");

  Options options;
  protobuf::TextFormat::ParseFromString(kOptimizationsDefaultOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

OptionsDatasetParams AllChainedDatasetsOptionsParams() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_11(mht_11_v, 360, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "AllChainedDatasetsOptionsParams");

  Options options;
  protobuf::TextFormat::ParseFromString(kAllChainedDatasetsOptions, &options);
  return OptionsDatasetParams(RangeDatasetParams(0, 10, 3),
                              options.SerializeAsString(),
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/"options_dataset_0");
}

FinalizeDatasetParams NoOptimizationFinalizeParams() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_12(mht_12_v, 373, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "NoOptimizationFinalizeParams");

  return FinalizeDatasetParams(NoOptimizationOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"options_dataset_0");
}

FinalizeDatasetParams MaxIntraOpParallelismParams() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_13(mht_13_v, 383, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "MaxIntraOpParallelismParams");

  return FinalizeDatasetParams(MaxIntraOpParallelismOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"MaxIntraOpParallelismDatasetOp");
}

FinalizeDatasetParams PrivateThreadPoolParams() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_14(mht_14_v, 393, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "PrivateThreadPoolParams");

  return FinalizeDatasetParams(PrivateThreadPoolOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"PrivateThreadPoolDatasetOp");
}

FinalizeDatasetParams ModelParams() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_15(mht_15_v, 403, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "ModelParams");

  return FinalizeDatasetParams(ModelOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"ModelDatasetOp");
}

FinalizeDatasetParams OptimizationsDefaultParams() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_16(mht_16_v, 413, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "OptimizationsDefaultParams");

  return FinalizeDatasetParams(OptimizationsDefaultOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"private_thread_pool");
}

FinalizeDatasetParams AllChainedDatasetsParams() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfinalize_dataset_op_testDTcc mht_17(mht_17_v, 423, "", "./tensorflow/core/kernels/data/finalize_dataset_op_test.cc", "AllChainedDatasetsParams");

  return FinalizeDatasetParams(AllChainedDatasetsOptionsParams(),
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/"ModelDataset/_9");
}

TEST_F(FinalizeDatasetOpTest, NoOptimizationNodeName) {
  auto test_case_params = NoOptimizationFinalizeParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings({"OptionsDataset", "RangeDataset"});
}

std::vector<GetNextTestCase<FinalizeDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/NoOptimizationFinalizeParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/MaxIntraOpParallelismParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/PrivateThreadPoolParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/ModelParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/OptimizationsDefaultParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/AllChainedDatasetsParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})}};
}

ITERATOR_GET_NEXT_TEST_P(FinalizeDatasetOpTest, FinalizeDatasetParams,
                         GetNextTestCases())

TEST_F(FinalizeDatasetOpTest, MaxIntraOpParallelismNodeName) {
  auto test_case_params = MaxIntraOpParallelismParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings(
      {"MaxIntraOpParallelismDataset", "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, PrivateThreadPoolNodeName) {
  auto test_case_params = PrivateThreadPoolParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings(
      {"PrivateThreadPoolDataset", "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, ModelNodeName) {
  auto test_case_params = ModelParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings(
      {"ModelDataset", "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, OptimizationsDefaultNodeName) {
  auto test_case_params = OptimizationsDefaultParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings({"PrivateThreadPoolDataset",
                                   "MaxIntraOpParallelismDataset",
                                   "OptionsDataset", "RangeDataset"});
}

TEST_F(FinalizeDatasetOpTest, AllChainedDatasetsNodeName) {
  auto test_case_params = AllChainedDatasetsParams();
  TF_ASSERT_OK(Initialize(test_case_params));
  std::vector<const DatasetBase*> inputs;
  Status s = dataset_->InputDatasets(&inputs);
  TF_ASSERT_OK(CheckDatasetNodeName(test_case_params.node_name()));
  CheckDatasetPipelineTypeStrings({"ModelDataset", "PrivateThreadPoolDataset",
                                   "MaxIntraOpParallelismDataset",
                                   "OptionsDataset", "RangeDataset"});
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
