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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/io_ops.h"

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"

namespace tensorflow {
namespace data {
namespace experimental {

constexpr char kSaveDatasetV2NodeName[] = "save_dataset_v2";

class SaveDatasetV2Params : public DatasetParams {
 public:
  template <typename T>
  SaveDatasetV2Params(T input_dataset_params, const tstring& path,
                      const std::string& compression,
                      FunctionDefHelper::AttrValueWrapper shard_func,
                      std::vector<FunctionDef> func_lib, bool use_shard_func,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name, DataTypeVector type_arguments)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        path_(path),
        compression_(compression),
        shard_func_(shard_func),
        func_lib_(std::move(func_lib)),
        use_shard_func_(use_shard_func),
        type_arguments_(std::move(type_arguments)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + (std::string)path + "\"");
   mht_0_v.push_back("compression: \"" + compression + "\"");
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "SaveDatasetV2Params");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors;
    input_tensors.emplace_back(CreateTensor<tstring>(TensorShape({}), {path_}));
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(SaveDatasetV2Op::kInputDataset);
    input_names->emplace_back(SaveDatasetV2Op::kPath);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back(SaveDatasetV2Op::kCompression, compression_);
    attr_vector->emplace_back(SaveDatasetV2Op::kShardFunc, shard_func_);
    attr_vector->emplace_back(SaveDatasetV2Op::kUseShardFunc, use_shard_func_);
    attr_vector->emplace_back(SaveDatasetV2Op::kShardFuncTarguments,
                              type_arguments_);
    attr_vector->emplace_back(SaveDatasetV2Op::kOutputTypes, output_dtypes_);
    attr_vector->emplace_back(SaveDatasetV2Op::kOutputShapes, output_shapes_);
    return Status::OK();
  }

  string path() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_3(mht_3_v, 264, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "path");
 return path_; }

  string dataset_type() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_4(mht_4_v, 269, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "dataset_type");
 return SaveDatasetV2Op::kDatasetType; }

  string op_name() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_5(mht_5_v, 274, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "op_name");
 return "SaveDatasetV2"; }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

 private:
  std::string path_;
  std::string compression_;
  FunctionDefHelper::AttrValueWrapper shard_func_;
  std::vector<FunctionDef> func_lib_;
  bool use_shard_func_;
  DataTypeVector type_arguments_;
};

class SaveDatasetV2OpTest : public DatasetOpsTestBase {
 public:
  Status Initialize(const DatasetParams& dataset_params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_6(mht_6_v, 292, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "Initialize");

    TF_RETURN_IF_ERROR(DatasetOpsTestBase::Initialize(dataset_params));
    auto params = static_cast<const SaveDatasetV2Params&>(dataset_params);
    save_filename_ = params.path();
    return Status::OK();
  }

 protected:
  std::string save_filename_;
};

// Test case 1. Basic save parameters.
SaveDatasetV2Params SaveDatasetV2Params1() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "SaveDatasetV2Params1");

  return SaveDatasetV2Params(
      RangeDatasetParams(0, 10, 2),
      /*path=*/io::JoinPath(testing::TmpDir(), "save_data"),
      /*compression=*/"",
      /*shard_func=*/
      FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo()},
      /*use_shard_func=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kSaveDatasetV2NodeName,
      /*type_arguments=*/{});
}

// Test case 2. Tests custom compression settings and uses shard func.
SaveDatasetV2Params SaveDatasetV2Params2() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSio_ops_testDTcc mht_8(mht_8_v, 326, "", "./tensorflow/core/kernels/data/experimental/io_ops_test.cc", "SaveDatasetV2Params2");

  return SaveDatasetV2Params(
      RangeDatasetParams(0, 5, 1),
      /*path=*/io::JoinPath(testing::TmpDir(), "save_data"),
      /*compression=*/"GZIP",
      /*shard_func=*/
      FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_INT64}}),
      /*func_lib=*/{test::function::XTimesTwo()},
      /*use_shard_func=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})},
      /*node_name=*/kSaveDatasetV2NodeName,
      /*type_arguments=*/{});
}

std::vector<GetNextTestCase<SaveDatasetV2Params>> GetNextTestCases() {
  return {{/*dataset_params=*/
           SaveDatasetV2Params1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}, {8}})},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}})}};
}

class ParameterizedGetNextTest : public SaveDatasetV2OpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<SaveDatasetV2Params>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  // Test the write mode.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;

  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(SaveDatasetV2OpTest, ParameterizedGetNextTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(SaveDatasetV2OpTest, DatasetNodeName) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(SaveDatasetV2OpTest, DatasetTypeString) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString("SaveDatasetV2"));
}

TEST_F(SaveDatasetV2OpTest, DatasetOutputDtypes) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes(dataset_params.output_dtypes()));
}

std::vector<DatasetOutputDtypesTestCase<SaveDatasetV2Params>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*expected_output_dtypes=*/{DT_INT64}},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SaveDatasetV2OpTest, SaveDatasetV2Params,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<SaveDatasetV2Params>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SaveDatasetV2OpTest, SaveDatasetV2Params,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<SaveDatasetV2Params>> CardinalityTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*expected_cardinality=*/5},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*expected_cardinality=*/5}};
}

DATASET_CARDINALITY_TEST_P(SaveDatasetV2OpTest, SaveDatasetV2Params,
                           CardinalityTestCases())

TEST_F(SaveDatasetV2OpTest, IteratorPrefix) {
  auto dataset_params = SaveDatasetV2Params1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      SaveDatasetV2Op::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<SaveDatasetV2Params>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/SaveDatasetV2Params1(),
           /*breakpoints=*/{0, 2, 4, 6, 8},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {2}, {4}, {6}, {8}})},
          {/*dataset_params=*/SaveDatasetV2Params2(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}})}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public SaveDatasetV2OpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<SaveDatasetV2Params>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_CASE_P(SaveDatasetV2OpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
