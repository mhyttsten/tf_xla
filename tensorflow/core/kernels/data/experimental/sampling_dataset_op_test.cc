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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/sampling_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "sampling_dataset";
constexpr int64_t kRandomSeed = 42;
constexpr int64_t kRandomSeed2 = 7;

class SamplingDatasetParams : public DatasetParams {
 public:
  template <typename T>
  SamplingDatasetParams(T input_dataset_params, float rate,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        rate_(rate) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "SamplingDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    Tensor rate = CreateTensor<float>(TensorShape({}), {rate_});
    Tensor seed_tensor = CreateTensor<int64_t>(TensorShape({}), {seed_tensor_});
    Tensor seed2_tensor =
        CreateTensor<int64_t>(TensorShape({}), {seed2_tensor_});
    return {rate, seed_tensor, seed2_tensor};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "GetInputNames");

    *input_names = {SamplingDatasetOp::kInputDataset, SamplingDatasetOp::kRate,
                    SamplingDatasetOp::kSeed, SamplingDatasetOp::kSeed2};

    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "GetAttributes");

    *attr_vector = {{SamplingDatasetOp::kOutputTypes, output_dtypes_},
                    {SamplingDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "dataset_type");

    return SamplingDatasetOp::kDatasetType;
  }

 private:
  // Target sample rate, range (0,1], wrapped in a scalar Tensor
  float rate_;
  // Boxed versions of kRandomSeed and kRandomSeed2.
  int64_t seed_tensor_ = kRandomSeed;
  int64_t seed2_tensor_ = kRandomSeed2;
};

class SamplingDatasetOpTest : public DatasetOpsTestBase {};

SamplingDatasetParams OneHundredPercentSampleParams() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "OneHundredPercentSampleParams");

  return SamplingDatasetParams(RangeDatasetParams(0, 3, 1),
                               /*rate=*/1.0,
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/kNodeName);
}

SamplingDatasetParams TenPercentSampleParams() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "TenPercentSampleParams");

  return SamplingDatasetParams(RangeDatasetParams(0, 20, 1),
                               /*rate=*/0.1,
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/kNodeName);
}

SamplingDatasetParams ZeroPercentSampleParams() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsampling_dataset_op_testDTcc mht_6(mht_6_v, 280, "", "./tensorflow/core/kernels/data/experimental/sampling_dataset_op_test.cc", "ZeroPercentSampleParams");

  return SamplingDatasetParams(RangeDatasetParams(0, 20, 1),
                               /*rate=*/0.0,
                               /*output_dtypes=*/{DT_INT64},
                               /*output_shapes=*/{PartialTensorShape({})},
                               /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<SamplingDatasetParams>> GetNextTestCases() {
  return {
      // Test case 1: 100% sample should return all inputs
      {/*dataset_params=*/OneHundredPercentSampleParams(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape({}),
                                                   {{0}, {1}, {2}})},

      // Test case 2: 10% sample should return about 10% of inputs, and the
      // specific inputs returned shouldn't change across build environments.
      {/*dataset_params=*/TenPercentSampleParams(),
       /*expected_outputs=*/CreateTensors<int64_t>(TensorShape({}),
                                                   {{9}, {11}, {19}})},

      // Test case 3: 0% sample should return nothing and should not crash.
      {/*dataset_params=*/ZeroPercentSampleParams(), /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                         GetNextTestCases())

std::vector<DatasetNodeNameTestCase<SamplingDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                         DatasetNodeNameTestCases())

std::vector<DatasetTypeStringTestCase<SamplingDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               SamplingDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                           DatasetTypeStringTestCases())

std::vector<DatasetOutputDtypesTestCase<SamplingDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<SamplingDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<SamplingDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/OneHundredPercentSampleParams(),
           /*expected_cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/TenPercentSampleParams(),
           /*expected,cardinality=*/kUnknownCardinality},
          {/*dataset_params=*/ZeroPercentSampleParams(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<SamplingDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<SamplingDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                              IteratorOutputShapesTestCases())

std::vector<IteratorPrefixTestCase<SamplingDatasetParams>>
IteratorOutputPrefixTestCases() {
  return {{/*dataset_params=*/TenPercentSampleParams(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               SamplingDatasetOp::kDatasetType,
               TenPercentSampleParams().iterator_prefix())}};
}

ITERATOR_PREFIX_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                       IteratorOutputPrefixTestCases())

std::vector<IteratorSaveAndRestoreTestCase<SamplingDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/OneHundredPercentSampleParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {1}, {2}})},
          {/*dataset_params=*/TenPercentSampleParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{9}, {11}, {19}})},
          {/*dataset_params=*/ZeroPercentSampleParams(),
           /*breakpoints=*/{0, 2, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(SamplingDatasetOpTest, SamplingDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
