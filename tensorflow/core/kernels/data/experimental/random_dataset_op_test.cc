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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/random_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "random_dataset";
constexpr char kIteratorPrefix[] = "Iterator";

// Number of random samples generated per test
constexpr int kCount = 10;

// Generate the first `count` random numbers that the kernel should produce
// for a given seed/seed2 combo.
// For compatibility with the test harness, return value is a vector of scalar
// Tensors.
std::vector<Tensor> GenerateExpectedData(int64_t seed, int64_t seed2,
                                         int count) {
  std::vector<Tensor> ret;
  auto parent_generator = random::PhiloxRandom(seed, seed2);
  auto generator =
      random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator);

  for (int i = 0; i < count; ++i) {
    ret.push_back(CreateTensor<int64_t>(TensorShape({}), {generator()}));
  }
  return ret;
}

class RandomDatasetParams : public DatasetParams {
 public:
  RandomDatasetParams(int64_t seed, int64_t seed2, DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        seed_(CreateTensor<int64_t>(TensorShape({}), {seed})),
        seed2_(CreateTensor<int64_t>(TensorShape({}), {seed2})) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "RandomDatasetParams");
}

  virtual std::vector<Tensor> GetInputTensors() const override {
    return {seed_, seed2_};
  }

  virtual Status GetInputNames(
      std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "GetInputNames");

    *input_names = {RandomDatasetOp::kSeed, RandomDatasetOp::kSeed2};
    return Status::OK();
  }

  virtual Status GetAttributes(AttributeVector* attributes) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "GetAttributes");

    *attributes = {{"output_types", output_dtypes_},
                   {"output_shapes", output_shapes_},
                   {"metadata", ""}};
    return Status::OK();
  }

  virtual string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "dataset_type");

    return RandomDatasetOp::kDatasetType;
  }

 private:
  Tensor seed_;
  Tensor seed2_;
};

class RandomDatasetOpTest : public DatasetOpsTestBase {};

RandomDatasetParams FortyTwo() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "FortyTwo");

  return {/*seed=*/42,
          /*seed2=*/42,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just first seed relative to FortyTwo
RandomDatasetParams ChangeSeed() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "ChangeSeed");

  return {/*seed=*/1000,
          /*seed2=*/42,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// Change just second seed relative to FortyTwo
RandomDatasetParams ChangeSeed2() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrandom_dataset_op_testDTcc mht_6(mht_6_v, 290, "", "./tensorflow/core/kernels/data/experimental/random_dataset_op_test.cc", "ChangeSeed2");

  return {/*seed=*/42,
          /*seed2=*/1000,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

class ParameterizedGetNextTest : public RandomDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<RandomDatasetParams>> {};

std::vector<GetNextTestCase<RandomDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_outputs=*/GenerateExpectedData(42, 42, kCount)},
          {/*dataset_params=*/ChangeSeed(),
           /*expected_outputs=*/GenerateExpectedData(1000, 42, kCount)},
          {/*dataset_params=*/ChangeSeed2(),
           /*expected_outputs=*/GenerateExpectedData(42, 1000, kCount)}};
}

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  // Can't use DatasetOpsTestBase::CheckIteratorGetNext because the kernel
  // under test produces unbounded input.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (out_tensors.size() < test_case.expected_outputs.size()) {
    std::vector<Tensor> next;
    TF_ASSERT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));

    ASSERT_FALSE(end_of_sequence);  // Dataset should never stop

    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_ASSERT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(
    RandomDatasetOpTest, ParameterizedGetNextTest,
    ::testing::ValuesIn(
        std::vector<GetNextTestCase<RandomDatasetParams>>(GetNextTestCases())));

std::vector<DatasetNodeNameTestCase<RandomDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/FortyTwo(), /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                         DatasetNodeNameTestCases());

std::vector<DatasetTypeStringTestCase<RandomDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               RandomDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                           DatasetTypeStringTestCases());

std::vector<DatasetOutputDtypesTestCase<RandomDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {
      {/*dataset_params=*/FortyTwo(), /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                             DatasetOutputDtypesTestCases());

std::vector<DatasetOutputShapesTestCase<RandomDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                             DatasetOutputShapesTestCases());

std::vector<CardinalityTestCase<RandomDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_cardinality=*/kInfiniteCardinality}};
}

DATASET_CARDINALITY_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                           CardinalityTestCases());

std::vector<IteratorOutputDtypesTestCase<RandomDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {
      {/*dataset_params=*/FortyTwo(), /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                              IteratorOutputDtypesTestCases());

std::vector<IteratorOutputShapesTestCase<RandomDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                              IteratorOutputShapesTestCases());

std::vector<IteratorPrefixTestCase<RandomDatasetParams>>
IteratorOutputPrefixTestCases() {
  return {{/*dataset_params=*/FortyTwo(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               RandomDatasetOp::kDatasetType, kIteratorPrefix)}};
}

ITERATOR_PREFIX_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                       IteratorOutputPrefixTestCases());

std::vector<IteratorSaveAndRestoreTestCase<RandomDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/FortyTwo(), /*breakpoints=*/{2, 5, 8},
           /*expected_outputs=*/GenerateExpectedData(42, 42, 9 /* 8 + 1 */)}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(RandomDatasetOpTest, RandomDatasetParams,
                                 IteratorSaveAndRestoreTestCases());

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
