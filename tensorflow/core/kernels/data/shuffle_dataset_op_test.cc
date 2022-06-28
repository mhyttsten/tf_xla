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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/shuffle_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kShuffleNodeName[] = "shuffle_dataset";
constexpr char kShuffleAndRepeatNodeName[] = "shuffle_and_repeat_dataset";

class ShuffleDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ShuffleDatasetParams(T input_dataset_params, int64_t buffer_size,
                       int64_t seed, int64_t seed2, int64_t count,
                       bool reshuffle_each_iteration,
                       DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        buffer_size_(buffer_size),
        seed_(seed),
        seed2_(seed2),
        count_(count),
        reshuffle_each_iteration_(reshuffle_each_iteration) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = {
        CreateTensor<int64_t>(TensorShape({}), {buffer_size_}),
        CreateTensor<int64_t>(TensorShape({}), {seed_}),
        CreateTensor<int64_t>(TensorShape({}), {seed2_})};
    if (count_ != 1) {
      input_tensors.emplace_back(
          CreateTensor<int64_t>(TensorShape({}), {count_}));
    }
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    input_names->emplace_back(ShuffleDatasetOpBase::kInputDataset);
    input_names->emplace_back(ShuffleDatasetOpBase::kBufferSize);
    input_names->emplace_back(ShuffleDatasetOpBase::kSeed);
    input_names->emplace_back(ShuffleDatasetOpBase::kSeed2);
    if (count_ != 1) {
      input_names->emplace_back(ShuffleAndRepeatDatasetOp::kCount);
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("reshuffle_each_iteration",
                              reshuffle_each_iteration_);
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_3(mht_3_v, 263, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "dataset_type");

    if (count_ != 1) {
      return ShuffleAndRepeatDatasetOp::kDatasetType;
    }
    return ShuffleDatasetOp::kDatasetType;
  }

  int64_t count() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "count");
 return count_; }

 private:
  int64_t buffer_size_;
  int64_t seed_;
  int64_t seed2_;
  int64_t count_;
  bool reshuffle_each_iteration_;
};

class ShuffleDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: test shuffle_dataset with reshuffle_each_iteration = false.
ShuffleDatasetParams ShuffleDatasetParams1() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_5(mht_5_v, 289, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams1");

  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/3,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 2: test shuffle_dataset with reshuffle_each_iteration = true.
ShuffleDatasetParams ShuffleDatasetParams2() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_6(mht_6_v, 305, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams2");

  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 3: similar with the test case 2 but a smaller buffer size than
// the input dataset.
ShuffleDatasetParams ShuffleDatasetParams3() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_7(mht_7_v, 322, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams3");

  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/2,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 4: similar with the test case 2 but has different seeds.
ShuffleDatasetParams ShuffleDatasetParams4() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_8(mht_8_v, 338, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams4");

  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/10,
                              /*seed=*/2,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 5: test shuffle_dataset with buffer_size = 1 &
// reshuffle_each_iteration = true.
ShuffleDatasetParams ShuffleDatasetParams5() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_9(mht_9_v, 355, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams5");

  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/1,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 6: test shuffle_dataset with an empty input dataset.
ShuffleDatasetParams ShuffleDatasetParams6() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_10(mht_10_v, 371, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams6");

  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 7: test shuffle_and_repeat_dataset with buffer_size = 10 &
// count = 2.
ShuffleDatasetParams ShuffleDatasetParams7() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_11(mht_11_v, 388, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams7");

  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/2,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

// Test case 8: test shuffle_and_repeat_dataset with buffer_size = 10 &
// count = -1
ShuffleDatasetParams ShuffleDatasetParams8() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_12(mht_12_v, 405, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParams8");

  return ShuffleDatasetParams(RangeDatasetParams(0, 3, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/-1,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

ShuffleDatasetParams ShuffleDatasetParamsWithInvalidBufferSize() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_13(mht_13_v, 420, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleDatasetParamsWithInvalidBufferSize");

  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/-1,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

ShuffleDatasetParams ShuffleAndRepeatDatasetParamsWithInvalidBufferSize() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_14(mht_14_v, 435, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleAndRepeatDatasetParamsWithInvalidBufferSize");

  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/-1,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/2,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

ShuffleDatasetParams ShuffleAndRepeatDatasetParamsWithInvalidCount() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_op_testDTcc mht_15(mht_15_v, 450, "", "./tensorflow/core/kernels/data/shuffle_dataset_op_test.cc", "ShuffleAndRepeatDatasetParamsWithInvalidCount");

  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/0,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

template <typename T>
struct GetNextTestCase {
  T dataset_params;
  std::vector<Tensor> expected_shuffle_outputs;
  std::vector<Tensor> expected_reshuffle_outputs;
};

std::vector<GetNextTestCase<ShuffleDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/ShuffleDatasetParams1(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{2}, {3}, {0}, {5}, {6}, {4}, {7}, {8}, {9}, {1}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{2}, {3}, {0}, {5}, {6}, {4}, {7}, {8}, {9}, {1}})},
      {/*dataset_params=*/ShuffleDatasetParams2(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{2}, {6}, {1}, {3}, {9}, {5}, {0}, {8}, {7}, {4}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{1}, {6}, {0}, {5}, {2}, {7}, {4}, {3}, {9}, {8}})},
      {/*dataset_params=*/ShuffleDatasetParams3(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{0}, {2}, {1}, {3}, {5}, {6}, {4}, {7}, {8}, {9}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{1}, {0}, {2}, {3}, {4}, {5}, {6}, {7}, {9}, {8}})},
      {/*dataset_params=*/ShuffleDatasetParams4(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{3}, {0}, {8}, {1}, {5}, {4}, {7}, {2}, {6}, {9}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{4}, {6}, {9}, {0}, {1}, {8}, {2}, {7}, {3}, {5}})},
      {/*dataset_params=*/ShuffleDatasetParams5(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/ShuffleDatasetParams6(),
       /*expected_shuffle_outputs=*/{},
       /*expected_reshuffle_outputs=*/{}},
      {/*dataset_params=*/ShuffleDatasetParams7(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5},
                             {9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5},
            {9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5}})},
      {/*dataset_params=*/ShuffleDatasetParams8(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0},
            {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0},
            {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}})}};
}

class ParameterizedGetNextTest : public ShuffleDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<ShuffleDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> shuffled_out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    shuffled_out_tensors.insert(shuffled_out_tensors.end(), next.begin(),
                                next.end());
    // For the forever-repeat case, we test only a finite number of steps of
    // the infinite sequence.
    if (test_case.dataset_params.count() == -1 &&
        shuffled_out_tensors.size() ==
            test_case.expected_shuffle_outputs.size()) {
      break;
    }
  }

  // Reshuffle the dataset.
  end_of_sequence = false;
  TF_ASSERT_OK(dataset_->MakeIterator(
      iterator_ctx_.get(), /*parent=*/nullptr,
      test_case.dataset_params.iterator_prefix(), &iterator_));
  std::vector<Tensor> reshuffled_out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    reshuffled_out_tensors.insert(reshuffled_out_tensors.end(), next.begin(),
                                  next.end());
    // For the forever-repeat case, we test only a finite number of steps of
    // the infinite sequence.
    if (test_case.dataset_params.count() == -1 &&
        reshuffled_out_tensors.size() ==
            test_case.expected_shuffle_outputs.size()) {
      break;
    }
  }

  TF_EXPECT_OK(ExpectEqual(shuffled_out_tensors,
                           test_case.expected_shuffle_outputs,
                           /*compare_order=*/true));
  TF_EXPECT_OK(ExpectEqual(reshuffled_out_tensors,
                           test_case.expected_reshuffle_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_CASE_P(ShuffleDatasetOpTest, ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

std::vector<DatasetNodeNameTestCase<ShuffleDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*expected_node_name=*/kShuffleNodeName},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*expected_node_name=*/kShuffleAndRepeatNodeName}};
}

DATASET_NODE_NAME_TEST_P(ShuffleDatasetOpTest, ShuffleDatasetParams,
                         DatasetNodeNameTestCases())

std::vector<DatasetTypeStringTestCase<ShuffleDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               ShuffleDatasetOp::kDatasetType)},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*expected_dataset_type_string=*/
           name_utils::OpName(ShuffleAndRepeatDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(ShuffleDatasetOpTest, ShuffleDatasetParams,
                           DatasetTypeStringTestCases())

TEST_F(ShuffleDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ShuffleDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<ShuffleDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams2(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams3(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams4(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams5(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams6(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*expected_cardinality=*/20},
          {/*dataset_params=*/ShuffleDatasetParams8(),
           /*expected_cardinality=*/kInfiniteCardinality}};
}

DATASET_CARDINALITY_TEST_P(ShuffleDatasetOpTest, ShuffleDatasetParams,
                           CardinalityTestCases())

TEST_F(ShuffleDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ShuffleDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(ShuffleDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ShuffleDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<Tensor> expected_shuffle_outputs;
};

std::vector<IteratorSaveAndRestoreTestCase<ShuffleDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{2}, {3}, {0}, {5}, {6}, {4}, {7}, {8}, {9}, {1}})},
          {/*dataset_params=*/ShuffleDatasetParams2(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{2}, {6}, {1}, {3}, {9}, {5}, {0}, {8}, {7}, {4}})},
          {/*dataset_params=*/ShuffleDatasetParams3(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{0}, {2}, {1}, {3}, {5}, {6}, {4}, {7}, {8}, {9}})},
          {/*dataset_params=*/ShuffleDatasetParams4(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{3}, {0}, {8}, {1}, {5}, {4}, {7}, {2}, {6}, {9}})},
          {/*dataset_params=*/ShuffleDatasetParams5(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
          {/*dataset_params=*/ShuffleDatasetParams6(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/{}},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*breakpoints=*/{0, 5, 22},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5},
                {9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5}})},
          {/*dataset_params=*/ShuffleDatasetParams8(),
           /*breakpoints=*/{0, 5, 20},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0},
                {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}})}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public ShuffleDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<ShuffleDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
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

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_shuffle_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_CASE_P(ShuffleDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(ShuffleDatasetOpTest, InvalidArguments) {
  std::vector<ShuffleDatasetParams> dataset_params_vec(
      {ShuffleDatasetParamsWithInvalidBufferSize(),
       ShuffleAndRepeatDatasetParamsWithInvalidBufferSize(),
       ShuffleAndRepeatDatasetParamsWithInvalidCount()});
  for (const auto& dataset_params : dataset_params_vec) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
