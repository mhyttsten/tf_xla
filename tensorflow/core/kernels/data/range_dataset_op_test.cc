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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/range_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

class RangeDatasetOpTest : public DatasetOpsTestBase {};

RangeDatasetParams PositiveStepRangeDatasetParams() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/kernels/data/range_dataset_op_test.cc", "PositiveStepRangeDatasetParams");

  return RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3);
}

RangeDatasetParams NegativeStepRangeDatasetParams() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc mht_1(mht_1_v, 201, "", "./tensorflow/core/kernels/data/range_dataset_op_test.cc", "NegativeStepRangeDatasetParams");

  return RangeDatasetParams(/*start=*/10, /*stop=*/0, /*step=*/-3);
}

RangeDatasetParams ZeroStepRangeDatasetParams() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc mht_2(mht_2_v, 208, "", "./tensorflow/core/kernels/data/range_dataset_op_test.cc", "ZeroStepRangeDatasetParams");

  return RangeDatasetParams(/*start=*/10, /*stop=*/0, /*step=*/0);
}

RangeDatasetParams RangeDatasetParams1() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc mht_3(mht_3_v, 215, "", "./tensorflow/core/kernels/data/range_dataset_op_test.cc", "RangeDatasetParams1");

  return RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3,
                            /*output_dtypes=*/{DT_INT32});
}

RangeDatasetParams RangeDatasetParams2() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_op_testDTcc mht_4(mht_4_v, 223, "", "./tensorflow/core/kernels/data/range_dataset_op_test.cc", "RangeDatasetParams2");

  return RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3,
                            /*output_dtypes=*/{DT_INT64});
}

std::vector<GetNextTestCase<RangeDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})}};
}

ITERATOR_GET_NEXT_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                         GetNextTestCases())

TEST_F(RangeDatasetOpTest, DatasetNodeName) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(range_dataset_params.node_name()));
}

TEST_F(RangeDatasetOpTest, DatasetTypeString) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(RangeDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<RangeDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/RangeDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT32}},
          {/*dataset_params=*/RangeDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                             DatasetOutputDtypesTestCases())

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<RangeDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*expected_cardinality=*/4}};
}

DATASET_CARDINALITY_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<RangeDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/RangeDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT32}},
          {/*dataset_params=*/RangeDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                              IteratorOutputDtypesTestCases())

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(RangeDatasetOpTest, IteratorPrefix) {
  auto range_dataset_params = PositiveStepRangeDatasetParams();
  TF_ASSERT_OK(Initialize(range_dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      RangeDatasetOp::kDatasetType, range_dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<RangeDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/PositiveStepRangeDatasetParams(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})},
          {/*dataset_params=*/NegativeStepRangeDatasetParams(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(RangeDatasetOpTest, RangeDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(RangeDatasetOpTest, ZeroStep) {
  auto range_dataset_params = ZeroStepRangeDatasetParams();
  EXPECT_EQ(Initialize(range_dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(RangeDatasetOpTest, SplitProviderPositiveStep) {
  auto params = RangeDatasetParams(/*start=*/0, /*stop=*/10, /*step=*/3,
                                   /*output_dtypes=*/{DT_INT64});
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}), {{0}, {3}, {6}, {9}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/2, /*shard_index=*/1,
      CreateTensors<int64_t>(TensorShape({}), {{3}, {9}})));
}

TEST_F(RangeDatasetOpTest, SplitProviderNegativeStep) {
  auto params = RangeDatasetParams(/*start=*/10, /*stop=*/0, /*step=*/-3,
                                   /*output_dtypes=*/{DT_INT64});
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}), {{10}, {7}, {4}, {1}})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/2, /*shard_index=*/0,
      CreateTensors<int64_t>(TensorShape({}), {{10}, {4}})));
}

TEST_F(RangeDatasetOpTest, SplitProviderEmpty) {
  auto params = RangeDatasetParams(/*start=*/0, /*stop=*/0, /*step=*/1,
                                   /*output_dtypes=*/{DT_INT64});
  TF_ASSERT_OK(InitializeRuntime(params));
  TF_EXPECT_OK(CheckSplitProviderFullIteration(
      params, CreateTensors<int64_t>(TensorShape({}), {})));
  TF_EXPECT_OK(CheckSplitProviderShardedIteration(
      params, /*num_shards=*/3, /*shard_index=*/2,
      CreateTensors<int64_t>(TensorShape({}), {})));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
