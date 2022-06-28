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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/zip_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "zip_dataset";

class ZipDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ZipDatasetParams(std::vector<T> input_dataset_params,
                   DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   int num_input_datasets, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_input_datasets_(num_input_datasets) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/data/zip_dataset_op_test.cc", "ZipDatasetParams");

    for (auto& params : input_dataset_params) {
      input_dataset_params_.push_back(absl::make_unique<T>(params));
    }

    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params[0].dataset_type(),
                                   input_dataset_params[0].iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override { return {}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/data/zip_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    for (int i = 0; i < num_input_datasets_; ++i) {
      input_names->emplace_back(
          absl::StrCat(ZipDatasetOp::kDatasetType, "_", i));
    }
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/data/zip_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("N", num_input_datasets_);
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/kernels/data/zip_dataset_op_test.cc", "dataset_type");
 return ZipDatasetOp::kDatasetType; }

 private:
  int32 num_input_datasets_;
};

class ZipDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: the input datasets with same number of outputs.
ZipDatasetParams ZipDatasetParams1() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc mht_4(mht_4_v, 255, "", "./tensorflow/core/kernels/data/zip_dataset_op_test.cc", "ZipDatasetParams1");

  return ZipDatasetParams(
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 13, 1)},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

// Test case 2: the input datasets with different number of outputs.
ZipDatasetParams ZipDatasetParams2() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSzip_dataset_op_testDTcc mht_5(mht_5_v, 269, "", "./tensorflow/core/kernels/data/zip_dataset_op_test.cc", "ZipDatasetParams2");

  return ZipDatasetParams(
      std::vector<RangeDatasetParams>{RangeDatasetParams(0, 3, 1),
                                      RangeDatasetParams(10, 15, 1)},
      /*output_dtypes=*/{DT_INT64, DT_INT64},
      /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
      /*num_input_datasets=*/2,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ZipDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})}};
}

ITERATOR_GET_NEXT_TEST_P(ZipDatasetOpTest, ZipDatasetParams, GetNextTestCases())

TEST_F(ZipDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ZipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ZipDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ZipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(ZipDatasetOp::kDatasetType)));
}

std::vector<DatasetOutputDtypesTestCase<ZipDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<ZipDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ZipDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<ZipDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_dtypes=*/{DT_INT64, DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<ZipDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ZipDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = ZipDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ZipDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ZipDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ZipDatasetParams1(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})},
          {/*dataset_params=*/ZipDatasetParams2(),
           /*breakpoints=*/{0, 1, 4},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape{},
                                  {{0}, {10}, {1}, {11}, {2}, {12}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ZipDatasetOpTest, ZipDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
