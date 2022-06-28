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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc() {
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
#include "tensorflow/core/kernels/data/cache_dataset_ops.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "cache_dataset";
constexpr char kFileDatasetPrefix[] = "File";
constexpr char kMemoryDatasetPrefix[] = "Memory";

class CacheDatasetParams : public DatasetParams {
 public:
  template <typename T>
  CacheDatasetParams(T input_dataset_params, string filename,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        filename_(filename) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "CacheDatasetParams");

    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    Tensor filename_tensor =
        CreateTensor<tstring>(TensorShape({}), {filename_});
    return {filename_tensor};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "GetInputNames");

    *input_names = {CacheDatasetOp::kInputDataset, CacheDatasetOp::kFileName};
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "GetAttributes");

    *attr_vector = {{"output_types", output_dtypes_},
                    {"output_shapes", output_shapes_},
                    {"metadata", ""}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "dataset_type");
 return CacheDatasetOp::kDatasetType; }

  string filename() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_4(mht_4_v, 249, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "filename");
 return filename_; }

 private:
  string filename_;
};

class CacheDatasetOpTest : public DatasetOpsTestBase {
 public:
  Status Initialize(const DatasetParams& dataset_params) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_5(mht_5_v, 260, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "Initialize");

    TF_RETURN_IF_ERROR(DatasetOpsTestBase::Initialize(dataset_params));
    auto params = static_cast<const CacheDatasetParams&>(dataset_params);
    cache_filename_ = params.filename();
    return Status::OK();
  }

  ~CacheDatasetOpTest() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_6(mht_6_v, 270, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "~CacheDatasetOpTest");

    if (!cache_filename_.empty()) {
      std::vector<string> cache_files;
      Status s = device_->env()->GetMatchingPaths(
          strings::StrCat(cache_filename_, "*"), &cache_files);
      if (!s.ok()) {
        LOG(WARNING) << "Failed to get matching files on " << cache_filename_
                     << "* : " << s.ToString();
      }
      for (const string& path : cache_files) {
        s = device_->env()->DeleteFile(path);
        if (!s.ok()) {
          LOG(WARNING) << "Failed to delete " << path << " : " << s.ToString();
        }
      }
    }
  }

 protected:
  tstring cache_filename_;
};

// Test case 1: cache data in file.
CacheDatasetParams CacheDatasetParams1() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_7(mht_7_v, 296, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "CacheDatasetParams1");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return CacheDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*filename=*/io::JoinPath(testing::TmpDir(), "cache_data"),
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3, 1})}, kNodeName);
}

// Test case 2: cache empty data in file.
CacheDatasetParams CacheDatasetParams2() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_8(mht_8_v, 312, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "CacheDatasetParams2");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{0}, {})},
      /*node_name=*/"tensor_slice");
  return CacheDatasetParams(
      std::move(tensor_slice_dataset_params),
      /*filename=*/io::JoinPath(testing::TmpDir(), "cache_data"),
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({})}, kNodeName);
}

// Test case 3: cache data in memory.
CacheDatasetParams CacheDatasetParams3() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_9(mht_9_v, 327, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "CacheDatasetParams3");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{3, 3, 1},
                                            {0, 1, 2, 3, 4, 5, 6, 7, 8})},
      /*node_name=*/"tensor_slice");
  return CacheDatasetParams(std::move(tensor_slice_dataset_params),
                            /*filename=*/"",
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({3, 1})},
                            kNodeName);
}

// Test case 4: cache empty data in memory.
CacheDatasetParams CacheDatasetParams4() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPScache_dataset_ops_testDTcc mht_10(mht_10_v, 343, "", "./tensorflow/core/kernels/data/cache_dataset_ops_test.cc", "CacheDatasetParams4");

  auto tensor_slice_dataset_params = TensorSliceDatasetParams(
      /*components=*/{CreateTensor<int64_t>(TensorShape{0}, {})},
      /*node_name=*/"tensor_slice");
  return CacheDatasetParams(std::move(tensor_slice_dataset_params),
                            /*filename=*/"",
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            kNodeName);
}

std::vector<GetNextTestCase<CacheDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/CacheDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({3, 1}),
                                  {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/CacheDatasetParams2(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/CacheDatasetParams3(),
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({3, 1}),
                                  {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/CacheDatasetParams4(),
           /*expected_outputs=*/{}}};
}

class ParameterizedGetNextTest : public CacheDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<CacheDatasetParams>> {};

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

  // Test the read mode.
  TF_ASSERT_OK(dataset_->MakeIterator(
      iterator_ctx_.get(), /*parent=*/nullptr,
      test_case.dataset_params.iterator_prefix(), &iterator_));
  end_of_sequence = false;
  out_tensors.clear();
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }
  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_SUITE_P(CacheDatasetOpTest, ParameterizedGetNextTest,
                         ::testing::ValuesIn(GetNextTestCases()));

TEST_F(CacheDatasetOpTest, DatasetNodeName) {
  auto dataset_params = CacheDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(CacheDatasetOpTest, DatasetTypeString) {
  auto dataset_params = CacheDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(CacheDatasetOp::kDatasetType)));
}

TEST_F(CacheDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = CacheDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<CacheDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/CacheDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({3, 1})}},
          {/*dataset_params=*/CacheDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/CacheDatasetParams3(),
           /*expected_output_shapes=*/
           /*expected_output_shapes=*/{PartialTensorShape({3, 1})}},
          {/*dataset_params=*/CacheDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(CacheDatasetOpTest, CacheDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<CacheDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/CacheDatasetParams1(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/CacheDatasetParams2(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/CacheDatasetParams3(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/CacheDatasetParams4(),
           /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(CacheDatasetOpTest, CacheDatasetParams,
                           CardinalityTestCases())

TEST_F(CacheDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = CacheDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<CacheDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/CacheDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({3, 1})}},
          {/*dataset_params=*/CacheDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({})}},
          {/*dataset_params=*/CacheDatasetParams3(),
           /*expected_output_shapes=*/
           /*expected_output_shapes=*/{PartialTensorShape({3, 1})}},
          {/*dataset_params=*/CacheDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(CacheDatasetOpTest, CacheDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(CacheDatasetOpTest, IteratorPrefix) {
  auto dataset_params = CacheDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams iterator_prefix_params;
  iterator_prefix_params.dataset_prefix =
      cache_filename_.empty() ? kMemoryDatasetPrefix : kFileDatasetPrefix;
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      CacheDatasetOp::kDatasetType, dataset_params.iterator_prefix(),
      iterator_prefix_params)));
}

std::vector<IteratorSaveAndRestoreTestCase<CacheDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/CacheDatasetParams1(),
           /*breakpoints=*/{0, 2, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({3, 1}),
                                  {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/CacheDatasetParams2(),
           /*breakpoints=*/{0, 2, 4, 11},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/CacheDatasetParams3(),
           /*breakpoints=*/{0, 2, 4, 11},
           /*expected_outputs=*/
           CreateTensors<int64_t>(TensorShape({3, 1}),
                                  {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/CacheDatasetParams4(),
           /*breakpoints=*/{0, 2, 4, 11},
           /*expected_outputs=*/{}}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public CacheDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<CacheDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, SaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  // For MemoryIterator in the read mode, the cache needs to be completed
  // before it has been read.
  if (cache_filename_.empty()) {
    while (!end_of_sequence) {
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
    }
    end_of_sequence = false;
    out_tensors.clear();
    TF_ASSERT_OK(dataset_->MakeIterator(
        iterator_ctx_.get(), /*parent=*/nullptr,
        test_case.dataset_params.iterator_prefix(), &iterator_));
  }

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));
  int cur_iteration = 0;
  auto expected_outputs_it = test_case.expected_outputs.begin();
  for (int breakpoint : test_case.breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration <= breakpoint) {
      out_tensors.clear();
      TF_EXPECT_OK(iterator_->GetNext(iterator_ctx_.get(), &out_tensors,
                                      &end_of_sequence));
      if (!end_of_sequence) {
        EXPECT_LT(expected_outputs_it, test_case.expected_outputs.end());
        TF_EXPECT_OK(ExpectEqual(out_tensors.back(), *expected_outputs_it));
        expected_outputs_it++;
      }
      cur_iteration++;
    }

    if (breakpoint >= dataset_->Cardinality()) {
      EXPECT_TRUE(end_of_sequence);
      EXPECT_EQ(expected_outputs_it, test_case.expected_outputs.end());
    } else {
      EXPECT_FALSE(end_of_sequence);
    }
  }
}

INSTANTIATE_TEST_CASE_P(CacheDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

}  // namespace
}  // namespace data
}  // namespace tensorflow
