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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/fixed_length_record_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "fixed_length_record_dataset";
constexpr int kOpVersion = 2;

tstring LocalTempFilename() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_0(mht_0_v, 192, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "LocalTempFilename");

  std::string path;
  CHECK(Env::Default()->LocalTempFilename(&path));
  return tstring(path);
}

class FixedLengthRecordDatasetParams : public DatasetParams {
 public:
  FixedLengthRecordDatasetParams(const std::vector<tstring>& filenames,
                                 int64_t header_bytes, int64_t record_bytes,
                                 int64_t footer_bytes, int64_t buffer_size,
                                 CompressionType compression_type,
                                 string node_name)
      : DatasetParams({DT_STRING}, {PartialTensorShape({})},
                      std::move(node_name)),
        filenames_(filenames),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        buffer_size_(buffer_size),
        compression_type_(compression_type) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "FixedLengthRecordDatasetParams");

    op_version_ = 2;
  }

  std::vector<Tensor> GetInputTensors() const override {
    int num_files = filenames_.size();
    return {
        CreateTensor<tstring>(TensorShape({num_files}), filenames_),
        CreateTensor<int64_t>(TensorShape({}), {header_bytes_}),
        CreateTensor<int64_t>(TensorShape({}), {record_bytes_}),
        CreateTensor<int64_t>(TensorShape({}), {footer_bytes_}),
        CreateTensor<int64_t>(TensorShape({}), {buffer_size_}),
        CreateTensor<tstring>(TensorShape({}), {ToString(compression_type_)})};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "GetInputNames");

    input_names->clear();
    *input_names = {FixedLengthRecordDatasetOp::kFileNames,
                    FixedLengthRecordDatasetOp::kHeaderBytes,
                    FixedLengthRecordDatasetOp::kRecordBytes,
                    FixedLengthRecordDatasetOp::kFooterBytes,
                    FixedLengthRecordDatasetOp::kBufferSize,
                    FixedLengthRecordDatasetOp::kCompressionType};
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "GetAttributes");

    attr_vector->clear();
    attr_vector->emplace_back("metadata", "");
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_4(mht_4_v, 257, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "dataset_type");

    return FixedLengthRecordDatasetOp::kDatasetType;
  }

 private:
  std::vector<tstring> filenames_;
  int64_t header_bytes_;
  int64_t record_bytes_;
  int64_t footer_bytes_;
  int64_t buffer_size_;
  CompressionType compression_type_;
};

class FixedLengthRecordDatasetOpTest : public DatasetOpsTestBase {};

Status CreateTestFiles(const std::vector<tstring>& filenames,
                       const std::vector<string>& contents,
                       CompressionType compression_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "CreateTestFiles");

  if (filenames.size() != contents.size()) {
    return tensorflow::errors::InvalidArgument(
        "The number of files does not match with the contents");
  }
  if (compression_type == CompressionType::UNCOMPRESSED) {
    for (int i = 0; i < filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(WriteDataToFile(filenames[i], contents[i].data()));
    }
  } else {
    CompressionParams params;
    params.output_buffer_size = 10;
    params.compression_type = compression_type;
    for (int i = 0; i < filenames.size(); ++i) {
      TF_RETURN_IF_ERROR(
          WriteDataToFile(filenames[i], contents[i].data(), params));
    }
  }
  return Status::OK();
}

// Test case 1: multiple fixed-length record files with ZLIB compression.
FixedLengthRecordDatasetParams FixedLengthRecordDatasetParams1() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_6(mht_6_v, 302, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "FixedLengthRecordDatasetParams1");

  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  std::vector<string> contents = {
      absl::StrCat("HHHHH", "111", "222", "333", "FF"),
      absl::StrCat("HHHHH", "aaa", "bbb", "FF")};
  CompressionType compression_type = CompressionType::ZLIB;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }

  return FixedLengthRecordDatasetParams(filenames,
                                        /*header_bytes=*/5,
                                        /*record_bytes=*/3,
                                        /*footer_bytes=*/2,
                                        /*buffer_size=*/10,
                                        /*compression_type=*/compression_type,
                                        /*node_name=*/kNodeName);
}

// Test case 2: multiple fixed-length record files with GZIP compression.
FixedLengthRecordDatasetParams FixedLengthRecordDatasetParams2() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_7(mht_7_v, 326, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "FixedLengthRecordDatasetParams2");

  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  std::vector<string> contents = {
      absl::StrCat("HHHHH", "111", "222", "333", "FF"),
      absl::StrCat("HHHHH", "aaa", "bbb", "FF")};
  CompressionType compression_type = CompressionType::GZIP;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return FixedLengthRecordDatasetParams(filenames,
                                        /*header_bytes=*/5,
                                        /*record_bytes=*/3,
                                        /*footer_bytes=*/2,
                                        /*buffer_size=*/10,
                                        /*compression_type=*/compression_type,
                                        /*node_name=*/kNodeName);
}

// Test case 3: multiple fixed-length record files without compression.
FixedLengthRecordDatasetParams FixedLengthRecordDatasetParams3() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_op_testDTcc mht_8(mht_8_v, 349, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op_test.cc", "FixedLengthRecordDatasetParams3");

  std::vector<tstring> filenames = {LocalTempFilename(), LocalTempFilename()};
  std::vector<string> contents = {
      absl::StrCat("HHHHH", "111", "222", "333", "FF"),
      absl::StrCat("HHHHH", "aaa", "bbb", "FF")};
  CompressionType compression_type = CompressionType::UNCOMPRESSED;
  if (!CreateTestFiles(filenames, contents, compression_type).ok()) {
    VLOG(WARNING) << "Failed to create the test files: "
                  << absl::StrJoin(filenames, ", ");
  }
  return FixedLengthRecordDatasetParams(filenames,
                                        /*header_bytes=*/5,
                                        /*record_bytes=*/3,
                                        /*footer_bytes=*/2,
                                        /*buffer_size=*/10,
                                        /*compression_type=*/compression_type,
                                        /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<FixedLengthRecordDatasetParams>>
GetNextTestCases() {
  return {
      {/*dataset_params=*/FixedLengthRecordDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams2(),
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams3(),
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})}};
}

ITERATOR_GET_NEXT_TEST_P(FixedLengthRecordDatasetOpTest,
                         FixedLengthRecordDatasetParams, GetNextTestCases())

TEST_F(FixedLengthRecordDatasetOpTest, DatasetNodeName) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetTypeString) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(FixedLengthRecordDatasetOp::kDatasetType, params)));
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_STRING}));
}

TEST_F(FixedLengthRecordDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

TEST_F(FixedLengthRecordDatasetOpTest, Cardinality) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetCardinality(kUnknownCardinality));
}

TEST_F(FixedLengthRecordDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_STRING}));
}

TEST_F(FixedLengthRecordDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(FixedLengthRecordDatasetOpTest, IteratorPrefix) {
  auto dataset_params = FixedLengthRecordDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  name_utils::IteratorPrefixParams iterator_prefix_params;
  iterator_prefix_params.op_version = kOpVersion;
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      FixedLengthRecordDatasetOp::kDatasetType,
      dataset_params.iterator_prefix(), iterator_prefix_params)));
}

std::vector<IteratorSaveAndRestoreTestCase<FixedLengthRecordDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/FixedLengthRecordDatasetParams1(),
       /*breakpoints=*/{0, 2, 6},
       /*expected_outputs=*/
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams2(),
       /*breakpoints=*/{0, 2, 6},
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})},
      {/*dataset_params=*/FixedLengthRecordDatasetParams3(),
       /*breakpoints=*/{0, 2, 6},
       CreateTensors<tstring>(TensorShape({}),
                              {{"111"}, {"222"}, {"333"}, {"aaa"}, {"bbb"}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(FixedLengthRecordDatasetOpTest,
                                 FixedLengthRecordDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

}  // namespace
}  // namespace data
}  // namespace tensorflow
