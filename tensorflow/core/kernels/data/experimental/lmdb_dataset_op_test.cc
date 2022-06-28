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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/lmdb_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "lmdb_dataset";
constexpr char kIteratorPrefix[] = "Iterator";
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr char kDataFileName[] = "data_bigendian.mdb";
#else
constexpr char kDataFileName[] = "data.mdb";
#endif
constexpr char kDataFileLoc[] = "core/lib/lmdb/testdata";

class LMDBDatasetParams : public DatasetParams {
 public:
  LMDBDatasetParams(std::vector<tstring> filenames,
                    DataTypeVector output_dtypes,
                    std::vector<PartialTensorShape> output_shapes,
                    string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      kNodeName),
        filenames_(CreateTensor<tstring>(
            TensorShape({static_cast<int>(filenames.size())}), filenames)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("node_name: \"" + node_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "LMDBDatasetParams");
}

  std::vector<Tensor> GetInputTensors() const override { return {filenames_}; }

  Status GetInputNames(std::vector<string>* input_names) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "GetInputNames");

    *input_names = {LMDBDatasetOp::kFileNames};
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attributes) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "GetAttributes");

    *attributes = {{LMDBDatasetOp::kOutputTypes, output_dtypes_},
                   {LMDBDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  string dataset_type() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "dataset_type");
 return LMDBDatasetOp::kDatasetType; }

 private:
  // Names of binary database files to read, boxed up inside a Tensor of
  // strings
  Tensor filenames_;
};

class LMDBDatasetOpTest : public DatasetOpsTestBase {};

// Copy our test data file to the current test program's temporary
// directory, and return the full path to the copied file.
// This copying is necessary because LMDB creates lock files adjacent
// to files that it reads.
tstring MaybeCopyDataFile() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "MaybeCopyDataFile");

  tstring src_loc =
      io::JoinPath(testing::TensorFlowSrcRoot(), kDataFileLoc, kDataFileName);
  tstring dest_loc = io::JoinPath(testing::TmpDir(), kDataFileName);

  FileSystem* fs;  // Pointer to singleton
  TF_EXPECT_OK(Env::Default()->GetFileSystemForFile(src_loc, &fs));

  // FileSystem::FileExists currently returns Status::OK() if the file
  // exists and errors::NotFound() if the file doesn't exist. There's no
  // indication in the code or docs about whether other error codes may be
  // added in the future, so we code defensively here.
  Status exists_status = fs->FileExists(dest_loc);
  if (exists_status.code() == error::NOT_FOUND) {
    TF_EXPECT_OK(fs->CopyFile(src_loc, dest_loc));
  } else {
    TF_EXPECT_OK(exists_status);
  }

  return dest_loc;
}

LMDBDatasetParams SingleValidInput() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "SingleValidInput");

  return {/*filenames=*/{MaybeCopyDataFile()},
          /*output_dtypes=*/{DT_STRING, DT_STRING},
          /*output_shapes=*/{PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams TwoValidInputs() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_6(mht_6_v, 288, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "TwoValidInputs");

  return {/*filenames*/ {MaybeCopyDataFile(), MaybeCopyDataFile()},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams EmptyInput() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_7(mht_7_v, 298, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "EmptyInput");

  return {/*filenames*/ {},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams InvalidPathAtStart() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_8(mht_8_v, 308, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "InvalidPathAtStart");

  return {/*filenames*/ {"This is not a valid filename", MaybeCopyDataFile()},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

LMDBDatasetParams InvalidPathInMiddle() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSlmdb_dataset_op_testDTcc mht_9(mht_9_v, 318, "", "./tensorflow/core/kernels/data/experimental/lmdb_dataset_op_test.cc", "InvalidPathInMiddle");

  return {/*filenames*/ {MaybeCopyDataFile(), "This is not a valid filename",
                         MaybeCopyDataFile()},
          /*output_dtypes*/ {DT_STRING, DT_STRING},
          /*output_shapes*/ {PartialTensorShape({}), PartialTensorShape({})},
          /*node_name=*/kNodeName};
}

// The tensors we expect to see each time we read through the input data file.
std::vector<GetNextTestCase<LMDBDatasetParams>> GetNextTestCases() {
  const std::vector<Tensor> kFileOutput = CreateTensors<tstring>(
      TensorShape({}),
      {
          // Each call to GetNext() produces two scalar string tensors, but the
          // test harness expects to receive a flat vector
          {"0"}, {"a"},  //
          {"1"}, {"b"},  //
          {"2"}, {"c"},  //
          {"3"}, {"d"},  //
          {"4"}, {"e"},  //
          {"5"}, {"f"},  //
          {"6"}, {"g"},  //
          {"7"}, {"h"},  //
          {"8"}, {"i"},  //
          {"9"}, {"j"},  //
      });

  // STL vectors don't have a "concatenate two vectors into a new vector"
  // operation, so...
  std::vector<Tensor> output_twice;
  output_twice.insert(output_twice.end(), kFileOutput.cbegin(),
                      kFileOutput.cend());
  output_twice.insert(output_twice.end(), kFileOutput.cbegin(),
                      kFileOutput.cend());

  return {
      {/*dataset_params=*/SingleValidInput(), /*expected_outputs=*/kFileOutput},
      {/*dataset_params=*/TwoValidInputs(), /*expected_outputs=*/output_twice},
      {/*dataset_params=*/EmptyInput(), /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                         GetNextTestCases());

TEST_F(LMDBDatasetOpTest, InvalidPathAtStart) {
  auto dataset_params = InvalidPathAtStart();
  TF_ASSERT_OK(Initialize(dataset_params));

  // Errors about invalid files are only raised when attempting to read data.
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  std::vector<Tensor> next;

  Status get_next_status =
      iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence);

  EXPECT_EQ(get_next_status.code(), error::INVALID_ARGUMENT);
}

TEST_F(LMDBDatasetOpTest, InvalidPathInMiddle) {
  auto dataset_params = InvalidPathInMiddle();
  TF_ASSERT_OK(Initialize(dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;

  // First 10 rows should be ok
  for (int i = 0; i < 10; ++i) {
    std::vector<Tensor> next;
    TF_ASSERT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    EXPECT_FALSE(end_of_sequence);
  }

  // Next read operation should raise an error
  std::vector<Tensor> next;
  Status get_next_status =
      iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence);
  EXPECT_EQ(get_next_status.code(), error::INVALID_ARGUMENT);
}

std::vector<DatasetNodeNameTestCase<LMDBDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                         DatasetNodeNameTestCases());

std::vector<DatasetTypeStringTestCase<LMDBDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               LMDBDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                           DatasetTypeStringTestCases());

std::vector<DatasetOutputDtypesTestCase<LMDBDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_output_dtypes=*/{DT_STRING, DT_STRING}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                             DatasetOutputDtypesTestCases());

std::vector<DatasetOutputShapesTestCase<LMDBDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                             DatasetOutputShapesTestCases());

std::vector<CardinalityTestCase<LMDBDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_cardinality=*/kUnknownCardinality}};
}

DATASET_CARDINALITY_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                           CardinalityTestCases());

std::vector<IteratorOutputDtypesTestCase<LMDBDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_output_dtypes=*/{DT_STRING, DT_STRING}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                              IteratorOutputDtypesTestCases());

std::vector<IteratorOutputShapesTestCase<LMDBDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_output_shapes=*/{PartialTensorShape({}),
                                       PartialTensorShape({})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                              IteratorOutputShapesTestCases());

std::vector<IteratorPrefixTestCase<LMDBDatasetParams>>
IteratorOutputPrefixTestCases() {
  return {{/*dataset_params=*/SingleValidInput(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               LMDBDatasetOp::kDatasetType, kIteratorPrefix)}};
}

ITERATOR_PREFIX_TEST_P(LMDBDatasetOpTest, LMDBDatasetParams,
                       IteratorOutputPrefixTestCases());

// No test of save and restore; save/restore functionality is not implemented
// for this dataset.

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
