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
class MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/tensor_slice_writer.h"

#include <array>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace tensorflow {

namespace checkpoint {

class TensorSliceWriteTestHelper {
 public:
  static void CheckEntries(const string& fname);
  static void GetData(TensorSliceReader::Table* table, const string& name,
                      const TensorSlice& slice, SavedSlice* ss);
};

namespace {

// Testing that an array is what is expected
void ExpectIdenticalFloatArrays(const float* expected, int size,
                                const float* actual) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/util/tensor_slice_writer_test.cc", "ExpectIdenticalFloatArrays");

  // TODO(yangke): copy some of the Dump* functions over
  //  LOG(INFO) << "Expected = " << DumpFloatArray(expected, size);
  //  LOG(INFO) << "Actual   = " << DumpFloatArray(actual, size);
  for (int i = 0; i < size; ++i) {
    EXPECT_NEAR(expected[i], actual[i], 1e-6);
  }
}

template <typename T, typename U>
void ExpectIdenticalIntArrays(const T* expected, int size, const U* actual) {
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], static_cast<T>(actual[i]));
  }
}

// Nifty routine to get the size of an array
template <typename T, unsigned SIZE>
inline size_t ArraySize(const T (&v)[SIZE]) {
  return SIZE;
}

// A simple test on writing a few tensor slices
// TODO(yangke): refactor into smaller tests: will do as we add more stuff to
// the writer.
TEST(TensorSliceWriteTest, SimpleWrite) {
  const string filename = io::JoinPath(testing::TmpDir(), "checkpoint");

  TensorSliceWriter writer(filename, CreateTableTensorSliceBuilder);

  // Add some int32 tensor slices
  {
    TensorShape shape({5, 10});
    TensorSlice slice = TensorSlice::ParseOrDie("-:0,1");
    const int32 data[] = {0, 1, 2, 3, 4};
    TF_CHECK_OK(writer.Add("test", shape, slice, data));
  }

  // Two slices share the same tensor name
  {
    TensorShape shape({5, 10});
    TensorSlice slice = TensorSlice::ParseOrDie("-:3,1");
    const int32 data[] = {10, 11, 12, 13, 14};
    TF_CHECK_OK(writer.Add("test", shape, slice, data));
  }

  // Another slice from a different float tensor -- it has a different name and
  // should be inserted in front of the previous tensor
  {
    TensorShape shape({3, 2});
    TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    const float data[] = {1.2, 1.3, 1.4, 2.1, 2.2, 2.3};
    TF_CHECK_OK(writer.Add("AA", shape, slice, data));
  }

  // A slice with int64 data
  {
    TensorShape shape({5, 10});
    TensorSlice slice = TensorSlice::ParseOrDie("-:3,1");
    const int64_t data[] = {10, 11, 12, 13, 14};
    TF_CHECK_OK(writer.Add("int64", shape, slice, data));
  }

  // A slice with int16 data
  {
    TensorShape shape({5, 10});
    TensorSlice slice = TensorSlice::ParseOrDie("-:3,1");
    const int16 data[] = {10, 11, 12, 13, 14};
    TF_CHECK_OK(writer.Add("int16", shape, slice, data));
  }

  TF_CHECK_OK(writer.Finish());

  // Now we examine the checkpoint file manually.
  TensorSliceWriteTestHelper::CheckEntries(filename);
}

}  // namespace

void TensorSliceWriteTestHelper::GetData(TensorSliceReader::Table* table,
                                         const string& name,
                                         const TensorSlice& slice,
                                         SavedSlice* ss) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc mht_1(mht_1_v, 303, "", "./tensorflow/core/util/tensor_slice_writer_test.cc", "TensorSliceWriteTestHelper::GetData");

  string key = EncodeTensorNameSlice(name, slice);
  string value;
  EXPECT_TRUE(table->Get(key, &value));
  SavedTensorSlices sts;
  EXPECT_TRUE(ParseProtoUnlimited(&sts, value));
  EXPECT_FALSE(sts.has_meta());
  *ss = sts.data();
  EXPECT_EQ(name, ss->name());
  TensorSlice slice2(ss->slice());
  EXPECT_EQ(slice.DebugString(), slice2.DebugString());
}

void TensorSliceWriteTestHelper::CheckEntries(const string& fname) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc mht_2(mht_2_v, 320, "", "./tensorflow/core/util/tensor_slice_writer_test.cc", "TensorSliceWriteTestHelper::CheckEntries");

  TensorSliceReader::Table* tptr;
  TF_CHECK_OK(OpenTableTensorSliceReader(fname, &tptr));
  std::unique_ptr<TensorSliceReader::Table> table(tptr);
  CHECK_NOTNULL(table.get());

  // We expect a block of SavedTensorSlices
  string value;
  ASSERT_TRUE(table->Get(kSavedTensorSlicesKey, &value));
  {
    SavedTensorSlices sts;
    EXPECT_TRUE(ParseProtoUnlimited(&sts, value));
    // We also expect two entries for the tensors
    EXPECT_TRUE(sts.has_meta());
    EXPECT_EQ(4, sts.meta().tensor_size());
    // We should have written nontrivial version information
    EXPECT_LT(0, TF_CHECKPOINT_VERSION);
    EXPECT_EQ(TF_CHECKPOINT_VERSION, sts.meta().versions().producer());
    EXPECT_EQ(TF_CHECKPOINT_VERSION_MIN_CONSUMER,
              sts.meta().versions().min_consumer());
    // We don't expect any data in the first block.
    EXPECT_FALSE(sts.has_data());
    // The two tensors should be stored in the same order as they are first
    // created.
    {
      // The two slices of the "test" tensor
      const SavedSliceMeta& ssm = sts.meta().tensor(0);
      EXPECT_EQ("test", ssm.name());
      TensorShapeProto expected_shape_proto;
      protobuf::TextFormat::ParseFromString(
          "dim { size: 5 } "
          "dim { size: 10 }",
          &expected_shape_proto);
      EXPECT_EQ(ssm.shape().ShortDebugString(),
                expected_shape_proto.ShortDebugString());
      EXPECT_EQ(DT_INT32, ssm.type());
      EXPECT_EQ(2, ssm.slice_size());
      TensorSlice s0(ssm.slice(0));
      TensorSlice s1(ssm.slice(1));
      EXPECT_EQ("-:0,1", s0.DebugString());
      EXPECT_EQ("-:3,1", s1.DebugString());
    }
    {
      // The "AA" tensor
      const SavedSliceMeta& ssm = sts.meta().tensor(1);
      EXPECT_EQ("AA", ssm.name());
      TensorShapeProto expected_shape_proto;
      protobuf::TextFormat::ParseFromString(
          "dim { size: 3 } "
          "dim { size: 2 }",
          &expected_shape_proto);
      EXPECT_EQ(ssm.shape().ShortDebugString(),
                expected_shape_proto.ShortDebugString());
      EXPECT_EQ(DT_FLOAT, ssm.type());
      EXPECT_EQ(1, ssm.slice_size());
      TensorSlice s0(ssm.slice(0));
      EXPECT_EQ("-:-", s0.DebugString());
    }
    {
      // The "int64" tensor
      const SavedSliceMeta& ssm = sts.meta().tensor(2);
      EXPECT_EQ("int64", ssm.name());
      TensorShapeProto expected_shape_proto;
      protobuf::TextFormat::ParseFromString(
          "dim { size: 5 } "
          "dim { size: 10 }",
          &expected_shape_proto);
      EXPECT_EQ(ssm.shape().ShortDebugString(),
                expected_shape_proto.ShortDebugString());
      EXPECT_EQ(DT_INT64, ssm.type());
      EXPECT_EQ(1, ssm.slice_size());
      TensorSlice s0(ssm.slice(0));
      EXPECT_EQ("-:3,1", s0.DebugString());
    }
    {
      // The "int16" tensor
      const SavedSliceMeta& ssm = sts.meta().tensor(3);
      EXPECT_EQ("int16", ssm.name());
      TensorShapeProto expected_shape_proto;
      protobuf::TextFormat::ParseFromString(
          "dim { size: 5 } "
          "dim { size: 10 }",
          &expected_shape_proto);
      EXPECT_EQ(ssm.shape().ShortDebugString(),
                expected_shape_proto.ShortDebugString());
      EXPECT_EQ(DT_INT16, ssm.type());
      EXPECT_EQ(1, ssm.slice_size());
      TensorSlice s0(ssm.slice(0));
      EXPECT_EQ("-:3,1", s0.DebugString());
    }
  }

  // We expect 5 blocks of tensor data
  {
    // Block 1: we expect it to be the full slice of the "AA" tensor
    SavedSlice ss;
    GetData(table.get(), "AA", TensorSlice(2), &ss);
    const float data[] = {1.2, 1.3, 1.4, 2.1, 2.2, 2.3};
    EXPECT_EQ(ArraySize(data), ss.data().float_val_size());
    ExpectIdenticalFloatArrays(data, ArraySize(data),
                               ss.data().float_val().data());
  }

  {
    // Block 2: we expect it to be the first slice of the "test" tensor
    SavedSlice ss;
    GetData(table.get(), "test", TensorSlice({{0, -1}, {0, 1}}), &ss);
    const int32 data[] = {0, 1, 2, 3, 4};
    EXPECT_EQ(ArraySize(data), ss.data().int_val_size());
    ExpectIdenticalIntArrays(data, ArraySize(data), ss.data().int_val().data());
  }

  {
    // Block 3: we expect it to be the second slice of the "test" tensor
    SavedSlice ss;
    GetData(table.get(), "test", TensorSlice({{0, -1}, {3, 1}}), &ss);
    const int32 data[] = {10, 11, 12, 13, 14};
    EXPECT_EQ(ArraySize(data), ss.data().int_val_size());
    ExpectIdenticalIntArrays(data, ArraySize(data), ss.data().int_val().data());
  }

  {
    // Block 4: we expect it to be the slice of the "int64" tensor
    SavedSlice ss;
    GetData(table.get(), "int64", TensorSlice({{0, -1}, {3, 1}}), &ss);
    const int64_t data[] = {10, 11, 12, 13, 14};
    EXPECT_EQ(ArraySize(data), ss.data().int64_val_size());
    ExpectIdenticalIntArrays(data, ArraySize(data),
                             ss.data().int64_val().data());
  }

  {
    // Block 5: we expect it to be the slice of the "int16" tensor
    SavedSlice ss;
    GetData(table.get(), "int16", TensorSlice({{0, -1}, {3, 1}}), &ss);
    const int16 data[] = {10, 11, 12, 13, 14};
    EXPECT_EQ(ArraySize(data), ss.data().int_val_size());
    ExpectIdenticalIntArrays(data, ArraySize(data), ss.data().int_val().data());
  }
}

template <typename DT>
size_t BytesPerElementHelper(DT value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_slice_writer_testDTcc mht_3(mht_3_v, 465, "", "./tensorflow/core/util/tensor_slice_writer_test.cc", "BytesPerElementHelper");

  SavedSlice ss;
  std::array<DT, 1> lo_data;
  std::fill(lo_data.begin(), lo_data.end(), value);
  TF_EXPECT_OK(
      TensorSliceWriter::SaveData(lo_data.data(), lo_data.size(), &ss));
  size_t lo_byte_size = ss.ByteSizeLong();

  std::array<DT, 1001> hi_data;
  std::fill(hi_data.begin(), hi_data.end(), value);
  TF_EXPECT_OK(
      TensorSliceWriter::SaveData(hi_data.data(), hi_data.size(), &ss));
  size_t hi_byte_size = ss.ByteSizeLong();

  return (hi_byte_size - lo_byte_size) / (hi_data.size() - lo_data.size());
}

TEST(TensorSliceWriteTest, CheckpointSize) {
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_BOOL),
            BytesPerElementHelper<bool>(false));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_BOOL),
            BytesPerElementHelper<bool>(true));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_FLOAT),
            BytesPerElementHelper<float>(-1.0));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_DOUBLE),
            BytesPerElementHelper<double>(-1.0));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_COMPLEX64),
            BytesPerElementHelper<complex64>(-1.0));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_COMPLEX128),
            BytesPerElementHelper<complex128>(-1.0));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_INT32),
            BytesPerElementHelper<int32>(-1));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_INT64),
            BytesPerElementHelper<int64_t>(-1));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_UINT16),
            BytesPerElementHelper<uint16>(std::numeric_limits<uint16>::max()));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_UINT8),
            BytesPerElementHelper<uint8>(std::numeric_limits<uint8>::max()));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_INT8),
            BytesPerElementHelper<int8>(-1));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_INT16),
            BytesPerElementHelper<int16>(-1));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_QINT8),
            BytesPerElementHelper<qint8>(-1));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_QUINT8),
            BytesPerElementHelper<quint8>(std::numeric_limits<uint8>::max()));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_QINT32),
            BytesPerElementHelper<qint32>(-1));
  EXPECT_EQ(TensorSliceWriter::MaxBytesPerElement(DT_HALF),
            BytesPerElementHelper<Eigen::half>(Eigen::half(-1.0)));
}

TEST(TensorSliceWriteTest, SizeErrors) {
  const string filename = io::JoinPath(testing::TmpDir(), "checkpoint");

  TensorSliceWriter writer(filename, CreateTableTensorSliceBuilder);

  // Add a 300MB int8 tensor slice, which will fail because it expands to 3GB.
  {
    TensorShape shape({300, 1000000});
    TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    const std::vector<int8> data(300000000, -1);
    Status s = writer.Add("test1", shape, slice, data.data());
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "Tensor slice is too large to serialize"));
  }

  // Add a large string tensor slice, which will fail.
  {
    TensorShape shape({256, 1024});
    TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    const std::vector<tstring> data(256 * 1024, std::string(8192, 'f'));
    Status s = writer.Add("test2", shape, slice, data.data());
    EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "Tensor slice is too large to serialize"));
  }
}

}  // namespace checkpoint

}  // namespace tensorflow
