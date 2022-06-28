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
class MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/table_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/tensor_bundle.pb.h"
#include "tensorflow/core/util/tensor_bundle/byte_swap.h"

namespace tensorflow {
using ::testing::ElementsAre;

namespace {

// Prepend the current test case's working temporary directory to <prefix>
string Prefix(const string& prefix) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "Prefix");

  return strings::StrCat(testing::TmpDir(), "/", prefix);
}

// Construct a data input directory by prepending the test data root
// directory to <prefix>
string TestdataPrefix(const string& prefix) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "TestdataPrefix");

  return strings::StrCat(testing::TensorFlowSrcRoot(),
                         "/core/util/tensor_bundle/testdata/", prefix);
}

template <typename T>
Tensor Constant(T v, TensorShape shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "Constant");

  Tensor ret(DataTypeToEnum<T>::value, shape);
  ret.flat<T>().setConstant(v);
  return ret;
}

template <typename T>
Tensor Constant_2x3(T v) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "Constant_2x3");

  return Constant(v, TensorShape({2, 3}));
}

Tensor ByteSwap(Tensor t) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_4(mht_4_v, 251, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "ByteSwap");

  Tensor ret = tensor::DeepCopy(t);
  TF_EXPECT_OK(ByteSwapTensor(&ret));
  return ret;
}

// Assert that <reader> has a tensor under <key> matching <expected_val> in
// terms of both shape, dtype, and value
template <typename T>
void Expect(BundleReader* reader, const string& key,
            const Tensor& expected_val) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_5(mht_5_v, 265, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "Expect");

  // Tests for Contains().
  EXPECT_TRUE(reader->Contains(key));
  // Tests for LookupDtypeAndShape().
  DataType dtype;
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupDtypeAndShape(key, &dtype, &shape));
  EXPECT_EQ(expected_val.dtype(), dtype);
  EXPECT_EQ(expected_val.shape(), shape);
  // Tests for Lookup(), checking tensor contents.
  Tensor val(expected_val.dtype(), shape);
  TF_ASSERT_OK(reader->Lookup(key, &val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

template <class T>
void ExpectVariant(BundleReader* reader, const string& key,
                   const Tensor& expected_t) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_6(mht_6_v, 286, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "ExpectVariant");

  // Tests for Contains().
  EXPECT_TRUE(reader->Contains(key));
  // Tests for LookupDtypeAndShape().
  DataType dtype;
  TensorShape shape;
  TF_ASSERT_OK(reader->LookupDtypeAndShape(key, &dtype, &shape));
  // Tests for Lookup(), checking tensor contents.
  EXPECT_EQ(expected_t.dtype(), dtype);
  EXPECT_EQ(expected_t.shape(), shape);
  Tensor actual_t(dtype, shape);
  TF_ASSERT_OK(reader->Lookup(key, &actual_t));
  for (int i = 0; i < expected_t.NumElements(); i++) {
    Variant actual_var = actual_t.flat<Variant>()(i);
    Variant expected_var = expected_t.flat<Variant>()(i);
    EXPECT_EQ(actual_var.TypeName(), expected_var.TypeName());
    auto* actual_val = actual_var.get<T>();
    auto* expected_val = expected_var.get<T>();
    EXPECT_EQ(*expected_val, *actual_val);
  }
}

template <typename T>
void ExpectNext(BundleReader* reader, const Tensor& expected_val) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_7(mht_7_v, 312, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "ExpectNext");

  EXPECT_TRUE(reader->Valid());
  reader->Next();
  TF_ASSERT_OK(reader->status());
  Tensor val;
  TF_ASSERT_OK(reader->ReadCurrent(&val));
  test::ExpectTensorEqual<T>(val, expected_val);
}

std::vector<string> AllTensorKeys(BundleReader* reader) {
  std::vector<string> ret;
  reader->Seek(kHeaderEntryKey);
  reader->Next();
  for (; reader->Valid(); reader->Next()) {
    ret.emplace_back(reader->key());
  }
  return ret;
}

// Writes out the metadata file of a bundle again, with the endianness marker
// bit flipped.
Status FlipEndiannessBit(const string& prefix) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_8(mht_8_v, 337, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "FlipEndiannessBit");

  Env* env = Env::Default();
  const string metadata_tmp_path = Prefix("some_tmp_path");
  std::unique_ptr<WritableFile> metadata_file;
  TF_RETURN_IF_ERROR(env->NewWritableFile(metadata_tmp_path, &metadata_file));
  // We create the builder lazily in case we run into an exception earlier, in
  // which case we'd forget to call Finish() and TableBuilder's destructor
  // would complain.
  std::unique_ptr<table::TableBuilder> builder;

  // Reads the existing metadata file, and fills the builder.
  {
    const string filename = MetaFilename(prefix);
    uint64 file_size;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    table::Table* table = nullptr;
    TF_RETURN_IF_ERROR(
        table::Table::Open(table::Options(), file.get(), file_size, &table));
    std::unique_ptr<table::Table> table_deleter(table);
    std::unique_ptr<table::Iterator> iter(table->NewIterator());

    // Reads the header entry.
    iter->Seek(kHeaderEntryKey);
    CHECK(iter->Valid());
    BundleHeaderProto header;
    CHECK(header.ParseFromArray(iter->value().data(), iter->value().size()));
    // Flips the endianness.
    if (header.endianness() == BundleHeaderProto::LITTLE) {
      header.set_endianness(BundleHeaderProto::BIG);
    } else {
      header.set_endianness(BundleHeaderProto::LITTLE);
    }
    builder.reset(
        new table::TableBuilder(table::Options(), metadata_file.get()));
    builder->Add(iter->key(), header.SerializeAsString());
    iter->Next();

    // Adds the non-header entries unmodified.
    for (; iter->Valid(); iter->Next())
      builder->Add(iter->key(), iter->value());
  }
  TF_RETURN_IF_ERROR(builder->Finish());
  TF_RETURN_IF_ERROR(env->RenameFile(metadata_tmp_path, MetaFilename(prefix)));
  return metadata_file->Close();
}

template <typename T>
void TestBasic() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_9(mht_9_v, 390, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "TestBasic");

  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("foo_003", Constant_2x3(T(3))));
    TF_EXPECT_OK(writer.Add("foo_000", Constant_2x3(T(0))));
    TF_EXPECT_OK(writer.Add("foo_002", Constant_2x3(T(2))));
    TF_EXPECT_OK(writer.Add("foo_001", Constant_2x3(T(1))));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "foo_000", Constant_2x3(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3(T(3)));
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    BundleWriter writer(Env::Default(), Prefix("bar"));
    TF_EXPECT_OK(writer.Add("bar_003", Constant_2x3(T(3))));
    TF_EXPECT_OK(writer.Add("bar_000", Constant_2x3(T(0))));
    TF_EXPECT_OK(writer.Add("bar_002", Constant_2x3(T(2))));
    TF_EXPECT_OK(writer.Add("bar_001", Constant_2x3(T(1))));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003"}));
    Expect<T>(&reader, "bar_003", Constant_2x3(T(3)));
    Expect<T>(&reader, "bar_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "bar_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "bar_000", Constant_2x3(T(0)));
  }
  {
    BundleReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  TF_ASSERT_OK(MergeBundles(Env::Default(), {Prefix("foo"), Prefix("bar")},
                            Prefix("merged")));
  {
    BundleReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003",
                             "foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "bar_000", Constant_2x3(T(0)));
    Expect<T>(&reader, "bar_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "bar_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "bar_003", Constant_2x3(T(3)));
    Expect<T>(&reader, "foo_000", Constant_2x3(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3(T(3)));
  }
  {
    BundleReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    ExpectNext<T>(&reader, Constant_2x3(T(0)));
    ExpectNext<T>(&reader, Constant_2x3(T(1)));
    ExpectNext<T>(&reader, Constant_2x3(T(2)));
    ExpectNext<T>(&reader, Constant_2x3(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
}

// Type-specific subroutine of SwapBytes test below
template <typename T>
void TestByteSwap(const T* forward, const T* swapped, int array_len) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_10(mht_10_v, 491, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "TestByteSwap");

  auto bytes_per_elem = sizeof(T);

  // Convert the entire array at once
  std::unique_ptr<T[]> forward_copy(new T[array_len]);
  std::memcpy(forward_copy.get(), forward, array_len * bytes_per_elem);
  TF_EXPECT_OK(ByteSwapArray(reinterpret_cast<char*>(forward_copy.get()),
                             bytes_per_elem, array_len));
  for (int i = 0; i < array_len; i++) {
    EXPECT_EQ(forward_copy.get()[i], swapped[i]);
  }

  // Then the array wrapped in a tensor
  auto shape = TensorShape({array_len});
  auto dtype = DataTypeToEnum<T>::value;
  Tensor forward_tensor(dtype, shape);
  Tensor swapped_tensor(dtype, shape);
  std::memcpy(const_cast<char*>(forward_tensor.tensor_data().data()), forward,
              array_len * bytes_per_elem);
  std::memcpy(const_cast<char*>(swapped_tensor.tensor_data().data()), swapped,
              array_len * bytes_per_elem);
  TF_EXPECT_OK(ByteSwapTensor(&forward_tensor));
  test::ExpectTensorEqual<T>(forward_tensor, swapped_tensor);
}

// Unit test of the byte-swapping operations that TensorBundle uses.
TEST(TensorBundleTest, SwapBytes) {
  // A bug in the compiler on MacOS causes ByteSwap() and FlipEndiannessBit()
  // to be removed from the executable if they are only called from templated
  // functions. As a workaround, we make some dummy calls here.
  // TODO(frreiss): Remove this workaround when the compiler bug is fixed.
  ByteSwap(Constant_2x3<int>(42));
  EXPECT_NE(Status::OK(), FlipEndiannessBit(Prefix("not_a_valid_prefix")));

  // Test patterns, manually swapped so that we aren't relying on the
  // correctness of our own byte-swapping macros when testing those macros.
  // At least one of the entries in each list has the sign bit set when
  // interpreted as a signed int.
  const int arr_len_16 = 4;
  const uint16_t forward_16[] = {0x1de5, 0xd017, 0xf1ea, 0xc0a1};
  const uint16_t swapped_16[] = {0xe51d, 0x17d0, 0xeaf1, 0xa1c0};
  const int arr_len_32 = 2;
  const uint32_t forward_32[] = {0x0ddba115, 0xf01dab1e};
  const uint32_t swapped_32[] = {0x15a1db0d, 0x1eab1df0};
  const int arr_len_64 = 2;
  const uint64_t forward_64[] = {0xf005ba11caba1000, 0x5ca1ab1ecab005e5};
  const uint64_t swapped_64[] = {0x0010baca11ba05f0, 0xe505b0ca1eaba15c};

  // 16-bit types
  TestByteSwap(forward_16, swapped_16, arr_len_16);
  TestByteSwap(reinterpret_cast<const int16_t*>(forward_16),
               reinterpret_cast<const int16_t*>(swapped_16), arr_len_16);
  TestByteSwap(reinterpret_cast<const bfloat16*>(forward_16),
               reinterpret_cast<const bfloat16*>(swapped_16), arr_len_16);

  // 32-bit types
  TestByteSwap(forward_32, swapped_32, arr_len_32);
  TestByteSwap(reinterpret_cast<const int32_t*>(forward_32),
               reinterpret_cast<const int32_t*>(swapped_32), arr_len_32);
  TestByteSwap(reinterpret_cast<const float*>(forward_32),
               reinterpret_cast<const float*>(swapped_32), arr_len_32);

  // 64-bit types
  // Cast to uint64*/int64* to make DataTypeToEnum<T> happy
  TestByteSwap(reinterpret_cast<const uint64*>(forward_64),
               reinterpret_cast<const uint64*>(swapped_64), arr_len_64);
  TestByteSwap(reinterpret_cast<const int64_t*>(forward_64),
               reinterpret_cast<const int64_t*>(swapped_64), arr_len_64);
  TestByteSwap(reinterpret_cast<const double*>(forward_64),
               reinterpret_cast<const double*>(swapped_64), arr_len_64);

  // Complex types.
  // Logic for complex number handling is only in ByteSwapTensor, so don't test
  // ByteSwapArray
  const float* forward_float = reinterpret_cast<const float*>(forward_32);
  const float* swapped_float = reinterpret_cast<const float*>(swapped_32);
  const double* forward_double = reinterpret_cast<const double*>(forward_64);
  const double* swapped_double = reinterpret_cast<const double*>(swapped_64);
  Tensor forward_complex64 = Constant_2x3<complex64>(
      std::complex<float>(forward_float[0], forward_float[1]));
  Tensor swapped_complex64 = Constant_2x3<complex64>(
      std::complex<float>(swapped_float[0], swapped_float[1]));
  Tensor forward_complex128 = Constant_2x3<complex128>(
      std::complex<double>(forward_double[0], forward_double[1]));
  Tensor swapped_complex128 = Constant_2x3<complex128>(
      std::complex<double>(swapped_double[0], swapped_double[1]));

  TF_EXPECT_OK(ByteSwapTensor(&forward_complex64));
  test::ExpectTensorEqual<complex64>(forward_complex64, swapped_complex64);

  TF_EXPECT_OK(ByteSwapTensor(&forward_complex128));
  test::ExpectTensorEqual<complex128>(forward_complex128, swapped_complex128);
}

// Basic test of alternate-endianness support. Generates a bundle in
// the opposite of the current system's endianness and attempts to
// read the bundle back in. Does not exercise sharding or access to
// nonaligned tensors. Does cover the major access types exercised
// in TestBasic.
template <typename T>
void TestEndianness() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_11(mht_11_v, 594, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "TestEndianness");

  {
    // Write out a TensorBundle in the opposite of this host's endianness.
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("foo_003", ByteSwap(Constant_2x3<T>(T(3)))));
    TF_EXPECT_OK(writer.Add("foo_000", ByteSwap(Constant_2x3<T>(T(0)))));
    TF_EXPECT_OK(writer.Add("foo_002", ByteSwap(Constant_2x3<T>(T(2)))));
    TF_EXPECT_OK(writer.Add("foo_001", ByteSwap(Constant_2x3<T>(T(1)))));
    TF_ASSERT_OK(writer.Finish());
    TF_ASSERT_OK(FlipEndiannessBit(Prefix("foo")));
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "foo_000", Constant_2x3<T>(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3<T>(T(3)));
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    BundleWriter writer(Env::Default(), Prefix("bar"));
    TF_EXPECT_OK(writer.Add("bar_003", ByteSwap(Constant_2x3<T>(T(3)))));
    TF_EXPECT_OK(writer.Add("bar_000", ByteSwap(Constant_2x3<T>(T(0)))));
    TF_EXPECT_OK(writer.Add("bar_002", ByteSwap(Constant_2x3<T>(T(2)))));
    TF_EXPECT_OK(writer.Add("bar_001", ByteSwap(Constant_2x3<T>(T(1)))));
    TF_ASSERT_OK(writer.Finish());
    TF_ASSERT_OK(FlipEndiannessBit(Prefix("bar")));
  }
  {
    BundleReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003"}));
    Expect<T>(&reader, "bar_003", Constant_2x3<T>(T(3)));
    Expect<T>(&reader, "bar_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "bar_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "bar_000", Constant_2x3<T>(T(0)));
  }
  {
    BundleReader reader(Env::Default(), Prefix("bar"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  TF_ASSERT_OK(MergeBundles(Env::Default(), {Prefix("foo"), Prefix("bar")},
                            Prefix("merged")));
  {
    BundleReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"bar_000", "bar_001", "bar_002", "bar_003",
                             "foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<T>(&reader, "bar_000", Constant_2x3<T>(T(0)));
    Expect<T>(&reader, "bar_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "bar_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "bar_003", Constant_2x3<T>(T(3)));
    Expect<T>(&reader, "foo_000", Constant_2x3<T>(T(0)));
    Expect<T>(&reader, "foo_001", Constant_2x3<T>(T(1)));
    Expect<T>(&reader, "foo_002", Constant_2x3<T>(T(2)));
    Expect<T>(&reader, "foo_003", Constant_2x3<T>(T(3)));
  }
  {
    BundleReader reader(Env::Default(), Prefix("merged"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(0)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(1)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(2)));
    ExpectNext<T>(&reader, Constant_2x3<T>(T(3)));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
}

template <typename T>
void TestNonStandardShapes() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_12(mht_12_v, 697, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "TestNonStandardShapes");

  {
    BundleWriter writer(Env::Default(), Prefix("nonstandard"));
    TF_EXPECT_OK(writer.Add("scalar", Constant(T(0), TensorShape())));
    TF_EXPECT_OK(
        writer.Add("non_standard0", Constant(T(0), TensorShape({0, 1618}))));
    TF_EXPECT_OK(
        writer.Add("non_standard1", Constant(T(0), TensorShape({16, 0, 18}))));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("nonstandard"));
    TF_ASSERT_OK(reader.status());
    Expect<T>(&reader, "scalar", Constant(T(0), TensorShape()));
    Expect<T>(&reader, "non_standard0", Constant(T(0), TensorShape({0, 1618})));
    Expect<T>(&reader, "non_standard1",
              Constant(T(0), TensorShape({16, 0, 18})));
  }
}

// Writes a bundle to disk with a bad "version"; checks for "expected_error".
void VersionTest(const VersionDef& version, StringPiece expected_error) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_13(mht_13_v, 721, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "VersionTest");

  const string path = Prefix("version_test");
  {
    // Prepare an empty bundle with the given version information.
    BundleHeaderProto header;
    *header.mutable_version() = version;

    // Write the metadata file to disk.
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewWritableFile(MetaFilename(path), &file));
    table::TableBuilder builder(table::Options(), file.get());
    builder.Add(kHeaderEntryKey, header.SerializeAsString());
    TF_ASSERT_OK(builder.Finish());
  }
  // Read it back in and verify that we get the expected error.
  BundleReader reader(Env::Default(), path);
  EXPECT_TRUE(errors::IsInvalidArgument(reader.status()));
  EXPECT_TRUE(
      absl::StartsWith(reader.status().error_message(), expected_error));
}

}  // namespace

TEST(TensorBundleTest, Basic) {
  TestBasic<float>();
  TestBasic<double>();
  TestBasic<int32>();
  TestBasic<uint8>();
  TestBasic<int16>();
  TestBasic<int8>();
  TestBasic<complex64>();
  TestBasic<complex128>();
  TestBasic<int64_t>();
  TestBasic<bool>();
  TestBasic<qint32>();
  TestBasic<quint8>();
  TestBasic<qint8>();
  TestBasic<bfloat16>();
}

TEST(TensorBundleTest, Endianness) {
  TestEndianness<float>();
  TestEndianness<double>();
  TestEndianness<int32>();
  TestEndianness<uint8>();
  TestEndianness<int16>();
  TestEndianness<int8>();
  TestEndianness<complex64>();
  TestEndianness<complex128>();
  TestEndianness<int64_t>();
  TestEndianness<bool>();
  TestEndianness<qint32>();
  TestEndianness<quint8>();
  TestEndianness<qint8>();
  TestEndianness<bfloat16>();
}

TEST(TensorBundleTest, PartitionedVariables) {
  const TensorShape kFullShape({5, 10});
  // Adds two slices.
  // First slice: column 0, all zeros.
  // Second slice: column 1 to rest, all ones.
  TensorSlice slice1 = TensorSlice::ParseOrDie("-:0,1");
  TensorSlice slice2 = TensorSlice::ParseOrDie("-:1,9");
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));

    TF_ASSERT_OK(writer.AddSlice("foo", kFullShape, slice1,
                                 Constant<float>(0., TensorShape({5, 1}))));
    TF_ASSERT_OK(writer.AddSlice("foo", kFullShape, slice2,
                                 Constant<float>(1., TensorShape({5, 9}))));
    TF_ASSERT_OK(writer.Finish());
  }
  // Reads in full.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    Tensor expected_val(DT_FLOAT, kFullShape);
    test::FillFn<float>(&expected_val, [](int offset) -> float {
      if (offset % 10 == 0) {
        return 0;  // First column zeros.
      }
      return 1;  // Other columns ones.
    });

    Tensor val(DT_FLOAT, kFullShape);
    TF_ASSERT_OK(reader.Lookup("foo", &val));
    test::ExpectTensorEqual<float>(val, expected_val);
  }
  // Reads all slices.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    std::vector<TensorSlice> slices;
    TF_ASSERT_OK(reader.LookupTensorSlices("foo", &slices));

    EXPECT_EQ(2, slices.size());
    EXPECT_EQ(slice1.DebugString(), slices[0].DebugString());
    EXPECT_EQ(slice2.DebugString(), slices[1].DebugString());
  }
  // Reads a slice consisting of first two columns, "cutting" both slices.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    // First two columns, "cutting" both slices.
    const TensorSlice distinct_slice = TensorSlice::ParseOrDie("-:0,2");
    Tensor expected_val(DT_FLOAT, TensorShape({5, 2}));
    test::FillFn<float>(&expected_val, [](int offset) -> float {
      if (offset % 2 == 0) {
        return 0;  // First column zeros.
      }
      return 1;  // Other columns ones.
    });

    Tensor val(DT_FLOAT, TensorShape({5, 2}));
    TF_ASSERT_OK(reader.LookupSlice("foo", distinct_slice, &val));
    test::ExpectTensorEqual<float>(val, expected_val);
  }
  // Reads a slice consisting of columns 2-4, "cutting" the second slice only.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());

    const TensorSlice distinct_slice = TensorSlice::ParseOrDie("-:2,2");
    Tensor val(DT_FLOAT, TensorShape({5, 2}));
    TF_ASSERT_OK(reader.LookupSlice("foo", distinct_slice, &val));
    test::ExpectTensorEqual<float>(val,
                                   Constant<float>(1., TensorShape({5, 2})));
  }
}

TEST(TensorBundleTest, EquivalentSliceTest) {
  const TensorShape kFullShape({5, 10});
  const Tensor kExpected(Constant<float>(1., kFullShape));
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(writer.AddSlice("no_extents", kFullShape,
                                 TensorSlice::ParseOrDie("-:-"), kExpected));
    TF_ASSERT_OK(writer.AddSlice("both_extents", kFullShape,
                                 TensorSlice::ParseOrDie("0,5:0,10"),
                                 kExpected));
    TF_ASSERT_OK(writer.Finish());
  }
  // Slices match exactly and are fully abbreviated.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("no_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
  // Slice match exactly and are fully specified.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("0,5:0,10");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("both_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
  // Stored slice has no extents, spec has extents.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("0,5:0,10");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("no_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
  // Stored slice has both extents, spec has no extents.
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    const TensorSlice slice = TensorSlice::ParseOrDie("-:-");
    Tensor val(DT_FLOAT, TensorShape(kFullShape));
    TF_ASSERT_OK(reader.LookupSlice("both_extents", slice, &val));
    test::ExpectTensorEqual<float>(val, kExpected);
  }
}

TEST(TensorBundleTest, NonStandardShapes) {
  TestNonStandardShapes<float>();
  TestNonStandardShapes<double>();
  TestNonStandardShapes<int32>();
  TestNonStandardShapes<uint8>();
  TestNonStandardShapes<int16>();
  TestNonStandardShapes<int8>();
  TestNonStandardShapes<complex64>();
  TestNonStandardShapes<complex128>();
  TestNonStandardShapes<int64_t>();
  TestNonStandardShapes<bool>();
  TestNonStandardShapes<qint32>();
  TestNonStandardShapes<quint8>();
  TestNonStandardShapes<qint8>();
  TestNonStandardShapes<bfloat16>();
}

TEST(TensorBundleTest, StringTensorsOldFormat) {
  // Test string tensor bundle made with previous version of code that use
  // varint32s to store string lengths (we now use varint64s).
  BundleReader reader(Env::Default(), TestdataPrefix("old_string_tensors/foo"));
  TF_ASSERT_OK(reader.status());
  EXPECT_EQ(AllTensorKeys(&reader),
            std::vector<string>({"floats", "scalar", "string_tensor", "strs"}));

  Expect<tstring>(&reader, "string_tensor",
                  Tensor(DT_STRING, TensorShape({1})));
  Expect<tstring>(&reader, "scalar", test::AsTensor<tstring>({"hello"}));
  Expect<tstring>(
      &reader, "strs",
      test::AsTensor<tstring>({"hello", "", "x01", string(1 << 10, 'c')}));
  Expect<float>(&reader, "floats", Constant_2x3<float>(16.18));
}

TEST(TensorBundleTest, StringTensors) {
  constexpr size_t kLongLength = static_cast<size_t>(UINT32_MAX) + 1;
  Tensor long_string_tensor(DT_STRING, TensorShape({1}));

  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(writer.Add("string_tensor",
                            Tensor(DT_STRING, TensorShape({1}))));  // Empty.
    TF_EXPECT_OK(writer.Add("scalar", test::AsTensor<tstring>({"hello"})));
    TF_EXPECT_OK(writer.Add(
        "strs",
        test::AsTensor<tstring>({"hello", "", "x01", string(1 << 25, 'c')})));

    // Requires a 64-bit length.
    tstring* backing_string = long_string_tensor.flat<tstring>().data();
    backing_string->resize_uninitialized(kLongLength);
    std::char_traits<char>::assign(backing_string->data(), kLongLength, 'd');
    TF_EXPECT_OK(writer.Add("long_scalar", long_string_tensor));

    // Mixes in some floats.
    TF_EXPECT_OK(writer.Add("floats", Constant_2x3<float>(16.18)));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(AllTensorKeys(&reader),
              std::vector<string>({"floats", "long_scalar", "scalar",
                                   "string_tensor", "strs"}));

    Expect<tstring>(&reader, "string_tensor",
                    Tensor(DT_STRING, TensorShape({1})));
    Expect<tstring>(&reader, "scalar", test::AsTensor<tstring>({"hello"}));
    Expect<tstring>(
        &reader, "strs",
        test::AsTensor<tstring>({"hello", "", "x01", string(1 << 25, 'c')}));

    Expect<float>(&reader, "floats", Constant_2x3<float>(16.18));

    // We don't use the Expect function so we can re-use the
    // `long_string_tensor` buffer for reading out long_scalar to keep memory
    // usage reasonable.
    EXPECT_TRUE(reader.Contains("long_scalar"));
    DataType dtype;
    TensorShape shape;
    TF_ASSERT_OK(reader.LookupDtypeAndShape("long_scalar", &dtype, &shape));
    EXPECT_EQ(DT_STRING, dtype);
    EXPECT_EQ(TensorShape({1}), shape);

    // Fill the string differently so that we can be sure the new one is read
    // in. Because fragmentation in tc-malloc and we have such a big tensor
    // of 4GB, therefore it is not ideal to free the buffer right now.
    // The rationale is to make allocation/free close to each other.
    tstring* backing_string = long_string_tensor.flat<tstring>().data();
    std::char_traits<char>::assign(backing_string->data(), kLongLength, 'e');

    // Read long_scalar and check it contains kLongLength 'd's.
    TF_ASSERT_OK(reader.Lookup("long_scalar", &long_string_tensor));
    ASSERT_EQ(backing_string, long_string_tensor.flat<tstring>().data());
    EXPECT_EQ(kLongLength, backing_string->length());
    for (size_t i = 0; i < kLongLength; i++) {
      // Not using ASSERT_EQ('d', c) because this way is twice as fast due to
      // compiler optimizations.
      if ((*backing_string)[i] != 'd') {
        FAIL() << "long_scalar is not full of 'd's as expected.";
        break;
      }
    }
  }
}

class VariantObject {
 public:
  VariantObject() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_14(mht_14_v, 1015, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "VariantObject");
}
  VariantObject(const string& metadata, int64_t value)
      : metadata_(metadata), value_(value) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("metadata: \"" + metadata + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_15(mht_15_v, 1021, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "VariantObject");
}

  string TypeName() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_16(mht_16_v, 1026, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "TypeName");
 return "TEST VariantObject"; }
  void Encode(VariantTensorData* data) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_17(mht_17_v, 1030, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "Encode");

    data->set_type_name(TypeName());
    data->set_metadata(metadata_);
    Tensor val_t = Tensor(DT_INT64, TensorShape({}));
    val_t.scalar<int64_t>()() = value_;
    *(data->add_tensors()) = val_t;
  }
  bool Decode(const VariantTensorData& data) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_18(mht_18_v, 1040, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "Decode");

    EXPECT_EQ(data.type_name(), TypeName());
    data.get_metadata(&metadata_);
    EXPECT_EQ(data.tensors_size(), 1);
    value_ = data.tensors(0).scalar<int64_t>()();
    return true;
  }
  bool operator==(const VariantObject other) const {
    return metadata_ == other.metadata_ && value_ == other.value_;
  }
  string metadata_;
  int64_t value_;
};

REGISTER_UNARY_VARIANT_DECODE_FUNCTION(VariantObject, "TEST VariantObject");

TEST(TensorBundleTest, VariantTensors) {
  {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_EXPECT_OK(
        writer.Add("variant_tensor",
                   test::AsTensor<Variant>({VariantObject("test", 10),
                                            VariantObject("test1", 20)})));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectVariant<VariantObject>(
        &reader, "variant_tensor",
        test::AsTensor<Variant>(
            {VariantObject("test", 10), VariantObject("test1", 20)}));
  }
}

TEST(TensorBundleTest, DirectoryStructure) {
  Env* env = Env::Default();
  // Writes two bundles.
  const std::vector<string> kBundlePrefixes = {Prefix("worker0"),
                                               Prefix("worker1")};
  for (int i = 0; i < 2; ++i) {
    BundleWriter writer(env, kBundlePrefixes[i]);
    TF_EXPECT_OK(
        writer.Add(strings::StrCat("tensor", i), Constant_2x3<float>(0.)));
    TF_ASSERT_OK(writer.Finish());
  }

  // Ensures we have the expected files.
  auto CheckDirFiles = [env](const string& bundle_prefix,
                             gtl::ArraySlice<string> expected_files) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("bundle_prefix: \"" + bundle_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_19(mht_19_v, 1093, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "lambda");

    StringPiece dir = io::Dirname(bundle_prefix);
    for (const string& expected_file : expected_files) {
      TF_EXPECT_OK(env->FileExists(io::JoinPath(dir, expected_file)));
    }
  };

  // Check we have:
  //   worker<i>.index
  //   worker<i>.data-00000-of-00001
  CheckDirFiles(kBundlePrefixes[0],
                {"worker0.index", "worker0.data-00000-of-00001"});
  CheckDirFiles(kBundlePrefixes[1],
                {"worker1.index", "worker1.data-00000-of-00001"});

  // Trivially "merge" one bundle to some other location (i.e., a renaming).
  const string kAnotherPrefix = Prefix("another");
  TF_ASSERT_OK(MergeBundles(env, {kBundlePrefixes[0]}, kAnotherPrefix));
  CheckDirFiles(kAnotherPrefix,
                {"another.index", "another.data-00000-of-00001"});

  // Performs actual merge of the two bundles.  Check we have:
  //   merged.index
  //   merged.data-00000-of-00002
  //   merged.data-00001-of-00002
  const string kMerged = Prefix("merged");
  TF_ASSERT_OK(
      MergeBundles(env, {kAnotherPrefix, kBundlePrefixes[1]}, kMerged));
  CheckDirFiles(kMerged, {"merged.index", "merged.data-00000-of-00002",
                          "merged.data-00001-of-00002"});
}

TEST(TensorBundleTest, SortForSequentialAccess) {
  Env* env = Env::Default();
  const std::vector<string> kBundlePrefixes = {Prefix("worker0"),
                                               Prefix("worker1")};
  BundleWriter writer0(env, kBundlePrefixes[0]);
  for (int i = 0; i < 3; ++i) {
    TF_EXPECT_OK(
        writer0.Add(strings::StrCat("tensor-0-", i), Constant_2x3<float>(0.)));
  }
  TF_ASSERT_OK(writer0.Finish());

  BundleWriter writer1(env, kBundlePrefixes[1]);
  for (int i = 2; i >= 0; --i) {
    TF_EXPECT_OK(
        writer1.Add(strings::StrCat("tensor-1-", i), Constant_2x3<float>(0.)));
  }
  TF_ASSERT_OK(writer1.Finish());

  const string kMerged = Prefix("merged");
  TF_ASSERT_OK(
      MergeBundles(env, {kBundlePrefixes[0], kBundlePrefixes[1]}, kMerged));

  // We now have:
  //   merged.data-00000-of-00002 with tensor-0-0, tensor-0-1, tensor-0-2
  //   merged.data-00001-of-00002 with tensor-1-2, tensor-1-1, tensor-1-0

  BundleReader reader(env, kMerged);
  TF_ASSERT_OK(reader.status());
  std::vector<string> tensor_names = {"tensor-1-0", "tensor-0-1", "tensor-1-2",
                                      "tensor-0-0", "tensor-1-1", "tensor-0-2"};
  TF_ASSERT_OK(reader.SortForSequentialAccess<string>(
      tensor_names, [](const string& element) { return element; }));
  EXPECT_THAT(tensor_names,
              ElementsAre("tensor-0-0", "tensor-0-1", "tensor-0-2",
                          "tensor-1-2", "tensor-1-1", "tensor-1-0"));
}

TEST(TensorBundleTest, Error) {
  {  // Dup keys.
    BundleWriter writer(Env::Default(), Prefix("dup"));
    TF_EXPECT_OK(writer.Add("foo", Constant_2x3(1.f)));
    EXPECT_FALSE(writer.Add("foo", Constant_2x3(2.f)).ok());
    EXPECT_TRUE(absl::StrContains(writer.status().ToString(), "duplicate key"));
    EXPECT_FALSE(writer.Finish().ok());
  }
  {  // Double finish
    BundleWriter writer(Env::Default(), Prefix("bad"));
    EXPECT_TRUE(writer.Finish().ok());
    EXPECT_FALSE(writer.Finish().ok());
  }
  {  // Not found.
    BundleReader reader(Env::Default(), Prefix("nonexist"));
    EXPECT_EQ(reader.status().code(), error::NOT_FOUND);
  }
}

TEST(TensorBundleTest, Checksum) {
  // Randomly flips a byte in [pos_lhs, end of data file), or exactly byte
  // pos_lhs if exact_pos == True.
  auto FlipByte = [](const string& prefix, int pos_lhs,
                     bool exact_pos = false) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_20(mht_20_v, 1189, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "lambda");

    DCHECK_GE(pos_lhs, 0);
    const string& datafile = DataFilename(Prefix(prefix), 0, 1);
    string data;
    TF_ASSERT_OK(ReadFileToString(Env::Default(), datafile, &data));

    int byte_pos = 0;
    if (!exact_pos) {
      std::mt19937 rng;
      std::uniform_int_distribution<int> dist(pos_lhs, data.size() - 1);
      byte_pos = dist(rng);
    } else {
      byte_pos = pos_lhs;
    }
    data[byte_pos] = ~data[byte_pos];
    TF_ASSERT_OK(WriteStringToFile(Env::Default(), datafile, data));
  };
  // The lookup should fail with a checksum-related message.
  auto ExpectLookupFails = [](const string& prefix, const string& key,
                              const string& expected_msg, Tensor& val) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("prefix: \"" + prefix + "\"");
   mht_21_v.push_back("key: \"" + key + "\"");
   mht_21_v.push_back("expected_msg: \"" + expected_msg + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_21(mht_21_v, 1214, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "lambda");

    BundleReader reader(Env::Default(), Prefix(prefix));
    Status status = reader.Lookup(key, &val);
    EXPECT_TRUE(errors::IsDataLoss(status));
    EXPECT_TRUE(absl::StrContains(status.ToString(), expected_msg));
  };

  // Corrupts a float tensor.
  {
    BundleWriter writer(Env::Default(), Prefix("singleton"));
    TF_EXPECT_OK(writer.Add("foo", Constant_2x3(1.f)));
    TF_ASSERT_OK(writer.Finish());

    FlipByte("singleton", 0 /* corrupts any byte */);
    Tensor val(DT_FLOAT, TensorShape({2, 3}));
    ExpectLookupFails("singleton", "foo",
                      "Checksum does not match" /* expected fail msg */, val);
  }
  // Corrupts a string tensor.
  {
    auto WriteStrings = []() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_22(mht_22_v, 1237, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "lambda");

      BundleWriter writer(Env::Default(), Prefix("strings"));
      TF_EXPECT_OK(
          writer.Add("foo", test::AsTensor<tstring>({"hello", "world"})));
      TF_ASSERT_OK(writer.Finish());
    };
    // Corrupts the first two bytes, which are the varint32-encoded lengths
    // of the two string elements.  Should hit mismatch on length cksum.
    for (int i = 0; i < 2; ++i) {
      WriteStrings();
      FlipByte("strings", i, true /* corrupts exactly byte i */);
      Tensor val(DT_STRING, TensorShape({2}));
      ExpectLookupFails(
          "strings", "foo",
          "length checksum does not match" /* expected fail msg */, val);
    }
    // Corrupts the string bytes, should hit an overall cksum mismatch.
    WriteStrings();
    FlipByte("strings", 2 /* corrupts starting from byte 2 */);
    Tensor val(DT_STRING, TensorShape({2}));
    ExpectLookupFails("strings", "foo",
                      "Checksum does not match" /* expected fail msg */, val);
  }
}

TEST(TensorBundleTest, TruncatedTensorContents) {
  Env* env = Env::Default();
  BundleWriter writer(env, Prefix("end"));
  TF_EXPECT_OK(writer.Add("key", Constant_2x3<float>(1.0)));
  TF_ASSERT_OK(writer.Finish());

  // Truncates the data file by one byte, so that we hit EOF.
  const string datafile = DataFilename(Prefix("end"), 0, 1);
  string data;
  TF_ASSERT_OK(ReadFileToString(env, datafile, &data));
  ASSERT_TRUE(!data.empty());
  TF_ASSERT_OK(WriteStringToFile(env, datafile,
                                 StringPiece(data.data(), data.size() - 1)));

  BundleReader reader(env, Prefix("end"));
  TF_ASSERT_OK(reader.status());
  Tensor val(DT_FLOAT, TensorShape({2, 3}));
  EXPECT_TRUE(errors::IsOutOfRange(reader.Lookup("key", &val)));
}

TEST(TensorBundleTest, HeaderEntry) {
  {
    BundleWriter writer(Env::Default(), Prefix("b"));
    TF_EXPECT_OK(writer.Add("key", Constant_2x3<float>(1.0)));
    TF_ASSERT_OK(writer.Finish());
  }

  // Extracts out the header.
  BundleHeaderProto header;
  {
    BundleReader reader(Env::Default(), Prefix("b"));
    TF_ASSERT_OK(reader.status());
    reader.Seek(kHeaderEntryKey);
    ASSERT_TRUE(reader.Valid());
    ASSERT_TRUE(ParseProtoUnlimited(&header, reader.value().data(),
                                    reader.value().size()));
  }

  // num_shards
  EXPECT_EQ(1, header.num_shards());
  // endianness
  if (port::kLittleEndian) {
    EXPECT_EQ(BundleHeaderProto::LITTLE, header.endianness());
  } else {
    EXPECT_EQ(BundleHeaderProto::BIG, header.endianness());
  }
  // version
  EXPECT_GT(kTensorBundleVersion, 0);
  EXPECT_EQ(kTensorBundleVersion, header.version().producer());
  EXPECT_EQ(kTensorBundleMinConsumer, header.version().min_consumer());
}

TEST(TensorBundleTest, VersionTest) {
  // Min consumer.
  {
    VersionDef versions;
    versions.set_producer(kTensorBundleVersion + 1);
    versions.set_min_consumer(kTensorBundleVersion + 1);
    VersionTest(
        versions,
        strings::StrCat("Checkpoint min consumer version ",
                        kTensorBundleVersion + 1, " above current version ",
                        kTensorBundleVersion, " for TensorFlow"));
  }
  // Min producer.
  {
    VersionDef versions;
    versions.set_producer(kTensorBundleMinProducer - 1);
    VersionTest(
        versions,
        strings::StrCat("Checkpoint producer version ",
                        kTensorBundleMinProducer - 1, " below min producer ",
                        kTensorBundleMinProducer, " supported by TensorFlow"));
  }
  // Bad consumer.
  {
    VersionDef versions;
    versions.set_producer(kTensorBundleVersion + 1);
    versions.add_bad_consumers(kTensorBundleVersion);
    VersionTest(
        versions,
        strings::StrCat(
            "Checkpoint disallows consumer version ", kTensorBundleVersion,
            ".  Please upgrade TensorFlow: this version is likely buggy."));
  }
}

class TensorBundleAlignmentTest : public ::testing::Test {
 protected:
  template <typename T>
  void ExpectAlignment(BundleReader* reader, const string& key, int alignment) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_23(mht_23_v, 1356, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "ExpectAlignment");

    BundleEntryProto full_tensor_entry;
    TF_ASSERT_OK(reader->GetBundleEntryProto(key, &full_tensor_entry));
    EXPECT_EQ(0, full_tensor_entry.offset() % alignment);
  }
};

TEST_F(TensorBundleAlignmentTest, AlignmentTest) {
  {
    BundleWriter::Options opts;
    opts.data_alignment = 42;
    BundleWriter writer(Env::Default(), Prefix("foo"), opts);
    TF_EXPECT_OK(writer.Add("foo_003", Constant_2x3<float>(3)));
    TF_EXPECT_OK(writer.Add("foo_000", Constant_2x3<float>(0)));
    TF_EXPECT_OK(writer.Add("foo_002", Constant_2x3<float>(2)));
    TF_EXPECT_OK(writer.Add("foo_001", Constant_2x3<float>(1)));
    TF_ASSERT_OK(writer.Finish());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    EXPECT_EQ(
        AllTensorKeys(&reader),
        std::vector<string>({"foo_000", "foo_001", "foo_002", "foo_003"}));
    Expect<float>(&reader, "foo_000", Constant_2x3<float>(0));
    Expect<float>(&reader, "foo_001", Constant_2x3<float>(1));
    Expect<float>(&reader, "foo_002", Constant_2x3<float>(2));
    Expect<float>(&reader, "foo_003", Constant_2x3<float>(3));
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectNext<float>(&reader, Constant_2x3<float>(0));
    ExpectNext<float>(&reader, Constant_2x3<float>(1));
    ExpectNext<float>(&reader, Constant_2x3<float>(2));
    ExpectNext<float>(&reader, Constant_2x3<float>(3));
    EXPECT_TRUE(reader.Valid());
    reader.Next();
    EXPECT_FALSE(reader.Valid());
  }
  {
    BundleReader reader(Env::Default(), Prefix("foo"));
    TF_ASSERT_OK(reader.status());
    ExpectAlignment<float>(&reader, "foo_000", 42);
    ExpectAlignment<float>(&reader, "foo_001", 42);
    ExpectAlignment<float>(&reader, "foo_002", 42);
    ExpectAlignment<float>(&reader, "foo_003", 42);
  }
}

static void BM_BundleAlignment(::testing::benchmark::State& state) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_24(mht_24_v, 1409, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "BM_BundleAlignment");

  {
    const int alignment = state.range(0);
    const int tensor_size = state.range(1);
    BundleWriter::Options opts;
    opts.data_alignment = alignment;
    BundleWriter writer(Env::Default(), Prefix("foo"), opts);
    TF_CHECK_OK(writer.Add("small", Constant(true, TensorShape({1}))));
    TF_CHECK_OK(writer.Add("big", Constant(32.1, TensorShape({tensor_size}))));
    TF_CHECK_OK(writer.Finish());
  }
  BundleReader reader(Env::Default(), Prefix("foo"));
  TF_CHECK_OK(reader.status());
  for (auto s : state) {
    Tensor t;
    TF_CHECK_OK(reader.Lookup("big", &t));
  }
}

BENCHMARK(BM_BundleAlignment)->ArgPair(1, 512);
BENCHMARK(BM_BundleAlignment)->ArgPair(1, 4096);
BENCHMARK(BM_BundleAlignment)->ArgPair(1, 1048576);
BENCHMARK(BM_BundleAlignment)->ArgPair(4096, 512);
BENCHMARK(BM_BundleAlignment)->ArgPair(4096, 4096);
BENCHMARK(BM_BundleAlignment)->ArgPair(4096, 1048576);

static void BM_BundleWriterSmallTensor(::testing::benchmark::State& state) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_25(mht_25_v, 1438, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "BM_BundleWriterSmallTensor");

  const int64_t bytes = state.range(0);
  Tensor t = Constant(static_cast<int8>('a'), TensorShape{bytes});
  BundleWriter writer(Env::Default(), Prefix("foo"));
  int suffix = 0;
  for (auto s : state) {
    TF_CHECK_OK(writer.Add(strings::StrCat("small", suffix++), t));
  }
}

BENCHMARK(BM_BundleWriterSmallTensor)->Range(1, 1 << 20);

static void BM_BundleWriterLargeTensor(::testing::benchmark::State& state) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSutilPStensor_bundlePStensor_bundle_testDTcc mht_26(mht_26_v, 1453, "", "./tensorflow/core/util/tensor_bundle/tensor_bundle_test.cc", "BM_BundleWriterLargeTensor");

  const int mb = state.range(0);
  const int64_t bytes = static_cast<int64_t>(mb) * (1 << 20);
  Tensor t = Constant(static_cast<int8>('a'), TensorShape{bytes});
  for (auto s : state) {
    BundleWriter writer(Env::Default(), Prefix("foo"));
    TF_CHECK_OK(writer.Add("big", t));
  }
}

BENCHMARK(BM_BundleWriterLargeTensor)->Arg(1 << 10);
BENCHMARK(BM_BundleWriterLargeTensor)->Arg(4 << 10);

}  // namespace tensorflow
