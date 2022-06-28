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
class MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc() {
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

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace io {
namespace {

// Construct a string of the specified length made out of the supplied
// partial string.
string BigString(const string& partial_string, size_t n) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("partial_string: \"" + partial_string + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/lib/io/recordio_test.cc", "BigString");

  string result;
  while (result.size() < n) {
    result.append(partial_string);
  }
  result.resize(n);
  return result;
}

// Construct a string from a number
string NumberString(int n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/lib/io/recordio_test.cc", "NumberString");

  char buf[50];
  snprintf(buf, sizeof(buf), "%d.", n);
  return string(buf);
}

// Return a skewed potentially long string
string RandomSkewedString(int i, random::SimplePhilox* rnd) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/lib/io/recordio_test.cc", "RandomSkewedString");

  return BigString(NumberString(i), rnd->Skewed(17));
}

class StringDest : public WritableFile {
 public:
  explicit StringDest(string* contents) : contents_(contents) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/lib/io/recordio_test.cc", "StringDest");
}

  Status Close() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_4(mht_4_v, 240, "", "./tensorflow/core/lib/io/recordio_test.cc", "Close");
 return Status::OK(); }
  Status Flush() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_5(mht_5_v, 244, "", "./tensorflow/core/lib/io/recordio_test.cc", "Flush");
 return Status::OK(); }
  Status Sync() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_6(mht_6_v, 248, "", "./tensorflow/core/lib/io/recordio_test.cc", "Sync");
 return Status::OK(); }
  Status Append(StringPiece slice) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_7(mht_7_v, 252, "", "./tensorflow/core/lib/io/recordio_test.cc", "Append");

    contents_->append(slice.data(), slice.size());
    return Status::OK();
  }
#if defined(TF_CORD_SUPPORT)
  Status Append(const absl::Cord& data) override {
    contents_->append(std::string(data));
    return Status::OK();
  }
#endif
  Status Tell(int64_t* pos) override {
    *pos = contents_->size();
    return Status::OK();
  }

 private:
  string* contents_;
};

class StringSource : public RandomAccessFile {
 public:
  explicit StringSource(string* contents)
      : contents_(contents), force_error_(false) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_8(mht_8_v, 277, "", "./tensorflow/core/lib/io/recordio_test.cc", "StringSource");
}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_9(mht_9_v, 284, "", "./tensorflow/core/lib/io/recordio_test.cc", "Read");

    if (force_error_) {
      force_error_ = false;
      return errors::DataLoss("read error");
    }

    if (offset >= contents_->size()) {
      return errors::OutOfRange("end of file");
    }

    if (contents_->size() < offset + n) {
      n = contents_->size() - offset;
    }
    *result = StringPiece(contents_->data() + offset, n);
    return Status::OK();
  }

  void force_error() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_10(mht_10_v, 304, "", "./tensorflow/core/lib/io/recordio_test.cc", "force_error");
 force_error_ = true; }

 private:
  string* contents_;
  mutable bool force_error_;
};

class RecordioTest : public ::testing::Test {
 private:
  string contents_;
  StringDest dest_;
  StringSource source_;
  bool reading_;
  uint64 readpos_;
  RecordWriter* writer_;
  RecordReader* reader_;

 public:
  RecordioTest()
      : dest_(&contents_),
        source_(&contents_),
        reading_(false),
        readpos_(0),
        writer_(new RecordWriter(&dest_)),
        reader_(new RecordReader(&source_)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_11(mht_11_v, 331, "", "./tensorflow/core/lib/io/recordio_test.cc", "RecordioTest");
}

  ~RecordioTest() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_12(mht_12_v, 336, "", "./tensorflow/core/lib/io/recordio_test.cc", "~RecordioTest");

    delete writer_;
    delete reader_;
  }

  void Write(const string& msg) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("msg: \"" + msg + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_13(mht_13_v, 345, "", "./tensorflow/core/lib/io/recordio_test.cc", "Write");

    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    TF_ASSERT_OK(writer_->WriteRecord(StringPiece(msg)));
  }

#if defined(TF_CORD_SUPPORT)
  void Write(const absl::Cord& msg) {
    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    TF_ASSERT_OK(writer_->WriteRecord(msg));
  }
#endif

  size_t WrittenBytes() const { return contents_.size(); }

  string Read() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_14(mht_14_v, 362, "", "./tensorflow/core/lib/io/recordio_test.cc", "Read");

    if (!reading_) {
      reading_ = true;
    }
    tstring record;
    Status s = reader_->ReadRecord(&readpos_, &record);
    if (s.ok()) {
      return record;
    } else if (errors::IsOutOfRange(s)) {
      return "EOF";
    } else {
      return s.ToString();
    }
  }

  void IncrementByte(int offset, int delta) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_15(mht_15_v, 380, "", "./tensorflow/core/lib/io/recordio_test.cc", "IncrementByte");
 contents_[offset] += delta; }

  void SetByte(int offset, char new_byte) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("new_byte: '" + std::string(1, new_byte) + "'");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_16(mht_16_v, 386, "", "./tensorflow/core/lib/io/recordio_test.cc", "SetByte");
 contents_[offset] = new_byte; }

  void ShrinkSize(int bytes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_17(mht_17_v, 391, "", "./tensorflow/core/lib/io/recordio_test.cc", "ShrinkSize");
 contents_.resize(contents_.size() - bytes); }

  void FixChecksum(int header_offset, int len) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_18(mht_18_v, 396, "", "./tensorflow/core/lib/io/recordio_test.cc", "FixChecksum");

    // Compute crc of type/len/data
    uint32_t crc = crc32c::Value(&contents_[header_offset + 6], 1 + len);
    crc = crc32c::Mask(crc);
    core::EncodeFixed32(&contents_[header_offset], crc);
  }

  void ForceError() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_19(mht_19_v, 406, "", "./tensorflow/core/lib/io/recordio_test.cc", "ForceError");
 source_.force_error(); }

  void StartReadingAt(uint64_t initial_offset) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_20(mht_20_v, 411, "", "./tensorflow/core/lib/io/recordio_test.cc", "StartReadingAt");
 readpos_ = initial_offset; }

  void CheckOffsetPastEndReturnsNoRecords(uint64_t offset_past_end) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_21(mht_21_v, 416, "", "./tensorflow/core/lib/io/recordio_test.cc", "CheckOffsetPastEndReturnsNoRecords");

    Write("foo");
    Write("bar");
    Write(BigString("x", 10000));
    reading_ = true;
    uint64 offset = WrittenBytes() + offset_past_end;
    tstring record;
    Status s = reader_->ReadRecord(&offset, &record);
    ASSERT_TRUE(errors::IsOutOfRange(s)) << s;
  }
};

TEST_F(RecordioTest, Empty) { ASSERT_EQ("EOF", Read()); }

TEST_F(RecordioTest, ReadWrite) {
  Write("foo");
  Write("bar");
  Write("");
  Write("xxxx");
  ASSERT_EQ("foo", Read());
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("", Read());
  ASSERT_EQ("xxxx", Read());
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());  // Make sure reads at eof work
}

#if defined(TF_CORD_SUPPORT)
TEST_F(RecordioTest, ReadWriteCords) {
  Write(absl::Cord("foo"));
  Write(absl::Cord("bar"));
  Write(absl::Cord(""));
  Write(absl::Cord("xxxx"));
  ASSERT_EQ("foo", Read());
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("", Read());
  ASSERT_EQ("xxxx", Read());
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());  // Make sure reads at eof work
}
#endif

TEST_F(RecordioTest, ManyRecords) {
  for (int i = 0; i < 100000; i++) {
    Write(NumberString(i));
  }
  for (int i = 0; i < 100000; i++) {
    ASSERT_EQ(NumberString(i), Read());
  }
  ASSERT_EQ("EOF", Read());
}

TEST_F(RecordioTest, RandomRead) {
  const int N = 500;
  {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    for (int i = 0; i < N; i++) {
      Write(RandomSkewedString(i, &rnd));
    }
  }
  {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    for (int i = 0; i < N; i++) {
      ASSERT_EQ(RandomSkewedString(i, &rnd), Read());
    }
  }
  ASSERT_EQ("EOF", Read());
}

void TestNonSequentialReads(const RecordWriterOptions& writer_options,
                            const RecordReaderOptions& reader_options) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_22(mht_22_v, 491, "", "./tensorflow/core/lib/io/recordio_test.cc", "TestNonSequentialReads");

  string contents;
  StringDest dst(&contents);
  RecordWriter writer(&dst, writer_options);
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK(writer.WriteRecord(NumberString(i))) << i;
  }
  TF_ASSERT_OK(writer.Close());

  StringSource file(&contents);
  RecordReader reader(&file, reader_options);

  tstring record;
  // First read sequentially to fill in the offsets table.
  uint64 offsets[10] = {0};
  uint64 offset = 0;
  for (int i = 0; i < 10; ++i) {
    offsets[i] = offset;
    TF_ASSERT_OK(reader.ReadRecord(&offset, &record)) << i;
  }

  // Read randomly: First go back to record #3 then forward to #8.
  offset = offsets[3];
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ("3.", record);
  EXPECT_EQ(offsets[4], offset);

  offset = offsets[8];
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ("8.", record);
  EXPECT_EQ(offsets[9], offset);
}

TEST_F(RecordioTest, NonSequentialReads) {
  TestNonSequentialReads(RecordWriterOptions(), RecordReaderOptions());
}

TEST_F(RecordioTest, NonSequentialReadsWithReadBuffer) {
  RecordReaderOptions options;
  options.buffer_size = 1 << 10;
  TestNonSequentialReads(RecordWriterOptions(), options);
}

TEST_F(RecordioTest, NonSequentialReadsWithCompression) {
  TestNonSequentialReads(
      RecordWriterOptions::CreateRecordWriterOptions("ZLIB"),
      RecordReaderOptions::CreateRecordReaderOptions("ZLIB"));
}

// Tests of all the error paths in log_reader.cc follow:
void AssertHasSubstr(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_23(mht_23_v, 544, "", "./tensorflow/core/lib/io/recordio_test.cc", "AssertHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << s << " does not contain " << expected;
}

void TestReadError(const RecordWriterOptions& writer_options,
                   const RecordReaderOptions& reader_options) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecordio_testDTcc mht_24(mht_24_v, 553, "", "./tensorflow/core/lib/io/recordio_test.cc", "TestReadError");

  const string wrote = BigString("well hello there!", 100);
  string contents;
  StringDest dst(&contents);
  TF_ASSERT_OK(RecordWriter(&dst, writer_options).WriteRecord(wrote));

  StringSource file(&contents);
  RecordReader reader(&file, reader_options);

  uint64 offset = 0;
  tstring read;
  file.force_error();
  Status status = reader.ReadRecord(&offset, &read);
  ASSERT_TRUE(errors::IsDataLoss(status));
  ASSERT_EQ(0, offset);

  // A failed Read() shouldn't update the offset, and thus a retry shouldn't
  // lose the record.
  status = reader.ReadRecord(&offset, &read);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_GT(offset, 0);
  EXPECT_EQ(wrote, read);
}

TEST_F(RecordioTest, ReadError) {
  TestReadError(RecordWriterOptions(), RecordReaderOptions());
}

TEST_F(RecordioTest, ReadErrorWithBuffering) {
  RecordReaderOptions options;
  options.buffer_size = 1 << 20;
  TestReadError(RecordWriterOptions(), options);
}

TEST_F(RecordioTest, ReadErrorWithCompression) {
  TestReadError(RecordWriterOptions::CreateRecordWriterOptions("ZLIB"),
                RecordReaderOptions::CreateRecordReaderOptions("ZLIB"));
}

TEST_F(RecordioTest, CorruptLength) {
  Write("foo");
  IncrementByte(6, 100);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, CorruptLengthCrc) {
  Write("foo");
  IncrementByte(10, 100);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, CorruptData) {
  Write("foo");
  IncrementByte(14, 10);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, CorruptDataCrc) {
  Write("foo");
  IncrementByte(WrittenBytes() - 1, 10);
  AssertHasSubstr(Read(), "corrupted record");
}

TEST_F(RecordioTest, ReadEnd) { CheckOffsetPastEndReturnsNoRecords(0); }

TEST_F(RecordioTest, ReadPastEnd) { CheckOffsetPastEndReturnsNoRecords(5); }

}  // namespace
}  // namespace io
}  // namespace tensorflow
