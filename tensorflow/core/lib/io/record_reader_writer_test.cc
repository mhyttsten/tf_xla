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
class MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_reader_writer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_reader_writer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_reader_writer_testDTcc() {
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

#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"

#include <zlib.h>
#include <vector>
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static std::vector<int> BufferSizes() {
  return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 65536};
}

namespace {

io::RecordReaderOptions GetMatchingReaderOptions(
    const io::RecordWriterOptions& options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_reader_writer_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/lib/io/record_reader_writer_test.cc", "GetMatchingReaderOptions");

  if (options.compression_type == io::RecordWriterOptions::ZLIB_COMPRESSION) {
    return io::RecordReaderOptions::CreateRecordReaderOptions("ZLIB");
  }
  return io::RecordReaderOptions::CreateRecordReaderOptions("");
}

uint64 GetFileSize(const string& fname) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_reader_writer_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/lib/io/record_reader_writer_test.cc", "GetFileSize");

  Env* env = Env::Default();
  uint64 fsize;
  TF_CHECK_OK(env->GetFileSize(fname, &fsize));
  return fsize;
}

void VerifyFlush(const io::RecordWriterOptions& options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_reader_writer_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/lib/io/record_reader_writer_test.cc", "VerifyFlush");

  std::vector<string> records = {
      "abcdefghijklmnopqrstuvwxyz",
      "ZYXWVUTSRQPONMLKJIHGFEDCBA0123456789!@#$%^&*()",
      "G5SyohOL9UmXofSOOwWDrv9hoLLMYPJbG9r38t3uBRcHxHj2PdKcPDuZmKW62RIY",
      "aaaaaaaaaaaaaaaaaaaaaaaaaa",
  };

  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_flush_test";

  std::unique_ptr<WritableFile> file;
  TF_CHECK_OK(env->NewWritableFile(fname, &file));
  io::RecordWriter writer(file.get(), options);

  std::unique_ptr<RandomAccessFile> read_file;
  TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
  io::RecordReaderOptions read_options = GetMatchingReaderOptions(options);
  io::RecordReader reader(read_file.get(), read_options);

  EXPECT_EQ(GetFileSize(fname), 0);
  for (size_t i = 0; i < records.size(); i++) {
    uint64 start_size = GetFileSize(fname);

    // Write a new record.
    TF_EXPECT_OK(writer.WriteRecord(records[i]));
    TF_CHECK_OK(writer.Flush());
    TF_CHECK_OK(file->Flush());

    // Verify that file size has changed after file flush.
    uint64 new_size = GetFileSize(fname);
    EXPECT_GT(new_size, start_size);

    // Verify that file has all records written so far and no more.
    uint64 offset = 0;
    tstring record;
    for (size_t j = 0; j <= i; j++) {
      // Check that j'th record is written correctly.
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ(record, records[j]);
    }

    // Verify that file has no more records.
    CHECK_EQ(reader.ReadRecord(&offset, &record).code(), error::OUT_OF_RANGE);
  }
}

}  // namespace

TEST(RecordReaderWriterTest, TestFlush) {
  io::RecordWriterOptions options;
  VerifyFlush(options);
}

TEST(RecordReaderWriterTest, TestZlibSyncFlush) {
  io::RecordWriterOptions options;
  options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
  // The default flush_mode is Z_NO_FLUSH and only writes to the file when the
  // buffer is full or the file is closed, which makes testing harder.
  // By using Z_SYNC_FLUSH the test can verify Flush does write out records of
  // approximately the right size at the right times.
  options.zlib_options.flush_mode = Z_SYNC_FLUSH;

  VerifyFlush(options);
}

TEST(RecordReaderWriterTest, TestBasics) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_test";

  for (auto buf_size : BufferSizes()) {
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      tstring record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);

      io::RecordReader::Metadata md;
      TF_ASSERT_OK(reader.GetMetadata(&md));
      EXPECT_EQ(2, md.stats.entries);
      EXPECT_EQ(7, md.stats.data_size);
      // Two entries have 16 bytes of header/footer each.
      EXPECT_EQ(39, md.stats.file_size);
    }
  }
}

TEST(RecordReaderWriterTest, TestSkipBasic) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_skip_basic_test";

  for (auto buf_size : BufferSizes()) {
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_EXPECT_OK(writer.WriteRecord("hij"));
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      int num_skipped;
      tstring record;
      TF_CHECK_OK(reader.SkipRecords(&offset, 2, &num_skipped));
      EXPECT_EQ(2, num_skipped);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("hij", record);
    }
  }
}

TEST(RecordReaderWriterTest, TestSkipOutOfRange) {
  Env* env = Env::Default();
  string fname =
      testing::TmpDir() + "/record_reader_writer_skip_out_of_range_test";

  for (auto buf_size : BufferSizes()) {
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      int num_skipped;
      tstring record;
      Status s = reader.SkipRecords(&offset, 3, &num_skipped);
      EXPECT_EQ(2, num_skipped);
      EXPECT_EQ(error::OUT_OF_RANGE, s.code());
    }
  }
}

TEST(RecordReaderWriterTest, TestSnappy) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_snappy_test";

  for (auto buf_size : BufferSizes()) {
    // Snappy compression needs output buffer size > 1.
    if (buf_size == 1) continue;
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.compression_type = io::RecordWriterOptions::SNAPPY_COMPRESSION;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.compression_type = io::RecordReaderOptions::SNAPPY_COMPRESSION;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      tstring record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);
    }
  }
}

TEST(RecordReaderWriterTest, TestZlib) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_zlib_test";

  for (auto buf_size : BufferSizes()) {
    // Zlib compression needs output buffer size > 1.
    if (buf_size == 1) continue;
    {
      std::unique_ptr<WritableFile> file;
      TF_CHECK_OK(env->NewWritableFile(fname, &file));

      io::RecordWriterOptions options;
      options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
      options.zlib_options.output_buffer_size = buf_size;
      io::RecordWriter writer(file.get(), options);
      TF_EXPECT_OK(writer.WriteRecord("abc"));
      TF_EXPECT_OK(writer.WriteRecord("defg"));
      TF_CHECK_OK(writer.Flush());
    }

    {
      std::unique_ptr<RandomAccessFile> read_file;
      // Read it back with the RecordReader.
      TF_CHECK_OK(env->NewRandomAccessFile(fname, &read_file));
      io::RecordReaderOptions options;
      options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
      options.zlib_options.input_buffer_size = buf_size;
      io::RecordReader reader(read_file.get(), options);
      uint64 offset = 0;
      tstring record;
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("abc", record);
      TF_CHECK_OK(reader.ReadRecord(&offset, &record));
      EXPECT_EQ("defg", record);
    }
  }
}

TEST(RecordReaderWriterTest, TestUseAfterClose) {
  Env* env = Env::Default();
  string fname = testing::TmpDir() + "/record_reader_writer_flush_close_test";

  {
    std::unique_ptr<WritableFile> file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));

    io::RecordWriterOptions options;
    options.compression_type = io::RecordWriterOptions::ZLIB_COMPRESSION;
    io::RecordWriter writer(file.get(), options);
    TF_EXPECT_OK(writer.WriteRecord("abc"));
    TF_CHECK_OK(writer.Flush());
    TF_CHECK_OK(writer.Close());

    CHECK_EQ(writer.WriteRecord("abc").code(), error::FAILED_PRECONDITION);
    CHECK_EQ(writer.Flush().code(), error::FAILED_PRECONDITION);

    // Second call to close is fine.
    TF_CHECK_OK(writer.Close());
  }
}

}  // namespace tensorflow
