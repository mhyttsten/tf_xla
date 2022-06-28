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
class MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc() {
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

#include "tensorflow/core/lib/io/buffered_inputstream.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace io {
namespace {

static std::vector<int> BufferSizes() {
  return {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 65536};
}

// This class will only return OutOfRange error once to make sure that
// BufferedInputStream is able to cache the error.
class ReadOnceInputStream : public InputStreamInterface {
 public:
  ReadOnceInputStream() : start_(true) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/lib/io/buffered_inputstream_test.cc", "ReadOnceInputStream");
}

  virtual Status ReadNBytes(int64_t bytes_to_read, tstring* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/lib/io/buffered_inputstream_test.cc", "ReadNBytes");

    if (bytes_to_read < 11) {
      return errors::InvalidArgument("Not reading all bytes: ", bytes_to_read);
    }
    if (start_) {
      *result = "0123456789";
      start_ = false;
      return errors::OutOfRange("Out of range.");
    }
    return errors::InvalidArgument(
        "Redudant call to ReadNBytes after an OutOfRange error.");
  }

  int64_t Tell() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/lib/io/buffered_inputstream_test.cc", "Tell");
 return start_ ? 0 : 10; }

  // Resets the stream to the beginning.
  Status Reset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/lib/io/buffered_inputstream_test.cc", "Reset");

    start_ = true;
    return Status::OK();
  }

 private:
  bool start_;
};

TEST(BufferedInputStream, ReadLine_Empty) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, ""));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine1) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\nline two\nline three\n"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_NoTrailingNewLine) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_EmptyLines) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\n\n\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, ReadLine_CRLF) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname,
                                 "line one\r\n\r\n\r\nline two\r\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line one");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line three");
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadLine(&line)));
  }
}

TEST(BufferedInputStream, SkipLine1) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(
      WriteStringToFile(env, fname, "line one\nline two\nline three\n"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.SkipLine());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipLine()));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipLine()));
  }
}

TEST(BufferedInputStream, SkipLine_NoTrailingNewLine) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\nline two\nline three"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
    TF_ASSERT_OK(in.SkipLine());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipLine()));
    // A second call should also return end of file
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipLine()));
  }
}

TEST(BufferedInputStream, SkipLine_EmptyLines) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "line one\n\n\nline two"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    BufferedInputStream in(input_stream.get(), buf_size);
    string line;
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "");
    TF_ASSERT_OK(in.SkipLine());
    TF_ASSERT_OK(in.ReadLine(&line));
    EXPECT_EQ(line, "line two");
  }
}

TEST(BufferedInputStream, ReadNBytes) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    tstring read;
    BufferedInputStream in(input_stream.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, OutOfRangeCache) {
  for (auto buf_size : BufferSizes()) {
    if (buf_size < 11) {
      continue;
    }
    ReadOnceInputStream input_stream;
    tstring read;
    BufferedInputStream in(&input_stream, buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK((in.ReadNBytes(7, &read)));
    EXPECT_EQ(read, "3456789");
    EXPECT_EQ(10, in.Tell());
    Status s = in.ReadNBytes(5, &read);
    // Make sure the read is failing with OUT_OF_RANGE error. If it is failing
    // with other errors, it is not caching the OUT_OF_RANGE properly.
    EXPECT_EQ(error::OUT_OF_RANGE, s.code()) << s;
    EXPECT_EQ(read, "");
    // Empty read shouldn't cause an error even at the end of the file.
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
  }
}

TEST(BufferedInputStream, SkipNBytes) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    tstring read;
    BufferedInputStream in(input_stream.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(1, &read));
    EXPECT_EQ(read, "7");
    EXPECT_EQ(8, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, ReadNBytesRandomAccessFile) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    tstring read;
    BufferedInputStream in(file.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(3, &read));
    EXPECT_EQ(read, "012");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(7, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "789");
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(0, &read));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, SkipNBytesRandomAccessFile) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    tstring read;
    BufferedInputStream in(file.get(), buf_size);
    EXPECT_EQ(0, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(3));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(3, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(2, &read));
    EXPECT_EQ(read, "34");
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(0));
    EXPECT_EQ(5, in.Tell());
    TF_ASSERT_OK(in.SkipNBytes(2));
    EXPECT_EQ(7, in.Tell());
    TF_ASSERT_OK(in.ReadNBytes(1, &read));
    EXPECT_EQ(read, "7");
    EXPECT_EQ(8, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.SkipNBytes(5)));
    EXPECT_EQ(10, in.Tell());
    EXPECT_TRUE(errors::IsOutOfRange(in.ReadNBytes(5, &read)));
    EXPECT_EQ(read, "");
    EXPECT_EQ(10, in.Tell());
  }
}

TEST(BufferedInputStream, Seek) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file.get()));
    tstring read;
    BufferedInputStream in(input_stream.get(), buf_size);

    // Seek forward
    TF_ASSERT_OK(in.Seek(3));
    EXPECT_EQ(3, in.Tell());

    // Read 4 bytes
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "3456");
    EXPECT_EQ(7, in.Tell());

    // Seek backwards
    TF_ASSERT_OK(in.Seek(1));
    TF_ASSERT_OK(in.ReadNBytes(4, &read));
    EXPECT_EQ(read, "1234");
    EXPECT_EQ(5, in.Tell());
  }
}

TEST(BufferedInputStream, Seek_NotReset) {
  // This test verifies seek backwards within the buffer doesn't reset
  // input_stream
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, "0123456789"));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file.get()));
  tstring read;
  BufferedInputStream in(input_stream.get(), 3);

  TF_ASSERT_OK(in.ReadNBytes(4, &read));
  int before_tell = input_stream.get()->Tell();
  EXPECT_EQ(before_tell, 6);
  // Seek backwards
  TF_ASSERT_OK(in.Seek(3));
  int after_tell = input_stream.get()->Tell();
  EXPECT_EQ(before_tell, after_tell);
}

TEST(BufferedInputStream, ReadAll_Empty) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  const string expected = "";
  TF_ASSERT_OK(WriteStringToFile(env, fname, expected));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    RandomAccessInputStream input_stream(file.get());
    BufferedInputStream in(&input_stream, buf_size);
    string contents;
    TF_ASSERT_OK(in.ReadAll(&contents));
    EXPECT_EQ(expected, contents);
  }
}

TEST(BufferedInputStream, ReadAll_Text) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  const string expected = "line one\nline two\nline three";
  TF_ASSERT_OK(WriteStringToFile(env, fname, expected));
  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  for (auto buf_size : BufferSizes()) {
    RandomAccessInputStream input_stream(file.get());
    BufferedInputStream in(&input_stream, buf_size);
    string contents;
    TF_ASSERT_OK(in.ReadAll(&contents));
    EXPECT_EQ(expected, contents);
  }
}

void BM_BufferedReaderSmallReads(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstream_testDTcc mht_4(mht_4_v, 707, "", "./tensorflow/core/lib/io/buffered_inputstream_test.cc", "BM_BufferedReaderSmallReads");

  const int buff_size = state.range(0);
  const int file_size = state.range(1);
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  const string file_elem = "0123456789";
  std::unique_ptr<WritableFile> write_file;
  TF_ASSERT_OK(env->NewWritableFile(fname, &write_file));
  for (int i = 0; i < file_size; ++i) {
    TF_ASSERT_OK(write_file->Append(file_elem));
  }
  TF_ASSERT_OK(write_file->Close());

  std::unique_ptr<RandomAccessFile> file;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file));

  tstring result;

  int itr = 0;
  for (auto s : state) {
    BufferedInputStream in(file.get(), buff_size);
    for (int64_t i = 0; i < 10 * file_size; ++i) {
      TF_ASSERT_OK(in.ReadNBytes(1, &result))
          << "i: " << i << " itr: " << itr << " buff_size: " << buff_size
          << " file size: " << file_size;
    }
    ++itr;
  }
}
BENCHMARK(BM_BufferedReaderSmallReads)
    ->ArgPair(1, 5)
    ->ArgPair(1, 1024)
    ->ArgPair(10, 5)
    ->ArgPair(10, 1024)
    ->ArgPair(1024, 1024)
    ->ArgPair(1024 * 1024, 1024)
    ->ArgPair(1024 * 1024, 1024 * 1024)
    ->ArgPair(256 * 1024 * 1024, 1024);

}  // anonymous namespace
}  // namespace io
}  // namespace tensorflow
