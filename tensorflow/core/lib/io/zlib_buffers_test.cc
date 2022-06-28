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
class MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc() {
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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace io {

static std::vector<int> InputBufferSizes() {
  return {10, 100, 200, 500, 1000, 10000};
}

static std::vector<int> OutputBufferSizes() { return {100, 200, 500, 1000}; }

static std::vector<int> NumCopies() { return {1, 50, 500}; }

static string GetRecord() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "GetRecord");

  static const string lorem_ipsum =
      "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
      " Fusce vehicula tincidunt libero sit amet ultrices. Vestibulum non "
      "felis augue. Duis vitae augue id lectus lacinia congue et ut purus. "
      "Donec auctor, nisl at dapibus volutpat, diam ante lacinia dolor, vel"
      "dignissim lacus nisi sed purus. Duis fringilla nunc ac lacus sagittis"
      " efficitur. Praesent tincidunt egestas eros, eu vehicula urna ultrices"
      " et. Aliquam erat volutpat. Maecenas vehicula risus consequat risus"
      " dictum, luctus tincidunt nibh imperdiet. Aenean bibendum ac erat"
      " cursus scelerisque. Cras lacinia in enim dapibus iaculis. Nunc porta"
      " felis lectus, ac tincidunt massa pharetra quis. Fusce feugiat dolor"
      " vel ligula rutrum egestas. Donec vulputate quam eros, et commodo"
      " purus lobortis sed.";
  return lorem_ipsum;
}

static string GenTestString(int copies = 1) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "GenTestString");

  string result = "";
  for (int i = 0; i < copies; i++) {
    result += GetRecord();
  }
  return result;
}

typedef io::ZlibCompressionOptions CompressionOptions;

void TestAllCombinations(CompressionOptions input_options,
                         CompressionOptions output_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "TestAllCombinations");

  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    // Write to compressed file
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        std::unique_ptr<WritableFile> file_writer;
        TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
        tstring result;

        ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                             output_options);
        TF_ASSERT_OK(out.Init());

        TF_ASSERT_OK(out.Append(StringPiece(data)));
        TF_ASSERT_OK(out.Close());
        TF_ASSERT_OK(file_writer->Flush());
        TF_ASSERT_OK(file_writer->Close());

        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);
        TF_ASSERT_OK(in.ReadNBytes(data.size(), &result));
        EXPECT_EQ(result, data);
      }
    }
  }
}

TEST(ZlibBuffers, DefaultOptions) {
  TestAllCombinations(CompressionOptions::DEFAULT(),
                      CompressionOptions::DEFAULT());
}

TEST(ZlibBuffers, RawDeflate) {
  TestAllCombinations(CompressionOptions::RAW(), CompressionOptions::RAW());
}

TEST(ZlibBuffers, Gzip) {
  TestAllCombinations(CompressionOptions::GZIP(), CompressionOptions::GZIP());
}

void TestMultipleWrites(uint8 input_buf_size, uint8 output_buf_size,
                        int num_writes, bool with_flush = false) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_3(mht_3_v, 291, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "TestMultipleWrites");

  Env* env = Env::Default();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  CompressionOptions output_options = CompressionOptions::DEFAULT();

  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  string data = GenTestString();
  std::unique_ptr<WritableFile> file_writer;
  string actual_result;
  string expected_result;

  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_ASSERT_OK(out.Init());

  for (int i = 0; i < num_writes; i++) {
    TF_ASSERT_OK(out.Append(StringPiece(data)));
    if (with_flush) {
      TF_ASSERT_OK(out.Flush());
    }
    strings::StrAppend(&expected_result, data);
  }
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);

  for (int i = 0; i < num_writes; i++) {
    tstring decompressed_output;
    TF_ASSERT_OK(in.ReadNBytes(data.size(), &decompressed_output));
    strings::StrAppend(&actual_result, decompressed_output);
  }

  EXPECT_EQ(actual_result, expected_result);
}

TEST(ZlibBuffers, MultipleWritesWithoutFlush) {
  TestMultipleWrites(200, 200, 10);
}

TEST(ZlibBuffers, MultipleWriteCallsWithFlush) {
  TestMultipleWrites(200, 200, 10, true);
}

TEST(ZlibInputStream, FailsToReadIfWindowBitsAreIncompatible) {
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  CompressionOptions output_options = CompressionOptions::DEFAULT();
  CompressionOptions input_options = CompressionOptions::DEFAULT();
  int input_buf_size = 200, output_buf_size = 200;
  output_options.window_bits = MAX_WBITS;
  // inflate() has smaller history buffer.
  input_options.window_bits = output_options.window_bits - 1;

  string data = GenTestString(10);
  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  tstring result;
  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_ASSERT_OK(out.Init());

  TF_ASSERT_OK(out.Append(StringPiece(data)));
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  std::unique_ptr<RandomAccessFile> file_reader;
  TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
  std::unique_ptr<RandomAccessInputStream> input_stream(
      new RandomAccessInputStream(file_reader.get()));
  ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                     input_options);
  Status read_status = in.ReadNBytes(data.size(), &result);
  CHECK_EQ(read_status.code(), error::DATA_LOSS);
  CHECK(read_status.error_message().find("inflate() failed") != string::npos);
}

void WriteCompressedFile(Env* env, const string& fname, int input_buf_size,
                         int output_buf_size,
                         const CompressionOptions& output_options,
                         const string& data) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("fname: \"" + fname + "\"");
   mht_4_v.push_back("data: \"" + data + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_4(mht_4_v, 386, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "WriteCompressedFile");

  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));

  ZlibOutputBuffer out(file_writer.get(), input_buf_size, output_buf_size,
                       output_options);
  TF_ASSERT_OK(out.Init());

  TF_ASSERT_OK(out.Append(StringPiece(data)));
  TF_ASSERT_OK(out.Close());
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());
}

void TestTell(CompressionOptions input_options,
              CompressionOptions output_options) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_5(mht_5_v, 404, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "TestTell");

  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        // Write the compressed file.
        WriteCompressedFile(env, fname, input_buf_size, output_buf_size,
                            output_options, data);

        // Boiler-plate to set up ZlibInputStream.
        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);

        tstring first_half(string(data, 0, data.size() / 2));
        tstring bytes_read;

        // Read the first half of the uncompressed file and expect that Tell()
        // returns half the uncompressed length of the file.
        TF_ASSERT_OK(in.ReadNBytes(first_half.size(), &bytes_read));
        EXPECT_EQ(in.Tell(), first_half.size());
        EXPECT_EQ(bytes_read, first_half);

        // Read the remaining half of the uncompressed file and expect that
        // Tell() points past the end of file.
        tstring second_half;
        TF_ASSERT_OK(
            in.ReadNBytes(data.size() - first_half.size(), &second_half));
        EXPECT_EQ(in.Tell(), data.size());
        bytes_read.append(second_half);

        // Expect that the file is correctly read.
        EXPECT_EQ(bytes_read, data);
      }
    }
  }
}

void TestSkipNBytes(CompressionOptions input_options,
                    CompressionOptions output_options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_6(mht_6_v, 452, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "TestSkipNBytes");

  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  for (auto file_size : NumCopies()) {
    string data = GenTestString(file_size);
    for (auto input_buf_size : InputBufferSizes()) {
      for (auto output_buf_size : OutputBufferSizes()) {
        // Write the compressed file.
        WriteCompressedFile(env, fname, input_buf_size, output_buf_size,
                            output_options, data);

        // Boiler-plate to set up ZlibInputStream.
        std::unique_ptr<RandomAccessFile> file_reader;
        TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
        std::unique_ptr<RandomAccessInputStream> input_stream(
            new RandomAccessInputStream(file_reader.get()));
        ZlibInputStream in(input_stream.get(), input_buf_size, output_buf_size,
                           input_options);

        size_t data_half_size = data.size() / 2;
        string second_half(data, data_half_size, data.size() - data_half_size);

        // Skip past the first half of the file and expect Tell() returns
        // correctly.
        TF_ASSERT_OK(in.SkipNBytes(data_half_size));
        EXPECT_EQ(in.Tell(), data_half_size);

        // Expect that second half is read correctly and Tell() returns past
        // end of file after reading complete file.
        tstring bytes_read;
        TF_ASSERT_OK(in.ReadNBytes(second_half.size(), &bytes_read));
        EXPECT_EQ(bytes_read, second_half);
        EXPECT_EQ(in.Tell(), data.size());
      }
    }
  }
}

void TestSoftErrorOnDecompress(CompressionOptions input_options) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_buffers_testDTcc mht_7(mht_7_v, 494, "", "./tensorflow/core/lib/io/zlib_buffers_test.cc", "TestSoftErrorOnDecompress");

  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));

  input_options.soft_fail_on_error = true;

  std::unique_ptr<WritableFile> file_writer;
  TF_ASSERT_OK(env->NewWritableFile(fname, &file_writer));
  TF_ASSERT_OK(file_writer->Append("nonsense non-gzip data"));
  TF_ASSERT_OK(file_writer->Flush());
  TF_ASSERT_OK(file_writer->Close());

  // Test `ReadNBytes` returns an error.
  {
    std::unique_ptr<RandomAccessFile> file_reader;
    TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file_reader.get()));
    ZlibInputStream in(input_stream.get(), 100, 100, input_options);

    tstring unused;
    EXPECT_TRUE(errors::IsDataLoss(in.ReadNBytes(5, &unused)));
  }

  // Test `SkipNBytes` returns an error.
  {
    std::unique_ptr<RandomAccessFile> file_reader;
    TF_ASSERT_OK(env->NewRandomAccessFile(fname, &file_reader));
    std::unique_ptr<RandomAccessInputStream> input_stream(
        new RandomAccessInputStream(file_reader.get()));
    ZlibInputStream in(input_stream.get(), 100, 100, input_options);

    EXPECT_TRUE(errors::IsDataLoss(in.SkipNBytes(5)));
  }
}

TEST(ZlibInputStream, TellDefaultOptions) {
  TestTell(CompressionOptions::DEFAULT(), CompressionOptions::DEFAULT());
}

TEST(ZlibInputStream, TellRawDeflate) {
  TestTell(CompressionOptions::RAW(), CompressionOptions::RAW());
}

TEST(ZlibInputStream, TellGzip) {
  TestTell(CompressionOptions::GZIP(), CompressionOptions::GZIP());
}

TEST(ZlibInputStream, SkipNBytesDefaultOptions) {
  TestSkipNBytes(CompressionOptions::DEFAULT(), CompressionOptions::DEFAULT());
}

TEST(ZlibInputStream, SkipNBytesRawDeflate) {
  TestSkipNBytes(CompressionOptions::RAW(), CompressionOptions::RAW());
}

TEST(ZlibInputStream, SkipNBytesGzip) {
  TestSkipNBytes(CompressionOptions::GZIP(), CompressionOptions::GZIP());
}

TEST(ZlibInputStream, TestSoftErrorOnDecompressDefaultOptions) {
  TestSoftErrorOnDecompress(CompressionOptions::DEFAULT());
}

TEST(ZlibInputStream, TestSoftErrorOnDecompressRaw) {
  TestSoftErrorOnDecompress(CompressionOptions::RAW());
}

TEST(ZlibInputStream, TestSoftErrorOnDecompressGzip) {
  TestSoftErrorOnDecompress(CompressionOptions::GZIP());
}

}  // namespace io
}  // namespace tensorflow
