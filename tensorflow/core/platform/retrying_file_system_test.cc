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
class MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc() {
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

#include "tensorflow/core/platform/retrying_file_system.h"

#include <fstream>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

typedef std::vector<std::tuple<string, Status>> ExpectedCalls;

ExpectedCalls CreateRetriableErrors(const string& method, int n) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "CreateRetriableErrors");

  ExpectedCalls expected_calls;
  expected_calls.reserve(n);
  for (int i = 0; i < n; i++) {
    expected_calls.emplace_back(std::make_tuple(
        method, errors::Unavailable(strings::StrCat("Retriable error #", i))));
  }
  return expected_calls;
}

// A class to manage call expectations on mock implementations.
class MockCallSequence {
 public:
  explicit MockCallSequence(const ExpectedCalls& calls) : calls_(calls) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "MockCallSequence");
}

  ~MockCallSequence() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "~MockCallSequence");

    EXPECT_TRUE(calls_.empty())
        << "Not all expected calls have been made, "
        << "the next expected call: " << std::get<0>(calls_.front());
  }

  Status ConsumeNextCall(const string& method) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "ConsumeNextCall");

    EXPECT_FALSE(calls_.empty()) << "No more calls were expected.";
    auto call = calls_.front();
    calls_.erase(calls_.begin());
    EXPECT_EQ(std::get<0>(call), method) << "Unexpected method called.";
    return std::get<1>(call);
  }

 private:
  ExpectedCalls calls_;
};

class MockRandomAccessFile : public RandomAccessFile {
 public:
  explicit MockRandomAccessFile(const ExpectedCalls& calls) : calls_(calls) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "MockRandomAccessFile");
}
  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Name");

    return calls_.ConsumeNextCall("Name");
  }
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("scratch: \"" + (scratch == nullptr ? std::string("nullptr") : std::string((char*)scratch)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_6(mht_6_v, 259, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Read");

    return calls_.ConsumeNextCall("Read");
  }

 private:
  mutable MockCallSequence calls_;
};

class MockWritableFile : public WritableFile {
 public:
  explicit MockWritableFile(const ExpectedCalls& calls) : calls_(calls) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_7(mht_7_v, 272, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "MockWritableFile");
}
  Status Append(StringPiece data) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_8(mht_8_v, 276, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Append");

    return calls_.ConsumeNextCall("Append");
  }
  Status Close() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_9(mht_9_v, 282, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Close");
 return calls_.ConsumeNextCall("Close"); }
  Status Flush() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_10(mht_10_v, 286, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Flush");
 return calls_.ConsumeNextCall("Flush"); }
  Status Name(StringPiece* result) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_11(mht_11_v, 290, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Name");

    return calls_.ConsumeNextCall("Name");
  }
  Status Sync() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_12(mht_12_v, 296, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Sync");
 return calls_.ConsumeNextCall("Sync"); }
  Status Tell(int64_t* position) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_13(mht_13_v, 300, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Tell");

    return calls_.ConsumeNextCall("Tell");
  }

 private:
  mutable MockCallSequence calls_;
};

class MockFileSystem : public FileSystem {
 public:
  explicit MockFileSystem(const ExpectedCalls& calls, bool* flushed = nullptr)
      : calls_(calls), flushed_(flushed) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_14(mht_14_v, 314, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "MockFileSystem");
}

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_15(mht_15_v, 324, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "NewRandomAccessFile");

    *result = std::move(random_access_file_to_return);
    return calls_.ConsumeNextCall("NewRandomAccessFile");
  }

  Status NewWritableFile(const string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_16(mht_16_v, 334, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "NewWritableFile");

    *result = std::move(writable_file_to_return);
    return calls_.ConsumeNextCall("NewWritableFile");
  }

  Status NewAppendableFile(const string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_17(mht_17_v, 344, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "NewAppendableFile");

    *result = std::move(writable_file_to_return);
    return calls_.ConsumeNextCall("NewAppendableFile");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_18(mht_18_v, 355, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "NewReadOnlyMemoryRegionFromFile");

    return calls_.ConsumeNextCall("NewReadOnlyMemoryRegionFromFile");
  }

  Status FileExists(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_19(mht_19_v, 363, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "FileExists");

    return calls_.ConsumeNextCall("FileExists");
  }

  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_20(mht_20_v, 372, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "GetChildren");

    return calls_.ConsumeNextCall("GetChildren");
  }

  Status GetMatchingPaths(const string& dir, TransactionToken* token,
                          std::vector<string>* result) override {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_21(mht_21_v, 381, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "GetMatchingPaths");

    return calls_.ConsumeNextCall("GetMatchingPaths");
  }

  Status Stat(const string& fname, TransactionToken* token,
              FileStatistics* stat) override {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_22(mht_22_v, 390, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "Stat");

    return calls_.ConsumeNextCall("Stat");
  }

  Status DeleteFile(const string& fname, TransactionToken* token) override {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_23(mht_23_v, 398, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "DeleteFile");

    return calls_.ConsumeNextCall("DeleteFile");
  }

  Status CreateDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_24(mht_24_v, 406, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "CreateDir");

    return calls_.ConsumeNextCall("CreateDir");
  }

  Status DeleteDir(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_25(mht_25_v, 414, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "DeleteDir");

    return calls_.ConsumeNextCall("DeleteDir");
  }

  Status GetFileSize(const string& fname, TransactionToken* token,
                     uint64* file_size) override {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("fname: \"" + fname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_26(mht_26_v, 423, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "GetFileSize");

    return calls_.ConsumeNextCall("GetFileSize");
  }

  Status RenameFile(const string& src, const string& target,
                    TransactionToken* token) override {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("src: \"" + src + "\"");
   mht_27_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_27(mht_27_v, 433, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "RenameFile");

    return calls_.ConsumeNextCall("RenameFile");
  }

  Status IsDirectory(const string& dirname, TransactionToken* token) override {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_28(mht_28_v, 441, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "IsDirectory");

    return calls_.ConsumeNextCall("IsDirectory");
  }

  Status DeleteRecursively(const string& dirname, TransactionToken* token,
                           int64_t* undeleted_files,
                           int64_t* undeleted_dirs) override {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("dirname: \"" + dirname + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_29(mht_29_v, 451, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "DeleteRecursively");

    return calls_.ConsumeNextCall("DeleteRecursively");
  }

  void FlushCaches(TransactionToken* token) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSplatformPSretrying_file_system_testDTcc mht_30(mht_30_v, 458, "", "./tensorflow/core/platform/retrying_file_system_test.cc", "FlushCaches");

    if (flushed_) {
      *flushed_ = true;
    }
  }

  std::unique_ptr<WritableFile> writable_file_to_return;
  std::unique_ptr<RandomAccessFile> random_access_file_to_return;

 private:
  MockCallSequence calls_;
  bool* flushed_ = nullptr;
};

TEST(RetryingFileSystemTest, NewRandomAccessFile_ImmediateSuccess) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls({std::make_tuple("Name", Status::OK()),
                                     std::make_tuple("Read", Status::OK())});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  TF_EXPECT_OK(random_access_file->Name(&result));
  EXPECT_EQ(result, "");

  char scratch[10];
  TF_EXPECT_OK(random_access_file->Read(0, 10, &result, scratch));
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Read", errors::Unavailable("Something is wrong")),
       std::make_tuple("Read", errors::Unavailable("Wrong again")),
       std::make_tuple("Read", Status::OK())});
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  TF_EXPECT_OK(random_access_file->Read(0, 10, &result, scratch));
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_AllRetriesFailed) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls = CreateRetriableErrors("Read", 11);
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  const auto& status = random_access_file->Read(0, 10, &result, scratch);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, NewRandomAccessFile_NoRetriesForSomeErrors) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls({
      std::make_tuple("Read",
                      errors::FailedPrecondition("Failed precondition")),
  });
  std::unique_ptr<RandomAccessFile> base_file(
      new MockRandomAccessFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewRandomAccessFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->random_access_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped random access file.
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_EXPECT_OK(
      fs.NewRandomAccessFile("filename.txt", nullptr, &random_access_file));

  // Use it and check the results.
  StringPiece result;
  char scratch[10];
  EXPECT_EQ("Failed precondition",
            random_access_file->Read(0, 10, &result, scratch).error_message());
}

TEST(RetryingFileSystemTest, NewWritableFile_ImmediateSuccess) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls({std::make_tuple("Name", Status::OK()),
                                     std::make_tuple("Sync", Status::OK()),
                                     std::make_tuple("Close", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  StringPiece result;
  TF_EXPECT_OK(writable_file->Name(&result));
  EXPECT_EQ(result, "");

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", Status::OK()),
       std::make_tuple("Close", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_SuccessWith3rdTry_ViaDestructor) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Close", errors::Unavailable("Something is wrong")),
       std::make_tuple("Close",
                       errors::Unavailable("Something is wrong again")),
       std::make_tuple("Close", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  writable_file.reset();  // Trigger Close() via destructor.
}

TEST(RetryingFileSystemTest, NewAppendableFile_SuccessWith3rdTry) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls(
      {std::make_tuple("Sync", errors::Unavailable("Something is wrong")),
       std::make_tuple("Sync", errors::Unavailable("Something is wrong again")),
       std::make_tuple("Sync", Status::OK()),
       std::make_tuple("Close", Status::OK())});
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewAppendableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped appendable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewAppendableFile("filename.txt", nullptr, &writable_file));

  // Use it and check the results.
  TF_EXPECT_OK(writable_file->Sync());
}

TEST(RetryingFileSystemTest, NewWritableFile_AllRetriesFailed) {
  // Configure the mock base random access file.
  ExpectedCalls expected_file_calls = CreateRetriableErrors("Sync", 11);
  expected_file_calls.emplace_back(std::make_tuple("Close", Status::OK()));
  std::unique_ptr<WritableFile> base_file(
      new MockWritableFile(expected_file_calls));

  // Configure the mock base file system.
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewWritableFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  base_fs->writable_file_to_return = std::move(base_file);
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  // Retrieve the wrapped writable file.
  std::unique_ptr<WritableFile> writable_file;
  TF_EXPECT_OK(fs.NewWritableFile("filename.txt", nullptr, &writable_file));

  // Use it and check the results.
  const auto& status = writable_file->Sync();
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest,
     NewReadOnlyMemoryRegionFromFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("NewReadOnlyMemoryRegionFromFile",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("NewReadOnlyMemoryRegionFromFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::unique_ptr<ReadOnlyMemoryRegion> result;
  TF_EXPECT_OK(
      fs.NewReadOnlyMemoryRegionFromFile("filename.txt", nullptr, &result));
}

TEST(RetryingFileSystemTest, NewReadOnlyMemoryRegionFromFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls =
      CreateRetriableErrors("NewReadOnlyMemoryRegionFromFile", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::unique_ptr<ReadOnlyMemoryRegion> result;
  const auto& status =
      fs.NewReadOnlyMemoryRegionFromFile("filename.txt", nullptr, &result);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, GetChildren_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetChildren",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetChildren", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetChildren("gs://path", nullptr, &result));
}

TEST(RetryingFileSystemTest, GetChildren_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("GetChildren", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  const auto& status = fs.GetChildren("gs://path", nullptr, &result);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, GetMatchingPaths_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetMatchingPaths",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetMatchingPaths", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  TF_EXPECT_OK(fs.GetMatchingPaths("gs://path/dir", nullptr, &result));
}

TEST(RetryingFileSystemTest, GetMatchingPaths_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls =
      CreateRetriableErrors("GetMatchingPaths", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  std::vector<string> result;
  const auto& status = fs.GetMatchingPaths("gs://path/dir", nullptr, &result);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, DeleteFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.DeleteFile("gs://path/file.txt", nullptr));
}

TEST(RetryingFileSystemTest, DeleteFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("DeleteFile", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.DeleteFile("gs://path/file.txt", nullptr);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, CreateDir_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("CreateDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("CreateDir", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.CreateDir("gs://path/newdir", nullptr));
}

TEST(RetryingFileSystemTest, CreateDir_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("CreateDir", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.CreateDir("gs://path/newdir", nullptr);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, DeleteDir_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteDir", errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteDir", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.DeleteDir("gs://path/dir", nullptr));
}

TEST(RetryingFileSystemTest, DeleteDir_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("DeleteDir", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.DeleteDir("gs://path/dir", nullptr);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, GetFileSize_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("GetFileSize",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("GetFileSize", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  uint64 size;
  TF_EXPECT_OK(fs.GetFileSize("gs://path/file.txt", nullptr, &size));
}

TEST(RetryingFileSystemTest, GetFileSize_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("GetFileSize", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  uint64 size;
  const auto& status = fs.GetFileSize("gs://path/file.txt", nullptr, &size);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, RenameFile_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("RenameFile", errors::Unavailable("Something is wrong")),
       std::make_tuple("RenameFile", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.RenameFile("old_name", "new_name", nullptr));
}

TEST(RetryingFileSystemTest, RenameFile_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("RenameFile", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.RenameFile("old_name", "new_name", nullptr);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, Stat_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("Stat", errors::Unavailable("Something is wrong")),
       std::make_tuple("Stat", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  FileStatistics stat;
  TF_EXPECT_OK(fs.Stat("file_name", nullptr, &stat));
}

TEST(RetryingFileSystemTest, Stat_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("Stat", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  FileStatistics stat;
  const auto& status = fs.Stat("file_name", nullptr, &stat);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, FileExists_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("FileExists", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.FileExists("file_name", nullptr);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, FileExists_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("FileExists", errors::Unavailable("Something is wrong")),
       std::make_tuple("FileExists", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.FileExists("gs://path/dir", nullptr));
}

TEST(RetryingFileSystemTest, IsDirectory_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("IsDirectory",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("IsDirectory", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  TF_EXPECT_OK(fs.IsDirectory("gs://path/dir", nullptr));
}

TEST(RetryingFileSystemTest, IsDirectory_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls = CreateRetriableErrors("IsDirectory", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));

  const auto& status = fs.IsDirectory("gs://path/dir", nullptr);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, DeleteRecursively_SuccessWith2ndTry) {
  ExpectedCalls expected_fs_calls(
      {std::make_tuple("DeleteRecursively",
                       errors::Unavailable("Something is wrong")),
       std::make_tuple("DeleteRecursively", Status::OK())});
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));
  int64_t undeleted_files, undeleted_dirs;

  TF_EXPECT_OK(fs.DeleteRecursively("gs://path/dir", nullptr, &undeleted_files,
                                    &undeleted_dirs));
}

TEST(RetryingFileSystemTest, DeleteRecursively_AllRetriesFailed) {
  ExpectedCalls expected_fs_calls =
      CreateRetriableErrors("DeleteRecursively", 11);
  std::unique_ptr<MockFileSystem> base_fs(
      new MockFileSystem(expected_fs_calls));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));
  int64_t undeleted_files, undeleted_dirs;

  const auto& status = fs.DeleteRecursively("gs://path/dir", nullptr,
                                            &undeleted_files, &undeleted_dirs);
  EXPECT_TRUE(absl::StrContains(status.error_message(), "Retriable error #10"))
      << status;
}

TEST(RetryingFileSystemTest, FlushCaches) {
  ExpectedCalls none;
  bool flushed = false;
  std::unique_ptr<MockFileSystem> base_fs(new MockFileSystem(none, &flushed));
  RetryingFileSystem<MockFileSystem> fs(
      std::move(base_fs), RetryConfig(0 /* init_delay_time_us */));
  fs.FlushCaches(nullptr);
  EXPECT_TRUE(flushed);
}

}  // namespace
}  // namespace tensorflow
