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
class MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc() {
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

#include "tensorflow/core/platform/env.h"

#include <sys/stat.h>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

string CreateTestFile(Env* env, const string& filename, int length) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/platform/env_test.cc", "CreateTestFile");

  string input(length, 0);
  for (int i = 0; i < length; i++) input[i] = i;
  TF_EXPECT_OK(WriteStringToFile(env, filename, input));
  return input;
}

GraphDef CreateTestProto() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/platform/env_test.cc", "CreateTestProto");

  GraphDef g;
  NodeDef* node = g.add_node();
  node->set_name("name1");
  node->set_op("op1");
  node = g.add_node();
  node->set_name("name2");
  node->set_op("op2");
  return g;
}

static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/platform/env_test.cc", "ExpectHasSubstr");

  EXPECT_TRUE(absl::StrContains(s, expected))
      << "'" << s << "' does not contain '" << expected << "'";
}

}  // namespace

string BaseDir() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/platform/env_test.cc", "BaseDir");
 return io::JoinPath(testing::TmpDir(), "base_dir"); }

class DefaultEnvTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/platform/env_test.cc", "SetUp");
 TF_CHECK_OK(env_->CreateDir(BaseDir())); }

  void TearDown() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_5(mht_5_v, 252, "", "./tensorflow/core/platform/env_test.cc", "TearDown");

    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(
        env_->DeleteRecursively(BaseDir(), &undeleted_files, &undeleted_dirs));
  }

  Env* env_ = Env::Default();
};

TEST_F(DefaultEnvTest, IncompleteReadOutOfRange) {
  const string filename = io::JoinPath(BaseDir(), "out_of_range");
  const string input = CreateTestFile(env_, filename, 2);
  std::unique_ptr<RandomAccessFile> f;
  TF_EXPECT_OK(env_->NewRandomAccessFile(filename, &f));

  // Reading past EOF should give an OUT_OF_RANGE error
  StringPiece result;
  char scratch[3];
  EXPECT_EQ(error::OUT_OF_RANGE, f->Read(0, 3, &result, scratch).code());
  EXPECT_EQ(input, result);

  // Exact read to EOF works.
  TF_EXPECT_OK(f->Read(0, 2, &result, scratch));
  EXPECT_EQ(input, result);
}

TEST_F(DefaultEnvTest, ReadFileToString) {
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1, (256 << 20) + 100}) {
    const string filename =
        io::JoinPath(BaseDir(), "bar", "..", strings::StrCat("file", length));

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);

    // Read the file back and check equality
    string output;
    TF_EXPECT_OK(ReadFileToString(env_, filename, &output));
    EXPECT_EQ(length, output.size());
    EXPECT_EQ(input, output);

    // Obtain stats.
    FileStatistics stat;
    TF_EXPECT_OK(env_->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_FALSE(stat.is_directory);
  }
}

TEST_F(DefaultEnvTest, ReadWriteBinaryProto) {
  const GraphDef proto = CreateTestProto();
  const string filename = strings::StrCat(BaseDir(), "binary_proto");

  // Write the binary proto
  TF_EXPECT_OK(WriteBinaryProto(env_, filename, proto));

  // Read the binary proto back in and make sure it's the same.
  GraphDef result;
  TF_EXPECT_OK(ReadBinaryProto(env_, filename, &result));
  EXPECT_EQ(result.DebugString(), proto.DebugString());

  // Reading as text or binary proto should also work.
  GraphDef result2;
  TF_EXPECT_OK(ReadTextOrBinaryProto(env_, filename, &result2));
  EXPECT_EQ(result2.DebugString(), proto.DebugString());
}

TEST_F(DefaultEnvTest, ReadWriteTextProto) {
  const GraphDef proto = CreateTestProto();
  const string filename = strings::StrCat(BaseDir(), "text_proto");

  // Write the text proto
  string as_text;
  EXPECT_TRUE(protobuf::TextFormat::PrintToString(proto, &as_text));
  TF_EXPECT_OK(WriteStringToFile(env_, filename, as_text));

  // Read the text proto back in and make sure it's the same.
  GraphDef result;
  TF_EXPECT_OK(ReadTextProto(env_, filename, &result));
  EXPECT_EQ(result.DebugString(), proto.DebugString());

  // Reading as text or binary proto should also work.
  GraphDef result2;
  TF_EXPECT_OK(ReadTextOrBinaryProto(env_, filename, &result2));
  EXPECT_EQ(result2.DebugString(), proto.DebugString());
}

TEST_F(DefaultEnvTest, FileToReadonlyMemoryRegion) {
  for (const int length : {1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    const string filename =
        io::JoinPath(BaseDir(), strings::StrCat("file", length));

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);

    // Create the region.
    std::unique_ptr<ReadOnlyMemoryRegion> region;
    TF_EXPECT_OK(env_->NewReadOnlyMemoryRegionFromFile(filename, &region));
    ASSERT_NE(region, nullptr);
    EXPECT_EQ(length, region->length());
    EXPECT_EQ(input, string(reinterpret_cast<const char*>(region->data()),
                            region->length()));
    FileStatistics stat;
    TF_EXPECT_OK(env_->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_FALSE(stat.is_directory);
  }
}

TEST_F(DefaultEnvTest, DeleteRecursively) {
  // Build a directory structure rooted at root_dir.
  // root_dir -> dirs: child_dir1, child_dir2; files: root_file1, root_file2
  // child_dir1 -> files: child1_file1
  // child_dir2 -> empty
  const string parent_dir = io::JoinPath(BaseDir(), "root_dir");
  const string child_dir1 = io::JoinPath(parent_dir, "child_dir1");
  const string child_dir2 = io::JoinPath(parent_dir, "child_dir2");
  TF_EXPECT_OK(env_->CreateDir(parent_dir));
  const string root_file1 = io::JoinPath(parent_dir, "root_file1");
  const string root_file2 = io::JoinPath(parent_dir, "root_file2");
  const string root_file3 = io::JoinPath(parent_dir, ".root_file3");
  CreateTestFile(env_, root_file1, 100);
  CreateTestFile(env_, root_file2, 100);
  CreateTestFile(env_, root_file3, 100);
  TF_EXPECT_OK(env_->CreateDir(child_dir1));
  const string child1_file1 = io::JoinPath(child_dir1, "child1_file1");
  CreateTestFile(env_, child1_file1, 100);
  TF_EXPECT_OK(env_->CreateDir(child_dir2));

  int64_t undeleted_files, undeleted_dirs;
  TF_EXPECT_OK(
      env_->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs));
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(0, undeleted_dirs);
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(root_file1).code());
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(root_file2).code());
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(root_file3).code());
  EXPECT_EQ(error::Code::NOT_FOUND, env_->FileExists(child1_file1).code());
}

TEST_F(DefaultEnvTest, DeleteRecursivelyFail) {
  // Try to delete a non-existent directory.
  const string parent_dir = io::JoinPath(BaseDir(), "root_dir");

  int64_t undeleted_files, undeleted_dirs;
  Status s =
      env_->DeleteRecursively(parent_dir, &undeleted_files, &undeleted_dirs);
  EXPECT_EQ(error::Code::NOT_FOUND, s.code());
  EXPECT_EQ(0, undeleted_files);
  EXPECT_EQ(1, undeleted_dirs);
}

TEST_F(DefaultEnvTest, RecursivelyCreateDir) {
  const string create_path = io::JoinPath(BaseDir(), "a", "b", "c", "d");
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));  // repeat creation.
  TF_EXPECT_OK(env_->FileExists(create_path));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirEmpty) {
  TF_CHECK_OK(env_->RecursivelyCreateDir(""));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirSubdirsExist) {
  // First create a/b.
  const string subdir_path = io::JoinPath(BaseDir(), "a", "b");
  TF_CHECK_OK(env_->CreateDir(io::JoinPath(BaseDir(), "a")));
  TF_CHECK_OK(env_->CreateDir(subdir_path));
  TF_EXPECT_OK(env_->FileExists(subdir_path));

  // Now try to recursively create a/b/c/d/
  const string create_path = io::JoinPath(BaseDir(), "a", "b", "c", "d");
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env_->RecursivelyCreateDir(create_path));  // repeat creation.
  TF_EXPECT_OK(env_->FileExists(create_path));
  TF_EXPECT_OK(env_->FileExists(io::JoinPath(BaseDir(), "a", "b", "c")));
}

TEST_F(DefaultEnvTest, LocalFileSystem) {
  // Test filename with file:// syntax.
  int expected_num_files = 0;
  std::vector<string> matching_paths;
  for (const int length : {0, 1, 1212, 2553, 4928, 8196, 9000, (1 << 20) - 1,
                           1 << 20, (1 << 20) + 1}) {
    string filename = io::JoinPath(BaseDir(), strings::StrCat("len", length));

    filename = strings::StrCat("file://", filename);

    // Write a file with the given length
    const string input = CreateTestFile(env_, filename, length);
    ++expected_num_files;

    // Ensure that GetMatchingPaths works as intended.
    TF_EXPECT_OK(env_->GetMatchingPaths(
        // Try it with the "file://" URI scheme.
        strings::StrCat("file://", io::JoinPath(BaseDir(), "l*")),
        &matching_paths));
    EXPECT_EQ(expected_num_files, matching_paths.size());
    TF_EXPECT_OK(env_->GetMatchingPaths(
        // Try it without any URI scheme.
        io::JoinPath(BaseDir(), "l*"), &matching_paths));
    EXPECT_EQ(expected_num_files, matching_paths.size());

    // Read the file back and check equality
    string output;
    TF_EXPECT_OK(ReadFileToString(env_, filename, &output));
    EXPECT_EQ(length, output.size());
    EXPECT_EQ(input, output);

    FileStatistics stat;
    TF_EXPECT_OK(env_->Stat(filename, &stat));
    EXPECT_EQ(length, stat.length);
    EXPECT_FALSE(stat.is_directory);
  }
}

TEST_F(DefaultEnvTest, SleepForMicroseconds) {
  const int64_t start = env_->NowMicros();
  const int64_t sleep_time = 1e6 + 5e5;
  env_->SleepForMicroseconds(sleep_time);
  const int64_t delta = env_->NowMicros() - start;

  // Subtract 200 from the sleep_time for this check because NowMicros can
  // sometimes give slightly inconsistent values between the start and the
  // finish (e.g. because the two calls run on different CPUs).
  EXPECT_GE(delta, sleep_time - 200);
}

class TmpDirFileSystem : public NullFileSystem {
 public:
  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status FileExists(const string& dir, TransactionToken* token) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_6(mht_6_v, 489, "", "./tensorflow/core/platform/env_test.cc", "FileExists");

    StringPiece scheme, host, path;
    io::ParseURI(dir, &scheme, &host, &path);
    if (path.empty()) return errors::NotFound(dir, " not found");
    // The special "flushed" file exists only if the filesystem's caches have
    // been flushed.
    if (path == "/flushed") {
      if (flushed_) {
        return Status::OK();
      } else {
        return errors::NotFound("FlushCaches() not called yet");
      }
    }
    return Env::Default()->FileExists(io::JoinPath(BaseDir(), path));
  }

  Status CreateDir(const string& dir, TransactionToken* token) override {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_7(mht_7_v, 509, "", "./tensorflow/core/platform/env_test.cc", "CreateDir");

    StringPiece scheme, host, path;
    io::ParseURI(dir, &scheme, &host, &path);
    if (scheme != "tmpdirfs") {
      return errors::FailedPrecondition("scheme must be tmpdirfs");
    }
    if (host != "testhost") {
      return errors::FailedPrecondition("host must be testhost");
    }
    Status status = Env::Default()->CreateDir(io::JoinPath(BaseDir(), path));
    if (status.ok()) {
      // Record that we have created this directory so `IsDirectory` works.
      created_directories_.push_back(std::string(path));
    }
    return status;
  }

  Status IsDirectory(const string& dir, TransactionToken* token) override {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_8(mht_8_v, 530, "", "./tensorflow/core/platform/env_test.cc", "IsDirectory");

    StringPiece scheme, host, path;
    io::ParseURI(dir, &scheme, &host, &path);
    for (const auto& existing_dir : created_directories_)
      if (existing_dir == path) return Status::OK();
    return errors::NotFound(dir, " not found");
  }

  void FlushCaches(TransactionToken* token) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_9(mht_9_v, 541, "", "./tensorflow/core/platform/env_test.cc", "FlushCaches");
 flushed_ = true; }

 private:
  bool flushed_ = false;
  std::vector<std::string> created_directories_ = {"/"};
};

REGISTER_FILE_SYSTEM("tmpdirfs", TmpDirFileSystem);

TEST_F(DefaultEnvTest, FlushFileSystemCaches) {
  Env* env = Env::Default();
  const string flushed =
      strings::StrCat("tmpdirfs://", io::JoinPath("testhost", "flushed"));
  EXPECT_EQ(error::Code::NOT_FOUND, env->FileExists(flushed).code());
  TF_EXPECT_OK(env->FlushFileSystemCaches());
  TF_EXPECT_OK(env->FileExists(flushed));
}

TEST_F(DefaultEnvTest, RecursivelyCreateDirWithUri) {
  Env* env = Env::Default();
  const string create_path = strings::StrCat(
      "tmpdirfs://", io::JoinPath("testhost", "a", "b", "c", "d"));
  EXPECT_EQ(error::Code::NOT_FOUND, env->FileExists(create_path).code());
  TF_CHECK_OK(env->RecursivelyCreateDir(create_path));
  TF_CHECK_OK(env->RecursivelyCreateDir(create_path));  // repeat creation.
  TF_EXPECT_OK(env->FileExists(create_path));
}

TEST_F(DefaultEnvTest, GetExecutablePath) {
  Env* env = Env::Default();
  TF_EXPECT_OK(env->FileExists(env->GetExecutablePath()));
}

TEST_F(DefaultEnvTest, LocalTempFilename) {
  Env* env = Env::Default();
  string filename;
  EXPECT_TRUE(env->LocalTempFilename(&filename));
  EXPECT_FALSE(env->FileExists(filename).ok());

  // Write something to the temporary file.
  std::unique_ptr<WritableFile> file_to_write;
  TF_CHECK_OK(env->NewWritableFile(filename, &file_to_write));
#if defined(PLATFORM_GOOGLE)
  TF_CHECK_OK(file_to_write->Append("Nu"));
  TF_CHECK_OK(file_to_write->Append(absl::Cord("ll")));
#else
  // TODO(ebrevdo): Remove this version.
  TF_CHECK_OK(file_to_write->Append("Null"));
#endif
  TF_CHECK_OK(file_to_write->Close());
  TF_CHECK_OK(env->FileExists(filename));

  // Open the file in append mode, check that Tell() reports the appropriate
  // offset.
  std::unique_ptr<WritableFile> file_to_append;
  TF_CHECK_OK(env->NewAppendableFile(filename, &file_to_append));
  int64_t pos;
  TF_CHECK_OK(file_to_append->Tell(&pos));
  ASSERT_EQ(4, pos);

  // Read from the temporary file and check content.
  std::unique_ptr<RandomAccessFile> file_to_read;
  TF_CHECK_OK(env->NewRandomAccessFile(filename, &file_to_read));
  StringPiece content;
  char scratch[1024];
  CHECK_EQ(
      error::OUT_OF_RANGE,
      file_to_read->Read(/*offset=*/0, /*n=*/1024, &content, scratch).code());
  EXPECT_EQ("Null", content);

  // Delete the temporary file.
  TF_CHECK_OK(env->DeleteFile(filename));
  EXPECT_FALSE(env->FileExists(filename).ok());
}

TEST_F(DefaultEnvTest, CreateUniqueFileName) {
  Env* env = Env::Default();

  string prefix = "tempfile-prefix-";
  string suffix = ".tmp";
  string filename = prefix;

  EXPECT_TRUE(env->CreateUniqueFileName(&filename, suffix));

  EXPECT_TRUE(absl::StartsWith(filename, prefix));
  EXPECT_TRUE(str_util::EndsWith(filename, suffix));
}

TEST_F(DefaultEnvTest, GetProcessId) {
  Env* env = Env::Default();
  EXPECT_NE(env->GetProcessId(), 0);
}

TEST_F(DefaultEnvTest, GetThreadInformation) {
  Env* env = Env::Default();
  // TODO(fishx): Turn on this test for Apple.
#if !defined(__APPLE__)
  EXPECT_NE(env->GetCurrentThreadId(), 0);
#endif
  string thread_name;
  bool res = env->GetCurrentThreadName(&thread_name);
#if defined(PLATFORM_WINDOWS) || defined(__ANDROID__)
  EXPECT_FALSE(res);
#elif !defined(__APPLE__)
  EXPECT_TRUE(res);
  EXPECT_GT(thread_name.size(), 0);
#endif
}

TEST_F(DefaultEnvTest, GetChildThreadInformation) {
  Env* env = Env::Default();
  Thread* child_thread = env->StartThread({}, "tf_child_thread", [env]() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSplatformPSenv_testDTcc mht_10(mht_10_v, 655, "", "./tensorflow/core/platform/env_test.cc", "lambda");

  // TODO(fishx): Turn on this test for Apple.
#if !defined(__APPLE__)
    EXPECT_NE(env->GetCurrentThreadId(), 0);
#endif
    string thread_name;
    bool res = env->GetCurrentThreadName(&thread_name);
    EXPECT_TRUE(res);
    ExpectHasSubstr(thread_name, "tf_child_thread");
  });
  delete child_thread;
}

}  // namespace tensorflow
