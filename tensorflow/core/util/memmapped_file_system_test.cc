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
class MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_testDTcc() {
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
#include "tensorflow/core/util/memmapped_file_system.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/memmapped_file_system_writer.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif

namespace tensorflow {

namespace {

// Names of files in memmapped environment.
constexpr char kTensor1FileName[] = "memmapped_package://t1";
constexpr char kTensor2FileName[] = "memmapped_package://t2";
constexpr char kProtoFileName[] = "memmapped_package://b";
constexpr int kTestGraphDefVersion = 666;

Status CreateMemmappedFileSystemFile(const string& filename, bool corrupted,
                                     Tensor* test_tensor) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/util/memmapped_file_system_test.cc", "CreateMemmappedFileSystemFile");

  Env* env = Env::Default();
  MemmappedFileSystemWriter writer;
  TF_RETURN_IF_ERROR(writer.InitializeToFile(env, filename));

  // Try to write a tensor and proto.
  test::FillFn<float>(test_tensor,
                      [](int i) { return static_cast<float>(i * i); });

  TF_RETURN_IF_ERROR(writer.SaveTensor(*test_tensor, kTensor1FileName));

  // Create a proto with some fields.
  GraphDef graph_def;
  graph_def.mutable_versions()->set_producer(kTestGraphDefVersion);
  graph_def.mutable_versions()->set_min_consumer(kTestGraphDefVersion);
  TF_RETURN_IF_ERROR(writer.SaveProtobuf(graph_def, kProtoFileName));

  // Save a tensor after the proto to check that alignment works.
  test::FillFn<float>(test_tensor,
                      [](int i) { return static_cast<float>(i) * i * i; });
  TF_RETURN_IF_ERROR(writer.SaveTensor(*test_tensor, kTensor2FileName));

  if (!corrupted) {
    // Flush and close the file.
    TF_RETURN_IF_ERROR(writer.FlushAndClose());
  }
  return Status::OK();
}

TEST(MemmappedFileSystemTest, SimpleTest) {
  const TensorShape test_tensor_shape = {10, 200};
  Tensor test_tensor(DT_FLOAT, test_tensor_shape);
  const string dir = testing::TmpDir();
  const string filename = io::JoinPath(dir, "memmapped_env_test");
  TF_ASSERT_OK(CreateMemmappedFileSystemFile(filename, false, &test_tensor));

  // Check that we can memmap the created file.
  MemmappedEnv memmapped_env(Env::Default());
  TF_ASSERT_OK(memmapped_env.InitializeFromFile(filename));
  // Try to load a proto from the file.
  GraphDef test_graph_def;
  TF_EXPECT_OK(
      ReadBinaryProto(&memmapped_env, kProtoFileName, &test_graph_def));
  EXPECT_EQ(kTestGraphDefVersion, test_graph_def.versions().producer());
  EXPECT_EQ(kTestGraphDefVersion, test_graph_def.versions().min_consumer());
  // Check that we can correctly get a tensor memory.
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region;
  TF_ASSERT_OK(memmapped_env.NewReadOnlyMemoryRegionFromFile(kTensor2FileName,
                                                             &memory_region));

  // The memory region can be bigger but not less than Tensor size.
  ASSERT_GE(memory_region->length(), test_tensor.TotalBytes());
  EXPECT_EQ(test_tensor.tensor_data(),
            StringPiece(static_cast<const char*>(memory_region->data()),
                        test_tensor.TotalBytes()));
  // Check that GetFileSize works.
  uint64 file_size = 0;
  TF_ASSERT_OK(memmapped_env.GetFileSize(kTensor2FileName, &file_size));
  EXPECT_EQ(test_tensor.TotalBytes(), file_size);

  // Check that Stat works.
  FileStatistics stat;
  TF_ASSERT_OK(memmapped_env.Stat(kTensor2FileName, &stat));
  EXPECT_EQ(test_tensor.TotalBytes(), stat.length);

  // Check that if file not found correct error message returned.
  EXPECT_EQ(
      error::NOT_FOUND,
      memmapped_env.NewReadOnlyMemoryRegionFromFile("bla-bla", &memory_region)
          .code());

  // Check FileExists.
  TF_EXPECT_OK(memmapped_env.FileExists(kTensor2FileName));
  EXPECT_EQ(error::Code::NOT_FOUND,
            memmapped_env.FileExists("bla-bla-bla").code());
}

TEST(MemmappedFileSystemTest, NotInitialized) {
  MemmappedEnv memmapped_env(Env::Default());
  std::unique_ptr<ReadOnlyMemoryRegion> memory_region;
  EXPECT_EQ(
      error::FAILED_PRECONDITION,
      memmapped_env
          .NewReadOnlyMemoryRegionFromFile(kTensor1FileName, &memory_region)
          .code());
  std::unique_ptr<RandomAccessFile> file;
  EXPECT_EQ(error::FAILED_PRECONDITION,
            memmapped_env.NewRandomAccessFile(kProtoFileName, &file).code());
}

TEST(MemmappedFileSystemTest, Corrupted) {
  // Create a corrupted file (it is not closed it properly).
  const TensorShape test_tensor_shape = {100, 200};
  Tensor test_tensor(DT_FLOAT, test_tensor_shape);
  const string dir = testing::TmpDir();
  const string filename = io::JoinPath(dir, "memmapped_env_corrupted_test");
  TF_ASSERT_OK(CreateMemmappedFileSystemFile(filename, true, &test_tensor));
  MemmappedFileSystem memmapped_env;
  ASSERT_NE(memmapped_env.InitializeFromFile(Env::Default(), filename),
            Status::OK());
}

TEST(MemmappedFileSystemTest, ProxyToDefault) {
  MemmappedEnv memmapped_env(Env::Default());
  const string dir = testing::TmpDir();
  const string filename = io::JoinPath(dir, "test_file");
  // Check that we can create write and read ordinary file.
  std::unique_ptr<WritableFile> writable_file_temp;
  TF_ASSERT_OK(memmapped_env.NewAppendableFile(filename, &writable_file_temp));
  // Making sure to clean up after the test finishes.
  const auto adh = [&memmapped_env, &filename](WritableFile* f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSmemmapped_file_system_testDTcc mht_1(mht_1_v, 323, "", "./tensorflow/core/util/memmapped_file_system_test.cc", "lambda");

    delete f;
    TF_CHECK_OK(memmapped_env.DeleteFile(filename));
  };
  std::unique_ptr<WritableFile, decltype(adh)> writable_file(
      writable_file_temp.release(), adh);
  const string test_string = "bla-bla-bla";
  TF_ASSERT_OK(writable_file->Append(test_string));
  TF_ASSERT_OK(writable_file->Close());
  uint64 file_length = 0;
  TF_EXPECT_OK(memmapped_env.GetFileSize(filename, &file_length));
  EXPECT_EQ(test_string.length(), file_length);
  FileStatistics stat;
  TF_EXPECT_OK(memmapped_env.Stat(filename, &stat));
  EXPECT_EQ(test_string.length(), stat.length);
  std::unique_ptr<RandomAccessFile> random_access_file;
  TF_ASSERT_OK(
      memmapped_env.NewRandomAccessFile(filename, &random_access_file));
}

}  // namespace
}  // namespace tensorflow
