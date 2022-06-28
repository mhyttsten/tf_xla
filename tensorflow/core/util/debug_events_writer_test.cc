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
class MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc() {
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

#include "tensorflow/core/util/debug_events_writer.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {
namespace tfdbg {

// shorthand
Env* env() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/util/debug_events_writer_test.cc", "env");
 return Env::Default(); }

class DebugEventsWriterTest : public ::testing::Test {
 public:
  static string GetDebugEventFileName(DebugEventsWriter* writer,
                                      DebugEventFileType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/util/debug_events_writer_test.cc", "GetDebugEventFileName");

    return writer->FileName(type);
  }

  static void ReadDebugEventProtos(DebugEventsWriter* writer,
                                   DebugEventFileType type,
                                   std::vector<DebugEvent>* protos) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/util/debug_events_writer_test.cc", "ReadDebugEventProtos");

    protos->clear();
    const string filename = writer->FileName(type);
    std::unique_ptr<RandomAccessFile> debug_events_file;
    TF_CHECK_OK(env()->NewRandomAccessFile(filename, &debug_events_file));
    io::RecordReader* reader = new io::RecordReader(debug_events_file.get());

    uint64 offset = 0;
    DebugEvent actual;
    while (ReadDebugEventProto(reader, &offset, &actual)) {
      protos->push_back(actual);
    }

    delete reader;
  }

  static bool ReadDebugEventProto(io::RecordReader* reader, uint64* offset,
                                  DebugEvent* proto) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/util/debug_events_writer_test.cc", "ReadDebugEventProto");

    tstring record;
    Status s = reader->ReadRecord(offset, &record);
    if (!s.ok()) {
      return false;
    }
    return ParseProtoUnlimited(proto, record);
  }

  void SetUp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_4(mht_4_v, 251, "", "./tensorflow/core/util/debug_events_writer_test.cc", "SetUp");

    dump_root_ = io::JoinPath(
        testing::TmpDir(),
        strings::Printf("%010lld", static_cast<long long>(env()->NowMicros())));
    tfdbg_run_id_ = "test_tfdbg_run_id";
  }

  void TearDown() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_5(mht_5_v, 261, "", "./tensorflow/core/util/debug_events_writer_test.cc", "TearDown");

    if (env()->IsDirectory(dump_root_).ok()) {
      int64_t undeleted_files = 0;
      int64_t undeleted_dirs = 0;
      TF_ASSERT_OK(env()->DeleteRecursively(dump_root_, &undeleted_files,
                                            &undeleted_dirs));
      ASSERT_EQ(0, undeleted_files);
      ASSERT_EQ(0, undeleted_dirs);
    }
  }

  string dump_root_;
  string tfdbg_run_id_;
};

TEST_F(DebugEventsWriterTest, GetDebugEventsWriterSameRootGivesSameObject) {
  // Test the per-dump_root_ singleton pattern.
  DebugEventsWriter* writer_1 = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  DebugEventsWriter* writer_2 = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  EXPECT_EQ(writer_1, writer_2);
}

TEST_F(DebugEventsWriterTest, ConcurrentGetDebugEventsWriterSameDumpRoot) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 4);

  std::vector<DebugEventsWriter*> writers;
  mutex mu;
  auto fn = [this, &writers, &mu]() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
        dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
    {
      mutex_lock l(mu);
      writers.push_back(writer);
    }
  };
  for (size_t i = 0; i < 4; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  EXPECT_EQ(writers.size(), 4);
  EXPECT_EQ(writers[0], writers[1]);
  EXPECT_EQ(writers[1], writers[2]);
  EXPECT_EQ(writers[2], writers[3]);
}

TEST_F(DebugEventsWriterTest, ConcurrentGetDebugEventsWriterDiffDumpRoots) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 3);

  std::atomic_int_fast64_t counter(0);
  std::vector<DebugEventsWriter*> writers;
  mutex mu;
  auto fn = [this, &counter, &writers, &mu]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_7(mht_7_v, 323, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    const string new_dump_root =
        io::JoinPath(dump_root_, strings::Printf("%ld", counter.fetch_add(1)));
    DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
        new_dump_root, tfdbg_run_id_,
        DebugEventsWriter::kDefaultCyclicBufferSize);
    {
      mutex_lock l(mu);
      writers.push_back(writer);
    }
  };
  for (size_t i = 0; i < 3; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  EXPECT_EQ(writers.size(), 3);
  EXPECT_NE(writers[0], writers[1]);
  EXPECT_NE(writers[0], writers[2]);
  EXPECT_NE(writers[1], writers[2]);
}

TEST_F(DebugEventsWriterTest, GetDebugEventsWriterDifferentRoots) {
  // Test the DebugEventsWriters for different directories are different.
  DebugEventsWriter* writer_1 = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  const string dump_root_2 = io::JoinPath(dump_root_, "subdirectory");
  DebugEventsWriter* writer_2 = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_2, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  EXPECT_NE(writer_1, writer_2);
}

TEST_F(DebugEventsWriterTest, GetAndInitDebugEventsWriter) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());
  TF_ASSERT_OK(writer->Close());

  // Verify the metadata file's content.
  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  // Check the content of the file version string.
  const string file_version = actuals[0].debug_metadata().file_version();
  EXPECT_EQ(file_version.find(DebugEventsWriter::kVersionPrefix), 0);
  EXPECT_GT(file_version.size(), strlen(DebugEventsWriter::kVersionPrefix));
  // Check the tfdbg run ID.
  EXPECT_EQ(actuals[0].debug_metadata().tfdbg_run_id(), "test_tfdbg_run_id");

  // Verify that the .source_files file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  // Verify that the .stack_frames file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
}

TEST_F(DebugEventsWriterTest, CallingCloseWithoutInitIsOkay) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, CallingCloseTwiceIsOkay) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Close());
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, ConcurrentInitCalls) {
  // Test that concurrent calls to Init() works correctly.
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 4);
  auto fn = [&writer]() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_8(mht_8_v, 402, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");
 TF_ASSERT_OK(writer->Init()); };
  for (size_t i = 0; i < 3; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  // Verify the metadata file's content.
  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  // Check the content of the file version string.
  const string file_version = actuals[0].debug_metadata().file_version();
  EXPECT_EQ(file_version.find(DebugEventsWriter::kVersionPrefix), 0);
  EXPECT_GT(file_version.size(), strlen(DebugEventsWriter::kVersionPrefix));
  EXPECT_EQ(actuals[0].debug_metadata().tfdbg_run_id(), "test_tfdbg_run_id");

  // Verify that the .source_files file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  // Verify that the .stack_frames file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
}

TEST_F(DebugEventsWriterTest, InitTwiceDoesNotCreateNewMetadataFile) {
  // Test that Init() is idempotent.
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  EXPECT_EQ(actuals[0].debug_metadata().tfdbg_run_id(), "test_tfdbg_run_id");
  EXPECT_GE(actuals[0].debug_metadata().file_version().size(), 0);

  string metadata_path_1 =
      GetDebugEventFileName(writer, DebugEventFileType::METADATA);
  TF_ASSERT_OK(writer->Init());
  EXPECT_EQ(GetDebugEventFileName(writer, DebugEventFileType::METADATA),
            metadata_path_1);
  TF_ASSERT_OK(writer->Close());

  // Verify the metadata file's content.
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  EXPECT_EQ(actuals[0].debug_metadata().tfdbg_run_id(), "test_tfdbg_run_id");
  EXPECT_GE(actuals[0].debug_metadata().file_version().size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteSourceFile) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  SourceFile* source_file_1 = new SourceFile();
  source_file_1->set_file_path("/home/tf_programs/main.py");
  source_file_1->set_host_name("localhost.localdomain");
  source_file_1->add_lines("import tensorflow as tf");
  source_file_1->add_lines("");
  source_file_1->add_lines("print(tf.constant([42.0]))");
  source_file_1->add_lines("");
  TF_ASSERT_OK(writer->WriteSourceFile(source_file_1));

  SourceFile* source_file_2 = new SourceFile();
  source_file_2->set_file_path("/home/tf_programs/train.py");
  source_file_2->set_host_name("localhost.localdomain");
  source_file_2->add_lines("import tensorflow.keras as keras");
  source_file_2->add_lines("");
  source_file_2->add_lines("model = keras.Sequential()");
  TF_ASSERT_OK(writer->WriteSourceFile(source_file_2));

  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 2);
  EXPECT_GT(actuals[0].wall_time(), 0);
  EXPECT_GT(actuals[1].wall_time(), actuals[0].wall_time());

  SourceFile actual_source_file_1 = actuals[0].source_file();
  EXPECT_EQ(actual_source_file_1.file_path(), "/home/tf_programs/main.py");
  EXPECT_EQ(actual_source_file_1.host_name(), "localhost.localdomain");
  EXPECT_EQ(actual_source_file_1.lines().size(), 4);
  EXPECT_EQ(actual_source_file_1.lines()[0], "import tensorflow as tf");
  EXPECT_EQ(actual_source_file_1.lines()[1], "");
  EXPECT_EQ(actual_source_file_1.lines()[2], "print(tf.constant([42.0]))");
  EXPECT_EQ(actual_source_file_1.lines()[3], "");

  SourceFile actual_source_file_2 = actuals[1].source_file();
  EXPECT_EQ(actual_source_file_2.file_path(), "/home/tf_programs/train.py");
  EXPECT_EQ(actual_source_file_2.host_name(), "localhost.localdomain");
  EXPECT_EQ(actual_source_file_2.lines().size(), 3);
  EXPECT_EQ(actual_source_file_2.lines()[0],
            "import tensorflow.keras as keras");
  EXPECT_EQ(actual_source_file_2.lines()[1], "");
  EXPECT_EQ(actual_source_file_2.lines()[2], "model = keras.Sequential()");

  // Verify no cross talk in the other non-execution debug-event files.
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteStackFramesFile) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  StackFrameWithId* stack_frame_1 = new StackFrameWithId();
  stack_frame_1->set_id("deadbeaf");
  GraphDebugInfo::FileLineCol* file_line_col =
      stack_frame_1->mutable_file_line_col();
  file_line_col->set_file_index(12);
  file_line_col->set_line(20);
  file_line_col->set_col(2);
  file_line_col->set_func("my_func");
  file_line_col->set_code("  x = y + z");

  StackFrameWithId* stack_frame_2 = new StackFrameWithId();
  stack_frame_2->set_id("eeeeeeec");
  file_line_col = stack_frame_2->mutable_file_line_col();
  file_line_col->set_file_index(12);
  file_line_col->set_line(21);
  file_line_col->set_col(4);
  file_line_col->set_func("my_func");
  file_line_col->set_code("  x = x ** 2.0");

  TF_ASSERT_OK(writer->WriteStackFrameWithId(stack_frame_1));
  TF_ASSERT_OK(writer->WriteStackFrameWithId(stack_frame_2));
  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 2);
  EXPECT_GT(actuals[0].wall_time(), 0);
  EXPECT_GT(actuals[1].wall_time(), actuals[0].wall_time());

  StackFrameWithId actual_stack_frame_1 = actuals[0].stack_frame_with_id();
  EXPECT_EQ(actual_stack_frame_1.id(), "deadbeaf");
  GraphDebugInfo::FileLineCol file_line_col_1 =
      actual_stack_frame_1.file_line_col();
  EXPECT_EQ(file_line_col_1.file_index(), 12);
  EXPECT_EQ(file_line_col_1.line(), 20);
  EXPECT_EQ(file_line_col_1.col(), 2);
  EXPECT_EQ(file_line_col_1.func(), "my_func");
  EXPECT_EQ(file_line_col_1.code(), "  x = y + z");

  StackFrameWithId actual_stack_frame_2 = actuals[1].stack_frame_with_id();
  EXPECT_EQ(actual_stack_frame_2.id(), "eeeeeeec");
  GraphDebugInfo::FileLineCol file_line_col_2 =
      actual_stack_frame_2.file_line_col();
  EXPECT_EQ(file_line_col_2.file_index(), 12);
  EXPECT_EQ(file_line_col_2.line(), 21);
  EXPECT_EQ(file_line_col_2.col(), 4);
  EXPECT_EQ(file_line_col_2.func(), "my_func");
  EXPECT_EQ(file_line_col_2.code(), "  x = x ** 2.0");

  // Verify no cross talk in the other non-execution debug-event files.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteGraphOpCreationAndDebuggedGraph) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  GraphOpCreation* graph_op_creation = new GraphOpCreation();
  graph_op_creation->set_op_type("MatMul");
  graph_op_creation->set_op_name("Dense_1/MatMul");
  TF_ASSERT_OK(writer->WriteGraphOpCreation(graph_op_creation));

  DebuggedGraph* debugged_graph = new DebuggedGraph();
  debugged_graph->set_graph_id("deadbeaf");
  debugged_graph->set_graph_name("my_func_graph");
  TF_ASSERT_OK(writer->WriteDebuggedGraph(debugged_graph));

  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 2);
  EXPECT_GT(actuals[0].wall_time(), 0);
  EXPECT_GT(actuals[1].wall_time(), actuals[0].wall_time());

  GraphOpCreation actual_op_creation = actuals[0].graph_op_creation();
  EXPECT_EQ(actual_op_creation.op_type(), "MatMul");
  EXPECT_EQ(actual_op_creation.op_name(), "Dense_1/MatMul");

  DebuggedGraph actual_debugged_graph = actuals[1].debugged_graph();
  EXPECT_EQ(actual_debugged_graph.graph_id(), "deadbeaf");
  EXPECT_EQ(actual_debugged_graph.graph_name(), "my_func_graph");

  // Verify no cross talk in the other non-execution debug-event files.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, ConcurrentWriteCallsToTheSameFile) {
  const size_t kConcurrentWrites = 100;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_9(mht_9_v, 629, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    const string file_path = strings::Printf(
        "/home/tf_programs/program_%.3ld.py", counter.fetch_add(1));
    SourceFile* source_file = new SourceFile();
    source_file->set_file_path(file_path);
    source_file->set_host_name("localhost.localdomain");
    TF_ASSERT_OK(writer->WriteSourceFile(source_file));
  };
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites);
  std::vector<string> file_paths;
  std::vector<string> host_names;
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    file_paths.push_back(actuals[i].source_file().file_path());
    host_names.push_back(actuals[i].source_file().host_name());
  }
  std::sort(file_paths.begin(), file_paths.end());
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    EXPECT_EQ(file_paths[i],
              strings::Printf("/home/tf_programs/program_%.3ld.py", i));
    EXPECT_EQ(host_names[i], "localhost.localdomain");
  }
}

TEST_F(DebugEventsWriterTest, ConcurrentWriteAndFlushCallsToTheSameFile) {
  const size_t kConcurrentWrites = 100;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_10(mht_10_v, 673, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    const string file_path = strings::Printf(
        "/home/tf_programs/program_%.3ld.py", counter.fetch_add(1));
    SourceFile* source_file = new SourceFile();
    source_file->set_file_path(file_path);
    source_file->set_host_name("localhost.localdomain");
    TF_ASSERT_OK(writer->WriteSourceFile(source_file));
    TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  };
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites);
  std::vector<string> file_paths;
  std::vector<string> host_names;
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    file_paths.push_back(actuals[i].source_file().file_path());
    host_names.push_back(actuals[i].source_file().host_name());
  }
  std::sort(file_paths.begin(), file_paths.end());
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    EXPECT_EQ(file_paths[i],
              strings::Printf("/home/tf_programs/program_%.3ld.py", i));
    EXPECT_EQ(host_names[i], "localhost.localdomain");
  }
}

TEST_F(DebugEventsWriterTest, ConcurrentWriteCallsToTheDifferentFiles) {
  const int32_t kConcurrentWrites = 30;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 10);
  std::atomic_int_fast32_t counter(0);
  auto fn = [&writer, &counter]() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_11(mht_11_v, 718, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    const int32_t index = counter.fetch_add(1);
    if (index % 3 == 0) {
      SourceFile* source_file = new SourceFile();
      source_file->set_file_path(
          strings::Printf("/home/tf_programs/program_%.2d.py", index));
      source_file->set_host_name("localhost.localdomain");
      TF_ASSERT_OK(writer->WriteSourceFile(source_file));
    } else if (index % 3 == 1) {
      StackFrameWithId* stack_frame = new StackFrameWithId();
      stack_frame->set_id(strings::Printf("e%.2d", index));
      TF_ASSERT_OK(writer->WriteStackFrameWithId(stack_frame));
    } else {
      GraphOpCreation* op_creation = new GraphOpCreation();
      op_creation->set_op_type("Log");
      op_creation->set_op_name(strings::Printf("Log_%.2d", index));
      TF_ASSERT_OK(writer->WriteGraphOpCreation(op_creation));
    }
  };
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites / 3);
  std::vector<string> file_paths;
  std::vector<string> host_names;
  for (int32_t i = 0; i < kConcurrentWrites / 3; ++i) {
    file_paths.push_back(actuals[i].source_file().file_path());
    host_names.push_back(actuals[i].source_file().host_name());
  }
  std::sort(file_paths.begin(), file_paths.end());
  for (int32_t i = 0; i < kConcurrentWrites / 3; ++i) {
    EXPECT_EQ(file_paths[i],
              strings::Printf("/home/tf_programs/program_%.2d.py", i * 3));
    EXPECT_EQ(host_names[i], "localhost.localdomain");
  }

  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites / 3);
  std::vector<string> stack_frame_ids;
  for (int32_t i = 0; i < kConcurrentWrites / 3; ++i) {
    stack_frame_ids.push_back(actuals[i].stack_frame_with_id().id());
  }
  std::sort(stack_frame_ids.begin(), stack_frame_ids.end());
  for (int32_t i = 0; i < kConcurrentWrites / 3; ++i) {
    EXPECT_EQ(stack_frame_ids[i], strings::Printf("e%.2d", i * 3 + 1));
  }

  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites / 3);
  std::vector<string> op_types;
  std::vector<string> op_names;
  for (int32_t i = 0; i < kConcurrentWrites / 3; ++i) {
    op_types.push_back(actuals[i].graph_op_creation().op_type());
    op_names.push_back(actuals[i].graph_op_creation().op_name());
  }
  std::sort(op_names.begin(), op_names.end());
  for (int32_t i = 0; i < kConcurrentWrites / 3; ++i) {
    EXPECT_EQ(op_types[i], "Log");
    EXPECT_EQ(op_names[i], strings::Printf("Log_%.2d", i * 3 + 2));
  }
}

TEST_F(DebugEventsWriterTest, WriteExecutionWithCyclicBufferNoFlush) {
  // Verify that no writing to disk happens until the flushing method is called.
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    Execution* execution = new Execution();
    execution->set_op_type("Log");
    execution->add_input_tensor_ids(i);
    TF_ASSERT_OK(writer->WriteExecution(execution));
  }

  std::vector<DebugEvent> actuals;
  // Before FlushExecutionFiles() is called, the file should be empty.
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), 0);

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, WriteExecutionWithCyclicBufferFlush) {
  // Verify that writing to disk happens when the flushing method is called.
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    Execution* execution = new Execution();
    execution->set_op_type("Log");
    execution->add_input_tensor_ids(i);
    TF_ASSERT_OK(writer->WriteExecution(execution));
  }

  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  // Expect there to be only the last kCyclicBufferSize debug events,
  // and the order should be correct.
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), kCyclicBufferSize);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    EXPECT_EQ(actuals[i].execution().op_type(), "Log");
    EXPECT_EQ(actuals[i].execution().input_tensor_ids().size(), 1);
    EXPECT_EQ(actuals[i].execution().input_tensor_ids()[0],
              kCyclicBufferSize + i);
  }

  // Second, write more than the capacity of the circular buffer,
  // in a concurrent fashion.
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_12(mht_12_v, 849, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    Execution* execution = new Execution();
    execution->set_op_type("Abs");
    execution->add_input_tensor_ids(counter.fetch_add(1));
    TF_ASSERT_OK(writer->WriteExecution(execution));
  };
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;
  TF_ASSERT_OK(writer->Close());

  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  // NOTE: This includes the files from the first stage above, because the
  // .execution file hasn't changed.
  EXPECT_EQ(actuals.size(), kCyclicBufferSize * 2);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    const size_t index = i + kCyclicBufferSize;
    EXPECT_EQ(actuals[index].execution().op_type(), "Abs");
    EXPECT_EQ(actuals[index].execution().input_tensor_ids().size(), 1);
    EXPECT_GE(actuals[index].execution().input_tensor_ids()[0], 0);
    EXPECT_LE(actuals[index].execution().input_tensor_ids()[0],
              kCyclicBufferSize * 2);
  }

  // Verify no cross-talk.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteGrahExecutionTraceWithCyclicBufferNoFlush) {
  // Check no writing to disk happens before the flushing method is called.
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(strings::Printf("graph_%.2ld", i));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  }

  std::vector<DebugEvent> actuals;
  // Before FlushExecutionFiles() is called, the file should be empty.
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 0);

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, WriteGrahExecutionTraceWithoutPreviousInitCall) {
  const size_t kCyclicBufferSize = -1;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, kCyclicBufferSize);
  // NOTE(cais): `writer->Init()` is not called here before
  // WriteGraphExecutionTrace() is called. This test checks that this is okay
  // and the `GraphExecutionTrace` gets written correctly even without `Init()`
  // being called first. This scenario can happen when a TF Graph with tfdbg
  // debug ops are executed on a remote TF server.

  GraphExecutionTrace* trace = new GraphExecutionTrace();
  trace->set_tfdbg_context_id(strings::Printf("graph_0"));
  TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_EQ(actuals[0].graph_execution_trace().tfdbg_context_id(), "graph_0");

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, WriteGrahExecutionTraceWithCyclicBufferFlush) {
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(strings::Printf("graph_%.2ld", i));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  }

  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  // Expect there to be only the last kCyclicBufferSize debug events,
  // and the order should be correct.
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), kCyclicBufferSize);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    EXPECT_EQ(actuals[i].graph_execution_trace().tfdbg_context_id(),
              strings::Printf("graph_%.2ld", i + kCyclicBufferSize));
  }

  // Second, write more than the capacity of the circular buffer,
  // in a concurrent fashion.
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSdebug_events_writer_testDTcc mht_13(mht_13_v, 971, "", "./tensorflow/core/util/debug_events_writer_test.cc", "lambda");

    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(
        strings::Printf("new_graph_%.2ld", counter.fetch_add(1)));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  };
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;
  TF_ASSERT_OK(writer->Close());

  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  // NOTE: This includes the files from the first stage above, because the
  // .graph_execution_traces file hasn't changed.
  EXPECT_EQ(actuals.size(), kCyclicBufferSize * 2);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    const size_t index = i + kCyclicBufferSize;
    EXPECT_EQ(actuals[index].graph_execution_trace().tfdbg_context_id().find(
                  "new_graph_"),
              0);
  }

  // Verify no cross-talk.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, RegisterDeviceAndGetIdTrace) {
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, DebugEventsWriter::kDefaultCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // Register and get some device IDs in a concurrent fashion.
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  int device_ids[8];
  for (int i = 0; i < 8; ++i) {
    thread_pool->Schedule([i, &writer, &device_ids]() {
      const string device_name = strings::Printf(
          "/job:localhost/replica:0/task:0/device:GPU:%d", i % 4);
      device_ids[i] = writer->RegisterDeviceAndGetId(device_name);
    });
  }
  delete thread_pool;
  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  // There should be only 4 unique device IDs, because there are only 4 unique
  // device names.
  EXPECT_EQ(device_ids[0], device_ids[4]);
  EXPECT_EQ(device_ids[1], device_ids[5]);
  EXPECT_EQ(device_ids[2], device_ids[6]);
  EXPECT_EQ(device_ids[3], device_ids[7]);
  // Assert that the four device IDs are all unique.
  EXPECT_EQ(absl::flat_hash_set<int>(device_ids, device_ids + 8).size(), 4);

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  // Due to the `% 4`, there are only 4 unique device names, even though there
  // are 8 threads each calling `RegisterDeviceAndGetId`.
  EXPECT_EQ(actuals.size(), 4);
  for (const DebugEvent& actual : actuals) {
    const string& device_name = actual.debugged_device().device_name();
    int device_index = -1;
    CHECK(absl::SimpleAtoi(device_name.substr(strlen(
                               "/job:localhost/replica:0/task:0/device:GPU:")),
                           &device_index));
    EXPECT_EQ(actual.debugged_device().device_id(), device_ids[device_index]);
  }
}

TEST_F(DebugEventsWriterTest, DisableCyclicBufferBehavior) {
  const size_t kCyclicBufferSize = 0;  // A value <= 0 disables cyclic behavior.
  DebugEventsWriter* writer = DebugEventsWriter::GetDebugEventsWriter(
      dump_root_, tfdbg_run_id_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  const size_t kNumEvents = 20;

  for (size_t i = 0; i < kNumEvents; ++i) {
    Execution* execution = new Execution();
    execution->set_op_type("Log");
    execution->add_input_tensor_ids(i);
    TF_ASSERT_OK(writer->WriteExecution(execution));
  }
  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), kNumEvents);
  for (size_t i = 0; i < kNumEvents; ++i) {
    EXPECT_EQ(actuals[i].execution().op_type(), "Log");
    EXPECT_EQ(actuals[i].execution().input_tensor_ids().size(), 1);
    EXPECT_EQ(actuals[i].execution().input_tensor_ids()[0], i);
  }

  for (size_t i = 0; i < kNumEvents; ++i) {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(strings::Printf("graph_%.2ld", i));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  }
  TF_ASSERT_OK(writer->FlushExecutionFiles());

  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), kNumEvents);
  for (size_t i = 0; i < kNumEvents; ++i) {
    EXPECT_EQ(actuals[i].graph_execution_trace().tfdbg_context_id(),
              strings::Printf("graph_%.2ld", i));
  }

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

}  // namespace tfdbg
}  // namespace tensorflow
