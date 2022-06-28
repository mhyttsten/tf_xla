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
class MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc() {
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

#include "tensorflow/core/util/events_writer.h"

#include <math.h>
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

// shorthand
Env* env() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/util/events_writer_test.cc", "env");
 return Env::Default(); }

void WriteSimpleValue(EventsWriter* writer, double wall_time, int64_t step,
                      const string& tag, float simple_value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/util/events_writer_test.cc", "WriteSimpleValue");

  Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_simple_value(simple_value);
  writer->WriteEvent(event);
}

void WriteFile(EventsWriter* writer) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/util/events_writer_test.cc", "WriteFile");

  WriteSimpleValue(writer, 1234, 34, "foo", 3.14159);
  WriteSimpleValue(writer, 2345, 35, "bar", -42);
}

static bool ReadEventProto(io::RecordReader* reader, uint64* offset,
                           Event* proto) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/util/events_writer_test.cc", "ReadEventProto");

  tstring record;
  Status s = reader->ReadRecord(offset, &record);
  if (!s.ok()) {
    return false;
  }
  return ParseProtoUnlimited(proto, record);
}

void VerifyFile(const string& filename) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/util/events_writer_test.cc", "VerifyFile");

  CHECK(env()->FileExists(filename).ok());
  std::unique_ptr<RandomAccessFile> event_file;
  TF_CHECK_OK(env()->NewRandomAccessFile(filename, &event_file));
  io::RecordReader* reader = new io::RecordReader(event_file.get());

  uint64 offset = 0;

  Event actual;
  CHECK(ReadEventProto(reader, &offset, &actual));
  VLOG(1) << actual.ShortDebugString();
  // Wall time should be within 5s of now.

  double current_time = env()->NowMicros() / 1000000.0;
  EXPECT_LT(fabs(actual.wall_time() - current_time), 5);
  // Should have the current version number.
  EXPECT_EQ(actual.file_version(),
            strings::StrCat(EventsWriter::kVersionPrefix,
                            EventsWriter::kCurrentVersion));

  Event expected;
  CHECK(ReadEventProto(reader, &offset, &actual));
  VLOG(1) << actual.ShortDebugString();
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "wall_time: 1234 step: 34 "
      "summary { value { tag: 'foo' simple_value: 3.14159 } }",
      &expected));
  // TODO(keveman): Enable this check
  // EXPECT_THAT(expected, EqualsProto(actual));

  CHECK(ReadEventProto(reader, &offset, &actual));
  VLOG(1) << actual.ShortDebugString();
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "wall_time: 2345 step: 35 "
      "summary { value { tag: 'bar' simple_value: -42 } }",
      &expected));
  // TODO(keveman): Enable this check
  // EXPECT_THAT(expected, EqualsProto(actual));

  TF_CHECK_OK(env()->DeleteFile(filename));
  delete reader;
}

string GetDirName(const string& suffix) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSevents_writer_testDTcc mht_5(mht_5_v, 294, "", "./tensorflow/core/util/events_writer_test.cc", "GetDirName");

  return io::JoinPath(testing::TmpDir(), suffix);
}

TEST(EventWriter, WriteFlush) {
  string file_prefix = GetDirName("/writeflush_test");
  EventsWriter writer(file_prefix);
  WriteFile(&writer);
  TF_EXPECT_OK(writer.Flush());
  string filename = writer.FileName();
  VerifyFile(filename);
}

TEST(EventWriter, WriteClose) {
  string file_prefix = GetDirName("/writeclose_test");
  EventsWriter writer(file_prefix);
  WriteFile(&writer);
  TF_EXPECT_OK(writer.Close());
  string filename = writer.FileName();
  VerifyFile(filename);
}

TEST(EventWriter, WriteDelete) {
  string file_prefix = GetDirName("/writedelete_test");
  EventsWriter* writer = new EventsWriter(file_prefix);
  WriteFile(writer);
  string filename = writer->FileName();
  delete writer;
  VerifyFile(filename);
}

TEST(EventWriter, FailFlush) {
  string file_prefix = GetDirName("/failflush_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  WriteFile(&writer);
  TF_EXPECT_OK(env()->FileExists(filename));
  TF_ASSERT_OK(env()->DeleteFile(filename));
  EXPECT_TRUE(writer.Flush().ok());
}

TEST(EventWriter, FailClose) {
  string file_prefix = GetDirName("/failclose_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  WriteFile(&writer);
  TF_EXPECT_OK(env()->FileExists(filename));
  TF_ASSERT_OK(env()->DeleteFile(filename));
  EXPECT_TRUE(writer.Close().ok());
}

TEST(EventWriter, InitWriteClose) {
  string file_prefix = GetDirName("/initwriteclose_test");
  EventsWriter writer(file_prefix);
  TF_EXPECT_OK(writer.Init());
  string filename0 = writer.FileName();
  TF_EXPECT_OK(env()->FileExists(filename0));
  WriteFile(&writer);
  TF_EXPECT_OK(writer.Close());
  string filename1 = writer.FileName();
  EXPECT_EQ(filename0, filename1);
  VerifyFile(filename1);
}

TEST(EventWriter, NameWriteClose) {
  string file_prefix = GetDirName("/namewriteclose_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  TF_EXPECT_OK(env()->FileExists(filename));
  WriteFile(&writer);
  TF_EXPECT_OK(writer.Close());
  VerifyFile(filename);
}

TEST(EventWriter, NameClose) {
  string file_prefix = GetDirName("/nameclose_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  TF_EXPECT_OK(writer.Close());
  TF_EXPECT_OK(env()->FileExists(filename));
  TF_ASSERT_OK(env()->DeleteFile(filename));
}

TEST(EventWriter, FileDeletionBeforeWriting) {
  string file_prefix = GetDirName("/fdbw_test");
  EventsWriter writer(file_prefix);
  string filename0 = writer.FileName();
  TF_EXPECT_OK(env()->FileExists(filename0));
  env()->SleepForMicroseconds(
      2000000);  // To make sure timestamp part of filename will differ.
  TF_ASSERT_OK(env()->DeleteFile(filename0));
  TF_EXPECT_OK(writer.Init());  // Init should reopen file.
  WriteFile(&writer);
  TF_EXPECT_OK(writer.Flush());
  string filename1 = writer.FileName();
  EXPECT_NE(filename0, filename1);
  VerifyFile(filename1);
}

}  // namespace
}  // namespace tensorflow
