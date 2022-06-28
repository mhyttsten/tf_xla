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
class MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/summary/summary_file_writer.h"

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

class FakeClockEnv : public EnvWrapper {
 public:
  FakeClockEnv() : EnvWrapper(Env::Default()), current_millis_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "FakeClockEnv");
}
  void AdvanceByMillis(const uint64 millis) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "AdvanceByMillis");
 current_millis_ += millis; }
  uint64 NowMicros() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_2(mht_2_v, 211, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "NowMicros");
 return current_millis_ * 1000; }
  uint64 NowSeconds() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_3(mht_3_v, 215, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "NowSeconds");
 return current_millis_ * 1000; }

 private:
  uint64 current_millis_;
};

class SummaryFileWriterTest : public ::testing::Test {
 protected:
  Status SummaryTestHelper(
      const string& test_name,
      const std::function<Status(SummaryWriterInterface*)>& writer_fn,
      const std::function<void(const Event&)>& test_fn) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("test_name: \"" + test_name + "\"");
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_4(mht_4_v, 230, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "SummaryTestHelper");

    static std::set<string>* tests = new std::set<string>();
    CHECK(tests->insert(test_name).second) << ": " << test_name;

    SummaryWriterInterface* writer;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 1, testing::TmpDir(), test_name,
                                        &env_, &writer));
    core::ScopedUnref deleter(writer);

    TF_CHECK_OK(writer_fn(writer));
    TF_CHECK_OK(writer->Flush());

    std::vector<string> files;
    TF_CHECK_OK(env_.GetChildren(testing::TmpDir(), &files));
    bool found = false;
    for (const string& f : files) {
      if (absl::StrContains(f, test_name)) {
        if (found) {
          return errors::Unknown("Found more than one file for ", test_name);
        }
        found = true;
        std::unique_ptr<RandomAccessFile> read_file;
        TF_CHECK_OK(env_.NewRandomAccessFile(io::JoinPath(testing::TmpDir(), f),
                                             &read_file));
        io::RecordReader reader(read_file.get(), io::RecordReaderOptions());
        tstring record;
        uint64 offset = 0;
        TF_CHECK_OK(
            reader.ReadRecord(&offset,
                              &record));  // The first event is irrelevant
        TF_CHECK_OK(reader.ReadRecord(&offset, &record));
        Event e;
        e.ParseFromString(record);
        test_fn(e);
      }
    }
    if (!found) {
      return errors::Unknown("Found no file for ", test_name);
    }
    return Status::OK();
  }

  FakeClockEnv env_;
};

TEST_F(SummaryFileWriterTest, WriteTensor) {
  TF_CHECK_OK(SummaryTestHelper("tensor_test",
                                [](SummaryWriterInterface* writer) {
                                  Tensor one(DT_FLOAT, TensorShape({}));
                                  one.scalar<float>()() = 1.0;
                                  TF_RETURN_IF_ERROR(writer->WriteTensor(
                                      2, one, "name",
                                      SummaryMetadata().SerializeAsString()));
                                  TF_RETURN_IF_ERROR(writer->Flush());
                                  return Status::OK();
                                },
                                [](const Event& e) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_5(mht_5_v, 289, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");

                                  EXPECT_EQ(e.step(), 2);
                                  CHECK_EQ(e.summary().value_size(), 1);
                                  EXPECT_EQ(e.summary().value(0).tag(), "name");
                                }));
  TF_CHECK_OK(SummaryTestHelper(
      "string_tensor_test",
      [](SummaryWriterInterface* writer) {
        Tensor hello(DT_STRING, TensorShape({}));
        hello.scalar<tstring>()() = "hello";
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            2, hello, "name", SummaryMetadata().SerializeAsString()));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_6(mht_6_v, 307, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");

        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name");
        EXPECT_EQ(e.summary().value(0).tensor().dtype(), DT_STRING);
        EXPECT_EQ(e.summary().value(0).tensor().string_val()[0], "hello");
      }));
}

TEST_F(SummaryFileWriterTest, WriteScalar) {
  TF_CHECK_OK(SummaryTestHelper(
      "scalar_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteScalar(2, one, "name"));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_7(mht_7_v, 329, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");

        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name");
        EXPECT_EQ(e.summary().value(0).simple_value(), 1.0);
      }));
}

TEST_F(SummaryFileWriterTest, WriteHistogram) {
  TF_CHECK_OK(SummaryTestHelper("hist_test",
                                [](SummaryWriterInterface* writer) {
                                  Tensor one(DT_FLOAT, TensorShape({}));
                                  one.scalar<float>()() = 1.0;
                                  TF_RETURN_IF_ERROR(
                                      writer->WriteHistogram(2, one, "name"));
                                  TF_RETURN_IF_ERROR(writer->Flush());
                                  return Status::OK();
                                },
                                [](const Event& e) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_8(mht_8_v, 350, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");

                                  EXPECT_EQ(e.step(), 2);
                                  CHECK_EQ(e.summary().value_size(), 1);
                                  EXPECT_EQ(e.summary().value(0).tag(), "name");
                                  EXPECT_TRUE(e.summary().value(0).has_histo());
                                }));
}

namespace {

// Create a 1x1 monochrome image consisting of a single pixel oof the given
// type.
template <typename T>
static Status CreateImage(SummaryWriterInterface* writer) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_9(mht_9_v, 366, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "CreateImage");

  Tensor bad_color(DT_UINT8, TensorShape({1}));
  bad_color.scalar<uint8>()() = 0;
  Tensor one(DataTypeToEnum<T>::v(), TensorShape({1, 1, 1, 1}));
  one.scalar<T>()() = T(1);
  TF_RETURN_IF_ERROR(writer->WriteImage(2, one, "name", 1, bad_color));
  TF_RETURN_IF_ERROR(writer->Flush());
  return Status::OK();
}

// Verify that the event contains an image generated by CreateImage above.
static void CheckImage(const Event& e) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_10(mht_10_v, 380, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "CheckImage");

  EXPECT_EQ(e.step(), 2);
  CHECK_EQ(e.summary().value_size(), 1);
  EXPECT_EQ(e.summary().value(0).tag(), "name/image");
  CHECK(e.summary().value(0).has_image());
  EXPECT_EQ(e.summary().value(0).image().height(), 1);
  EXPECT_EQ(e.summary().value(0).image().width(), 1);
  EXPECT_EQ(e.summary().value(0).image().colorspace(), 1);
}

}  // namespace

TEST_F(SummaryFileWriterTest, WriteImageUInt8) {
  TF_CHECK_OK(
      SummaryTestHelper("image_test_uint8", CreateImage<uint8>, CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteImageFloat) {
  TF_CHECK_OK(
      SummaryTestHelper("image_test_float", CreateImage<float>, CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteImageHalf) {
  TF_CHECK_OK(SummaryTestHelper("image_test_half", CreateImage<Eigen::half>,
                                CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteImageDouble) {
  TF_CHECK_OK(
      SummaryTestHelper("image_test_double", CreateImage<double>, CheckImage));
}

TEST_F(SummaryFileWriterTest, WriteAudio) {
  TF_CHECK_OK(SummaryTestHelper(
      "audio_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({1, 1}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteAudio(2, one, "name", 1, 1));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_11(mht_11_v, 425, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");

        EXPECT_EQ(e.step(), 2);
        CHECK_EQ(e.summary().value_size(), 1);
        EXPECT_EQ(e.summary().value(0).tag(), "name/audio");
        CHECK(e.summary().value(0).has_audio());
      }));
}

TEST_F(SummaryFileWriterTest, WriteEvent) {
  TF_CHECK_OK(
      SummaryTestHelper("event_test",
                        [](SummaryWriterInterface* writer) {
                          std::unique_ptr<Event> e{new Event};
                          e->set_step(7);
                          e->mutable_summary()->add_value()->set_tag("hi");
                          TF_RETURN_IF_ERROR(writer->WriteEvent(std::move(e)));
                          TF_RETURN_IF_ERROR(writer->Flush());
                          return Status::OK();
                        },
                        [](const Event& e) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_12(mht_12_v, 447, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");

                          EXPECT_EQ(e.step(), 7);
                          CHECK_EQ(e.summary().value_size(), 1);
                          EXPECT_EQ(e.summary().value(0).tag(), "hi");
                        }));
}

TEST_F(SummaryFileWriterTest, WallTime) {
  env_.AdvanceByMillis(7023);
  TF_CHECK_OK(SummaryTestHelper(
      "wall_time_test",
      [](SummaryWriterInterface* writer) {
        Tensor one(DT_FLOAT, TensorShape({}));
        one.scalar<float>()() = 1.0;
        TF_RETURN_IF_ERROR(writer->WriteScalar(2, one, "name"));
        TF_RETURN_IF_ERROR(writer->Flush());
        return Status::OK();
      },
      [](const Event& e) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSsummaryPSsummary_file_writer_testDTcc mht_13(mht_13_v, 468, "", "./tensorflow/core/summary/summary_file_writer_test.cc", "lambda");
 EXPECT_EQ(e.wall_time(), 7.023); }));
}

TEST_F(SummaryFileWriterTest, AvoidFilenameCollision) {
  // Keep unique with all other test names in this file.
  string test_name = "avoid_filename_collision_test";
  int num_files = 10;
  for (int i = 0; i < num_files; i++) {
    SummaryWriterInterface* writer;
    TF_CHECK_OK(CreateSummaryFileWriter(1, 1, testing::TmpDir(), test_name,
                                        &env_, &writer));
    core::ScopedUnref deleter(writer);
  }
  std::vector<string> files;
  TF_CHECK_OK(env_.GetChildren(testing::TmpDir(), &files));
  // Filter `files` down to just those generated in this test.
  files.erase(std::remove_if(files.begin(), files.end(),
                             [test_name](string f) {
                               return !absl::StrContains(f, test_name);
                             }),
              files.end());
  EXPECT_EQ(num_files, files.size())
      << "files = [" << absl::StrJoin(files, ", ") << "]";
}

}  // namespace
}  // namespace tensorflow
