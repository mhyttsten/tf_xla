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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/journal.h"

#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

namespace {
using ::testing::HasSubstr;

bool NewJournalDir(std::string& journal_dir) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/data/service/journal_test.cc", "NewJournalDir");

  std::string filename = testing::TmpDir();
  if (!Env::Default()->CreateUniqueFileName(&filename, "journal_dir")) {
    return false;
  }
  journal_dir = filename;
  return true;
}

Update MakeCreateJobUpdate() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/data/service/journal_test.cc", "MakeCreateJobUpdate");

  Update update;
  CreateJobUpdate* create_job = update.mutable_create_job();
  create_job->set_dataset_id(3);
  create_job->set_job_id(8);
  create_job->mutable_processing_mode_def()->set_sharding_policy(
      ProcessingModeDef::OFF);
  return update;
}

Update MakeFinishTaskUpdate() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/data/service/journal_test.cc", "MakeFinishTaskUpdate");

  Update update;
  FinishTaskUpdate* finish_task = update.mutable_finish_task();
  finish_task->set_task_id(8);
  return update;
}

Update MakeRegisterDatasetUpdate() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/core/data/service/journal_test.cc", "MakeRegisterDatasetUpdate");

  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id(2);
  register_dataset->set_fingerprint(3);
  return update;
}

Status CheckJournalContent(StringPiece journal_dir,
                           const std::vector<Update>& expected) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournal_testDTcc mht_4(mht_4_v, 250, "", "./tensorflow/core/data/service/journal_test.cc", "CheckJournalContent");

  FileJournalReader reader(Env::Default(), journal_dir);
  for (const auto& update : expected) {
    Update result;
    bool end_of_journal = true;
    TF_RETURN_IF_ERROR(reader.Read(result, end_of_journal));
    EXPECT_FALSE(end_of_journal);
    // We can't use the testing::EqualsProto matcher because it is not available
    // in OSS.
    EXPECT_EQ(result.SerializeAsString(), update.SerializeAsString());
  }
  Update result;
  bool end_of_journal = false;
  TF_RETURN_IF_ERROR(reader.Read(result, end_of_journal));
  EXPECT_TRUE(end_of_journal);
  return Status::OK();
}
}  // namespace

TEST(Journal, RoundTripMultiple) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));
  std::vector<Update> updates = {MakeCreateJobUpdate(),
                                 MakeRegisterDatasetUpdate(),
                                 MakeFinishTaskUpdate()};
  FileJournalWriter writer(Env::Default(), journal_dir);
  for (const auto& update : updates) {
    TF_EXPECT_OK(writer.Write(update));
  }

  TF_EXPECT_OK(CheckJournalContent(journal_dir, updates));
}

TEST(Journal, AppendExistingJournal) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));
  std::vector<Update> updates = {MakeCreateJobUpdate(),
                                 MakeRegisterDatasetUpdate(),
                                 MakeFinishTaskUpdate()};
  for (const auto& update : updates) {
    FileJournalWriter writer(Env::Default(), journal_dir);
    TF_EXPECT_OK(writer.Write(update));
  }

  TF_EXPECT_OK(CheckJournalContent(journal_dir, updates));
}

TEST(Journal, MissingFile) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));
  FileJournalReader reader(Env::Default(), journal_dir);
  Update result;
  bool end_of_journal = true;
  Status s = reader.Read(result, end_of_journal);
  EXPECT_TRUE(errors::IsNotFound(s));
}

TEST(Journal, NonRecordData) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));

  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(journal_dir));
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewAppendableFile(
        DataServiceJournalFile(journal_dir, /*sequence_number=*/0), &file));
    TF_ASSERT_OK(file->Append("not record data"));
  }

  FileJournalReader reader(Env::Default(), journal_dir);
  Update result;
  bool end_of_journal = true;
  Status s = reader.Read(result, end_of_journal);
  EXPECT_THAT(s.error_message(), HasSubstr("corrupted record"));
  EXPECT_EQ(s.code(), error::DATA_LOSS);
}

TEST(Journal, InvalidRecordData) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));

  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(journal_dir));
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewAppendableFile(
        DataServiceJournalFile(journal_dir, /*sequence_number=*/0), &file));
    auto writer = absl::make_unique<io::RecordWriter>(file.get());
    TF_ASSERT_OK(writer->WriteRecord("not serializd proto"));
  }

  FileJournalReader reader(Env::Default(), journal_dir);
  Update result;
  bool end_of_journal = true;
  Status s = reader.Read(result, end_of_journal);
  EXPECT_THAT(s.error_message(), HasSubstr("Failed to parse journal record"));
  EXPECT_EQ(s.code(), error::DATA_LOSS);
}
}  // namespace data
}  // namespace tensorflow
