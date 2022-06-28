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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc() {
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

#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace data {

namespace {
constexpr StringPiece kJournal = "journal";

Status ParseSequenceNumber(const std::string& journal_file,
                           int64_t* sequence_number) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("journal_file: \"" + journal_file + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/data/service/journal.cc", "ParseSequenceNumber");

  if (!RE2::FullMatch(journal_file, ".*_(\\d+)", sequence_number)) {
    return errors::InvalidArgument("Failed to parse journal file name: ",
                                   journal_file);
  }
  return Status::OK();
}
}  // namespace

std::string DataServiceJournalFile(const std::string& journal_dir,
                                   int64_t sequence_number) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("journal_dir: \"" + journal_dir + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/data/service/journal.cc", "DataServiceJournalFile");

  return io::JoinPath(journal_dir,
                      absl::StrCat(kJournal, "_", sequence_number));
}

FileJournalWriter::FileJournalWriter(Env* env, const std::string& journal_dir)
    : env_(env), journal_dir_(journal_dir) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("journal_dir: \"" + journal_dir + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/data/service/journal.cc", "FileJournalWriter::FileJournalWriter");
}

Status FileJournalWriter::EnsureInitialized() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_3(mht_3_v, 233, "", "./tensorflow/core/data/service/journal.cc", "FileJournalWriter::EnsureInitialized");

  if (writer_) {
    return Status::OK();
  }
  std::vector<std::string> journal_files;
  TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(journal_dir_));
  TF_RETURN_IF_ERROR(env_->GetChildren(journal_dir_, &journal_files));
  int64_t latest_sequence_number = -1;
  for (const auto& file : journal_files) {
    int64_t sequence_number;
    TF_RETURN_IF_ERROR(ParseSequenceNumber(file, &sequence_number));
    latest_sequence_number = std::max(latest_sequence_number, sequence_number);
  }
  std::string journal_file =
      DataServiceJournalFile(journal_dir_, latest_sequence_number + 1);
  TF_RETURN_IF_ERROR(env_->NewAppendableFile(journal_file, &file_));
  writer_ = absl::make_unique<io::RecordWriter>(file_.get());
  VLOG(1) << "Created journal writer to write to " << journal_file;
  return Status::OK();
}

Status FileJournalWriter::Write(const Update& update) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_4(mht_4_v, 257, "", "./tensorflow/core/data/service/journal.cc", "FileJournalWriter::Write");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  std::string s = update.SerializeAsString();
  if (s.empty()) {
    return errors::Internal("Failed to serialize update ", update.DebugString(),
                            " to string");
  }
  TF_RETURN_IF_ERROR(writer_->WriteRecord(s));
  TF_RETURN_IF_ERROR(writer_->Flush());
  TF_RETURN_IF_ERROR(file_->Sync());
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Wrote journal entry: " << update.DebugString();
  }
  return Status::OK();
}

FileJournalReader::FileJournalReader(Env* env, StringPiece journal_dir)
    : env_(env), journal_dir_(journal_dir) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/data/service/journal.cc", "FileJournalReader::FileJournalReader");
}

Status FileJournalReader::EnsureInitialized() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_6(mht_6_v, 282, "", "./tensorflow/core/data/service/journal.cc", "FileJournalReader::EnsureInitialized");

  if (reader_) {
    return Status::OK();
  }
  return UpdateFile(DataServiceJournalFile(journal_dir_, 0));
}

Status FileJournalReader::Read(Update& update, bool& end_of_journal) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_7(mht_7_v, 292, "", "./tensorflow/core/data/service/journal.cc", "FileJournalReader::Read");

  TF_RETURN_IF_ERROR(EnsureInitialized());
  while (true) {
    tstring record;
    Status s = reader_->ReadRecord(&record);
    if (errors::IsOutOfRange(s)) {
      sequence_number_++;
      std::string next_journal_file =
          DataServiceJournalFile(journal_dir_, sequence_number_);
      if (errors::IsNotFound(env_->FileExists(next_journal_file))) {
        VLOG(3) << "Next journal file " << next_journal_file
                << " does not exist. End of journal reached.";
        end_of_journal = true;
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(UpdateFile(next_journal_file));
      continue;
    }
    TF_RETURN_IF_ERROR(s);
    if (!update.ParseFromString(record)) {
      return errors::DataLoss("Failed to parse journal record.");
    }
    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Read journal entry: " << update.DebugString();
    }
    end_of_journal = false;
    return Status::OK();
  }
}

Status FileJournalReader::UpdateFile(const std::string& filename) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSjournalDTcc mht_8(mht_8_v, 326, "", "./tensorflow/core/data/service/journal.cc", "FileJournalReader::UpdateFile");

  VLOG(1) << "Reading from journal file " << filename;
  TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(filename, &file_));
  io::RecordReaderOptions opts;
  opts.buffer_size = 2 << 20;  // 2MB
  reader_ = absl::make_unique<io::SequentialRecordReader>(file_.get(), opts);
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
