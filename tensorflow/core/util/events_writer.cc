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
class MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc() {
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

#include <stddef.h>  // for NULL

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {

EventsWriter::EventsWriter(const string& file_prefix)
    // TODO(jeff,sanjay): Pass in env and use that here instead of Env::Default
    : env_(Env::Default()),
      file_prefix_(file_prefix),
      num_outstanding_events_(0) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("file_prefix: \"" + file_prefix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::EventsWriter");
}

EventsWriter::~EventsWriter() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::~EventsWriter");

  Close().IgnoreError();  // Autoclose in destructor.
}

Status EventsWriter::Init() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::Init");
 return InitWithSuffix(""); }

Status EventsWriter::InitWithSuffix(const string& suffix) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("suffix: \"" + suffix + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_3(mht_3_v, 225, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::InitWithSuffix");

  file_suffix_ = suffix;
  return InitIfNeeded();
}

Status EventsWriter::InitIfNeeded() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_4(mht_4_v, 233, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::InitIfNeeded");

  if (recordio_writer_ != nullptr) {
    CHECK(!filename_.empty());
    if (!FileStillExists().ok()) {
      // Warn user of data loss and let .reset() below do basic cleanup.
      if (num_outstanding_events_ > 0) {
        LOG(WARNING) << "Re-initialization, attempting to open a new file, "
                     << num_outstanding_events_ << " events will be lost.";
      }
    } else {
      // No-op: File is present and writer is initialized.
      return Status::OK();
    }
  }

  int64_t time_in_seconds = env_->NowMicros() / 1000000;

  filename_ =
      strings::Printf("%s.out.tfevents.%010lld.%s%s", file_prefix_.c_str(),
                      static_cast<long long>(time_in_seconds),
                      port::Hostname().c_str(), file_suffix_.c_str());

  // Reset recordio_writer (which has a reference to recordio_file_) so final
  // Flush() and Close() call have access to recordio_file_.
  recordio_writer_.reset();

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      env_->NewWritableFile(filename_, &recordio_file_),
      "Creating writable file ", filename_);
  recordio_writer_.reset(new io::RecordWriter(recordio_file_.get()));
  if (recordio_writer_ == nullptr) {
    return errors::Unknown("Could not create record writer");
  }
  num_outstanding_events_ = 0;
  VLOG(1) << "Successfully opened events file: " << filename_;
  {
    // Write the first event with the current version, and flush
    // right away so the file contents will be easily determined.

    Event event;
    event.set_wall_time(time_in_seconds);
    event.set_file_version(strings::StrCat(kVersionPrefix, kCurrentVersion));
    WriteEvent(event);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(Flush(), "Flushing first event.");
  }
  return Status::OK();
}

string EventsWriter::FileName() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_5(mht_5_v, 284, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::FileName");

  if (filename_.empty()) {
    InitIfNeeded().IgnoreError();
  }
  return filename_;
}

void EventsWriter::WriteSerializedEvent(StringPiece event_str) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::WriteSerializedEvent");

  if (recordio_writer_ == nullptr) {
    if (!InitIfNeeded().ok()) {
      LOG(ERROR) << "Write failed because file could not be opened.";
      return;
    }
  }
  num_outstanding_events_++;
  recordio_writer_->WriteRecord(event_str).IgnoreError();
}

// NOTE(touts); This is NOT the function called by the Python code.
// Python calls WriteSerializedEvent(), see events_writer.i.
void EventsWriter::WriteEvent(const Event& event) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_7(mht_7_v, 310, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::WriteEvent");

  string record;
  event.AppendToString(&record);
  WriteSerializedEvent(record);
}

Status EventsWriter::Flush() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_8(mht_8_v, 319, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::Flush");

  if (num_outstanding_events_ == 0) return Status::OK();
  CHECK(recordio_file_ != nullptr) << "Unexpected NULL file";

  TF_RETURN_WITH_CONTEXT_IF_ERROR(recordio_writer_->Flush(), "Failed to flush ",
                                  num_outstanding_events_, " events to ",
                                  filename_);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(recordio_file_->Sync(), "Failed to sync ",
                                  num_outstanding_events_, " events to ",
                                  filename_);
  VLOG(1) << "Wrote " << num_outstanding_events_ << " events to disk.";
  num_outstanding_events_ = 0;
  return Status::OK();
}

Status EventsWriter::Close() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_9(mht_9_v, 337, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::Close");

  Status status = Flush();
  if (recordio_file_ != nullptr) {
    Status close_status = recordio_file_->Close();
    if (!close_status.ok()) {
      status = close_status;
    }
    recordio_writer_.reset(nullptr);
    recordio_file_.reset(nullptr);
  }
  num_outstanding_events_ = 0;
  return status;
}

Status EventsWriter::FileStillExists() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSevents_writerDTcc mht_10(mht_10_v, 354, "", "./tensorflow/core/util/events_writer.cc", "EventsWriter::FileStillExists");

  if (env_->FileExists(filename_).ok()) {
    return Status::OK();
  }
  // This can happen even with non-null recordio_writer_ if some other
  // process has removed the file.
  return errors::Unknown("The events file ", filename_, " has disappeared.");
}

}  // namespace tensorflow
