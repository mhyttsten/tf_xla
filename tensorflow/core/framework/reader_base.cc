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
class MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc() {
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

#include "tensorflow/core/framework/reader_base.h"

#include "tensorflow/core/framework/reader_base.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

// ReaderBase ------------------------------------------------------

ReaderBase::ReaderBase(const string& name) : name_(name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::ReaderBase");
}

int64_t ReaderBase::NumRecordsProduced() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::NumRecordsProduced");

  mutex_lock lock(mu_);
  return num_records_produced_;
}

int64_t ReaderBase::NumWorkUnitsCompleted() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_2(mht_2_v, 214, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::NumWorkUnitsCompleted");

  mutex_lock lock(mu_);
  return work_finished_;
}

Status ReaderBase::Reset() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::Reset");

  mutex_lock lock(mu_);
  return ResetLocked();
}

Status ReaderBase::ResetLocked() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_4(mht_4_v, 230, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::ResetLocked");

  work_started_ = 0;
  work_finished_ = 0;
  num_records_produced_ = 0;
  work_.clear();
  return Status::OK();
}

Status ReaderBase::SerializeState(tstring* state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_5(mht_5_v, 241, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::SerializeState");

  mutex_lock lock(mu_);
  return SerializeStateLocked(state);
}

Status ReaderBase::SerializeStateLocked(tstring* state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_6(mht_6_v, 249, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::SerializeStateLocked");

  return errors::Unimplemented("Reader SerializeState");
}

Status ReaderBase::RestoreState(const tstring& state) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("state: \"" + (std::string)state + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_7(mht_7_v, 257, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::RestoreState");

  mutex_lock lock(mu_);
  Status status = RestoreStateLocked(state);
  if (!status.ok()) {
    ResetLocked().IgnoreError();
  }
  return status;
}

Status ReaderBase::RestoreStateLocked(const tstring& state) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("state: \"" + (std::string)state + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_8(mht_8_v, 270, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::RestoreStateLocked");

  return errors::Unimplemented("Reader RestoreState");
}

int64_t ReaderBase::ReadUpTo(const int64_t num_records, QueueInterface* queue,
                             std::vector<tstring>* keys,
                             std::vector<tstring>* values,
                             OpKernelContext* context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_9(mht_9_v, 280, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::ReadUpTo");

  mutex_lock lock(mu_);
  int64_t records_produced_this_call = 0;
  while (true) {
    // Records produced by this iteration of the ReadUpToLocked call.
    int64_t num_records_produced = 0;
    int64_t remaining = num_records - records_produced_this_call;
    if (remaining == 0) {
      return records_produced_this_call;
    }
    if (!work_in_progress()) {
      work_ = GetNextWorkLocked(queue, context);
      if (!context->status().ok()) {
        return records_produced_this_call;
      }
      Status status = OnWorkStartedLocked();
      if (status.ok()) {
        work_started_++;
      } else {
        context->SetStatus(status);
        return records_produced_this_call;
      }
    }
    bool at_end = false;

    Status status =
        ReadUpToLocked(remaining, keys, values, &num_records_produced, &at_end);
    // This call so far.
    records_produced_this_call += num_records_produced;

    // In total, over the lifetime of the ReaderBase.
    num_records_produced_ += num_records_produced;

    if (!at_end && status.ok() && num_records_produced == 0) {
      status = errors::Internal(
          "ReadManyLocked() for ", name(),
          " must set *at_end=true, *num_produced > 0 or return an error.");
      context->SetStatus(status);
      return records_produced_this_call;
    }
    if (status.ok() && at_end) {
      status = OnWorkFinishedLocked();
      work_finished_ = work_started_;
      if (records_produced_this_call > 0) {
        return records_produced_this_call;
      }
    }
    if (!status.ok()) {
      context->SetStatus(status);
      return records_produced_this_call;
    }
  }
}

// Default implementation just reads one record at a time.
Status ReaderBase::ReadUpToLocked(int64_t num_records,
                                  std::vector<tstring>* keys,
                                  std::vector<tstring>* values,
                                  int64_t* num_read, bool* at_end) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_10(mht_10_v, 341, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::ReadUpToLocked");

  bool produced = false;
  tstring key;
  tstring value;
  Status status = ReadLocked(&key, &value, &produced, at_end);
  if (produced) {
    keys->push_back(std::move(key));
    values->push_back(std::move(value));
    *num_read = 1;
  } else {
    *num_read = 0;
  }
  return status;
}

void ReaderBase::Read(QueueInterface* queue, tstring* key, tstring* value,
                      OpKernelContext* context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_11(mht_11_v, 360, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::Read");

  mutex_lock lock(mu_);
  while (true) {
    if (!work_in_progress()) {
      work_ = GetNextWorkLocked(queue, context);
      if (!context->status().ok()) {
        return;
      }
      Status status = OnWorkStartedLocked();
      if (status.ok()) {
        work_started_++;
      } else {
        context->SetStatus(status);
        return;
      }
    }

    bool produced = false;
    bool at_end = false;
    Status status = ReadLocked(key, value, &produced, &at_end);

    if (!at_end && status.ok() && !produced) {
      status = errors::Internal(
          "ReadLocked() for ", name(),
          " must set *at_end=true, *produced=true, or return an error.");
    }
    if (!status.ok() && produced) {
      status = errors::Internal("ReadLocked() for ", name(),
                                " set *produced=true *and* returned an error: ",
                                status.error_message());
    }
    if (status.ok() && at_end) {
      status = OnWorkFinishedLocked();
      work_finished_ = work_started_;
    }
    if (!status.ok()) {
      context->SetStatus(status);
      return;
    }
    if (produced) {
      ++num_records_produced_;
      return;
    }
  }
}

string ReaderBase::GetNextWorkLocked(QueueInterface* queue,
                                     OpKernelContext* context) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_12(mht_12_v, 410, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::GetNextWorkLocked");

  string work;
  Notification n;
  queue->TryDequeue(
      context, [context, &n, &work](const QueueInterface::Tuple& tuple) {
        if (context->status().ok()) {
          if (tuple.size() != 1) {
            context->SetStatus(
                errors::InvalidArgument("Expected single component queue"));
          } else if (tuple[0].dtype() != DT_STRING) {
            context->SetStatus(errors::InvalidArgument(
                "Expected queue with single string component"));
          } else if (tuple[0].NumElements() != 1) {
            context->SetStatus(errors::InvalidArgument(
                "Expected to dequeue a one-element string tensor"));
          } else {
            work = tuple[0].flat<tstring>()(0);
          }
        }
        n.Notify();
      });
  n.WaitForNotification();
  return work;
}

void ReaderBase::SaveBaseState(ReaderBaseState* state) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_13(mht_13_v, 438, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::SaveBaseState");

  state->Clear();
  state->set_work_started(work_started_);
  state->set_work_finished(work_finished_);
  state->set_num_records_produced(num_records_produced_);
  state->set_current_work(work_.data(), work_.size());
}

tstring ReaderBase::KeyName(const tstring& key) const {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("key: \"" + (std::string)key + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_14(mht_14_v, 450, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::KeyName");

  return strings::StrCat(current_work(), ":", key);
}

Status ReaderBase::RestoreBaseState(const ReaderBaseState& state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSreader_baseDTcc mht_15(mht_15_v, 457, "", "./tensorflow/core/framework/reader_base.cc", "ReaderBase::RestoreBaseState");

  work_started_ = state.work_started();
  work_finished_ = state.work_finished();
  num_records_produced_ = state.num_records_produced();
  work_ = state.current_work();
  if (work_started_ < 0 || work_finished_ < 0 || num_records_produced_ < 0) {
#if defined(__ANDROID__) || defined(__EMSCRIPTEN__)
    const string debug_string = "<debug state not available>";
#else
    const string debug_string = state.DebugString();
#endif
    return errors::InvalidArgument(
        "Unexpected negative value when restoring in ", name(), ": ",
        debug_string);
  }
  if (work_started_ > work_finished_) {
#if defined(__ANDROID__) || (__EMSCRIPTEN__)
    const string debug_string = "<debug state not available>";
#else
    const string debug_string = state.DebugString();
#endif
    return errors::InvalidArgument(
        "Inconsistent work started vs. finished when restoring in ", name(),
        ": ", debug_string);
  }
  return Status::OK();
}

}  // namespace tensorflow
