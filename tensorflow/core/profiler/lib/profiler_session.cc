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
class MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/lib/profiler_session.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/convert/post_process_single_host_xplane.h"
#include "tensorflow/core/profiler/lib/profiler_collection.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/lib/profiler_lock.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#endif

namespace tensorflow {
namespace {

ProfileOptions GetOptions(const ProfileOptions& opts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/profiler/lib/profiler_session.cc", "GetOptions");

  if (opts.version()) return opts;
  ProfileOptions options = ProfilerSession::DefaultOptions();
  options.set_include_dataset_ops(opts.include_dataset_ops());
  return options;
}

};  // namespace

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
    const ProfileOptions& options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/profiler/lib/profiler_session.cc", "ProfilerSession::Create");

  return absl::WrapUnique(new ProfilerSession(options));
}

tensorflow::Status ProfilerSession::Status() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/profiler/lib/profiler_session.cc", "ProfilerSession::Status");

  mutex_lock l(mutex_);
  return status_;
}

#if !defined(IS_MOBILE_PLATFORM)
Status ProfilerSession::CollectDataInternal(profiler::XSpace* space) {
  mutex_lock l(mutex_);
  TF_RETURN_IF_ERROR(status_);
  LOG(INFO) << "Profiler session collecting data.";
  if (profilers_ != nullptr) {
    profilers_->Stop().IgnoreError();
    profilers_->CollectData(space).IgnoreError();
    profilers_.reset();  // data has been collected.
  }
  // Allow another session to start.
  profiler_lock_.ReleaseIfActive();
  return Status::OK();
}
#endif

Status ProfilerSession::CollectData(profiler::XSpace* space) {
#if !defined(IS_MOBILE_PLATFORM)
  TF_RETURN_IF_ERROR(CollectDataInternal(space));
  PostProcessSingleHostXSpace(space, start_time_ns_);
#endif
  return Status::OK();
}

ProfilerSession::ProfilerSession(const ProfileOptions& options)
#if defined(IS_MOBILE_PLATFORM)
    : status_(errors::Unimplemented(
          "Profiler is unimplemented for mobile platforms.")) {
#else
    : options_(GetOptions(options)) {
  auto profiler_lock = profiler::ProfilerLock::Acquire();
  if (!profiler_lock.ok()) {
    status_ = profiler_lock.status();
    return;
  }
  profiler_lock_ = *std::move(profiler_lock);

  LOG(INFO) << "Profiler session initializing.";
  // Sleep until it is time to start profiling.
  if (options_.start_timestamp_ns() > 0) {
    int64_t sleep_duration_ns =
        options_.start_timestamp_ns() - profiler::GetCurrentTimeNanos();
    if (sleep_duration_ns < 0) {
      LOG(WARNING) << "Profiling is late by " << -sleep_duration_ns
                   << " nanoseconds and will start immediately.";
    } else {
      LOG(INFO) << "Delaying start of profiler session by "
                << sleep_duration_ns;
      profiler::SleepForNanos(sleep_duration_ns);
    }
  }

  LOG(INFO) << "Profiler session started.";
  start_time_ns_ = profiler::GetCurrentTimeNanos();

  DCHECK(profiler_lock_.Active());
  profilers_ = absl::make_unique<profiler::ProfilerCollection>(
      profiler::CreateProfilers(options_));
  profilers_->Start().IgnoreError();
#endif
}

ProfilerSession::~ProfilerSession() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSlibPSprofiler_sessionDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/profiler/lib/profiler_session.cc", "ProfilerSession::~ProfilerSession");

#if !defined(IS_MOBILE_PLATFORM)
  LOG(INFO) << "Profiler session tear down.";
#endif
}

}  // namespace tensorflow
