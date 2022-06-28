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
class MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc() {
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

// Class method definitions for HostStream, the Stream implementation for
// the HostExecutor implementation.
#include "tensorflow/stream_executor/host/host_stream.h"

#include "absl/synchronization/notification.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/setround.h"

namespace stream_executor {
namespace host {

namespace {

port::ThreadOptions GetThreadOptions(size_t stack_size_in_bytes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_0(mht_0_v, 198, "", "./tensorflow/stream_executor/host/host_stream.cc", "GetThreadOptions");

  port::ThreadOptions options;
  options.stack_size = stack_size_in_bytes;
  return options;
}

}  // namespace

HostStream::HostStream(size_t stack_size_in_bytes)
    : thread_(port::Env::Default()->StartThread(
          GetThreadOptions(stack_size_in_bytes), "host_executor",
          [this]() { WorkLoop(); })) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_1(mht_1_v, 212, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::HostStream");
}

HostStream::~HostStream() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_2(mht_2_v, 217, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::~HostStream");

  {
    absl::MutexLock lock(&mu_);
    work_queue_.push(nullptr);
  }
  // thread_'s destructor blocks until the thread finishes running.
  thread_.reset();
}

bool HostStream::EnqueueTask(std::function<void()> task) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_3(mht_3_v, 229, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::EnqueueTask");

  return EnqueueTaskWithStatus([task = std::move(task)]() {
    task();
    return port::Status::OK();
  });
}

bool HostStream::EnqueueTaskWithStatus(std::function<port::Status()> task) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_4(mht_4_v, 239, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::EnqueueTaskWithStatus");

  CHECK(task != nullptr);
  absl::MutexLock lock(&mu_);
  work_queue_.push(std::move(task));
  return true;
}

bool HostStream::WorkAvailable() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_5(mht_5_v, 249, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::WorkAvailable");
 return !work_queue_.empty(); }

void HostStream::WorkLoop() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_6(mht_6_v, 254, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::WorkLoop");

  // Set denormal and rounding behavior to match the default TF ThreadPool
  // behavior.
  // TODO(phawkins, jlebar): it's not clear this is the best place to set this.
  tensorflow::port::ScopedFlushDenormal flush;
  tensorflow::port::ScopedSetRound round(FE_TONEAREST);
  while (true) {
    std::queue<std::function<port::Status()>> queue;
    {
      absl::MutexLock lock(&mu_);
      mu_.Await(absl::Condition(this, &HostStream::WorkAvailable));
      std::swap(queue, work_queue_);
    }
    while (!queue.empty()) {
      std::function<port::Status()>& fn = queue.front();
      if (!fn) {
        return;
      }
      status_.Update(fn());
      queue.pop();
    }
  }
}

port::Status HostStream::BlockUntilDone() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPShostPShost_streamDTcc mht_7(mht_7_v, 281, "", "./tensorflow/stream_executor/host/host_stream.cc", "HostStream::BlockUntilDone");

  absl::Notification done;
  port::Status status;
  EnqueueTask([&done, &status, this]() {
    // This task is always executed synchronously before 'status_' is updated
    // with the result of the task (always OK() in this case), so we don't need
    // to worry about locking access to 'status_'.
    status = status_;
    status_ = port::Status::OK();
    done.Notify();
  });
  done.WaitForNotification();
  return status;
}

}  // namespace host

}  // namespace stream_executor
