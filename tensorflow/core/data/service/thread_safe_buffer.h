/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh() {
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


#include <deque>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace data {

// A thread-safe bounded buffer with cancellation support.
template <class T>
class ThreadSafeBuffer final {
 public:
  // Creates a buffer with the specified `buffer_size`.
  // REQUIRES: buffer_size > 0
  explicit ThreadSafeBuffer(size_t buffer_size);

  // Gets the next element. Blocks if the buffer is empty. Returns an error if
  // a non-OK status was pushed or the buffer has been cancelled.
  StatusOr<T> Pop();

  // Writes the next element. Blocks if the buffer is full. Returns an error if
  // the buffer has been cancelled.
  Status Push(StatusOr<T> value);

  // Cancels the buffer with `status` and notifies waiting threads. After
  // cancelling, all `Push` and `Pop` calls will return `status`.
  // REQUIRES: !status.ok()
  void Cancel(Status status);

 private:
  const size_t buffer_size_;

  mutex mu_;
  condition_variable ready_to_pop_;
  condition_variable ready_to_push_;
  std::deque<StatusOr<T>> results_ TF_GUARDED_BY(mu_);
  Status status_ TF_GUARDED_BY(mu_) = Status::OK();

  TF_DISALLOW_COPY_AND_ASSIGN(ThreadSafeBuffer);
};

template <class T>
ThreadSafeBuffer<T>::ThreadSafeBuffer(size_t buffer_size)
    : buffer_size_(buffer_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh mht_0(mht_0_v, 232, "", "./tensorflow/core/data/service/thread_safe_buffer.h", "ThreadSafeBuffer<T>::ThreadSafeBuffer");

  DCHECK_GT(buffer_size, 0)
      << "ThreadSafeBuffer must have a postive buffer size. Got " << buffer_size
      << ".";
}

template <class T>
StatusOr<T> ThreadSafeBuffer<T>::Pop() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh mht_1(mht_1_v, 242, "", "./tensorflow/core/data/service/thread_safe_buffer.h", "ThreadSafeBuffer<T>::Pop");

  mutex_lock l(mu_);
  while (status_.ok() && results_.empty()) {
    ready_to_pop_.wait(l);
  }
  if (!status_.ok()) {
    return status_;
  }
  StatusOr<T> result = std::move(results_.front());
  results_.pop_front();
  ready_to_push_.notify_one();
  return result;
}

template <class T>
Status ThreadSafeBuffer<T>::Push(StatusOr<T> value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh mht_2(mht_2_v, 260, "", "./tensorflow/core/data/service/thread_safe_buffer.h", "ThreadSafeBuffer<T>::Push");

  mutex_lock l(mu_);
  while (status_.ok() && results_.size() >= buffer_size_) {
    ready_to_push_.wait(l);
  }
  if (!status_.ok()) {
    return status_;
  }
  results_.push_back(std::move(value));
  ready_to_pop_.notify_one();
  return Status::OK();
}

template <class T>
void ThreadSafeBuffer<T>::Cancel(Status status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSthread_safe_bufferDTh mht_3(mht_3_v, 277, "", "./tensorflow/core/data/service/thread_safe_buffer.h", "ThreadSafeBuffer<T>::Cancel");

  DCHECK(!status.ok())
      << "Cancelling ThreadSafeBuffer requires a non-OK status. Got " << status;
  mutex_lock l(mu_);
  status_ = std::move(status);
  ready_to_push_.notify_all();
  ready_to_pop_.notify_all();
}

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_THREAD_SAFE_BUFFER_H_
