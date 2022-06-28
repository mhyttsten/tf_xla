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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XFEED_QUEUE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XFEED_QUEUE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh() {
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
#include <functional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU outfeed implementation settles, consider
// folding back the cpu and gpu outfeed implementations into a generic
// one if possible.

// Manages a thread-safe queue of buffers.
template <typename BufferType>
class XfeedQueue {
 public:
  // Adds a tree of buffers to the queue. The individual buffers correspond to
  // the elements of a tuple and may be nullptr if the buffer is a tuple index
  // buffer.
  void EnqueueDestination(BufferType buffers) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "EnqueueDestination");

    absl::MutexLock l(&mu_);
    enqueued_buffers_.push_back(std::move(buffers));
    enqueue_cv_.Signal();

    EnqueueHook();
  }

  // Blocks until the queue is non-empty, then returns the buffer at the head of
  // the queue.
  BufferType BlockingGetNextDestination() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "BlockingGetNextDestination");

    for (const auto& callback : before_get_next_dest_callbacks_) {
      callback();
    }

    bool became_empty;
    BufferType current_buffer;
    {
      absl::MutexLock l(&mu_);
      while (enqueued_buffers_.empty()) {
        enqueue_cv_.Wait(&mu_);
      }
      current_buffer = std::move(enqueued_buffers_.front());
      enqueued_buffers_.pop_front();
      DequeueHook();
      became_empty = enqueued_buffers_.empty();
    }
    if (became_empty) {
      for (const auto& callback : on_empty_callbacks_) {
        callback();
      }
    }
    return current_buffer;
  }

  void RegisterOnEmptyCallback(std::function<void()> callback) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_2(mht_2_v, 252, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "RegisterOnEmptyCallback");

    on_empty_callbacks_.push_back(std::move(callback));
  }
  void RegisterBeforeGetNextDestinationCallback(
      std::function<void()> callback) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "RegisterBeforeGetNextDestinationCallback");

    before_get_next_dest_callbacks_.push_back(std::move(callback));
  }

  virtual ~XfeedQueue() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_4(mht_4_v, 266, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "~XfeedQueue");
}

 protected:
  virtual void DequeueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {}
  virtual void EnqueueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) {}

  absl::Mutex mu_;

  // The queue of trees of buffers. Buffer* queue contents are not owned.
  std::deque<BufferType> enqueued_buffers_ ABSL_GUARDED_BY(mu_);

 private:
  // Condition variable that is signaled every time a buffer is enqueued.
  absl::CondVar enqueue_cv_;

  // List of callbacks which will be called when 'enqueued_buffers_' becomes
  // empty.
  std::vector<std::function<void()>> on_empty_callbacks_;

  // List of callbacks which will be called before BlockingGetNextDestination()
  // is called. This lets you e.g. call EnqueueDestination() for each call to
  // BlockingGetNextDestination().
  std::vector<std::function<void()>> before_get_next_dest_callbacks_;
};

// Like XfeedQueue but with a maximum capacity.  Clients can call
// `BlockUntilEnqueueSlotAvailable` to block until there are fewer than
// `max_pending_xfeeds_` capacity pending infeed items.
//
// We introduce a separate `BlockUntilEnqueueSlotAvailable` (as opposed to
// overriding `EnqueueDestination` to block) because we want to block before we
// copy the buffer to GPU memory, in order to bound the memory consumption due
// to pending infeeds.
template <typename BufferType>
class BlockingXfeedQueue : public XfeedQueue<BufferType> {
 public:
  explicit BlockingXfeedQueue(int max_pending_xfeeds)
      : max_pending_xfeeds_(max_pending_xfeeds) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_5(mht_5_v, 306, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "BlockingXfeedQueue");
}

  void BlockUntilEnqueueSlotAvailable() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSxfeed_queueDTh mht_6(mht_6_v, 311, "", "./tensorflow/compiler/xla/service/gpu/xfeed_queue.h", "BlockUntilEnqueueSlotAvailable");

    absl::MutexLock l{&this->mu_};
    while (pending_buffers_ + this->enqueued_buffers_.size() >=
           max_pending_xfeeds_) {
      VLOG(2) << "Capacity "
              << (pending_buffers_ + this->enqueued_buffers_.size())
              << " >= max capacity " << max_pending_xfeeds_;
      dequeue_cv_.Wait(&this->mu_);
    }

    pending_buffers_++;
  }

 protected:
  void EnqueueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    pending_buffers_--;
  }

  void DequeueHook() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_) override {
    dequeue_cv_.Signal();
  }

 private:
  const int max_pending_xfeeds_;

  // Condition variable that is signaled every time a buffer is dequeued.
  absl::CondVar dequeue_cv_;

  // Keeps track of the number of buffers reserved but not added to
  // enqueued_buffers_.
  int pending_buffers_ ABSL_GUARDED_BY(this->mu_) = 0;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_XFEED_QUEUE_H_
