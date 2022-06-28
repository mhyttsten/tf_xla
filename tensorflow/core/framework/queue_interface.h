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

#ifndef TENSORFLOW_CORE_FRAMEWORK_QUEUE_INTERFACE_H_
#define TENSORFLOW_CORE_FRAMEWORK_QUEUE_INTERFACE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSqueue_interfaceDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSqueue_interfaceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSqueue_interfaceDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// All implementations must be thread-safe.
class QueueInterface : public ResourceBase {
 public:
  typedef std::vector<Tensor> Tuple;
  typedef AsyncOpKernel::DoneCallback DoneCallback;
  typedef std::function<void(const Tuple&)> CallbackWithTuple;

  virtual Status ValidateTuple(const Tuple& tuple) = 0;
  virtual Status ValidateManyTuple(const Tuple& tuple) = 0;

  // Stashes a function object for future execution, that will eventually
  // enqueue the tuple of tensors into the queue, and returns immediately. The
  // function object is guaranteed to call 'callback'.
  virtual void TryEnqueue(const Tuple& tuple, OpKernelContext* ctx,
                          DoneCallback callback) = 0;

  // Same as above, but the component tensors are sliced along the 0th dimension
  // to make multiple queue-element components.
  virtual void TryEnqueueMany(const Tuple& tuple, OpKernelContext* ctx,
                              DoneCallback callback) = 0;

  // Stashes a function object for future execution, that will eventually
  // dequeue an element from the queue and call 'callback' with that tuple
  // element as argument.
  virtual void TryDequeue(OpKernelContext* ctx, CallbackWithTuple callback) = 0;

  // Same as above, but the stashed function object will attempt to dequeue
  // num_elements items.  If allow_small_batch is true, and the Queue is
  // closed but at least 1 element is available, there is no blocking
  // and between 1 and num_elements items are immediately returned.
  // If the queue does not support the allow_small_batch flag will
  // return an Unimplemented error.
  virtual void TryDequeueMany(int num_elements, OpKernelContext* ctx,
                              bool allow_small_batch,
                              CallbackWithTuple callback) = 0;

  // Signals that no more elements will be enqueued, and optionally
  // cancels pending Enqueue(Many) operations.
  //
  // After calling this function, subsequent calls to Enqueue(Many)
  // will fail. If `cancel_pending_enqueues` is true, all pending
  // calls to Enqueue(Many) will fail as well.
  //
  // After calling this function, all current and subsequent calls to
  // Dequeue(Many) will fail instead of blocking (though they may
  // succeed if they can be satisfied by the elements in the queue at
  // the time it was closed).
  virtual void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
                     DoneCallback callback) = 0;

  // Returns true if a given queue is closed and false if it is open.
  virtual bool is_closed() const = 0;

  // Assuming *this represents a shared queue, verify that it matches
  // another instantiation indicated by node_def.
  virtual Status MatchesNodeDef(const NodeDef& node_def) = 0;

  // Returns the number of elements in the queue.
  virtual int32 size() const = 0;

  virtual const DataTypeVector& component_dtypes() const = 0;

  string DebugString() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSqueue_interfaceDTh mht_0(mht_0_v, 261, "", "./tensorflow/core/framework/queue_interface.h", "DebugString");

    return strings::StrCat("A Queue of size: ", size());
  }

 protected:
  virtual ~QueueInterface() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSqueue_interfaceDTh mht_1(mht_1_v, 269, "", "./tensorflow/core/framework/queue_interface.h", "~QueueInterface");
}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_QUEUE_INTERFACE_H_
