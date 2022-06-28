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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_idsDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_idsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_idsDTh() {
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
#include <unordered_set>
#include <vector>

#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// RecentRequestIds tracks recent 64-bit request_ids. When maximum capacity is
// reached, the oldest request_id is evicted. Thread safe.
//
// Some RPCs like RecvTensor are unsafe to retry. For example, RecvTensor pairs
// one sender and one receiver, and the receiver waits for the sender's tensor.
// Retried RecvTensor requests are problematic, because the original RecvTensor
// request may have consumed the sender's tensor, so a retried request might
// block forever. RecentRequestIds identifies retried requests, so we can fail
// them instead of blocking forever.
//
// Internally, recent request_ids are stored in two data structures: a set and a
// circular buffer. The set is used for efficient lookups, and the circular
// buffer tracks the oldest request_id. When the buffer is full, the new
// request_id replaces the oldest request_id in the circular buffer, and the
// oldest request_id is removed from the set.
class RecentRequestIds {
 public:
  // num_tracked_request_ids should be much larger than the number of RPCs that
  // can be received in a small time window. For example, we observed a peak RPC
  // rate of ~700 RecvTensor RPC/s when training inception v3 on TPUs, so we
  // currently set num_tracked_request_ids to 100,000 for RecvTensor.
  RecentRequestIds(int num_tracked_request_ids);

  // Returns OK iff request_id has not been seen in the last
  // num_tracked_request_ids insertions. For backwards compatibility, this
  // always returns OK for request_id 0. The method_name and the request's
  // ShortDebugString are added to returned errors.
  Status TrackUnique(int64_t request_id, const string& method_name,
                     const protobuf::Message& request);
  // Overloaded version of the above function for wrapped protos.
  template <typename RequestWrapper>
  Status TrackUnique(int64_t request_id, const string& method_name,
                     const RequestWrapper* wrapper);

 private:
  bool Insert(int64_t request_id);

  mutex mu_;
  // next_index_ indexes into circular_buffer_, and points to the next storage
  // space to use. When the buffer is full, next_index_ points at the oldest
  // request_id.
  int next_index_ TF_GUARDED_BY(mu_) = 0;
  std::vector<int64_t> circular_buffer_ TF_GUARDED_BY(mu_);
  std::unordered_set<int64_t> set_ TF_GUARDED_BY(mu_);
};

// Implementation details

template <typename RequestWrapper>
Status RecentRequestIds::TrackUnique(int64_t request_id,
                                     const string& method_name,
                                     const RequestWrapper* wrapper) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("method_name: \"" + method_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrecent_request_idsDTh mht_0(mht_0_v, 254, "", "./tensorflow/core/distributed_runtime/recent_request_ids.h", "RecentRequestIds::TrackUnique");

  if (Insert(request_id)) {
    return Status::OK();
  } else {
    return errors::Aborted("The same ", method_name,
                           " request was received twice. ",
                           wrapper->ToProto().ShortDebugString());
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RECENT_REQUEST_IDS_H_
