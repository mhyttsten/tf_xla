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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RENDEZVOUS_H_
#define TENSORFLOW_CORE_FRAMEWORK_RENDEZVOUS_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh() {
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

#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class DeviceMgr;

// A Rendezvous is an abstraction for passing tensors from producers
// to consumers. A rendezvous is a table of channels. Each channel is
// keyed by a rendezvous key. The key encodes a pair of <producer,
// consumer>, where the producer and the consumer are tensorflow
// devices.
//
// The producer calls the Send() method to send one tensor over one
// named channel. The consumer calls the Recv() method to receive one
// tensor from a named channel. A sequence of tensors can be passed
// from the producer to the consumer.  The consumer receives them in
// the order as the producer sends them.
//
// A consumer may safely request the tensor before or after it has
// been produced.  A consumer has the choice of making a blocking call
// or providing a callback: in either case, the consumer receives the
// Tensor as soon as it is available.  A producer never blocks.
class RendezvousInterface {
 public:
  struct Args {
    DeviceContext* device_context = nullptr;
    AllocatorAttributes alloc_attrs;
    CancellationManager* cancellation_manager = nullptr;  // not owned.
  };

  // Parses the key constructed by CreateKey and parse src/dst device
  // names into structures respectively.
  struct ParsedKey {
    StringPiece src_device;
    DeviceNameUtils::ParsedName src;
    uint64 src_incarnation = 0;
    StringPiece dst_device;
    DeviceNameUtils::ParsedName dst;
    StringPiece edge_name;

    ParsedKey() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_0(mht_0_v, 236, "", "./tensorflow/core/framework/rendezvous.h", "ParsedKey");
}
    ParsedKey(const ParsedKey& b) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/framework/rendezvous.h", "ParsedKey");
 *this = b; }

    ParsedKey& operator=(const ParsedKey& b);
    StringPiece FullKey() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_2(mht_2_v, 246, "", "./tensorflow/core/framework/rendezvous.h", "FullKey");
 return buf_; }

   private:
    friend class Rendezvous;
    friend class SendOp;
    friend class RecvOp;
    std::string buf_;
  };

  // The caller is a tensor producer and it sends a message (a tensor
  // "val" and a bool "is_dead") under the given "key".
  //
  // {val, is_dead} is bundled as a message sent and received.
  // Typically, is_dead is set by some control flow nodes
  // (e.g., a not-taken branch).  args is passed by Send to the
  // Recv function to communicate any information that the Recv
  // function might need.  This is typically only necessary for
  // Send/Recv on the same worker.
  //
  // Send() never blocks.
  virtual Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
                      const bool is_dead) = 0;

  // Callback provided by a tensor consumer waiting on the rendezvous.
  // It will be invoked when the tensor is available, or when a non-OK
  // status arises in the production of that tensor.  It also gets
  // two Rendezvous::Args, one provided by the sender, the other by the
  // receiver, which may be needed when a non-CPU device is in use
  // by either side.
  typedef std::function<void(const Status&, const Args&, const Args&,
                             const Tensor&, const bool)>
      DoneCallback;

  virtual void RecvAsync(const ParsedKey& key, const Args& args,
                         DoneCallback done) = 0;

  // Synchronous wrapper for RecvAsync.
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead, int64_t timeout_ms);
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead);

  // Aborts all pending and future Send/Recv with the given "status".
  //
  // StartAbort() does not wait for ongoing calls to finish.
  // REQUIRES: !status.ok()
  virtual void StartAbort(const Status& status) = 0;

 protected:
  virtual ~RendezvousInterface();

  virtual bool is_cross_process() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_3(mht_3_v, 300, "", "./tensorflow/core/framework/rendezvous.h", "is_cross_process");
 return false; }
  friend class ProcessFunctionLibraryRuntime;
};

// A reference-counted implementation of RendezvousInterface.
//
// This class is used in cases where a rendezvous may be shared between multiple
// threads with no clear owner.
class Rendezvous : public RendezvousInterface, public core::RefCounted {
 public:
  class Factory {
   public:
    // Default to a factory that evaluates to false.
    Factory() : valid_(false) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_4(mht_4_v, 316, "", "./tensorflow/core/framework/rendezvous.h", "Factory");
}

    Factory(std::function<Status(const int64_t, const DeviceMgr*, Rendezvous**)>
                create_fn,
            std::function<Status(const int64_t)> cleanup_fn)
        : valid_(true),
          create_fn_(std::move(create_fn)),
          cleanup_fn_(std::move(cleanup_fn)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_5(mht_5_v, 326, "", "./tensorflow/core/framework/rendezvous.h", "Factory");
}

    // If no clean up fn is provided, just put in a dummy.
    // For backwards compatibility.
    explicit Factory(
        std::function<Status(const int64_t, const DeviceMgr*, Rendezvous**)>
            create_fn)
        : valid_(true),
          create_fn_(std::move(create_fn)),
          cleanup_fn_([](const int64_t step_id) { return Status::OK(); }) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_6(mht_6_v, 338, "", "./tensorflow/core/framework/rendezvous.h", "Factory");
}

    explicit operator bool() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_7(mht_7_v, 343, "", "./tensorflow/core/framework/rendezvous.h", "bool");
 return valid_; }

    Status operator()(const int64_t step_id, const DeviceMgr* device_mgr,
                      Rendezvous** rendez) const {
      return create_fn_(step_id, device_mgr, rendez);
    }

    Status CleanUp(const int64_t step_id) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSrendezvousDTh mht_8(mht_8_v, 353, "", "./tensorflow/core/framework/rendezvous.h", "CleanUp");
 return cleanup_fn_(step_id); }

   private:
    bool valid_;
    std::function<Status(const int64_t, const DeviceMgr*, Rendezvous**)>
        create_fn_;
    std::function<Status(const int64_t)> cleanup_fn_;
  };

  // Constructs a rendezvous key for the tensor of "name" sent from
  // "src_device" to "dst_device". The tensor is generated in the frame
  // and iteration specified by "frame_iter".
  static std::string CreateKey(const std::string& src_device,
                               uint64 src_incarnation,
                               const std::string& dst_device,
                               const std::string& name,
                               const FrameAndIter& frame_iter);

  static Status ParseKey(StringPiece key, ParsedKey* out);
};

// Returns a Rendezvous instance that is limited to use only by
// producers and consumers in the local process.  The caller assumes
// ownership of one Ref() on the returned object.
Rendezvous* NewLocalRendezvous();

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_RENDEZVOUS_H_
