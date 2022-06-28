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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTh() {
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


#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class Device;
class DeviceContext;
class DeviceMgr;
class Tensor;

// EXPERIMENTAL: RDMA oriented producer/consumer rendezvous on a local
// Tensor value for which DMAHelper::CanUseDMA() is true, i.e. dense
// numeric types.  Similar to Rendezvous but never owns a Ref on the
// tensor, instead it uses an explicit callback to the producer when
// the consumer side is finished with the value.  This allows the
// producer to perform in-place updates on the source buffer or to take
// other actions that depend on knowing the consumer has passed a certain
// execution point.
class BufRendezvous {
 public:
  explicit BufRendezvous(uint64 step_id, const DeviceMgr* dev_mgr)
      : step_id_(step_id), dev_mgr_(dev_mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTh mht_0(mht_0_v, 214, "", "./tensorflow/core/common_runtime/buf_rendezvous.h", "BufRendezvous");
}

  virtual ~BufRendezvous();

  // Inform all waiting parties that this BufRendezvous is defunct because of
  // an error Status interrupting the Step.
  void StartAbort(const Status& s);

  struct Hook;
  // Provided by the consumer to be called when access to the buffer
  // is available.  If the Status arg is not OK, then hook will not
  // be populated.  Ownership of Hook passes to consumer with the
  // callback.
  typedef std::function<void(const Status&, Hook*)> ConsumerCallback;
  // Provided by the producer to be called when the consumer has finished
  // reading the buffer and will no longer access it.
  typedef std::function<void(const Status&)> ProducerCallback;

  struct Hook {
    Device* prod_dev;
    DeviceContext* prod_ctx;
    const Tensor* prod_value;
    AllocatorAttributes prod_attr;
    ProducerCallback prod_cb;
    ConsumerCallback cons_cb;
    CancellationManager* cancellation_manager;
    CancellationToken cancellation_token;
    explicit Hook(CancellationManager* cancellation_manager,
                  CancellationToken cancellation_token)
        : prod_dev(nullptr),
          prod_ctx(nullptr),
          prod_value(nullptr),
          prod_cb(nullptr),
          cons_cb(nullptr),
          cancellation_manager(cancellation_manager),
          cancellation_token(cancellation_token) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSbuf_rendezvousDTh mht_1(mht_1_v, 252, "", "./tensorflow/core/common_runtime/buf_rendezvous.h", "Hook");
}
    string DebugString() const;
  };

  // Called to advertise availability of a Tensor value corresponding
  // to key.  That value must stay valid until done is called.
  //
  // If a non-null cancellation manager is provided, this function registers a
  // callback to delete the hook and invoke provider/consumer callbacks with
  // cancelled error.
  void ProvideBuf(const string& key, Device* dev, DeviceContext* dev_ctx,
                  const Tensor* v, const AllocatorAttributes& attr,
                  const ProducerCallback& done,
                  CancellationManager* cancellation_manager);

  // Called to request access to a Tensor value corresponding to key.
  // Consumer is provided with a Hook as soon as available.
  //
  // This function also checks that the current incarnation number of the
  // `device` that produced this value matches the `incarnation` expected by the
  // consumer, and invokes `done` with `FailedPrecondition` status and
  // `nullptr` hook if it does not match.
  //
  // If a non-null cancellation manager is provided, this function registers a
  // callback to delete the hook and invoke provider/consumer callbacks with
  // cancelled error.
  virtual void ConsumeBuf(const string& key, const string& device,
                          const uint64 incarnation,
                          const ConsumerCallback& done,
                          CancellationManager* cancellation_manager);

  // Cancel the rendezvous entry corresponding to `key`.  Triggered by the
  // cancellation manager. No-op if the rendezvous was already successful.
  void CancelHook(const string& key);

  // Consumer must call this function when it's done reading the Hook provided
  // by the ConsumerCallback.  This function will invoke the producer callback
  // and then delete h.
  static void DoneWithHook(Hook* h);

  // Write the current contents of the table to the INFO log.
  void LogContents();

 protected:
  const uint64 step_id_;
  const DeviceMgr* const dev_mgr_;  // Not owned.
  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
  typedef absl::flat_hash_map<string, Hook*> HookTable;
  HookTable hook_table_ TF_GUARDED_BY(mu_);

  void PurgeTable(const Status& s, HookTable* table);
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
