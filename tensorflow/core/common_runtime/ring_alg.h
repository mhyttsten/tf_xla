/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RING_ALG_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RING_ALG_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTh() {
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
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/framework/collective.h"

namespace tensorflow {
class Device;

// Basic ring-algorithm implementation to be further specialized
// for specific collective functions.
class RingAlg : public CollectiveImplementationInterface {
 public:
  explicit RingAlg(CollectiveType type, const string& name);
  ~RingAlg() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSring_algDTh mht_0(mht_0_v, 203, "", "./tensorflow/core/common_runtime/ring_alg.h", "~RingAlg");
}

  // Establishes the requested number of subdivision permutations based on the
  // ring order implicit in the device order.
  Status InitializeCollectiveParams(CollectiveParams* col_params) override;

  // Initializes members of CollectiveContext not yet initialized, i.e. device
  // and device_locality.  Also saves the CollectiveContext in this object.
  Status InitializeCollectiveContext(
      std::shared_ptr<CollectiveContext> col_ctx) override;

 protected:
  // Called when a bad status is received that implies we should terminate
  // execution and return a bad status.
  void StartAbort(const Status& s);
  void Finish(bool ok);

  // Current status of a RingField
  enum RingFieldAction {
    RF_INIT = 0,    // Just initialized for a pass
    RF_RECV,        // Recv pending
    RF_REDUCE,      // Reduce pending
    RF_FINALIZE,    // FinalOp pending
    RF_SEND_READY,  // Ready to send
    RF_SEND,        // Send pending
    RF_DONE,        // No more work
  };

  // Tracks progress of actions on a single subfield of the entire tensor.
  struct RingField {
    int16 chunk_idx;     // major division index
    int16 subdiv_idx;    // minor division index
    int16 sc_idx;        // subchunk index
    int16 rank;          // rank within subdiv permutation
    int16 recv_dev_idx;  // dev from which value should be recv'd
    RingFieldAction action;
    bool second_pass;
    bool recv_is_remote = false;
    bool send_is_remote = false;
    bool do_send = false;   // is the value sent in this pass?
    bool do_recv = false;   // is the value recv'd in this pass?
    bool is_final = false;  // is the last field in the pass for this rank
    Tensor chunk;           // alias to field values
    Tensor tmp_chunk;
    Status status;
    string DebugString() const;
  };
  virtual void InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                             int field_idx);
  void AdvanceToSecondPass(RingField* rf);
  void DispatchSend(RingField* rf, const StatusCallback& done);
  void DispatchRecv(RingField* rf, const StatusCallback& done);

  // For constructing log messages for debugging.
  string FieldState();
  string TensorDebugString(const Tensor& tensor);

  // Producer/Consumer Queue of RingField structs.
  class PCQueue {
   public:
    void Enqueue(RingField* rf);
    RingField* Dequeue();

   private:
    mutex pcq_mu_;
    condition_variable cv_;
    int waiter_count_ TF_GUARDED_BY(pcq_mu_) = 0;
    std::deque<RingField*> deque_ TF_GUARDED_BY(pcq_mu_);
  };

  const CollectiveType type_;
  const string name_;
  std::shared_ptr<CollectiveContext> col_ctx_;
  const CollectiveParams* col_params_;  // Not owned
  StatusCallback done_;
  int group_size_;
  int num_subdivs_;
  Tensor group_size_tensor_;
  Notification group_size_tensor_ready_;
  std::unique_ptr<CollectiveAdapter> ca_;
  mutex status_mu_;
  Status status_ TF_GUARDED_BY(status_mu_);
  std::vector<RingField> rfv_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RING_ALG_H_
