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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh() {
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


#include <cstddef>

#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/shape_inference.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {

// RemoteExecuteNode is an implementation of EagerNode which enqueues
// an operation via RPC in a remote EagerService.
class RemoteExecuteNode : public AsyncRemoteExecuteNode {
 public:
  RemoteExecuteNode(EagerContext* eager_context,
                    std::unique_ptr<EnqueueRequest> request, Device* device,
                    uint64 context_view_id, EagerClient* eager_client,
                    CancellationManager* cancellation_manager,
                    const NodeDef& ndef, FunctionLibraryDefinition* lib_def,
                    const gtl::InlinedVector<TensorHandle*, 4>& inputs,
                    absl::Span<TensorHandle*> retvals)
      : AsyncRemoteExecuteNode(),
        eager_context_(eager_context),
        request_(std::move(request)),
        device_(device),
        context_view_id_(context_view_id),
        eager_client_(eager_client),
        cancellation_manager_(cancellation_manager),
        ndef_(ndef),
        lib_def_(lib_def),
        inputs_(inputs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_0(mht_0_v, 225, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "RemoteExecuteNode");

    // Copy the output handles, since the container for them might get
    // destroyed.
    for (auto handle : retvals) {
      handle->Ref();
      retvals_.push_back(handle);
    }

    // This is required to ensure that the tensor handles stay alive across the
    // execution.
    for (auto handle : inputs_) {
      handle->Ref();
    }
    eager_client_->Ref();

    needs_remote_inputs_ = false;
    for (const TensorHandle* input : inputs_) {
      // TODO(bramandia): Should this be op_device() instead?
      if (input->resource_device() != nullptr &&
          input->resource_device() != device_) {
        needs_remote_inputs_ = true;
        break;
      }
    }
  }

  ~RemoteExecuteNode() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_1(mht_1_v, 254, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "~RemoteExecuteNode");

    for (auto handle : retvals_) {
      handle->Unref();
    }

    for (auto handle : inputs_) {
      handle->Unref();
    }
    eager_client_->Unref();
  }

  Status Prepare() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_2(mht_2_v, 268, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "Prepare");

    return RunShapeInference(ndef_, *lib_def_, inputs_, retvals_);
  }

  void RunAsync(StatusCallback done) override;

  Status SyncExecutors() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_3(mht_3_v, 277, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "SyncExecutors");
 return eager_context_->SyncExecutors(); }

  void Abort(Status status) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_4(mht_4_v, 282, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "Abort");

    int i = 0;
    for (auto handle : retvals_) {
      handle->PoisonRemote(status, device_, context_view_id_);
      ++i;
    }
  }

  const EagerClient* eager_client() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_5(mht_5_v, 293, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "eager_client");
 return eager_client_; }

  bool needs_remote_inputs() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_6(mht_6_v, 298, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "needs_remote_inputs");
 return needs_remote_inputs_; }

  bool allow_multiple_pending_requests() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_7(mht_7_v, 303, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "allow_multiple_pending_requests");

    return eager_client_->allow_multiple_pending_requests();
  }

  string DebugString() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPSremote_execute_nodeDTh mht_8(mht_8_v, 310, "", "./tensorflow/core/distributed_runtime/eager/remote_execute_node.h", "DebugString");

    string out = "[RemoteExecuteNode]";
    strings::StrAppend(&out, " request: ", request_->DebugString());
    strings::StrAppend(&out, ", target_device: ", device_->name());
    return out;
  }

 private:
  EagerContext* eager_context_;  // Not owned, and must outlive this node.
  std::unique_ptr<EnqueueRequest> request_;
  Device* device_;             // Not owned
  uint64 context_view_id_;
  bool needs_remote_inputs_;
  EagerClient* eager_client_;  // Not owned, and must outlive this node.
  CancellationManager* cancellation_manager_;
  const NodeDef ndef_;
  const FunctionLibraryDefinition* lib_def_;
  gtl::InlinedVector<TensorHandle*, 4> inputs_;
  gtl::InlinedVector<TensorHandle*, 2> retvals_;
};

}  // namespace eager
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_EAGER_REMOTE_EXECUTE_NODE_H_
