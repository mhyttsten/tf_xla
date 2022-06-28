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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh() {
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


// clang-format off
// Required for IS_MOBILE_PLATFORM
#include <cstddef>
#include <memory>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_executor.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

class ExecuteNodeArgs : public EagerKernelArgs {
 public:
  explicit ExecuteNodeArgs(int count) : EagerKernelArgs(count) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "ExecuteNodeArgs");
}

  Status Init(EagerContext* ctx,
              const absl::InlinedVector<TensorHandle*, 4>& op_inputs,
              const core::RefCountPtr<KernelAndDevice>& kernel);

  Status GetLocalArg(const FunctionArgIndex& index, Tensor* val) const override;

  bool HasRemoteOrPackedInputs() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "HasRemoteOrPackedInputs");

    return has_remote_inputs_ || has_packed_inputs_;
  };

#if !defined(IS_MOBILE_PLATFORM)
  Status GetRemoteArg(const FunctionArgIndex& index,
                      eager::RemoteTensorHandle* val) const override {
    return serialize_remote_handle_(index, val);
  }
#endif  // IS_MOBILE_PLATFORM

 private:
#if !defined(IS_MOBILE_PLATFORM)
  // Returns whether `handle` is a remote handle or has a remote mirror on
  // `input_device`
  bool IsRemote(EagerContext* ctx, Device* input_device, TensorHandle* handle);
#endif  // IS_MOBILE_PLATFORM

  // Initialize a packed TensorHandle which is the `index`-th argument.
  Status InitPackedHandle(const int index, EagerContext* ctx,
                          Device* input_device, TensorHandle* packed_handle);

  bool has_remote_inputs_ = false;
  bool has_packed_inputs_ = false;
  // Maps from the index of a packed arg to a list of sub-args.
  absl::flat_hash_map<int, gtl::InlinedVector<TensorValue, 4>> packed_args_;
#if !defined(IS_MOBILE_PLATFORM)
  std::function<Status(const FunctionArgIndex&, eager::RemoteTensorHandle*)>
      serialize_remote_handle_;
#endif  // IS_MOBILE_PLATFORM
};

class ExecuteNode : public EagerNode {
 public:
  ExecuteNode(EagerContext* ctx,
              const absl::InlinedVector<TensorHandle*, 4>& inputs,
              const absl::optional<EagerFunctionParams>& eager_func_params,
              const core::RefCountPtr<KernelAndDevice>& kernel,
              GraphCollector* graph_collector,
              CancellationManager* cancellation_manager,
              absl::Span<TensorHandle*> retvals,
              absl::optional<ManagedStackTrace> stack_trace)
      : EagerNode(),
        ctx_(ctx),
        inputs_(inputs),
        eager_func_params_(eager_func_params),
        kernel_(kernel),
        graph_collector_(graph_collector),
        cancellation_manager_(cancellation_manager),
        retvals_(retvals),
        stack_trace_(stack_trace) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_2(mht_2_v, 285, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "ExecuteNode");
}

  Status Run() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_3(mht_3_v, 290, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "Run");

    int i = 0;
    for (TensorHandle* h : inputs_) {
      if (h->RefCountIsOne()) {
        const Device* d = ctx_->CanonicalDevice(kernel_->InputDevice(i));
        Status s = h->Unprotect(d);
        if (!s.ok()) {
          VLOG(1) << "Unable to unprotect tensor: " << s;
        }
      }
      ++i;
    }
    return EagerKernelExecute(ctx_, inputs_, eager_func_params_, kernel_,
                              graph_collector_, cancellation_manager_, retvals_,
                              stack_trace_);
  }

  void Abort(Status status) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_4(mht_4_v, 310, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "Abort");
}

  std::string DebugString() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_5(mht_5_v, 315, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "DebugString");

    std::string out = "[ExecuteNode]";
    strings::StrAppend(&out, " kernel: ", kernel_->name());
    return out;
  }

 private:
  EagerContext* ctx_;
  const absl::InlinedVector<TensorHandle*, 4>& inputs_;
  const absl::optional<EagerFunctionParams>& eager_func_params_;
  const core::RefCountPtr<KernelAndDevice>& kernel_;
  GraphCollector* graph_collector_;
  CancellationManager* const cancellation_manager_;
  absl::Span<TensorHandle*> retvals_;
  absl::optional<ManagedStackTrace> stack_trace_;
};

class AsyncExecuteNode : public EagerNode {
 public:
  AsyncExecuteNode(EagerContext* ctx,
                   const absl::InlinedVector<TensorHandle*, 4>& inputs,
                   const absl::optional<EagerFunctionParams>& eager_func_params,
                   core::RefCountPtr<KernelAndDevice> kernel,
                   GraphCollector* graph_collector,
                   CancellationManager* cancellation_manager,
                   absl::Span<TensorHandle*> retvals,
                   absl::optional<ManagedStackTrace> stack_trace)
      : EagerNode(),
        ctx_(ctx),
        inputs_(inputs),
        eager_func_params_(eager_func_params),
        kernel_(std::move(kernel)),
        graph_collector_(graph_collector),
        cancellation_manager_(cancellation_manager),
        stack_trace_(stack_trace) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_6(mht_6_v, 352, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "AsyncExecuteNode");

    // Copy the output handles, since the container for them might get
    // destroyed.
    for (auto handle : retvals) {
      handle->Ref();
      retvals_.push_back(handle);
    }

    // This is required to ensure that the tensor handles stay alive across
    // the execution.
    for (auto handle : inputs_) {
      handle->Ref();
    }
  }

  ~AsyncExecuteNode() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_7(mht_7_v, 370, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "~AsyncExecuteNode");

    for (auto handle : retvals_) {
      handle->Unref();
    }

    for (auto handle : inputs_) {
      handle->Unref();
    }
  }

  Status Run() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_8(mht_8_v, 383, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "Run");

    int i = 0;
    for (TensorHandle* h : inputs_) {
      if (h->RefCountIsOne()) {
        const Device* d = ctx_->CanonicalDevice(kernel_->InputDevice(i));
        Status s = h->Unprotect(d);
        if (!s.ok()) {
          VLOG(1) << "Unable to unprotect tensor: " << s;
        }
      }
      ++i;
    }
    Status status = EagerKernelExecute(
        ctx_, inputs_, eager_func_params_, kernel_, graph_collector_,
        cancellation_manager_, absl::MakeSpan(retvals_), stack_trace_);
    if (!status.ok()) {
      if (stack_trace_.has_value()) {
        Status with_stack_trace(status.code(), status.error_message(),
                                stack_trace_->ToStackFrames({}, {}));
        errors::CopyPayloads(status, with_stack_trace);
        status = std::move(with_stack_trace);
      }
      Abort(status);
      return status;
    }
    // If status is ok, EagerKernelExecute would have called SetTensor on
    // all the output handles.
    return Status::OK();
  }

  void Abort(Status status) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_9(mht_9_v, 416, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "Abort");

    int i = 0;
    for (auto handle : retvals_) {
      handle->Poison(status, ctx_->CanonicalDevice(kernel_->OutputDevice(i)));
      ++i;
    }
  }

  std::string DebugString() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTh mht_10(mht_10_v, 427, "", "./tensorflow/core/common_runtime/eager/execute_node.h", "DebugString");

    std::string out = "[AsyncExecuteNode]";
    strings::StrAppend(&out, " kernel: ", kernel_->name());
    return out;
  }

 private:
  EagerContext* ctx_;
  absl::InlinedVector<TensorHandle*, 4> inputs_;
  const absl::optional<EagerFunctionParams> eager_func_params_;
  core::RefCountPtr<KernelAndDevice> kernel_;
  GraphCollector* graph_collector_;
  CancellationManager* const cancellation_manager_;
  absl::optional<ManagedStackTrace> stack_trace_;
  absl::InlinedVector<TensorHandle*, 2> retvals_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EXECUTE_NODE_H_
