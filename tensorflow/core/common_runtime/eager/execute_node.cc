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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTcc() {
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
#include "tensorflow/core/common_runtime/eager/execute_node.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

#if !defined(IS_MOBILE_PLATFORM)
bool ExecuteNodeArgs::IsRemote(EagerContext* ctx, Device* input_device,
                               TensorHandle* handle) {
  uint64 context_view_id = ctx->GetContextViewId();
  if (handle->Type() == TensorHandle::REMOTE ||
      handle->HasRemoteMirror(input_device, context_view_id)) {
    if (!has_remote_inputs_) {
      has_remote_inputs_ = true;
    }
    return true;
  }
  return false;
}
#endif  // IS_MOBILE_PLATFORM

Status ExecuteNodeArgs::InitPackedHandle(const int index, EagerContext* ctx,
                                         Device* input_device,
                                         TensorHandle* packed_handle) {
  int num_handles = packed_handle->NumPackedHandles();
  packed_args_.emplace(index, gtl::InlinedVector<TensorValue, 4>(num_handles));
  TensorValue* packed_arg_flat = &(packed_args_[index][0]);
  for (int i = 0; i < num_handles; ++i) {
    TensorHandle* h = nullptr;
    TF_RETURN_IF_ERROR(packed_handle->ExtractPackedHandle(i, &h));
    // We have validated that h->device() is not a CustomDevice when
    // constructing a pack TensorHandle.
    const Status status = h->TensorValue(h->device(), &packed_arg_flat[i]);
    if (!status.ok()) {
#if !defined(IS_MOBILE_PLATFORM)
      if (IsRemote(ctx, input_device, h)) {
        continue;
      }
#endif  // IS_MOBILE_PLATFORM
      if (h->Type() == TensorHandle::PACKED) {
        return errors::InvalidArgument(
            "Nested packed handles are not supported");
      }
      return status;
    }
  }
  return Status::OK();
}

Status ExecuteNodeArgs::Init(
    EagerContext* ctx, const gtl::InlinedVector<TensorHandle*, 4>& op_inputs,
    const core::RefCountPtr<KernelAndDevice>& kernel) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTcc mht_0(mht_0_v, 235, "", "./tensorflow/core/common_runtime/eager/execute_node.cc", "ExecuteNodeArgs::Init");

  // If there are multiple references to a TensorHandle in 'op_inputs' we must
  // increment the reference count of the corresponding Tensor or risk it being
  // overwritten during kernel execution. The reference count is incremented
  // below when we insert a copy of the Tensor into protected_tensors, and will
  // be decremented once execution is complete.
  const int n_inputs = op_inputs.size();
  if (n_inputs > 0) {
    TensorHandle* const* op_inputs_flat = &op_inputs[0];
    TensorValue* tensor_args_flat = &tensor_args_[0];
    for (int i = 0; i < n_inputs; ++i) {
      TensorHandle* in = op_inputs_flat[i];
      Device* d = kernel->InputDevice(i);
      Status s = in->TensorValue(ctx->CanonicalDevice(d), &tensor_args_flat[i]);
      if (!s.ok()) {
#if !defined(IS_MOBILE_PLATFORM)
        if (IsRemote(ctx, d, in)) {
          continue;
        }
#endif
        if (in->Type() != TensorHandle::PACKED) {
          return s;
        }
        if (!has_packed_inputs_) {
          has_packed_inputs_ = true;
        }
        TF_RETURN_IF_ERROR(InitPackedHandle(i, ctx, d, in));
      }
    }
  }

#if !defined(IS_MOBILE_PLATFORM)
  if (has_remote_inputs_) {
    const bool is_function = kernel->IsFunction();
    serialize_remote_handle_ =
        [ctx, &op_inputs, is_function](
            const FunctionArgIndex& index,
            eager::RemoteTensorHandle* handle) -> Status {
      TensorHandle* h = op_inputs[index.index];
      if (op_inputs[index.index]->Type() == TensorHandle::PACKED) {
        TF_RETURN_IF_ERROR(
            op_inputs[index.index]->ExtractPackedHandle(index.sub_index, &h));
      }
      Device* device = h->device();
      // For a multi-device function, a remote RunComponentFunction request is
      // not sent through StreamingEnqueueAsync. It could arrive at a remote
      // worker before a remote execution request which produces an input of the
      // component function. So we wait until the remote input is ready before
      // serializing it.
      const bool wait_util_ready = is_function;
      return ctx->RemoteMgr()->SerializeRemoteTensorHandle(
          h, wait_util_ready, handle, device, device->name());
    };
  }
#endif  // !IS_MOBILE_PLATFORM
  return Status::OK();
}

Status ExecuteNodeArgs::GetLocalArg(const FunctionArgIndex& index,
                                    Tensor* val) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_nodeDTcc mht_1(mht_1_v, 297, "", "./tensorflow/core/common_runtime/eager/execute_node.cc", "ExecuteNodeArgs::GetLocalArg");

  Status s = EagerKernelArgs::GetLocalArg(index, val);
  if (s.ok()) {
    return Status::OK();
  }
  if (packed_args_.contains(index.index)) {
    Tensor* arg = packed_args_.at(index.index).at(index.sub_index).tensor;
    if (arg) {
      *val = *arg;
      return Status::OK();
    } else {
      return errors::NotFound("Argument (", index.index, ",", index.sub_index,
                              ") has no local tensor.");
    }
  } else {
    return s;
  }
}

}  // namespace tensorflow
