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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/eager/placement_utils.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/platform/errors.h"

namespace {

bool IsCPU(tensorflow::Device* d) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/common_runtime/eager/core.cc", "IsCPU");

  return d == nullptr || d->tensorflow_accelerator_device_info() == nullptr;
}

}  // namespace

namespace tensorflow {

// TODO(b/152902651): This should not depend on EagerContext. This can be
// resolved by storing ctx->HostCPU() in the TensorHandle class.
AbstractTensorInterface* TensorHandle::Resolve(Status* status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/common_runtime/eager/core.cc", "TensorHandle::Resolve");

  *status = WaitUnknownDevice();
  if (!status->ok()) {
    return nullptr;
  }
  if (Type() == REMOTE) {
    const tensorflow::Tensor* t = nullptr;
    TensorHandle* h_cpu = nullptr;
    *status = EagerCopyToDevice(this, ctx_, &ctx_->Executor(), ctx_->HostCPU(),
                                false, &h_cpu);
    if (!status->ok()) {
      return nullptr;
    }
    *status = h_cpu->Tensor(&t);
    if (!status->ok()) {
      h_cpu->Unref();
      return nullptr;
    }
    // TODO(b/153052876): Change TF_TensorFromTensor to just return an
    // AbstractTensorInterface
    TF_Tensor* tf_tensor = TF_TensorFromTensor(*t, status);
    AbstractTensorInterface* retval = tf_tensor->tensor;
    h_cpu->Unref();
    delete tf_tensor;
    return retval;
  } else if (Type() == LOCAL) {
    tensorflow::Tensor tensor;
    if (IsCPU(device()) || HasLocalMirror(nullptr)) {
      const tensorflow::Tensor* src = nullptr;
      if (HasLocalMirror(nullptr)) {
        *status = TensorFromDevice(nullptr, &src);
      } else {
        *status = Tensor(&src);
      }
      if (!status->ok()) return nullptr;

      tensor = *src;
    } else {
      *status = CopyToDevice(*ctx_, ctx_->HostCPU(), &tensor);
      if (!status->ok()) return nullptr;

      tensorflow::Tensor mirror = tensor;
      *status = AddLocalMirror(std::move(mirror), nullptr);
      if (!status->ok()) {
        // If a mirror was added since we called HasLocalMirror then drop the
        // newly copied tensor and use the previously added mirror.
        if (status->code() != error::Code::ALREADY_EXISTS) {
          return nullptr;
        }
        const tensorflow::Tensor* src = nullptr;
        *status = TensorFromDevice(nullptr, &src);
        if (!status->ok()) return nullptr;

        tensor = *src;
      }
    }
    // TODO(b/153052876): Change TF_TensorFromTensor to just return an
    // AbstractTensorInterface
    TF_Tensor* tf_tensor = TF_TensorFromTensor(tensor, status);
    AbstractTensorInterface* retval = tf_tensor->tensor;
    delete tf_tensor;
    return retval;
  } else {
    *status = errors::InvalidArgument(
        "Resolve() is not supoorted on packed TensorHandles.");
    return nullptr;
  }
}

ImmediateExecutionTensorHandle* EagerContext::CopyTensorHandleToDevice(
    ImmediateExecutionTensorHandle* handle, const char* device_name,
    Status* status) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_2(mht_2_v, 284, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerContext::CopyTensorHandleToDevice");

  ImmediateExecutionTensorHandle* result = nullptr;
  Device* device;
  *status = this->FindDeviceFromName(device_name, &device);
  if (!status->ok()) {
    *status =
        tensorflow::errors::InvalidArgument(device_name, " unknown device.");
    return nullptr;
  }

  TensorHandle* input = TensorHandleFromInterface(handle);
  *status =
      EagerCopyToDevice(input, this, &this->Executor(), device, false,
                        reinterpret_cast<tensorflow::TensorHandle**>(&result));
  if (status->ok()) {
    return result;
  }
  return nullptr;
}

// TODO(b/152902651): We unfortunately need to put this EagerContext function
// here to a circular BUILD dep issue. If we move this to context.cc, then we
// will have the circular dependency of:
//   context -> tensor_handle -> remote_tensor_handle_data -> context
ImmediateExecutionTensorHandle* EagerContext::CreateLocalHandle(
    AbstractTensorInterface* t) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_3(mht_3_v, 312, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerContext::CreateLocalHandle");

  Tensor tensor = TensorFromInterface(t);
  return TensorHandle::CreateLocalHandle(std::move(tensor), /*d=*/HostCPU(),
                                         /*op_device=*/nullptr, this);
}

ImmediateExecutionTensorHandle* EagerContext::CreateLocalHandleFromTFTensor(
    tensorflow::Tensor& t, const char* d_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("d_name: \"" + (d_name == nullptr ? std::string("nullptr") : std::string((char*)d_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_4(mht_4_v, 323, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerContext::CreateLocalHandleFromTFTensor");

  // If device name is not specified, create the TensorHandle on host cpu.
  if (d_name == nullptr)
    return TensorHandle::CreateLocalHandle(std::move(t), /*d=*/HostCPU(),
                                           /*op_device=*/nullptr, this);
  Device* d = nullptr;
  auto status = FindDeviceFromName(d_name, &d);
  if (!status.ok()) return nullptr;
  return TensorHandle::CreateLocalHandle(std::move(t), /*d=*/d,
                                         /*op_device=*/nullptr, this);
}

ImmediateExecutionTensorHandle* EagerContext::TFTensorHandleFromInterface(
    ImmediateExecutionTensorHandle* handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_5(mht_5_v, 339, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerContext::TFTensorHandleFromInterface");

  return handle;
}

// TODO(b/152902651): We have to keep this function here since EagerOperation
// depends on EagerContext. Thus, the context build target can't depend on
// EagerOperation.
ImmediateExecutionOperation* EagerContext::CreateOperation() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_6(mht_6_v, 349, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerContext::CreateOperation");

  return new EagerOperation(this);
}

Status EagerContext::RegisterFunction(AbstractFunction* f) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_7(mht_7_v, 356, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerContext::RegisterFunction");

  FunctionDef* fdef;
  TF_RETURN_IF_ERROR(f->GetFunctionDef(&fdef));
  if (!fdef) {
    return errors::InvalidArgument("GetFunctionDef returned nullptr.");
  }
  return AddFunctionDef(*fdef);
}

// TODO(b/152902651): Once we move many execute.cc functions into
// eager_operation.cc we can avoid a circular dependency between them.
Status EagerOperation::Execute(absl::Span<AbstractTensorHandle*> retvals,
                               int* num_retvals) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScoreDTcc mht_8(mht_8_v, 371, "", "./tensorflow/core/common_runtime/eager/core.cc", "EagerOperation::Execute");

  for (ImmediateExecutionTensorHandle* handle : inputs_) {
    if (TensorHandle::classof(handle)) {
      TF_RETURN_IF_ERROR(down_cast<TensorHandle*>(handle)->WaitUnknownDevice());
    }
  }

  // Run eager placement logic.
  class Device* device = absl::get<class Device*>(Device());
  if (device == nullptr) {
    TF_RETURN_IF_ERROR(eager::MaybePinToResourceDevice(&device, *this));
  }
  if (device == nullptr && ctx_.PinSmallOpsToCPU()) {
    bool pin_to_cpu;
    TF_RETURN_IF_ERROR(eager::MaybePinSmallOpsToCpu(
        &pin_to_cpu, Name(), GetInputs(), ctx_.HostCPU()->name()));
    if (pin_to_cpu) {
      device = ctx_.HostCPU();
    }
  }

  if (device != nullptr) {
    SetDevice(device);
  }
  // At this point all inputs and outputs are TensorHandles associated with
  // physical devices.
  tensorflow::TensorHandle** retval_array =
      reinterpret_cast<tensorflow::TensorHandle**>(retvals.data());
  return EagerExecute(this, retval_array, num_retvals);
}

}  //  namespace tensorflow
