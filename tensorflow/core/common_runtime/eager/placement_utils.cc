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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc() {
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

#include "tensorflow/core/common_runtime/eager/placement_utils.h"

#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace eager {

// These ops are not pinnable since they generate data. It can be slower to
// generate and then copy the data instead of just generating the data on the
// device directly.
static bool IsPinnableOp(StringPiece op_name) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/common_runtime/eager/placement_utils.cc", "IsPinnableOp");

  static const gtl::FlatSet<string>* unpinnable_ops = new gtl::FlatSet<string>({
      "RandomUniform",
      "RandomUniformInt",
      "RandomStandardNormal",
      "StatelessRandomUniform",
      "StatelessRandomUniformInt",
      "StatelessRandomUniformFullInt",
      "StatelessRandomNormal",
  });

  // XRT ops refer to per-device handles that are not safe to move between
  // devices.
  return unpinnable_ops->find(string(op_name)) == unpinnable_ops->end() &&
         !absl::StartsWith(op_name, "XRT");
}
// Validate if the remote device with the given incarnation is valid in the
// remote device manager of the current eager context.
static Status ValidateTensorHandleRemoteDevice(EagerContext* ctx,
                                               int64_t device_incarnation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/common_runtime/eager/placement_utils.cc", "ValidateTensorHandleRemoteDevice");

  if (ctx->remote_device_mgr()->ContainsDevice(device_incarnation)) {
    return Status::OK();
  }
  return errors::InvalidArgument(
      "Resource input tensor contains an invalid device. This might happen "
      "when the client has connected to a different cluster, or some remote "
      "workers have been restarted.");
}

bool IsColocationExempt(StringPiece op_name) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/common_runtime/eager/placement_utils.cc", "IsColocationExempt");

  const auto& exempt_ops = InputColocationExemptionRegistry::Global()->Get();
  return exempt_ops.find(string(op_name)) != exempt_ops.end();
}

bool IsFunction(StringPiece op_name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/common_runtime/eager/placement_utils.cc", "IsFunction");

  const OpDef* op_def = nullptr;
  Status s = OpDefForOp(string(op_name), &op_def);
  if (!s.ok()) {
    if (!errors::IsNotFound(s)) {
      LOG(WARNING) << "Looking up OpDef failed with error: " << s.ToString();
    }
    // Cannot find OpDef, it is a function.
    return true;
  }
  return false;
}

Status MaybePinSmallOpsToCpu(
    bool* result, StringPiece op_name,
    absl::Span<ImmediateExecutionTensorHandle* const> args,
    StringPiece cpu_device_name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/common_runtime/eager/placement_utils.cc", "MaybePinSmallOpsToCpu");

  if (IsFunction(op_name) || IsColocationExempt(op_name) ||
      !IsPinnableOp(op_name)) {
    *result = false;
    return Status::OK();
  }

  // Ops without inputs are usually ops that generate a tensor in some way and
  // usually require being present on whatever device they are scheduled on
  // - for e.g. VarHandleOp or _Recv).
  if (args.empty()) {
    *result = false;
    return Status::OK();
  }

  int i = 0;
  for (auto* arg : args) {
    Status s;
    const char* device_name = arg->DeviceName(&s);
    DataType dtype = arg->DataType();
    TF_RETURN_IF_ERROR(s);

    DVLOG(2) << "for op " << op_name << " input " << i << " "
             << DataTypeString(dtype) << " input device = " << device_name;

    // Input is on CPU.
    if (device_name != cpu_device_name) {
      *result = false;
      return Status::OK();
    }

    if (dtype != DataType::DT_INT32 && dtype != DataType::DT_INT64) {
      *result = false;
      return Status::OK();
    }

    int64_t num_elements;
    TF_RETURN_IF_ERROR(arg->NumElements(&num_elements));
    if (num_elements > 64) {
      *result = false;
      return Status::OK();
    }
    i++;
  }

  // TODO(nareshmodi): Is it possible there is no int32/int64 CPU kernel for
  // an op, but there is a GPU kernel?
  DVLOG(1) << "Forcing op " << op_name
           << " to be on the CPU since all input tensors have an "
              "int32/int64 dtype, and are small (less than 64 elements).";
  *result = true;
  return Status::OK();
}

Status MaybePinToResourceDevice(Device** device, const EagerOperation& op) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSplacement_utilsDTcc mht_5(mht_5_v, 321, "", "./tensorflow/core/common_runtime/eager/placement_utils.cc", "MaybePinToResourceDevice");

  if (op.colocation_exempt()) {
    return Status::OK();
  }
  EagerContext& ctx = op.EagerContext();
  const absl::InlinedVector<TensorHandle*, 4>* inputs;
  TF_RETURN_IF_ERROR(op.TensorHandleInputs(&inputs));
  Device* op_device = op.Device() == kVariantDeviceNull
                          ? ctx.HostCPU()
                          : absl::get<Device*>(op.Device());
  for (int i = 0; i < inputs->size(); ++i) {
    TensorHandle* tensor_handle = (*inputs)[i];
    if (tensor_handle->dtype == DT_RESOURCE) {
      if (tensor_handle->resource_remote_device_incarnation() != 0) {
        TF_RETURN_IF_ERROR(ValidateTensorHandleRemoteDevice(
            &ctx, tensor_handle->resource_remote_device_incarnation()));
      }
      Device* resource_device = tensor_handle->resource_device();
      DVLOG(2) << "for op " << op.Name() << " input " << i << " "
               << DataTypeString(tensor_handle->dtype)
               << " input device = " << resource_device->name()
               << ", op device = " << op_device->name();
      // We check for `op->Device() == nullptr` because it can be later
      // interpreted as unspecified device and a different device can
      // be selected based on device priority. If any input to an op
      // is a resource we must pin it to prevent different device selection.
      // TODO(iga): null device can mean "unspecified" or "CPU". Clean this up.
      if (resource_device != op_device || op.Device() == kVariantDeviceNull) {
        DVLOG(1) << (resource_device != op_device ? "Changing " : "Setting ")
                 << "device of operation " << op.Name() << " to "
                 << resource_device->name() << " because input #" << i
                 << " is a resource in this device.";
        *device = resource_device;
        return Status::OK();
        // No point in looking at other inputs. If there are other resources,
        // they must have the same device and we already declared the op to be
        // ineligible for CPU pinning.
      }
    }
  }
  return Status::OK();
}

}  // namespace eager
}  // namespace tensorflow
