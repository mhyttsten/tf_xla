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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/custom_device_op_handler.h"

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

void CustomDeviceOpHandler::Clear() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/common_runtime/eager/custom_device_op_handler.cc", "CustomDeviceOpHandler::Clear");
 custom_devices_.clear(); }

Status CustomDeviceOpHandler::RegisterCustomDevice(
    const string& device_name, std::unique_ptr<CustomDevice> device) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc mht_1(mht_1_v, 198, "", "./tensorflow/core/common_runtime/eager/custom_device_op_handler.cc", "CustomDeviceOpHandler::RegisterCustomDevice");

  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device_name, &parsed) ||
      !parsed.has_job || !parsed.has_replica || !parsed.has_task ||
      !parsed.has_type || !parsed.has_id) {
    return errors::InvalidArgument(
        device_name,
        " could not be parsed as a device name. Use the full "
        "/job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num> "
        "format.");
  }

  if (!custom_devices_.emplace(device_name, std::move(device)).second) {
    return errors::AlreadyExists(device_name,
                                 " already registered as a custom device.");
  }
  return Status::OK();
}

bool CustomDeviceOpHandler::FindCustomDeviceFromName(
    const string& name, CustomDevice** device) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/common_runtime/eager/custom_device_op_handler.cc", "CustomDeviceOpHandler::FindCustomDeviceFromName");

  auto dev_it = custom_devices_.find(name);
  if (dev_it == custom_devices_.end()) {
    return false;
  }
  *device = dev_it->second.get();
  return true;
}

Status CustomDeviceOpHandler::Execute(ImmediateExecutionOperation* op,
                                      ImmediateExecutionTensorHandle** retvals,
                                      int* num_retvals) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/common_runtime/eager/custom_device_op_handler.cc", "CustomDeviceOpHandler::Execute");

  tensorflow::CustomDevice* custom_device = nullptr;

  TF_RETURN_IF_ERROR(MaybePinToCustomDevice(&custom_device, *op));

  if (custom_device != nullptr) {
    return custom_device->Execute(op, retvals, num_retvals);
  }

  // The op will be placed on physical device. However, it contains custom
  // device tensor handles. The tensor handles will be copy to physical device
  // first.
  if (op->HasCustomDeviceInput()) {
    auto inputs = op->GetInputs();
    for (int i = 0; i < inputs.size(); ++i) {
      auto target_device = op->DeviceName();
      if (target_device.empty()) {
        target_device = op->GetContext()->HostCPUName();
      }
      // TODO(b/175427838): It would be nice to be able to use tensorflow::isa
      // here.
      if (tensorflow::CustomDeviceTensorHandle::classof(inputs[i])) {
        tensorflow::CustomDeviceTensorHandle* previous =
            tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
                inputs[i]);
        tensorflow::ImmediateExecutionTensorHandle* new_tensor;
        TF_RETURN_IF_ERROR(previous->device()->CopyTensorFromDevice(
            previous, target_device, &new_tensor));
        Status s = op->SetInput(i, new_tensor);
        new_tensor->Unref();
        TF_RETURN_IF_ERROR(s);
      }
    }
  }

  return op->Execute(
      absl::MakeSpan(
          reinterpret_cast<tensorflow::AbstractTensorHandle**>(retvals),
          *num_retvals),
      num_retvals);
}

ImmediateExecutionTensorHandle* CustomDeviceOpHandler::CopyTensorHandleToDevice(
    ImmediateExecutionContext* context, ImmediateExecutionTensorHandle* handle,
    const char* device_name, Status* status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc mht_4(mht_4_v, 284, "", "./tensorflow/core/common_runtime/eager/custom_device_op_handler.cc", "CustomDeviceOpHandler::CopyTensorHandleToDevice");

  *status = Status::OK();
  ImmediateExecutionTensorHandle* result = nullptr;
  tensorflow::CustomDevice* dev;

  if (FindCustomDeviceFromName(device_name, &dev)) {
    *status = dev->CopyTensorToDevice(handle, &result);
    if (status->ok()) {
      return result;
    }
    return nullptr;
  }

  // Target device is regular device. Check if the input is on custom
  // device
  const char* handle_device_name = handle->DeviceName(status);
  if (!status->ok()) {
    return nullptr;
  }
  if (FindCustomDeviceFromName(handle_device_name, &dev)) {
    *status = dev->CopyTensorFromDevice(handle, device_name, &result);
    if (status->ok()) {
      return result;
    }
    return nullptr;
  }

  // Both source and target device are regular device.
  return context->CopyTensorHandleToDevice(handle, device_name, status);
}

Status CustomDeviceOpHandler::MaybePinToCustomDevice(
    CustomDevice** device, const ImmediateExecutionOperation& op) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_device_op_handlerDTcc mht_5(mht_5_v, 319, "", "./tensorflow/core/common_runtime/eager/custom_device_op_handler.cc", "CustomDeviceOpHandler::MaybePinToCustomDevice");

  *device = nullptr;
  if (!FindCustomDeviceFromName(op.DeviceName(), device) &&
      !op.HasCustomDeviceInput()) {
    return Status::OK();
  }

  // Ops are placed on a custom device if there's no other explicit requested
  // placement and there is only one custom device in the op
  // inputs.
  //
  // Resource-dtype inputs take precedence over non-resource inputs and explicit
  // placements; this function pins ops with a resource-dtype custom device
  // input to that custom device.
  CustomDevice* first = nullptr;
  if (!op.GetInputs().empty()) {
    for (const ImmediateExecutionTensorHandle* generic_input : op.GetInputs()) {
      // TODO(b/175427838): It would be nice to be able to use tensorflow::isa
      // here.
      if (CustomDeviceTensorHandle::classof(generic_input)) {
        const CustomDeviceTensorHandle* input =
            down_cast<const CustomDeviceTensorHandle*>(generic_input);
        CustomDevice* current = input->device();
        if (first == nullptr) {
          first = current;
        } else if (first != current) {
          return errors::InvalidArgument(absl::StrCat(
              "If an operation has one of its inputs in a custom device, then "
              "all inputs should be on that same custom device or another "
              "physical device. Operation ",
              op.Name(),
              " has one input in custom "
              "device ",
              first->name(),
              " and at least one input in a different custom device ",
              current->name()));
        }
      }
    }
    for (const ImmediateExecutionTensorHandle* generic_input : op.GetInputs()) {
      if (generic_input->DataType() == DT_RESOURCE) {
        if (CustomDeviceTensorHandle::classof(generic_input)) {
          const CustomDeviceTensorHandle* input =
              down_cast<const CustomDeviceTensorHandle*>(generic_input);
          // There's only one custom device input, and it's a resource input, so
          // we'll force-place the op on to that custom device. As with physical
          // devices, this overrides any explicit placement for the op.
          *device = input->device();
          return Status::OK();
        } else {
          // Don't set a custom device if there's a physical-device resource
          // input.
          return Status::OK();
        }
      }
    }
  }
  // Since there are no resource-dtype inputs, we'll respect explicit placements
  // before considering input-based placement.
  if (*device == nullptr && op.DeviceName().empty() && first != nullptr) {
    // If there are non-resource inputs on a custom device we will default the
    // op to that custom device, but not override an explicit op placement.
    *device = first;
    return Status::OK();
  }
  return Status::OK();
}

}  // namespace tensorflow
