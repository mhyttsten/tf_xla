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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh() {
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

#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_operation.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class TensorHandle;
class EagerOperation;
class CustomDeviceTensorHandle;

// Custom devices intercept the execution of operations (the `Execute` method),
// typically implemented with one or more of the custom device's own executions.
class CustomDevice {
 public:
  virtual ~CustomDevice() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_0(mht_0_v, 205, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "~CustomDevice");
}
  virtual const string& name() = 0;
  virtual Status CopyTensorToDevice(
      ImmediateExecutionTensorHandle* tensor,
      ImmediateExecutionTensorHandle** result) = 0;

  virtual Status CopyTensorFromDevice(
      ImmediateExecutionTensorHandle* tensor, const string& target_device_name,
      ImmediateExecutionTensorHandle** result) = 0;

  virtual Status Execute(const ImmediateExecutionOperation* op,
                         ImmediateExecutionTensorHandle** retvals,
                         int* num_retvals) = 0;

  // Creates a packed TensorHandle from a group of custom device TensorHandles,
  // one of which is on this custom device.
  virtual Status Pack(absl::Span<ImmediateExecutionTensorHandle*> handles,
                      ImmediateExecutionTensorHandle** result) = 0;
};

// Custom devices do many of the same things as physical Devices, but have a
// much more restricted interface. We pass around ambiguous pointers since
// operations may be placed either on custom or physical devices.
using VariantDevice = absl::variant<Device*, CustomDevice*>;

// Indicates either HostCPU or an unset physical device. We never set a null
// CustomDevice*.
const VariantDevice kVariantDeviceNull = static_cast<Device*>(nullptr);

// A tensor handle produced by a custom device. Generally they can only be
// consumed by executing an operation on the same custom device that produced it
// originally, or by attempting to copy the handle off the custom device.
//
// TODO(allenl): Currently custom devices are tied to the eager C API. They
// should be renamed op handlers and subclass AbstractTensorHandle instead so
// they are eager/graph agnostic.
class CustomDeviceTensorHandle : public ImmediateExecutionTensorHandle {
 public:
  CustomDeviceTensorHandle(ImmediateExecutionContext* context,
                           CustomDevice* device, tensorflow::DataType dtype)
      : ImmediateExecutionTensorHandle(kCustomDevice),
        context_(context),
        device_(device),
        dtype_(dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_1(mht_1_v, 251, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "CustomDeviceTensorHandle");
}

  // TODO(allenl): Should this be a generic method of
  // ImmediateExecutionTensorHandle to support TFE_TensorHandleDevicePointer?
  virtual void* DevicePointer() const = 0;

  tensorflow::DataType DataType() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_2(mht_2_v, 260, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "DataType");
 return dtype_; }
  Status Shape(PartialTensorShape* shape) const override;
  Status NumElements(int64_t* num_elements) const override;

  const char* DeviceName(Status* status) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_3(mht_3_v, 267, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "DeviceName");

    return device_->name().c_str();
  }
  const char* BackingDeviceName(Status* status) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_4(mht_4_v, 273, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "BackingDeviceName");

    return device_->name().c_str();
  }
  CustomDevice* device() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_5(mht_5_v, 279, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "device");
 return device_; }
  const char* DeviceType(Status* status) const override;
  int DeviceId(Status* status) const override;

  AbstractTensorInterface* Resolve(Status* status) override;

  ImmediateExecutionTensorHandle* Copy() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_6(mht_6_v, 288, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "Copy");

    Ref();
    return this;
  }
  void Release() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_7(mht_7_v, 295, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "Release");
 Unref(); }

  // For LLVM style RTTI.
  static bool classof(const AbstractTensorHandle* ptr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScustom_deviceDTh mht_8(mht_8_v, 301, "", "./tensorflow/core/common_runtime/eager/custom_device.h", "classof");

    return ptr->getKind() == kCustomDevice;
  }

 protected:
  const DeviceNameUtils::ParsedName* ParsedName(Status* status) const;

  ImmediateExecutionContext* const context_;
  CustomDevice* const device_;
  const tensorflow::DataType dtype_;

  mutable absl::optional<DeviceNameUtils::ParsedName> parsed_name_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_CUSTOM_DEVICE_H_
