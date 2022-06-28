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
class MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc() {
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

// A simple logging device to test custom device registration.
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/test.h"

namespace {

struct LoggingDevice {
  tensorflow::string device_name;
  tensorflow::string underlying_device;
  // Set to true whenever a TensorHandle is copied onto the device
  bool* arrived_flag;
  // Set to true whenever an operation is executed
  bool* executed_flag;
  // If true, only explicit op placements are accepted. If false, uses
  // type-based dispatch.
  bool strict_scope_placement;
};

struct LoggedTensor {
  TFE_TensorHandle* tensor;
  LoggedTensor() = delete;
  explicit LoggedTensor(TFE_TensorHandle* tensor) : tensor(tensor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_0(mht_0_v, 213, "", "./tensorflow/c/eager/custom_device_testutil.cc", "LoggedTensor");
}
  ~LoggedTensor() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_1(mht_1_v, 217, "", "./tensorflow/c/eager/custom_device_testutil.cc", "~LoggedTensor");
 TFE_DeleteTensorHandle(tensor); }
};

int64_t LoggedTensorDim(void* data, int dim_index, TF_Status* status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_2(mht_2_v, 223, "", "./tensorflow/c/eager/custom_device_testutil.cc", "LoggedTensorDim");

  return TFE_TensorHandleDim(reinterpret_cast<LoggedTensor*>(data)->tensor,
                             dim_index, status);
}

int LoggedTensorNumDims(void* data, TF_Status* status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_3(mht_3_v, 231, "", "./tensorflow/c/eager/custom_device_testutil.cc", "LoggedTensorNumDims");

  return TFE_TensorHandleNumDims(reinterpret_cast<LoggedTensor*>(data)->tensor,
                                 status);
}

void LoggedTensorDeallocator(void* data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_4(mht_4_v, 239, "", "./tensorflow/c/eager/custom_device_testutil.cc", "LoggedTensorDeallocator");

  delete reinterpret_cast<LoggedTensor*>(data);
}

TFE_TensorHandle* MakeLoggedTensorHandle(
    TFE_Context* context, const tensorflow::string& logging_device_name,
    std::unique_ptr<LoggedTensor> t, TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_5(mht_5_v, 248, "", "./tensorflow/c/eager/custom_device_testutil.cc", "MakeLoggedTensorHandle");

  auto dtype = TFE_TensorHandleDataType(t->tensor);
  TFE_CustomDeviceTensorHandleMethods handle_methods;
  handle_methods.num_dims = &LoggedTensorNumDims;
  handle_methods.dim = &LoggedTensorDim;
  handle_methods.deallocator = &LoggedTensorDeallocator;
  return TFE_NewCustomDeviceTensorHandle(context, logging_device_name.c_str(),
                                         dtype, t.release(), handle_methods,
                                         status);
}

TFE_TensorHandle* CopyToLoggingDevice(TFE_Context* context,
                                      TFE_TensorHandle* tensor,
                                      TF_Status* status, void* device_info) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_6(mht_6_v, 264, "", "./tensorflow/c/eager/custom_device_testutil.cc", "CopyToLoggingDevice");

  LoggingDevice* dev = reinterpret_cast<LoggingDevice*>(device_info);
  TFE_TensorHandle* t = TFE_TensorHandleCopyToDevice(
      tensor, context, dev->underlying_device.c_str(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  auto dst = std::make_unique<LoggedTensor>(t);
  *(dev->arrived_flag) = true;
  return MakeLoggedTensorHandle(context, dev->device_name, std::move(dst),
                                status);
}

TFE_TensorHandle* CopyTensorFromLoggingDevice(TFE_Context* context,
                                              TFE_TensorHandle* tensor,
                                              const char* target_device_name,
                                              TF_Status* status,
                                              void* device_info) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("target_device_name: \"" + (target_device_name == nullptr ? std::string("nullptr") : std::string((char*)target_device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_7(mht_7_v, 283, "", "./tensorflow/c/eager/custom_device_testutil.cc", "CopyTensorFromLoggingDevice");

  TF_SetStatus(status, TF_INTERNAL,
               "Trying to copy a tensor out of a logging device.");
  return nullptr;
}

void LoggingDeviceExecute(const TFE_Op* original_op, int* num_outputs,
                          TFE_TensorHandle** outputs, TF_Status* s,
                          void* device_info) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_8(mht_8_v, 294, "", "./tensorflow/c/eager/custom_device_testutil.cc", "LoggingDeviceExecute");

  const char* requested_placement = TFE_OpGetDevice(original_op, s);
  if (TF_GetCode(s) != TF_OK) return;

  LoggingDevice* dev = reinterpret_cast<LoggingDevice*>(device_info);
  if (dev->strict_scope_placement && *requested_placement == '\0') {
    TF_SetStatus(s, TF_INTERNAL,
                 "Ops must be placed on the device explicitly, or their inputs "
                 "first copied to other devices.");
    return;
  }
  TFE_Context* context = TFE_OpGetContext(original_op, s);
  if (TF_GetCode(s) != TF_OK) return;
  const char* operation_name = TFE_OpGetName(original_op, s);
  if (TF_GetCode(s) != TF_OK) return;
  const TFE_OpAttrs* attributes = TFE_OpGetAttrs(original_op);

  TFE_Op* op(TFE_NewOp(context, operation_name, s));
  if (TF_GetCode(s) != TF_OK) return;
  TFE_OpAddAttrs(op, attributes);
  TFE_OpSetDevice(op, dev->underlying_device.c_str(), s);
  if (TF_GetCode(s) != TF_OK) return;
  int num_inputs = TFE_OpGetFlatInputCount(original_op, s);
  if (TF_GetCode(s) != TF_OK) return;
  for (int j = 0; j < num_inputs; ++j) {
    TFE_TensorHandle* input = TFE_OpGetFlatInput(original_op, j, s);
    if (TF_GetCode(s) != TF_OK) return;
    const char* input_device = TFE_TensorHandleDeviceName(input, s);
    if (TF_GetCode(s) != TF_OK) return;
    if (dev->device_name == input_device) {
      LoggedTensor* t = reinterpret_cast<LoggedTensor*>(
          TFE_TensorHandleDevicePointer(input, s));
      if (TF_GetCode(s) != TF_OK) return;
      TFE_OpAddInput(op, t->tensor, s);
    } else {
      TFE_OpAddInput(op, input, s);
    }
    if (TF_GetCode(s) != TF_OK) return;
  }
  std::vector<TFE_TensorHandle*> op_outputs(*num_outputs);
  TFE_Execute(op, op_outputs.data(), num_outputs, s);
  TFE_DeleteOp(op);
  if (TF_GetCode(s) != TF_OK) return;
  std::vector<TFE_TensorHandle*> unwrapped_outputs;
  unwrapped_outputs.reserve(op_outputs.size());
  for (auto* handle : op_outputs) {
    unwrapped_outputs.push_back(handle);
  }
  for (int i = 0; i < *num_outputs; ++i) {
    auto logged_tensor = std::make_unique<LoggedTensor>(unwrapped_outputs[i]);
    outputs[i] = MakeLoggedTensorHandle(context, dev->device_name,
                                        std::move(logged_tensor), s);
  }
  *(dev->executed_flag) = true;
}

void DeleteLoggingDevice(void* device_info) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_9(mht_9_v, 353, "", "./tensorflow/c/eager/custom_device_testutil.cc", "DeleteLoggingDevice");

  delete reinterpret_cast<LoggingDevice*>(device_info);
}

}  // namespace

void RegisterLoggingDevice(TFE_Context* context, const char* name,
                           bool strict_scope_placement, bool* arrived_flag,
                           bool* executed_flag, TF_Status* status) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_10(mht_10_v, 365, "", "./tensorflow/c/eager/custom_device_testutil.cc", "RegisterLoggingDevice");

  TFE_CustomDevice custom_device;
  custom_device.copy_tensor_to_device = &CopyToLoggingDevice;
  custom_device.copy_tensor_from_device = &CopyTensorFromLoggingDevice;
  custom_device.delete_device = &DeleteLoggingDevice;
  custom_device.execute = &LoggingDeviceExecute;
  LoggingDevice* device = new LoggingDevice;
  device->arrived_flag = arrived_flag;
  device->executed_flag = executed_flag;
  device->device_name = name;
  device->underlying_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  device->strict_scope_placement = strict_scope_placement;
  TFE_RegisterCustomDevice(context, custom_device, name, device, status);
}

TFE_TensorHandle* UnpackTensorHandle(TFE_TensorHandle* logged_tensor_handle,
                                     TF_Status* status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_11(mht_11_v, 384, "", "./tensorflow/c/eager/custom_device_testutil.cc", "UnpackTensorHandle");

  return reinterpret_cast<LoggedTensor*>(
             TFE_TensorHandleDevicePointer(logged_tensor_handle, status))
      ->tensor;
}

void AllocateLoggingDevice(const char* name, bool* arrived_flag,
                           bool* executed_flag, TFE_CustomDevice** device,
                           void** device_info) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPScustom_device_testutilDTcc mht_12(mht_12_v, 396, "", "./tensorflow/c/eager/custom_device_testutil.cc", "AllocateLoggingDevice");

  TFE_CustomDevice* custom_device = new TFE_CustomDevice;
  custom_device->copy_tensor_to_device = &CopyToLoggingDevice;
  custom_device->copy_tensor_from_device = &CopyTensorFromLoggingDevice;
  custom_device->delete_device = &DeleteLoggingDevice;
  custom_device->execute = &LoggingDeviceExecute;
  *device = custom_device;
  LoggingDevice* logging_device = new LoggingDevice;
  logging_device->arrived_flag = arrived_flag;
  logging_device->executed_flag = executed_flag;
  logging_device->device_name = name;
  logging_device->underlying_device =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  logging_device->strict_scope_placement = true;
  *device_info = reinterpret_cast<void*>(logging_device);
}
