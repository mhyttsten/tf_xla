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
class MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc() {
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

#include "tensorflow/c/eager/parallel_device/parallel_device_testlib.h"

#include <array>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/platform/test.h"

// NOTE(allenl): These tests currently go through TFE_Execute and so are
// integration testing rather than purely testing the parallel device. They
// correspond fairly well to the implementation, but testing the C++ directly is
// another option.

namespace tensorflow {
namespace parallel_device {

Variable* Variable::Create(TFE_Context* context, TF_DataType type,
                           const int64_t* dims, const int num_dims,
                           const char* device, TF_Status* status) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device: \"" + (device == nullptr ? std::string("nullptr") : std::string((char*)device)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_0(mht_0_v, 206, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Variable::Create");

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "VarHandleOp", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op.get(), "dtype", type);
  TFE_OpSetAttrShape(op.get(), "shape", dims, num_dims, status);
  TFE_OpSetAttrString(op.get(), "container", "", 0);
  // Use the special GUID for no buffer sharing
  //
  // TODO(allenl): Should we provide a better API for this? AFAIK this is the
  // only reasonable way to make variables with no aliasing using the eager C
  // API.
  std::string no_sharing = "cd2c89b7-88b7-44c8-ad83-06c2a9158347";
  TFE_OpSetAttrString(op.get(), "shared_name", no_sharing.c_str(),
                      no_sharing.length());
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op.get(), &var_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return new Variable(var_handle, type);
}

void Variable::Destroy(TFE_Context* context, TF_Status* status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_1(mht_1_v, 233, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Variable::Destroy");

  // Free the backing buffer for the variable.
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "DestroyResourceOp", status), &TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpAddInput(op.get(), handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* device = TFE_TensorHandleDeviceName(handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return;
  int num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return;
  // Delete the variable handle itself.
  TFE_DeleteTensorHandle(handle_);
}

TensorHandlePtr Variable::Read(TFE_Context* context, TF_Status* status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_2(mht_2_v, 254, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Variable::Read");

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "ReadVariableOp", status), &TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpAddInput(op.get(), handle_, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  const char* device = TFE_TensorHandleDeviceName(handle_, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op.get(), "dtype", type_);
  int num_retvals = 1;
  TFE_TensorHandle* var_value = nullptr;
  TFE_Execute(op.get(), &var_value, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(var_value);
}

void Variable::GeneralAssignment(const char* op_name, TFE_Context* context,
                                 TFE_TensorHandle* value, TF_Status* status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_3(mht_3_v, 277, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Variable::GeneralAssignment");

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, op_name, status), &TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetAttrType(op.get(), "dtype", type_);
  TFE_OpAddInput(op.get(), handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpAddInput(op.get(), value, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* device = TFE_TensorHandleDeviceName(handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetDevice(op.get(), device, status);

  int num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return;
}

void Variable::AssignAdd(TFE_Context* context, TFE_TensorHandle* value,
                         TF_Status* status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_4(mht_4_v, 299, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Variable::AssignAdd");

  GeneralAssignment("AssignAddVariableOp", context, value, status);
}

void Variable::Assign(TFE_Context* context, TFE_TensorHandle* value,
                      TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_5(mht_5_v, 307, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Variable::Assign");

  GeneralAssignment("AssignVariableOp", context, value, status);
}

// Passed to `TF_NewTensor` to indicate how an array of floats should be
// deleted.
static void FloatDeallocator(void* data, size_t, void* arg) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_6(mht_6_v, 316, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "FloatDeallocator");

  delete[] static_cast<float*>(data);
}

// Creates a TFE_TensorHandle with value `v`.
TensorHandlePtr FloatTensorHandle(float v, TF_Status* status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_7(mht_7_v, 324, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "FloatTensorHandle");

  const int num_bytes = sizeof(float);
  float* values = new float[1];
  values[0] = v;
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
      TF_NewTensor(TF_FLOAT, nullptr, 0, values, num_bytes, &FloatDeallocator,
                   nullptr),
      TF_DeleteTensor);
  return TensorHandlePtr(TFE_NewTensorHandle(tensor.get(), status));
}

// Creates a rank-one TFE_TensorHandle with value `v`.
TensorHandlePtr VectorFloatTensorHandle(const std::vector<float>& v,
                                        TF_Status* status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_8(mht_8_v, 340, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "VectorFloatTensorHandle");

  const int num_bytes = v.size() * sizeof(float);
  float* values = new float[v.size()];
  memcpy(values, v.data(), num_bytes);
  int64_t dims = v.size();
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
      TF_NewTensor(TF_FLOAT, &dims, 1 /* num_dims */, values, num_bytes,
                   &FloatDeallocator, nullptr),
      TF_DeleteTensor);
  return TensorHandlePtr(TFE_NewTensorHandle(tensor.get(), status));
}

// Helper to un-pack `num_replicas` TFE_TensorHandles from one parallel handle.
template <std::size_t num_replicas>
void ExtractPerDeviceValues(
    TFE_Context* context, TFE_TensorHandle* input,
    std::array<TensorHandlePtr, num_replicas>* components, TF_Status* status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_9(mht_9_v, 359, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "ExtractPerDeviceValues");

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "TPUReplicatedOutput", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetAttrInt(op.get(), "num_replicas", num_replicas);
  TFE_OpAddInput(op.get(), input, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* device = TFE_TensorHandleDeviceName(input, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return;

  TFE_TensorHandle* result_handles[num_replicas];
  int num_retvals = num_replicas;
  TFE_Execute(op.get(), result_handles, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return;
  for (int i = 0; i < num_replicas; ++i) {
    (*components)[i].reset(result_handles[i]);
  }
}

TensorHandlePtr Multiply(TFE_Context* context, TFE_TensorHandle* first,
                         TFE_TensorHandle* second, TF_Status* status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_10(mht_10_v, 384, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "Multiply");

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "Mul", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpAddInput(op.get(), first, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpAddInput(op.get(), second, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  const char* first_device = TFE_TensorHandleDeviceName(first, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetDevice(op.get(), first_device, status);

  TFE_TensorHandle* result_handle;
  int num_retvals = 1;
  TFE_Execute(op.get(), &result_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(result_handle);
}

// Create and modify a variable placed on a parallel device which composes
// `first_device` and `second_device`.
void BasicTestsForTwoDevices(TFE_Context* context, const char* first_device,
                             const char* second_device) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("first_device: \"" + (first_device == nullptr ? std::string("nullptr") : std::string((char*)first_device)) + "\"");
   mht_11_v.push_back("second_device: \"" + (second_device == nullptr ? std::string("nullptr") : std::string((char*)second_device)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_11(mht_11_v, 411, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "BasicTestsForTwoDevices");

  // Register the custom device
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{first_device, second_device};
  RegisterParallelDevice(context, device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a variable handle (uninitialized to start) placed on the parallel
  // device.
  std::function<void(Variable*)> variable_deleter = [&](Variable* to_delete) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTcc mht_12(mht_12_v, 426, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.cc", "lambda");

    to_delete->Destroy(context, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    delete to_delete;
  };
  std::unique_ptr<Variable, decltype(variable_deleter)> variable(
      Variable::Create(context, TF_FLOAT, /* Scalar */ {}, 0, device_name,
                       status.get()),
      variable_deleter);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Assign an initial value to the variable, mirroring it to each component
  // device.
  {
    TensorHandlePtr initial_value_cpu = FloatTensorHandle(20., status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TFE_TensorHandle*, 2> components{initial_value_cpu.get(),
                                                initial_value_cpu.get()};
    TensorHandlePtr initial_value =
        CreatePerDeviceValues(context, components, device_name, status.get());
    variable->Assign(context, initial_value.get(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  }

  // Read from the variable and verify that we have a parallel tensor.
  {
    TensorHandlePtr read = variable->Read(context, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TensorHandlePtr, 2> components;
    ExtractPerDeviceValues(context, read.get(), &components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    ExpectScalarEq<float>(components[0].get(), 20.);
    ExpectScalarEq<float>(components[1].get(), 20.);

    std::string first_device =
        TFE_TensorHandleBackingDeviceName(components[0].get(), status.get());
    ASSERT_EQ(underlying_devices[0], first_device);
    std::string second_device =
        TFE_TensorHandleBackingDeviceName(components[1].get(), status.get());
    ASSERT_EQ(underlying_devices[1], second_device);
  }

  // Add a parallel tensor with different values on each device to the variable.
  {
    TensorHandlePtr value_one(FloatTensorHandle(3., status.get()));
    TensorHandlePtr value_two(FloatTensorHandle(-2., status.get()));
    std::array<TFE_TensorHandle*, 2> components{value_one.get(),
                                                value_two.get()};
    TensorHandlePtr combined_value =
        CreatePerDeviceValues(context, components, device_name, status.get());
    variable->AssignAdd(context, combined_value.get(), status.get());
  }

  // Read the variable and verify that each component has the right modified
  // value.
  {
    TensorHandlePtr read = variable->Read(context, status.get());
    std::array<TensorHandlePtr, 2> components;
    ExtractPerDeviceValues(context, read.get(), &components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    ExpectScalarEq<float>(components[0].get(), 23.);
    ExpectScalarEq<float>(components[1].get(), 18.);

    std::string first_device =
        TFE_TensorHandleBackingDeviceName(components[0].get(), status.get());
    ASSERT_EQ(underlying_devices[0], first_device);
    std::string second_device =
        TFE_TensorHandleBackingDeviceName(components[1].get(), status.get());
    ASSERT_EQ(underlying_devices[1], second_device);
  }
}

}  // namespace parallel_device
}  // namespace tensorflow
