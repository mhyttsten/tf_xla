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

#ifndef TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_TESTLIB_H_
#define TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_TESTLIB_H_
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
class MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh {
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
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh() {
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


#include <array>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/parallel_device/parallel_device.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace parallel_device {

// A helper for performing common operations on variables. A much more
// restricted stand-in for tf.Variable in Python.
class Variable {
 public:
  // Construct a Variable from a resource-dtype TFE_TensorHandle and an
  // indication of the dtype of the variable's value.
  //
  // Note that creating this resource-dtype handle can fail, so `Create` is a
  // separate static method which returns a status.
  Variable(TFE_TensorHandle* handle, TF_DataType type)
      : handle_(handle), type_(type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh mht_0(mht_0_v, 211, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.h", "Variable");
}

  // Helper for constructing a resource handle and wrapping it in a `Variable`
  // object.
  static Variable* Create(TFE_Context* context, TF_DataType type,
                          const int64_t* dims, const int num_dims,
                          const char* device, TF_Status* status);
  // Dereferences the backing buffer for the variable. Note that since this can
  // fail (it runs operations), it must be called explicitly and the resulting
  // `status` checked.
  void Destroy(TFE_Context* context, TF_Status* status);

  // Reads from the variable.
  TensorHandlePtr Read(TFE_Context* context, TF_Status* status);
  // Assigns a new value to the variable.
  void Assign(TFE_Context* context, TFE_TensorHandle* value, TF_Status* status);
  // Adds `value` to the existing value of the variable.
  void AssignAdd(TFE_Context* context, TFE_TensorHandle* value,
                 TF_Status* status);

 private:
  // Helper for running any single-argument assignment ops (Assign, AssignAdd,
  // AssignSub, ...).
  void GeneralAssignment(const char* op_name, TFE_Context* context,
                         TFE_TensorHandle* value, TF_Status* status);

  // The a handle for the resource-dtype tensor pointing to the variable's
  // buffer.
  TFE_TensorHandle* handle_;
  // The dtype of the variable's buffer (input dtype for assignments, output
  // dtype of read operations).
  TF_DataType type_;
};

// Creates a TFE_TensorHandle with value `v`.
TensorHandlePtr FloatTensorHandle(float v, TF_Status* status);

// Creates a rank-one TFE_TensorHandle with value `v`.
TensorHandlePtr VectorFloatTensorHandle(const std::vector<float>& v,
                                        TF_Status* status);

// Helper to un-pack `num_replicas` TFE_TensorHandles from one parallel handle.
template <std::size_t num_replicas>
void ExtractPerDeviceValues(
    TFE_Context* context, TFE_TensorHandle* input,
    std::array<TensorHandlePtr, num_replicas>* components, TF_Status* status);

// Helper to pack `num_replicas` TFE_TensorHandles into one parallel handle.
template <std::size_t num_replicas>
TensorHandlePtr CreatePerDeviceValues(
    TFE_Context* context,
    const std::array<TFE_TensorHandle*, num_replicas>& components,
    const char* device, TF_Status* status);

TensorHandlePtr Multiply(TFE_Context* context, TFE_TensorHandle* first,
                         TFE_TensorHandle* second, TF_Status* status);

// Assert that `handle` is equal to `expected_value`.
template <typename value_type>
void ExpectScalarEq(TFE_TensorHandle* handle, value_type expected_value);

template <std::size_t num_devices>
void RegisterParallelDevice(
    TFE_Context* context, const char* device_name,
    const std::array<const char*, num_devices>& underlying_devices,
    TF_Status* status);

// Create and modify a variable placed on a parallel device which composes
// `first_device` and `second_device`.
void BasicTestsForTwoDevices(TFE_Context* context, const char* first_device,
                             const char* second_device);

// Implementations of templated functions ******************************

template <std::size_t num_replicas>
TensorHandlePtr CreatePerDeviceValues(
    TFE_Context* context,
    const std::array<TFE_TensorHandle*, num_replicas>& components,
    const char* device, TF_Status* status) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device: \"" + (device == nullptr ? std::string("nullptr") : std::string((char*)device)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh mht_1(mht_1_v, 293, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.h", "CreatePerDeviceValues");

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "TPUReplicatedInput", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrInt(op.get(), "N", num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    TFE_OpAddInput(op.get(), components[i], status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_TensorHandle* result_handle;
  int num_retvals = 1;
  TFE_Execute(op.get(), &result_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(result_handle);
}

template <typename value_type>
void ExpectScalarEq(TFE_TensorHandle* handle, value_type expected_value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh mht_2(mht_2_v, 316, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.h", "ExpectScalarEq");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> actual_value(
      TFE_TensorHandleResolve(handle, status.get()), TF_DeleteTensor);
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
  ASSERT_EQ(TF_TensorType(actual_value.get()),
            static_cast<TF_DataType>(DataTypeToEnum<value_type>().value));
  EXPECT_EQ(expected_value,
            *static_cast<value_type*>(TF_TensorData(actual_value.get())));
}

template <std::size_t num_devices>
void RegisterParallelDevice(
    TFE_Context* context, const char* device_name,
    const std::array<const char*, num_devices>& underlying_devices,
    TF_Status* status) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_testlibDTh mht_3(mht_3_v, 336, "", "./tensorflow/c/eager/parallel_device/parallel_device_testlib.h", "RegisterParallelDevice");

  TFE_CustomDevice device;
  void* device_info;
  tensorflow::parallel_device::AllocateParallelDevice(
      device_name, underlying_devices.data(), underlying_devices.size(),
      &device, &device_info);
  TFE_RegisterCustomDevice(context, device, device_name, device_info, status);
}

}  // namespace parallel_device
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_TESTLIB_H_
