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
class MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/dtensor/cc/dtensor_device.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace py = ::pybind11;
using tensorflow::dtensor::AddMesh;
using tensorflow::dtensor::AllocateDTensorDevice;
using tensorflow::dtensor::ClearTPUCoreIDs;
using tensorflow::dtensor::ExperimentalClearDefaultLayout;
using tensorflow::dtensor::ExperimentalClearDefaultMesh;
using tensorflow::dtensor::ExperimentalSetDefaultLayout;
using tensorflow::dtensor::ExperimentalSetDefaultMesh;
using tensorflow::dtensor::FetchLayout;
using tensorflow::dtensor::IsSparseDTensor;
using tensorflow::dtensor::Pack;
using tensorflow::dtensor::SetSameShapePolicy;
using tensorflow::dtensor::SetTPUCoreIDs;
using tensorflow::dtensor::SparsePack;
using tensorflow::dtensor::TPUCoreIDsToLocations;
using tensorflow::dtensor::TPUCoreLocationsToIDs;
using tensorflow::dtensor::Unpack;

void PyXDecref(PyObject* obj) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc mht_0(mht_0_v, 217, "", "./tensorflow/python/pywrap_dtensor_device.cc", "PyXDecref");
 Py_XDECREF(obj); }

void CallDelete_Device(PyObject* capsule) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc mht_1(mht_1_v, 222, "", "./tensorflow/python/pywrap_dtensor_device.cc", "CallDelete_Device");

  delete reinterpret_cast<TFE_CustomDevice*>(
      PyCapsule_GetPointer(capsule, "TFE_CustomDevice"));
}

void CallDelete_DeviceInfo(PyObject* capsule) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc mht_2(mht_2_v, 230, "", "./tensorflow/python/pywrap_dtensor_device.cc", "CallDelete_DeviceInfo");

  void (*destructor)(void*) =
      reinterpret_cast<void (*)(void*)>(PyCapsule_GetContext(capsule));
  destructor(PyCapsule_GetPointer(capsule, "TFE_CustomDevice_DeviceInfo"));
}

// Supports 2 cases:
//  i) input is an EagerTensor.
//  ii) input is an arbitrary python list/tuple.
void ConvertToTensor(TFE_Context* ctx, PyObject* input,
                     tensorflow::Safe_PyObjectPtr* output_handle,
                     TF_Status* status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSpywrap_dtensor_deviceDTcc mht_3(mht_3_v, 244, "", "./tensorflow/python/pywrap_dtensor_device.cc", "ConvertToTensor");

  if (EagerTensor_CheckExact(input)) {
    // Input is already a EagerTensor so increment the reference, since the
    // caller will use it through output_handle.
    Py_INCREF(input);
    output_handle->reset(input);
    return;
  }
  TFE_TensorHandle* handle =
      tensorflow::ConvertToEagerTensor(ctx, input, tensorflow::DT_INVALID);
  if (handle == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Failure converting to eager tensor.");
    return;
  }
  output_handle->reset(EagerTensorFromHandle(handle));
}

PYBIND11_MODULE(_pywrap_dtensor_device, m) {
  m.def("Allocate", [](const std::string& name) {
    TFE_CustomDevice* device = new TFE_CustomDevice;
    std::unique_ptr<PyObject, decltype(&PyXDecref)> device_capsule(
        PyCapsule_New(device, "TFE_CustomDevice", &CallDelete_Device),
        PyXDecref);
    void* device_info;
    AllocateDTensorDevice(name, device, &device_info);
    std::unique_ptr<PyObject, decltype(&PyXDecref)> device_info_capsule(
        PyCapsule_New(device_info, "TFE_CustomDevice_DeviceInfo",
                      &CallDelete_DeviceInfo),
        PyXDecref);
    // The PyCapsule destructor needs a pointer to the destructor for
    // DeviceInfo.
    PyCapsule_SetContext(device_info_capsule.get(),
                         reinterpret_cast<void*>(device->delete_device));
    if (PyErr_Occurred()) throw py::error_already_set();
    return pybind11::reinterpret_steal<pybind11::object>(
        PyTuple_Pack(2, device_capsule.get(), device_info_capsule.get()));
  });
  m.def("AddMesh", [](const py::capsule& device_info,
                      const std::string& serialized_mesh, bool is_async,
                      bool is_host_mesh) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    AddMesh(
        serialized_mesh,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        is_async, is_host_mesh, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def(
      "ExperimentalSetDefaultLayout",
      [](const py::capsule& device_info, const std::string& serialized_layout) {
        std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
            TF_NewStatus(), TF_DeleteStatus);
        ExperimentalSetDefaultLayout(
            serialized_layout,
            PyCapsule_GetPointer(device_info.ptr(),
                                 "TFE_CustomDevice_DeviceInfo"),
            status.get());
        if (TF_GetCode(status.get()) != TF_OK) {
          PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
          throw py::error_already_set();
        }
      });
  m.def("ExperimentalClearDefaultLayout", [](const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    ExperimentalClearDefaultLayout(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("ExperimentalSetDefaultMesh", [](const py::capsule& device_info,
                                         const std::string& serialized_mesh) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    ExperimentalSetDefaultMesh(
        serialized_mesh,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("ExperimentalClearDefaultMesh", [](const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    ExperimentalClearDefaultMesh(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("SetSameShapePolicy", [](const py::capsule& device_info, bool enabled) {
    SetSameShapePolicy(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        enabled);
  });
  m.def("SetTPUCoreIDs", [](const py::capsule& device_info,
                            const std::string& mesh_name,
                            const std::vector<int>& tpu_core_ids) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    SetTPUCoreIDs(
        mesh_name, tpu_core_ids,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("ClearTPUCoreIDs", [](const py::capsule& device_info) {
    ClearTPUCoreIDs(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"));
  });
  m.def("TPUCoreIDsToLocations", [](const py::handle& context,
                                    const py::capsule& device_info,
                                    const std::vector<int>& tpu_core_ids) {
    return TPUCoreIDsToLocations(
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr)),
        tpu_core_ids,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"));
  });
  m.def("TPUCoreLocationsToIDs",
        [](const py::handle& context, const py::capsule& device_info,
           const std::vector<std::vector<int>>& tpu_core_locations) {
          return TPUCoreLocationsToIDs(
              static_cast<TFE_Context*>(
                  PyCapsule_GetPointer(context.ptr(), nullptr)),
              tpu_core_locations,
              PyCapsule_GetPointer(device_info.ptr(),
                                   "TFE_CustomDevice_DeviceInfo"));
        });
  m.def("Pack", [](const py::handle& context, const py::handle& input_tensors,
                   const std::string& string_layout,
                   const py::capsule& device_info, const bool is_sparse) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    TFE_Context* ctx =
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr));
    // Convert each python object to safe py eagertensors.
    std::vector<tensorflow::Safe_PyObjectPtr> py_eager_tensor_handles;
    Py_ssize_t len = PyList_Size(input_tensors.ptr());
    py_eager_tensor_handles.resize(len);

    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PyList_GetItem(input_tensors.ptr(), i);
      ConvertToTensor(ctx, elem, &py_eager_tensor_handles[i], status.get());

      if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
        return tensorflow::PyoOrThrow(nullptr);
    }
    std::vector<TFE_TensorHandle*> input_vector;
    input_vector.resize(len);
    for (int i = 0; i < len; ++i)
      input_vector[i] = EagerTensor_Handle(py_eager_tensor_handles[i].get());
    TFE_TensorHandle* packed_tensor;
    if (is_sparse) {
      auto size = input_vector.size() / 3;
      packed_tensor = SparsePack(
          ctx,
          /*num_inputs=*/input_vector.size() / 3,
          /*indices=*/
          std::vector<TFE_TensorHandle*>(input_vector.begin(),
                                         input_vector.begin() + size)
              .data(),
          /*values=*/
          std::vector<TFE_TensorHandle*>(input_vector.begin() + size,
                                         input_vector.begin() + 2 * size)
              .data(),
          /*shapes=*/
          std::vector<TFE_TensorHandle*>(input_vector.begin() + 2 * size,
                                         input_vector.end())
              .data(),
          string_layout, device_info, status.get());
    } else {
      packed_tensor = Pack(ctx, input_vector.size(), input_vector.data(),
                           string_layout, device_info, status.get());
    }
    if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
      return tensorflow::PyoOrThrow(nullptr);
    // Convert c++ packed tensor handle into a python eager tensor object.
    tensorflow::Safe_PyObjectPtr flat_result(PyList_New(1));
    PyList_SET_ITEM(flat_result.get(), 0, EagerTensorFromHandle(packed_tensor));
    auto* result = PyList_GET_ITEM(flat_result.get(), 0);
    Py_INCREF(result);
    return tensorflow::PyoOrThrow(result);
  });
  m.def("Unpack", [](const py::handle& context,
                     const py::handle& dtensor_handle,
                     const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);

    TFE_TensorHandle* input_handle = EagerTensor_Handle(dtensor_handle.ptr());
    std::vector<TFE_TensorHandle*> unpacked_handles = Unpack(
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr)),
        input_handle, device_info, status.get());

    if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
      return tensorflow::PyoOrThrow(nullptr);
    // Convert all TFE_TensorHandles to py EagerTensor and
    // return a python list of them.
    int num_outputs = unpacked_handles.size();
    PyObject* result(PyList_New(num_outputs));
    for (int i = 0; i < num_outputs; ++i) {
      PyList_SET_ITEM(result, i, EagerTensorFromHandle(unpacked_handles[i]));
    }
    return tensorflow::PyoOrThrow(result);
  });
  m.def("FetchLayout",
        [](const py::handle& context, const py::handle& dtensor_handle,
           const py::capsule& device_info) -> py::object {
          std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
              TF_NewStatus(), TF_DeleteStatus);

          std::string layout_string =
              FetchLayout(static_cast<TFE_Context*>(
                              PyCapsule_GetPointer(context.ptr(), nullptr)),
                          EagerTensor_Handle(dtensor_handle.ptr()), device_info,
                          status.get());
          if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(),
                                                          nullptr))
            return tensorflow::PyoOrThrow(nullptr);
          return tensorflow::PyoOrThrow(
              PyUnicode_FromString(layout_string.c_str()));
        });
  m.def("IsSparseDTensor", [](const py::handle& context,
                              const py::handle& dtensor_handle,
                              const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);

    TFE_TensorHandle* input_handle = EagerTensor_Handle(dtensor_handle.ptr());
    bool is_sparse = IsSparseDTensor(
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr)),
        input_handle, device_info, status.get());

    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
    return is_sparse;
  });
}
