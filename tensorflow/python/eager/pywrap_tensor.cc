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
class MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/eager/pywrap_tensor.h"

#include <stdlib.h>
#include <string.h>

#include <cmath>

#include "structmember.h"  // NOLINT // For PyMemberDef
#include "pybind11/pybind11.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/python/eager/pywrap_tensor_conversion.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/py_seq_tensor.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

// forward declare
struct EagerTensor;
namespace tensorflow {

// Convert a TFE_TensorHandle to a Python numpy.ndarray object.
// The two may share underlying storage so changes to one may reflect in the
// other.
PyObject* TFE_TensorHandleToNumpy(TFE_TensorHandle* handle, TF_Status* status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_0(mht_0_v, 220, "", "./tensorflow/python/eager/pywrap_tensor.cc", "TFE_TensorHandleToNumpy");

  if (TFE_TensorHandleDataType(handle) == TF_RESOURCE) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Cannot convert a Tensor of dtype resource to a NumPy array.");
    return nullptr;
  }

  if (TFE_TensorHandleDataType(handle) == TF_VARIANT) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Cannot convert a Tensor of dtype variant to a NumPy array.");
    return nullptr;
  }
  tensorflow::Safe_TF_TensorPtr tensor = nullptr;
  Py_BEGIN_ALLOW_THREADS;
  tensor = tensorflow::make_safe(TFE_TensorHandleResolve(handle, status));
  Py_END_ALLOW_THREADS;
  if (!status->status.ok()) {
    return nullptr;
  }

  PyObject* ret = nullptr;
  auto cppstatus =
      tensorflow::TF_TensorToMaybeAliasedPyArray(std::move(tensor), &ret);
  tensorflow::Set_TF_Status_from_Status(status, cppstatus);
  if (!status->status.ok()) {
    Py_XDECREF(ret);
    return nullptr;
  }
  CHECK_NE(ret, nullptr);
  return ret;
}
}  // namespace tensorflow
namespace {

using tensorflow::TFE_TensorHandleToNumpy;

// An instance of _EagerTensorProfiler that will receive callbacks about
// events on eager tensors. This is set by TFE_Py_InitEagerTensor, if at all.
PyObject* eager_tensor_profiler = nullptr;

// Read-only dict. Please don't use this in any setting where the dict might
// actually get mutated. This is only used to pass empty kwargs when creating a
// new EagerTensor.
PyObject* EmptyDict() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_1(mht_1_v, 266, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EmptyDict");

  static PyObject* empty_dict = PyDict_New();
  return empty_dict;
}

PyObject* EmptyTuple() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_2(mht_2_v, 274, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EmptyTuple");

  static PyObject* empty_tuple = PyTuple_New(0);
  return empty_tuple;
}

TFE_Context* GetContextHandle(PyObject* py_context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_3(mht_3_v, 282, "", "./tensorflow/python/eager/pywrap_tensor.cc", "GetContextHandle");

  tensorflow::Safe_PyObjectPtr py_context_handle(
      PyObject_GetAttrString(py_context, "_handle"));
  if (py_context_handle == nullptr) {
    // Current Python code makes sure this never happens. If it does, or
    // becomes hard to maintain, we can call the ensure_initialized() method
    // here.
    PyErr_SetString(
        PyExc_TypeError,
        "Expected `context` argument in EagerTensor constructor to have a "
        "`_handle` attribute but it did not. Was eager Context initialized?");
    return nullptr;
  }

  auto* ctx = reinterpret_cast<TFE_Context*>(
      PyCapsule_GetPointer(py_context_handle.get(), nullptr));
  if (ctx == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "Expected context._handle to contain a PyCapsule "
                        "encoded pointer to TFE_Context. Got ",
                        Py_TYPE(py_context_handle.get())->tp_name)
                        .c_str());
  }
  return ctx;
}


// Helper function to convert `v` to a tensorflow::DataType and store it in
// `*out`. Returns true on success, false otherwise.
// Note that we assume that v is a python int (not long) representing a
// TF_DataType/tensorflow::DataType value.
bool PyIntToDataType(PyObject* v, tensorflow::DataType* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_4(mht_4_v, 317, "", "./tensorflow/python/eager/pywrap_tensor.cc", "PyIntToDataType");

#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(v)) {
    *out = static_cast<tensorflow::DataType>(PyInt_AS_LONG(v));
    return true;
  }
#else
  if (PyLong_Check(v)) {
    *out = static_cast<tensorflow::DataType>(PyLong_AsLong(v));
    return true;
  }
#endif
  return false;
}

// Helper function to create a python integer from TF_DataType.
PyObject* PyIntFromDataType(TF_DataType l) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_5(mht_5_v, 336, "", "./tensorflow/python/eager/pywrap_tensor.cc", "PyIntFromDataType");

#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(l);
#else
  return PyLong_FromLong(l);
#endif
}

// PyObject->tensorflow::DataType conversion function to be used with
// PyArg_Parse* APIs.
int ConvertDataType(PyObject* obj, tensorflow::DataType* dst) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_6(mht_6_v, 349, "", "./tensorflow/python/eager/pywrap_tensor.cc", "ConvertDataType");

  if (obj == Py_None) {
    *dst = tensorflow::DataType::DT_INVALID;
  } else if (!PyIntToDataType(obj, dst)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a DataType value for dtype. Got ", Py_TYPE(obj)->tp_name)
            .c_str());
    return 0;
  }

  return 1;
}

// Conversion function extracting a const char** device name from a PyObject.
// The function should be used with PyArg_Parse* APIs.
int ConvertDeviceName(PyObject* obj, const char** dst) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_7(mht_7_v, 369, "", "./tensorflow/python/eager/pywrap_tensor.cc", "ConvertDeviceName");

  if (obj == Py_None) {
    *dst = nullptr;
  } else {
    auto device_name = TFE_GetPythonString(obj);
    if (device_name == nullptr) {
      PyErr_Clear();
      PyErr_SetString(PyExc_TypeError, "Error parsing device argument.");
      return 0;
    }
    *dst = device_name;
  }

  return 1;
}

void RaiseExceptionTypeFromTFStatus(TF_Status* tf_status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_8(mht_8_v, 388, "", "./tensorflow/python/eager/pywrap_tensor.cc", "RaiseExceptionTypeFromTFStatus");

  auto status = tensorflow::StatusFromTF_Status(tf_status);
  SetRegisteredErrFromStatus(status);
}

}  // namespace

namespace tensorflow {
// This function checks whether the desired type is "compatible" with the
// inferred type. At a high level, compatibility means that all integral types
// are compatible with each other, and all floating types are compatible with
// each other.
//
// Type compatibility doesn't consider overflows (i.e. int64 is *always*
// compatible with int32). This is intended to match graph behavior.
bool IsCompatible(DataType desired, DataType returned) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_9(mht_9_v, 406, "", "./tensorflow/python/eager/pywrap_tensor.cc", "IsCompatible");

  if (desired == returned) return true;

  if (DataTypeIsInteger(desired) && DataTypeIsInteger(returned)) {
    return true;
  } else if (DataTypeIsFloating(desired) &&
             (DataTypeIsFloating(returned) || DataTypeIsInteger(returned))) {
    return true;
  } else if (DataTypeIsComplex(desired) &&
             (DataTypeIsComplex(returned) || DataTypeIsInteger(returned) ||
              DataTypeIsFloating(returned))) {
    return true;
  } else if (DataTypeIsQuantized(desired) && DataTypeIsInteger(returned)) {
    return true;
  }
  return false;
}

// TODO(nareshmodi): Move EagerCast and ReadVariableOp (which use the C API to
// execute TFE Ops) to a separate common library.
// Casts data referred to by `handle` from type `src_type_enum` to type
// `dst_type_enum`.
TFE_TensorHandle* EagerCast(TFE_Context* ctx, TFE_TensorHandle* handle,
                            TF_DataType src_type_enum,
                            TF_DataType dst_type_enum, TF_Status* out_status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_10(mht_10_v, 433, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerCast");

  if (ctx == nullptr) return nullptr;
  const char* op_name = "Cast";
  const char* device_name = "/device:CPU:0";
  TFE_Op* op = TFE_NewOp(ctx, op_name, out_status);
#define RETURN_ERROR  \
  {                   \
    TFE_DeleteOp(op); \
    return nullptr;   \
  }
  if (!out_status->status.ok()) RETURN_ERROR
  TFE_OpSetDevice(op, device_name, out_status);
  if (!out_status->status.ok()) RETURN_ERROR
  TFE_OpAddInput(op, handle, out_status);
  if (!out_status->status.ok()) RETURN_ERROR
  TFE_OpSetAttrType(op, "SrcT", src_type_enum);
  TFE_OpSetAttrType(op, "DstT", dst_type_enum);
  TFE_OpSetAttrBool(op, "Truncate", false);
  TFE_TensorHandle* output = nullptr;
  int num_outputs = 1;
  TFE_Execute(op, &output, &num_outputs, out_status);
  if (!out_status->status.ok() || num_outputs != 1 || output == nullptr) {
    if (output != nullptr) {
      TFE_DeleteTensorHandle(output);
    }
    RETURN_ERROR
  }
  TFE_DeleteOp(op);
  return output;
#undef RETURN_ERROR
}

Safe_TFE_TensorHandlePtr EagerConst(TFE_Context* ctx, TFE_TensorHandle* handle,
                                    const char* device_name,
                                    TF_Status* out_status) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_11(mht_11_v, 471, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerConst");

  const char* op_name = "_EagerConst";
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(ctx, op_name, out_status), TFE_DeleteOp);
  if (!out_status->status.ok()) return nullptr;
  TFE_OpSetDevice(op.get(), device_name, out_status);
  if (!out_status->status.ok()) return nullptr;
  TFE_OpAddInput(op.get(), handle, out_status);
  if (!out_status->status.ok()) return nullptr;
  TFE_OpSetAttrType(op.get(), "T", TFE_TensorHandleDataType(handle));
  TFE_TensorHandle* output = nullptr;
  int num_outputs = 1;
  TFE_Execute(op.get(), &output, &num_outputs, out_status);
  Safe_TFE_TensorHandlePtr result(output);
  if (!out_status->status.ok() || num_outputs != 1) {
    return nullptr;
  }
  return result;
}

TFE_TensorHandle* ConvertToEagerTensorUncached(TFE_Context* ctx,
                                               PyObject* value,
                                               tensorflow::DataType dtype,
                                               const char* device_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_12(mht_12_v, 498, "", "./tensorflow/python/eager/pywrap_tensor.cc", "ConvertToEagerTensorUncached");

  tensorflow::Safe_PyObjectPtr value_decrefer;
  if (PyArray_IsScalar(value, Generic)) {
    // Convert numpy scalars to numpy arrays.
    value = PyArray_FromScalar(value, nullptr);
    // The returned value needs to be DECREF'd, but the original value was
    // created in python code, and doesn't need to be DECREF'd.
    value_decrefer.reset(value);
  }

  Safe_TFE_TensorHandlePtr handle =
      make_safe(PySeqToTFE_TensorHandle(ctx, value, dtype));

  if (handle == nullptr) return nullptr;

  Safe_TF_StatusPtr status = make_safe(TF_NewStatus());
  TF_DataType handle_dtype = TFE_TensorHandleDataType(handle.get());
  if (dtype != tensorflow::DT_INVALID &&
      dtype != static_cast<DataType>(handle_dtype)) {
    if (tensorflow::IsCompatible(dtype, static_cast<DataType>(handle_dtype))) {
      handle = tensorflow::make_safe(
          tensorflow::EagerCast(ctx, handle.get(), handle_dtype,
                                static_cast<TF_DataType>(dtype), status.get()));
      if (!status->status.ok()) {
        PyErr_SetString(PyExc_TypeError,
                        absl::StrCat("Error while casting from dtype ",
                                     tensorflow::DataTypeString(
                                         static_cast<DataType>(handle_dtype)),
                                     " to ", tensorflow::DataTypeString(dtype),
                                     ". ", TF_Message(status.get()))
                            .c_str());
        return nullptr;
      }
    } else {
      tensorflow::Safe_PyObjectPtr value_str(PyObject_Repr(value));
      PyErr_SetString(
          PyExc_TypeError,
          absl::StrCat("Cannot convert ", TFE_GetPythonString(value_str.get()),
                       " to EagerTensor of dtype ",
                       tensorflow::DataTypeString(dtype))
              .c_str());
      return nullptr;
    }
  }

  // We always initially generate CPU:0 tensors. Copy to the current device.
  if (device_name != nullptr) {
    if (strstr(device_name, "/device:CPU:0") != nullptr) {
      // We always generate CPU:0 tensors, but we may need to change the device
      // slightly, as for example from /job:localhost/... to /job:worker/...
      //
      // Note that this is a shallow copy and will share the underlying buffer,
      // because we are copying to the same device.
      handle = make_safe(TFE_TensorHandleCopyToDevice(
          handle.get(), ctx, device_name, status.get()));
      const TF_Code code = TF_GetCode(status.get());
      if (code != TF_OK) {
        RaiseExceptionTypeFromTFStatus(status.get());
        return nullptr;
      }
    } else {
      /*Copy the constant to the current device. Identity is sometimes
        overloaded to allow copies like this, but using a different op allows
        devices to support constant creation without allowing copies via
        identity ops.

        Note that running this _EagerConst op limits mirroring of cached Python
        literals somewhat. Mirroring of constants themselves works:

        with tf.device("GPU:0"):
          tf.constant(1.)  # Cached on CPU:0, mirrored to GPU:0
        with tf.device("GPU:1"):
          tf.constant(1.)  # Cache hit for the CPU version, new mirror to GPU:1.
        with tf.device("GPU:1"):
          tf.constant(1.)  # Cache hit for the CPU version, cached mirror

        But mirrors for the output of `tf.constant` are not shared just because
        there was a cache hit for the input literal, because of _EagerConst:

        x = tf.constant(2.)  # Cached on CPU:0
        with tf.device("GPU:1"):
          tf.identity(x)  # `x` now mirrored to GPU:1
        y = tf.constant(2.)  # Cache hit for CPU version
        with tf.device("GPU:1"):
          tf.identity(y)  # `y` now mirrored on GPU:1 (new copy!)*/
      handle =
          tensorflow::EagerConst(ctx, handle.get(), device_name, status.get());
      const TF_Code code = TF_GetCode(status.get());
      if (code != TF_OK) {
        RaiseExceptionTypeFromTFStatus(status.get());
        return nullptr;
      }
    }
  }

  return handle.release();
}

TFE_TensorHandle* ConvertToEagerTensor(TFE_Context* ctx, PyObject* value,
                                       DataType dtype,
                                       const char* device_name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_13(mht_13_v, 602, "", "./tensorflow/python/eager/pywrap_tensor.cc", "ConvertToEagerTensor");

  // Reduce the overhead of allocation/transfer-to-device for scalars by
  // caching the corresponding handles. Note that currently only Python
  // scalars are cached.
  // TODO(slebedev): also cache singleton NumPy arrays and scalars?
  if (PyArray_IsPythonNumber(value)) {
    auto* cache = TFE_TensorHandleCache::Get();
    TFE_TensorHandle* handle = cache->Lookup(value, dtype, ctx, device_name);
    if (handle != nullptr) return handle;
    handle = ConvertToEagerTensorUncached(ctx, value, dtype, device_name);
    if (handle == nullptr) return nullptr;
    if (!PyFloat_Check(value) || std::isfinite(PyFloat_AS_DOUBLE(value))) {
      cache->Insert(value, dtype, ctx, device_name, handle);
    }
    return handle;
  } else {
    return ConvertToEagerTensorUncached(ctx, value, dtype, device_name);
  }
}

}  // namespace tensorflow

extern "C" {

static const int kMaxEagerTensorParentSize = 64;

// TODO(agarwal): store context handle in EagerTensor.
typedef struct EagerTensor {
  PyObject_HEAD;
  // Note that we leave kMaxEagerTensorParentSize bytes here for use by the
  // parent class. The parent class is set at runtime, so we don't know the
  // exact size at compile time.
  char unused[kMaxEagerTensorParentSize];
  TFE_TensorHandle* handle;
  int64_t id;
  // Indicates whether it's a packed tensor or not.
  bool is_packed;
  // This mirrors tensorflow.core.framework.ops.Tensor._handle_data Which will
  // be None for tensors of type other than DT_RESOURCE. For DT_RESOURCE
  // tensors, this will contain a serialized HandleData proto with shape
  // inference metadata about shapes and dtypes of resources accessible from
  // this handle.
  // Note that we assume that handle_data cannot participate in reference
  // cycles, and hence don't provide GC support for it.
  PyObject* handle_data;

  // This stores `_tensor_shape`, a cached `TensorShape` object, and is set the
  // first time that `_EagerTensorBase`'s `shape` property is called.
  PyObject* tensor_shape;

  // We store a status object here as an optimization to avoid allocating a new
  // Status objects on different functions that operate on EagerTensor and need
  // to use a TF_Status object. However note that accesses to `status` are not
  // thread-safe.
  TF_Status status;

  // The eager Context (from eager/context.py) used by this Tensor.
  // This is currently used only to make sure context outlives TensorHandles.
  PyObject* context;

  PyObject* weakreflist; /* List of weak references */

  // Per-instance attribute dictionary, to support monkey patching
  // (e.g. EagerTensor.assign when slicing variables). This dictionary is
  // created by CPython the first time an attribute is assigned, pointed to by
  // tp_dictoffset. Note that garbage collection is not enabled for
  // EagerTensors, so assigning objects to EagerTensor attributes which require
  // garbage collection is likely to cause issues.
  PyObject* dict;
} EagerTensor;

namespace {

// Returns true on success - successfully invoked or no profiler registered.
// Returns false if some error occurred.
bool MaybeInvokeCreatedOnEagerTensorProfiler(EagerTensor* created_tensor) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_14(mht_14_v, 680, "", "./tensorflow/python/eager/pywrap_tensor.cc", "MaybeInvokeCreatedOnEagerTensorProfiler");

  if (eager_tensor_profiler != nullptr) {
#if PY_MAJOR_VERSION < 3
    PyObject* created_method_name = PyString_InternFromString("created");
#else
    PyObject* created_method_name = PyUnicode_InternFromString("created");
#endif
    if (created_method_name == nullptr) {
      return false;
    }
    PyObject* result = PyObject_CallMethodObjArgs(
        eager_tensor_profiler, created_method_name, created_tensor, NULL);
    if (result == nullptr) {
      LOG(ERROR) << "Invoking created() on EagerTensor profiler failed";
      // While we can potentially continue because the error is related to
      // profiling, we choose to return an error because:
      //  - If profiling is used, the user likely wants to stop execution on
      //    profiling errors.
      //  - Error in profiling code might have left some state in an invalid
      //    form that can lead to an error later on. Better to fail fast.
      Py_DECREF(created_method_name);
      return false;
    }
    Py_DECREF(created_method_name);
    Py_DECREF(result);
  }
  return true;
}

}  // namespace

// tp_init for EagerTensor.
int EagerTensor_init(EagerTensor* self, PyObject* args, PyObject* kwds) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_15(mht_15_v, 715, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_init");

  self->id = get_uid();
  self->handle = nullptr;
  self->is_packed = false;
  Py_INCREF(Py_None);
  self->handle_data = Py_None;
  Py_INCREF(Py_None);
  self->tensor_shape = Py_None;
  self->status.status = tensorflow::Status::OK();
  self->dict = nullptr;
  self->weakreflist = nullptr;
  self->context = nullptr;
  PyObject* value;
  const char* device_name = nullptr;
  tensorflow::DataType dtype = tensorflow::DataType::DT_INVALID;
  const char* kwlist[] = {"value", "device", "dtype", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "OO&|O&", const_cast<char**>(kwlist), &value,
          ConvertDeviceName, &device_name, ConvertDataType, &dtype)) {
    return -1;
  }

  PyObject* py_context = GetPyEagerContext();
  if (py_context == nullptr) return -1;
  self->context = py_context;

  auto* handle = tensorflow::ConvertToEagerTensor(GetContextHandle(py_context),
                                                  value, dtype, device_name);
  if (handle == nullptr) return -1;
  self->handle = handle;

  if (!MaybeInvokeCreatedOnEagerTensorProfiler(self)) {
    return -1;
  }

  return 0;
}

// tp_dealloc for EagerTensor.
void EagerTensor_dealloc(EagerTensor* self) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_16(mht_16_v, 757, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_dealloc");

  // Unhook the object from python's GC so that the weakref deleter doesn't
  // try to re-delete this.
  PyObject_GC_UnTrack((PyObject*)self);

  // Clear weak references to self.
  // Needs to happen before any actual destruction.
  PyObject_ClearWeakRefs((PyObject*)self);

  Py_DECREF(self->handle_data);
  Py_DECREF(self->tensor_shape);
  // If an attribute dictionary has been created, release it. Note that this
  // is only ever created by CPython's attribute setting methods; we don't
  // create it ourselves.
  Py_CLEAR(self->dict);
  if (self->handle != nullptr) {
    // Destructor may call arbitrary functions that end up calling into
    // Python from another thread.
    Py_BEGIN_ALLOW_THREADS;
    TFE_DeleteTensorHandle(self->handle);
    Py_END_ALLOW_THREADS;
    self->handle = nullptr;
  }

  // Decref context after deleting the tensor handle.
  Py_XDECREF(self->context);

  // We have the global interpreter lock, so use this chance to perform delayed
  // refcount decrements.
  tensorflow::ClearDecrefCache();
  auto id = self->id;
  Py_TYPE(self)->tp_free(self);
  TFE_Py_TapeSetDeleteTrace(id);
}

// Getter for `_id`.
static PyObject* EagerTensor_getid(EagerTensor* self, void* closure) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_17(mht_17_v, 796, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_getid");

  return PyLong_FromLongLong(self->id);
}

// Getter for `_datatype_enum`.
static PyObject* EagerTensor_datatype_enum(EagerTensor* self) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_18(mht_18_v, 804, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_datatype_enum");

  return PyIntFromDataType(TFE_TensorHandleDataType(self->handle));
}

// Getter for `_shape_tuple`.
static PyObject* EagerTensor_shape_tuple(EagerTensor* self) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_19(mht_19_v, 812, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_shape_tuple");

  auto handle = self->handle;
  int n = TFE_TensorHandleNumDims(handle, &self->status);
  TF_Code code = TF_GetCode(&self->status);
  if (code != TF_OK) {
    RaiseExceptionTypeFromTFStatus(&self->status);
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  }
  PyObject* shape = PyTuple_New(n);
  if (PyErr_Occurred()) return nullptr;
  for (int i = 0; i < n; ++i) {
    int64_t dim_c_value = TFE_TensorHandleDim(handle, i, &self->status);
    PyObject* dim;
    // The C++ convention is -1 for unknown/variable axis lengths. Translate
    // that to the Python "None" convention. Unknown axis lengths are unusual
    // for eager tensors.
    if (dim_c_value < 0) {
      Py_IncRef(Py_None);
      dim = Py_None;
    } else {
      dim = PyLong_FromLongLong(dim_c_value);
    }
    code = TF_GetCode(&self->status);
    if (code != TF_OK || dim == nullptr ||
        PyTuple_SetItem(shape, i, dim) != 0) {
      if (code != TF_OK) {
        RaiseExceptionTypeFromTFStatus(&self->status);
      } else {
        PyErr_SetString(PyExc_RuntimeError, "Error while creating shape");
      }
      // Cleanup self->status before returning.
      self->status.status = tensorflow::Status::OK();
      Py_DECREF(shape);
      if (dim != nullptr) Py_DECREF(dim);
      return nullptr;
    }
  }
  return shape;
}

// Getter for `_rank`.
static PyObject* EagerTensor_rank(EagerTensor* self) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_20(mht_20_v, 858, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_rank");

  int num_dims = TFE_TensorHandleNumDims(self->handle, &self->status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status, nullptr)) {
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  }
#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(num_dims);
#else
  return PyLong_FromLong(num_dims);
#endif
}

// Getter for `_num_elements`.
static PyObject* EagerTensor_num_elements(EagerTensor* self) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_21(mht_21_v, 876, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_num_elements");

  auto handle = self->handle;
  int n = TFE_TensorHandleNumElements(handle, &self->status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status, nullptr)) {
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  }
  return PyLong_FromLongLong(n);
}

static PyObject* EagerTensor_handle_data(EagerTensor* self, void* unused) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_22(mht_22_v, 890, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_handle_data");

  Py_INCREF(self->handle_data);
  return self->handle_data;
}

static int EagerTensor_sethandle_data(EagerTensor* self, PyObject* value,
                                      void* unused) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_23(mht_23_v, 899, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_sethandle_data");

  Py_DECREF(self->handle_data);
  Py_INCREF(value);
  self->handle_data = value;
  return 0;
}

static PyObject* EagerTensor_tensor_shape(EagerTensor* self, void* unused) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_24(mht_24_v, 909, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_tensor_shape");

  Py_INCREF(self->tensor_shape);
  return self->tensor_shape;
}

static int EagerTensor_settensor_shape(EagerTensor* self, PyObject* value,
                                       void* unused) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_25(mht_25_v, 918, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_settensor_shape");

  Py_DECREF(self->tensor_shape);
  Py_INCREF(value);
  self->tensor_shape = value;
  return 0;
}

// Function `_copy_to_device`.
static PyObject* EagerTensor_copy_to_device(EagerTensor* self, PyObject* args,
                                            PyObject* kwds) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_26(mht_26_v, 930, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_copy_to_device");

  if (!_PyArg_NoKeywords("copy_to_device", kwds)) return nullptr;

  const char* device_name = nullptr;
  if (!PyArg_ParseTuple(args, "O&:copy_to_device", ConvertDeviceName,
                        &device_name)) {
    return nullptr;
  }

  // Note that this is a shallow copy and will share the underlying buffer
  // if copying to the same device.
  TFE_TensorHandle* handle = TFE_TensorHandleCopyToDevice(
      self->handle, GetContextHandle(self->context), device_name,
      &self->status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status,
                                                  PyExc_RuntimeError)) {
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  }

  return EagerTensorFromHandle(handle);
}

// Function `_numpy_internal`.
// Convert an EagerTensor to a Python numpy.ndarray object.
// The two may share underlying storage so changes to one may reflect in the
// other.
// Note that if `self` is not on CPU, we raise an Exception.
static PyObject* EagerTensor_numpy_internal(EagerTensor* self) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_27(mht_27_v, 962, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_numpy_internal");

  auto* py_array = TFE_TensorHandleToNumpy(self->handle, &self->status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status, nullptr)) {
    Py_XDECREF(py_array);
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  } else {
    return PyArray_Return(reinterpret_cast<PyArrayObject*>(py_array));
  }
}

// Function `_prefer_custom_summarizer`.
//
// A hint that callers should prefer `SummarizeValue` to resolving this handle
// and formatting the tensor.
static PyObject* EagerTensor_prefer_custom_summarizer(EagerTensor* self) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_28(mht_28_v, 981, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_prefer_custom_summarizer");

  if (tensorflow::unwrap(self->handle)->PreferCustomSummarizer()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// Function `_summarize_value`.
//
// Returns a string PyObject which summarizes the value of this tensor. It does
// not include a shape or dtype.
static PyObject* EagerTensor_summarize_value(EagerTensor* self) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_29(mht_29_v, 996, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_summarize_value");

  std::string summary;
  tensorflow::Status status =
      tensorflow::unwrap(self->handle)->SummarizeValue(summary);
  if (MaybeRaiseExceptionFromStatus(status, nullptr)) {
    return nullptr;
  }
  return PyUnicode_FromString(summary.c_str());
}

// Getter `device`.
static PyObject* EagerTensor_device(EagerTensor* self) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_30(mht_30_v, 1010, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_device");

  const char* device = TFE_TensorHandleDeviceName(self->handle, &self->status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status,
                                                  PyExc_ValueError)) {
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  }
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(device);
#else
  return PyBytes_FromString(device);
#endif
}

// Getter `backing_device`.
static PyObject* EagerTensor_backing_device(EagerTensor* self) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_31(mht_31_v, 1029, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_backing_device");

  const char* device =
      TFE_TensorHandleBackingDeviceName(self->handle, &self->status);
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status,
                                                  PyExc_ValueError)) {
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return nullptr;
  }
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(device);
#else
  return PyBytes_FromString(device);
#endif
}

// Getter `is_packed`.
static PyObject* EagerTensor_is_packed(EagerTensor* self) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_32(mht_32_v, 1049, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_is_packed");

  return PyBool_FromLong(self->is_packed);
}

static PyGetSetDef EagerTensor_getsetters[] = {
    {const_cast<char*>("_id"), (getter)EagerTensor_getid, nullptr,
     const_cast<char*>("Tensor ID."), nullptr},
    {const_cast<char*>("device"), (getter)EagerTensor_device, nullptr,
     const_cast<char*>("Device of op that produced the tensor."), nullptr},
    {const_cast<char*>("backing_device"), (getter)EagerTensor_backing_device,
     nullptr, const_cast<char*>("Device on which tensor's memory is resident."),
     nullptr},
    {const_cast<char*>("is_packed"), (getter)EagerTensor_is_packed, nullptr,
     const_cast<char*>("Whether the EagerTensor is a packed tensor or not."),
     nullptr},
    {const_cast<char*>("_handle_data"), (getter)EagerTensor_handle_data,
     (setter)EagerTensor_sethandle_data,
     const_cast<char*>("Shape/DType data if the EagerTensor is a DT_RESOURCE"),
     nullptr},
    {const_cast<char*>("_tensor_shape"), (getter)EagerTensor_tensor_shape,
     (setter)EagerTensor_settensor_shape,
     const_cast<char*>("Shape of the tensor."), nullptr},
    {nullptr} /* Sentinel */
};

#if PY_MAJOR_VERSION < 3
// Only used for Python2 since Python3 seems to set the __dict__ correctly.
static PyMemberDef EagerTensor_members[] = {
    {const_cast<char*>("__dict__"), T_OBJECT, offsetof(EagerTensor, dict),
     READONLY},
    {nullptr},
};
#endif

static PyMethodDef EagerTensor_methods[] = {
    {"_numpy_internal", (PyCFunction)EagerTensor_numpy_internal, METH_NOARGS,
     PyDoc_STR("Internal method to get a NumPy array for the tensor.")},
    {"_datatype_enum", (PyCFunction)EagerTensor_datatype_enum, METH_NOARGS,
     PyDoc_STR("The DType of the tensor as an enum.")},
    {"_shape_tuple", (PyCFunction)EagerTensor_shape_tuple, METH_NOARGS,
     PyDoc_STR("The shape of the tensor as a python tuple.")},
    {"_rank", (PyCFunction)EagerTensor_rank, METH_NOARGS,
     PyDoc_STR("The rank of the tensor.")},
    {"_copy_to_device", (PyCFunction)EagerTensor_copy_to_device,
     METH_VARARGS | METH_KEYWORDS,
     PyDoc_STR("Copies the tensor to the desired device.")},
    {"_num_elements", (PyCFunction)EagerTensor_num_elements, METH_NOARGS,
     PyDoc_STR("Number of elements in the tensor.")},
    {"_prefer_custom_summarizer",
     (PyCFunction)EagerTensor_prefer_custom_summarizer, METH_NOARGS,
     PyDoc_STR("Indicates whether _numpy_internal loses information.")},
    {"_summarize_value", (PyCFunction)EagerTensor_summarize_value, METH_NOARGS,
     PyDoc_STR("A string which summarizes the value of this tensor.")},
    {nullptr, nullptr},
};

static int EagerTensor_getbuffer(EagerTensor* self, Py_buffer* view,
                                 int flags) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_33(mht_33_v, 1109, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_getbuffer");

  if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
    PyErr_SetString(PyExc_BufferError, "EagerTensor is not writable.");
    return -1;
  }

  // TensorHandleToNumpy is zero-copy for everything but DT_RESOURCE and
  // DT_STRING so the following is only slightly slower than a NumPy-free
  // implementation.
  auto py_array = tensorflow::make_safe(
      TFE_TensorHandleToNumpy(self->handle, &self->status));
  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&self->status,
                                                  PyExc_BufferError)) {
    // Cleanup self->status before returning.
    self->status.status = tensorflow::Status::OK();
    return -1;
  }
  if (PyObject_GetBuffer(py_array.get(), view, flags) < 0) {
    return -1;
  }
  view->readonly = 1;
  return 0;
}

static PyBufferProcs EagerTensor_as_buffer = {
#if PY_MAJOR_VERSION < 3
    nullptr, nullptr, nullptr, nullptr,
#endif
    (getbufferproc)EagerTensor_getbuffer,
    // Never called because getbufferproc delegates to NumPy.
    (releasebufferproc) nullptr};

// Note that here we are trying to dynamically create a new class as a subclass
// of a "HEAPTYPE" class that is itself created in python code and passed in at
// runtime. This is fairly atypical and undocumented.
//
// We use the following strategy for this. Unfortunately, we have to use
// different approaches for python2.x vs python3.x
// For python2.x, we create the class as a static type and set its tp_base to
// the passed in type. Unfortunately setting tp_flags to include
// Py_TPFLAGS_HEAPTYPE does not work by itself since it needs some more
// initialization of the underlying PyHeapTypeObject and not doing that leads to
// some random crashes especially during garbage collection.
// python3.x explicitly disables a static subclass of a HEAPTYPE base class.
// However it provides a new function, PyType_FromSpecWithBases, to create
// types dynamically.

// Type object for EagerTensor. This is set by TFE_Py_InitEagerTensor.
PyTypeObject* EagerTensorType = nullptr;

#if PY_MAJOR_VERSION >= 3
static PyType_Slot EagerTensor_Type_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(EagerTensor_dealloc)},
    {Py_tp_methods, reinterpret_cast<void*>(EagerTensor_methods)},
    {Py_tp_getset, reinterpret_cast<void*>(EagerTensor_getsetters)},
    {Py_tp_init, reinterpret_cast<void*>(EagerTensor_init)},
    {0, nullptr},
};
#else

#define EAGER_TENSOR_TPFLAGS (Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_NEWBUFFER)

// TODO(agarwal): support active_trace.
static PyTypeObject _EagerTensorType = {
    // clang-format off
    PyVarObject_HEAD_INIT(nullptr, 0)
    // clang-format on
    "EagerTensor",                      /* tp_name */
    sizeof(EagerTensor),                /* tp_basicsize */
    0,                                  /* tp_itemsize */
    (destructor)EagerTensor_dealloc,    /* tp_dealloc */
#if PY_VERSION_HEX < 0x03080000
    nullptr,                            /* tp_print */
#else
    0, /* tp_vectorcall_offset */
#endif
    nullptr,                            /* tp_getattr */
    nullptr,                            /* tp_setattr */
    nullptr,                            /* tp_compare */
    nullptr,                            /* tp_repr */
    nullptr,                            /* tp_as_number */
    nullptr,                            /* tp_as_sequence */
    nullptr,                            /* tp_as_mapping */
    nullptr,                            /* tp_hash */
    nullptr,                            /* tp_call */
    nullptr,                            /* tp_str */
    nullptr,                            /* tp_getattro */
    nullptr,                            /* tp_setattro */
    &EagerTensor_as_buffer,             /* tp_as_buffer */
    EAGER_TENSOR_TPFLAGS,               /* tp_flags */
    nullptr,                            /* tp_doc */
    nullptr,                            /* tp_traverse */
    nullptr,                            /* tp_clear */
    nullptr,                            /* tp_richcompare */
    offsetof(EagerTensor, weakreflist), /* tp_weaklistoffset */
    nullptr,                            /* tp_iter */
    nullptr,                            /* tp_iternext */
    EagerTensor_methods,                /* tp_methods */
    EagerTensor_members,                /* tp_members */
    EagerTensor_getsetters,             /* tp_getset */
    nullptr,                            /* tp_base */
    nullptr,                            /* tp_dict */
    nullptr,                            /* tp_descr_get */
    nullptr,                            /* tp_descr_set */
    offsetof(EagerTensor, dict),        /* tp_dictoffset */
    (initproc)EagerTensor_init,         /* tp_init */
    nullptr,                            /* tp_alloc */
    nullptr,                            /* tp_new */
};

#endif

}  // extern "C"

bool EagerTensor_CheckExact(const PyObject* o) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_34(mht_34_v, 1226, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_CheckExact");

  return Py_TYPE(o) == EagerTensorType;
}

TFE_TensorHandle* EagerTensor_Handle(const PyObject* o) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_35(mht_35_v, 1233, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensor_Handle");

  return reinterpret_cast<const EagerTensor*>(o)->handle;
}

PyObject* EagerTensorFromHandle(TFE_TensorHandle* handle,
                                const bool is_packed) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_36(mht_36_v, 1241, "", "./tensorflow/python/eager/pywrap_tensor.cc", "EagerTensorFromHandle");

  if (handle == nullptr) {
    return nullptr;
  }
  EagerTensor* t = reinterpret_cast<EagerTensor*>(
      EagerTensorType->tp_new(EagerTensorType, EmptyTuple(), EmptyDict()));
  if (t != nullptr) {
    t->id = get_uid();
    t->is_packed = is_packed;
    Py_INCREF(Py_None);
    t->handle_data = Py_None;
    Py_INCREF(Py_None);
    t->tensor_shape = Py_None;
    t->handle = handle;
    t->status.status = tensorflow::Status::OK();
    t->weakreflist = nullptr;
    PyObject* py_context = GetPyEagerContext();
    if (py_context == nullptr) {
      LOG(ERROR) << "Cannot create an eager tensor before eager context has "
                    "been set or after it has been deleted";
      return nullptr;
    }
    t->context = py_context;

    if (!MaybeInvokeCreatedOnEagerTensorProfiler(t)) {
      return nullptr;
    }
  }
  return reinterpret_cast<PyObject*>(t);
}

int64_t PyEagerTensor_ID(const PyObject* tensor) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_37(mht_37_v, 1275, "", "./tensorflow/python/eager/pywrap_tensor.cc", "PyEagerTensor_ID");

  DCHECK(EagerTensor_CheckExact(tensor));
  return reinterpret_cast<const EagerTensor*>(tensor)->id;
}

tensorflow::DataType PyEagerTensor_Dtype(const PyObject* tensor) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_38(mht_38_v, 1283, "", "./tensorflow/python/eager/pywrap_tensor.cc", "PyEagerTensor_Dtype");

  DCHECK(EagerTensor_CheckExact(tensor));
  return static_cast<tensorflow::DataType>(TFE_TensorHandleDataType(
      reinterpret_cast<const EagerTensor*>(tensor)->handle));
}

int64_t PyEagerTensor_NumElements(PyObject* tensor) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_39(mht_39_v, 1292, "", "./tensorflow/python/eager/pywrap_tensor.cc", "PyEagerTensor_NumElements");

  DCHECK(EagerTensor_CheckExact(tensor));
  EagerTensor* as_c_eager_tensor = reinterpret_cast<EagerTensor*>(tensor);
  int64_t result = TFE_TensorHandleNumElements(as_c_eager_tensor->handle,
                                               &as_c_eager_tensor->status);

  if (tensorflow::MaybeRaiseExceptionFromTFStatus(&as_c_eager_tensor->status,
                                                  PyExc_ValueError)) {
    // Cleanup status before returning.
    as_c_eager_tensor->status.status = tensorflow::Status::OK();
    return -1;
  }

  return result;
}

PyObject* TFE_Py_InitEagerTensor(PyObject* base_class) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_40(mht_40_v, 1311, "", "./tensorflow/python/eager/pywrap_tensor.cc", "TFE_Py_InitEagerTensor");

  if (!PyType_Check(base_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `base_class` passed to ",
            "TFE_InitEagerTensor. Got ", Py_TYPE(base_class)->tp_name)
            .c_str());
    return nullptr;
  }
  // Note that we allocated kMaxEagerTensorParentSize bytes of unused space in
  // EagerTensor to allow for the space usage of the base class.
  PyTypeObject* base_class_type = reinterpret_cast<PyTypeObject*>(base_class);
  if (base_class_type->tp_basicsize > kMaxEagerTensorParentSize) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Unable to create subclass EagerTensor from base class ",
            Py_TYPE(base_class)->tp_name,
            ". Need its size to be <= ", kMaxEagerTensorParentSize)
            .c_str());
    return nullptr;
  }
  if (base_class_type->tp_itemsize != 0) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Unable to create subclass EagerTensor from base class ",
            Py_TYPE(base_class)->tp_name,
            " which supports variable length instances.")
            .c_str());
    return nullptr;
  }
  Py_INCREF(base_class);
#if PY_MAJOR_VERSION >= 3
  PyObject* bases = PyTuple_New(1);
  PyTuple_SET_ITEM(bases, 0, base_class);

  tensorflow::Safe_PyObjectPtr base_class_module(
      PyObject_GetAttrString(base_class, "__module__"));
  const char* module = nullptr;
  if (PyErr_Occurred()) {
    PyErr_Clear();
    module = "__builtin__";
  } else {
    module = PyBytes_AsString(base_class_module.get());
    if (module == nullptr) {
      PyErr_Clear();
      module = PyUnicode_AsUTF8(base_class_module.get());
      if (module == nullptr) {
        PyErr_Clear();
        module = "__builtin__";
      }
    }
  }

  // NOTE: The c_str from this string needs to outlast the function, hence is
  // static.
  static tensorflow::string fully_qualified_name =
      tensorflow::strings::StrCat(module, ".EagerTensor");

  static PyType_Spec EagerTensor_Type_spec = {
      fully_qualified_name.c_str(), sizeof(EagerTensor), 0,
      Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE, EagerTensor_Type_slots};

  EagerTensorType = reinterpret_cast<PyTypeObject*>(
      PyType_FromSpecWithBases(&EagerTensor_Type_spec, bases));
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (EagerTensorType == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Error while creating EagerTensorType");
    return nullptr;
  }
  EagerTensorType->tp_dictoffset = offsetof(EagerTensor, dict);
  EagerTensorType->tp_as_buffer = &EagerTensor_as_buffer;
#else
  _EagerTensorType.tp_base = base_class_type;

  if (PyType_Ready(&_EagerTensorType) < 0) {
    if (PyErr_Occurred()) return nullptr;
    PyErr_SetString(PyExc_RuntimeError,
                    "Error while creating EagerTensor type.");
    return nullptr;
  }
  EagerTensorType = &_EagerTensorType;
  Py_INCREF(EagerTensorType);
#endif
  return reinterpret_cast<PyObject*>(EagerTensorType);
}

PyObject* TFE_Py_SetEagerTensorProfiler(PyObject* profiler) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_41(mht_41_v, 1405, "", "./tensorflow/python/eager/pywrap_tensor.cc", "TFE_Py_SetEagerTensorProfiler");

  Py_XDECREF(eager_tensor_profiler);

  if (profiler == Py_None) {
    eager_tensor_profiler = nullptr;
  } else {
    eager_tensor_profiler = profiler;
    Py_INCREF(eager_tensor_profiler);
  }
  Py_RETURN_NONE;
}

PyObject* TFE_Py_TensorShapeSlice(PyObject* tensors, int slice_dim) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_42(mht_42_v, 1420, "", "./tensorflow/python/eager/pywrap_tensor.cc", "TFE_Py_TensorShapeSlice");

  if (!PyList_Check(tensors) && !PyTuple_Check(tensors)) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "tensors argument must be a list or a tuple. Got \"",
                        Py_TYPE(tensors)->tp_name, "\"")
                        .c_str());
    return nullptr;
  }
  if (slice_dim < 0) {
    PyErr_SetString(
        PyExc_ValueError,
        tensorflow::strings::StrCat("Slice dimension must be non-negative. "
                                    "Got ",
                                    slice_dim)
            .c_str());
    return nullptr;
  }

  PyObject* py_context = GetPyEagerContext();
  if (py_context == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, tensorflow::strings::StrCat(
                                            "Cannot create EagerTensor when "
                                            "EagerContext is not valid")
                                            .c_str());
    return nullptr;
  }

  TFE_Context* ctx = GetContextHandle(py_context);

  Py_ssize_t num_tensors = PySequence_Fast_GET_SIZE(tensors);
  PyObject** tensors_array = PySequence_Fast_ITEMS(tensors);
  int64_t num_tensors_int = static_cast<int64_t>(num_tensors);

  auto status = tensorflow::make_safe(TF_NewStatus());

  // Create an empty tensor.
  auto* tensor = tensorflow::unwrap(ctx)->CreateTensor(
      tensorflow::DT_INT32, /*dim_sizes=*/{num_tensors_int});

  if (num_tensors_int > 0) {
    int32_t* data = reinterpret_cast<int32_t*>(tensor->Data());

    // Fill the tensor with dims.
    for (Py_ssize_t i = 0; i < num_tensors; ++i) {
      PyObject* tensor_obj = tensors_array[i];
      if (!EagerTensor_CheckExact(tensor_obj)) {
        PyErr_SetString(
            PyExc_TypeError,
            tensorflow::strings::StrCat("Expected a list of EagerTensors but "
                                        "element ",
                                        i, " has type \"",
                                        Py_TYPE(tensor_obj)->tp_name, "\"")
                .c_str());
        return nullptr;
      }

      EagerTensor* t = reinterpret_cast<EagerTensor*>(tensor_obj);
      TFE_TensorHandle* handle = t->handle;
      int num_dims = TFE_TensorHandleNumDims(handle, status.get());
      if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(),
                                                      PyExc_ValueError)) {
        return nullptr;
      }
      if (slice_dim >= num_dims) {
        PyErr_SetString(
            PyExc_IndexError,
            tensorflow::strings::StrCat("Slice dimension (", slice_dim,
                                        ") must be smaller than rank of all "
                                        "tensors, but tensor at index ",
                                        i, " has rank ", num_dims)
                .c_str());
        return nullptr;
      }
      int64_t dim = TFE_TensorHandleDim(handle, slice_dim, status.get());
      if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(),
                                                      PyExc_ValueError)) {
        return nullptr;
      }
      data[i] = dim;
    }
  }

  TFE_TensorHandle* handle =
      tensorflow::wrap(tensorflow::unwrap(ctx)->CreateLocalHandle(tensor));

  if (!status->status.ok()) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat("Failed to construct new tensor handle: ",
                                    TF_Message(status.get()))
            .c_str());
    return nullptr;
  }

  return EagerTensorFromHandle(handle);
}

PyObject* TFE_Py_TensorShapeOnDevice(PyObject* tensor) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensorDTcc mht_43(mht_43_v, 1521, "", "./tensorflow/python/eager/pywrap_tensor.cc", "TFE_Py_TensorShapeOnDevice");

  if (!EagerTensor_CheckExact(tensor)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat("Expected an EagerTensors but got type \"",
                                    Py_TYPE(tensor)->tp_name, "\"")
            .c_str());
    return nullptr;
  }
  TFE_TensorHandle* handle = EagerTensor_Handle(tensor);

  auto status = tensorflow::make_safe(TF_NewStatus());
  TFE_TensorDebugInfo* debug_info =
      TFE_TensorHandleTensorDebugInfo(handle, status.get());
  if (!status->status.ok()) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat("Error retrieving tensor's device shape: ",
                                    TF_Message(status.get()))
            .c_str());
    return nullptr;
  }

  int rank = TFE_TensorDebugInfoOnDeviceNumDims(debug_info);
  PyObject* shape = PyTuple_New(rank);
  for (int i = 0; i < rank; ++i) {
    int64_t dim_size = TFE_TensorDebugInfoOnDeviceDim(debug_info, i);
    PyTuple_SET_ITEM(shape, i, PyLong_FromLongLong(dim_size));
  }
  TFE_DeleteTensorDebugInfo(debug_info);

  return shape;
}
