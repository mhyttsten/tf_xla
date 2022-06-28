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
class MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.h"

#include <stdarg.h>

#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/shared_library.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/util.h"

#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_TENSOR_BOUNDS_CHECK(i)                                    \
  if (i >= interpreter_->tensors_size() || i < 0) {                         \
    PyErr_Format(PyExc_ValueError,                                          \
                 "Invalid tensor index %d exceeds max tensor index %lu", i, \
                 interpreter_->tensors_size());                             \
    return nullptr;                                                         \
  }

#define TFLITE_PY_SUBGRAPH_TENSOR_BOUNDS_CHECK(i, subgraph_index)             \
  if (i >= interpreter_->subgraph(subgraph_index)->tensors_size() || i < 0) { \
    PyErr_Format(PyExc_ValueError,                                            \
                 "Invalid tensor index %d exceeds max tensor index %lu", i,   \
                 interpreter_->subgraph(subgraph_index)->tensors_size());     \
    return nullptr;                                                           \
  }

#define TFLITE_PY_SUBGRAPH_BOUNDS_CHECK(i)                                   \
  if (i >= interpreter_->subgraphs_size() || i < 0) {                        \
    PyErr_Format(PyExc_ValueError,                                           \
                 "Invalid subgraph index %d exceeds max subgraph index %lu", \
                 i, interpreter_->subgraphs_size());                         \
    return nullptr;                                                          \
  }

#define TFLITE_PY_NODES_BOUNDS_CHECK(i)                   \
  if (i >= interpreter_->nodes_size() || i < 0) {         \
    PyErr_Format(PyExc_ValueError, "Invalid node index"); \
    return nullptr;                                       \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace interpreter_wrapper {

namespace {

using python_utils::PyDecrefDeleter;

std::unique_ptr<Interpreter> CreateInterpreter(
    const InterpreterWrapper::Model* model,
    const tflite::MutableOpResolver& resolver, bool preserve_all_tensors) {
  if (!model) {
    return nullptr;
  }

  ::tflite::python::ImportNumpy();

  std::unique_ptr<Interpreter> interpreter;
  InterpreterOptions options;
  options.SetPreserveAllTensors(preserve_all_tensors);
  InterpreterBuilder builder(*model, resolver, &options);
  if (builder(&interpreter) != kTfLiteOk) {
    return nullptr;
  }
  return interpreter;
}

PyObject* PyArrayFromFloatVector(const float* data, npy_intp size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_0(mht_0_v, 279, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "PyArrayFromFloatVector");

  void* pydata = malloc(size * sizeof(float));
  memcpy(pydata, data, size * sizeof(float));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_FLOAT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

PyObject* PyArrayFromIntVector(const int* data, npy_intp size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_1(mht_1_v, 290, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "PyArrayFromIntVector");

  void* pydata = malloc(size * sizeof(int));
  memcpy(pydata, data, size * sizeof(int));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_INT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

PyObject* PyTupleFromQuantizationParam(const TfLiteQuantizationParams& param) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_2(mht_2_v, 301, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "PyTupleFromQuantizationParam");

  PyObject* result = PyTuple_New(2);
  PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(param.scale));
  PyTuple_SET_ITEM(result, 1, PyLong_FromLong(param.zero_point));
  return result;
}

PyObject* PyDictFromSparsityParam(const TfLiteSparsity& param) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_3(mht_3_v, 311, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "PyDictFromSparsityParam");

  PyObject* result = PyDict_New();
  PyDict_SetItemString(result, "traversal_order",
                       PyArrayFromIntVector(param.traversal_order->data,
                                            param.traversal_order->size));
  PyDict_SetItemString(
      result, "block_map",
      PyArrayFromIntVector(param.block_map->data, param.block_map->size));
  PyObject* dim_metadata = PyList_New(param.dim_metadata_size);
  for (int i = 0; i < param.dim_metadata_size; i++) {
    PyObject* dim_metadata_i = PyDict_New();
    if (param.dim_metadata[i].format == kTfLiteDimDense) {
      PyDict_SetItemString(dim_metadata_i, "format", PyLong_FromSize_t(0));
      PyDict_SetItemString(dim_metadata_i, "dense_size",
                           PyLong_FromSize_t(param.dim_metadata[i].dense_size));
    } else {
      PyDict_SetItemString(dim_metadata_i, "format", PyLong_FromSize_t(1));
      const auto* array_segments = param.dim_metadata[i].array_segments;
      const auto* array_indices = param.dim_metadata[i].array_indices;
      PyDict_SetItemString(
          dim_metadata_i, "array_segments",
          PyArrayFromIntVector(array_segments->data, array_segments->size));
      PyDict_SetItemString(
          dim_metadata_i, "array_indices",
          PyArrayFromIntVector(array_indices->data, array_indices->size));
    }
    PyList_SetItem(dim_metadata, i, dim_metadata_i);
  }
  PyDict_SetItemString(result, "dim_metadata", dim_metadata);
  return result;
}

bool RegisterCustomOpByName(const char* registerer_name,
                            tflite::MutableOpResolver* resolver,
                            std::string* error_msg) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("registerer_name: \"" + (registerer_name == nullptr ? std::string("nullptr") : std::string((char*)registerer_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_4(mht_4_v, 349, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "RegisterCustomOpByName");

  // Registerer functions take a pointer to a BuiltinOpResolver as an input
  // parameter and return void.
  // TODO(b/137576229): We should implement this functionality in a more
  // principled way.
  typedef void (*RegistererFunctionType)(tflite::MutableOpResolver*);

  // Look for the Registerer function by name.
  RegistererFunctionType registerer = reinterpret_cast<RegistererFunctionType>(
      SharedLibrary::GetSymbol(registerer_name));

  // Fail in an informative way if the function was not found.
  if (registerer == nullptr) {
    *error_msg =
        absl::StrFormat("Looking up symbol '%s' failed with error '%s'.",
                        registerer_name, SharedLibrary::GetError());
    return false;
  }

  // Call the registerer with the resolver.
  registerer(resolver);
  return true;
}

}  // namespace

static constexpr int kBuiltinOpResolver = 1;
static constexpr int kBuiltinRefOpResolver = 2;
static constexpr int kBuiltinOpResolverWithoutDefaultDelegates = 3;

InterpreterWrapper* InterpreterWrapper::CreateInterpreterWrapper(
    std::unique_ptr<InterpreterWrapper::Model> model, int op_resolver_id,
    std::unique_ptr<PythonErrorReporter> error_reporter,
    const std::vector<std::string>& registerers_by_name,
    const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
    std::string* error_msg, bool preserve_all_tensors) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_5(mht_5_v, 387, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::CreateInterpreterWrapper");

  if (!model) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  std::unique_ptr<tflite::MutableOpResolver> resolver;
  switch (op_resolver_id) {
    case kBuiltinOpResolver:
      resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
      break;
    case kBuiltinRefOpResolver:
      resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinRefOpResolver>();
      break;
    case kBuiltinOpResolverWithoutDefaultDelegates:
      resolver = absl::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
      break;
    default:
      // This should not never happen because the eventual caller in
      // interpreter.py should have passed a valid id here.
      TFLITE_DCHECK(false);
      return nullptr;
  }

  for (const auto& registerer : registerers_by_name) {
    if (!RegisterCustomOpByName(registerer.c_str(), resolver.get(), error_msg))
      return nullptr;
  }
  for (const auto& registerer : registerers_by_func) {
    registerer(reinterpret_cast<uintptr_t>(resolver.get()));
  }
  auto interpreter =
      CreateInterpreter(model.get(), *resolver, preserve_all_tensors);
  if (!interpreter) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  InterpreterWrapper* wrapper =
      new InterpreterWrapper(std::move(model), std::move(error_reporter),
                             std::move(resolver), std::move(interpreter));
  return wrapper;
}

InterpreterWrapper::InterpreterWrapper(
    std::unique_ptr<InterpreterWrapper::Model> model,
    std::unique_ptr<PythonErrorReporter> error_reporter,
    std::unique_ptr<tflite::MutableOpResolver> resolver,
    std::unique_ptr<Interpreter> interpreter)
    : model_(std::move(model)),
      error_reporter_(std::move(error_reporter)),
      resolver_(std::move(resolver)),
      interpreter_(std::move(interpreter)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_6(mht_6_v, 444, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::InterpreterWrapper");
}

InterpreterWrapper::~InterpreterWrapper() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_7(mht_7_v, 449, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::~InterpreterWrapper");
}

// LINT.IfChange
static constexpr int kUndeterminedSubgraphIndex = -1;
// LINT.ThenChange(//tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper_pybind11.cc)
PyObject* InterpreterWrapper::AllocateTensors(int subgraph_index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_8(mht_8_v, 457, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::AllocateTensors");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (subgraph_index == kUndeterminedSubgraphIndex) {
    TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  } else {
    TFLITE_PY_SUBGRAPH_BOUNDS_CHECK(subgraph_index);
    TFLITE_PY_CHECK(interpreter_->subgraph(subgraph_index)->AllocateTensors());
  }
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::Invoke(int subgraph_index) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_9(mht_9_v, 471, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::Invoke");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_SUBGRAPH_BOUNDS_CHECK(subgraph_index);

  // Release the GIL so that we can run multiple interpreters in parallel
  TfLiteStatus status_code = kTfLiteOk;
  Py_BEGIN_ALLOW_THREADS;  // To return can happen between this and end!
  tflite::Subgraph* subgraph = interpreter_->subgraph(subgraph_index);
  status_code = subgraph->Invoke();

  if (!interpreter_->allow_buffer_handle_output_) {
    for (int tensor_index : subgraph->outputs()) {
      subgraph->EnsureTensorDataIsReadable(tensor_index);
    }
  }
  Py_END_ALLOW_THREADS;

  TFLITE_PY_CHECK(
      status_code);  // don't move this into the Py_BEGIN/Py_End block

  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::InputIndices() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_10(mht_10_v, 497, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::InputIndices");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  PyObject* np_array = PyArrayFromIntVector(interpreter_->inputs().data(),
                                            interpreter_->inputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::OutputIndices() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_11(mht_11_v, 508, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::OutputIndices");

  PyObject* np_array = PyArrayFromIntVector(interpreter_->outputs().data(),
                                            interpreter_->outputs().size());

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::ResizeInputTensorImpl(int i, PyObject* value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_12(mht_12_v, 518, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::ResizeInputTensorImpl");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert numpy value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  if (PyArray_NDIM(array) != 1) {
    PyErr_Format(PyExc_ValueError, "Shape should be 1D instead of %d.",
                 PyArray_NDIM(array));
    return nullptr;
  }

  if (PyArray_TYPE(array) != NPY_INT32) {
    PyErr_Format(PyExc_ValueError, "Shape must be type int32 (was %d).",
                 PyArray_TYPE(array));
    return nullptr;
  }

  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(array),
                      NPY_ARRAY_OWNDATA);
  return PyArray_Return(reinterpret_cast<PyArrayObject*>(array));
}

PyObject* InterpreterWrapper::ResizeInputTensor(int i, PyObject* value,
                                                bool strict,
                                                int subgraph_index) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_13(mht_13_v, 553, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::ResizeInputTensor");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_SUBGRAPH_BOUNDS_CHECK(subgraph_index);

  PyArrayObject* array =
      reinterpret_cast<PyArrayObject*>(ResizeInputTensorImpl(i, value));
  if (array == nullptr) {
    return nullptr;
  }

  std::vector<int> dims(PyArray_SHAPE(array)[0]);
  memcpy(dims.data(), PyArray_BYTES(array), dims.size() * sizeof(int));

  if (strict) {
    TFLITE_PY_CHECK(interpreter_->subgraph(subgraph_index)
                        ->ResizeInputTensorStrict(i, dims));
  } else {
    TFLITE_PY_CHECK(
        interpreter_->subgraph(subgraph_index)->ResizeInputTensor(i, dims));
  }
  Py_RETURN_NONE;
}

int InterpreterWrapper::NumTensors() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_14(mht_14_v, 579, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::NumTensors");

  if (!interpreter_) {
    return 0;
  }
  return interpreter_->tensors_size();
}

std::string InterpreterWrapper::TensorName(int i) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_15(mht_15_v, 589, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorName");

  if (!interpreter_ || i >= interpreter_->tensors_size() || i < 0) {
    return "";
  }

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return tensor->name ? tensor->name : "";
}

PyObject* InterpreterWrapper::TensorType(int i) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_16(mht_16_v, 601, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorType");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->type == kTfLiteNoType) {
    PyErr_Format(PyExc_ValueError, "Tensor with no type found.");
    return nullptr;
  }

  int code = python_utils::TfLiteTypeToPyArrayType(tensor->type);
  if (code == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid tflite type code %d", code);
    return nullptr;
  }
  return PyArray_TypeObjectFromType(code);
}

PyObject* InterpreterWrapper::TensorSize(int i) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_17(mht_17_v, 622, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorSize");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->dims == nullptr) {
    PyErr_Format(PyExc_ValueError, "Tensor with no shape found.");
    return nullptr;
  }
  PyObject* np_array =
      PyArrayFromIntVector(tensor->dims->data, tensor->dims->size);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::TensorSizeSignature(int i) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_18(mht_18_v, 640, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorSizeSignature");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);

  const TfLiteTensor* tensor = interpreter_->tensor(i);
  const int32_t* size_signature_data = nullptr;
  int32_t size_signature_size = 0;
  if (tensor->dims_signature != nullptr && tensor->dims_signature->size != 0) {
    size_signature_data = tensor->dims_signature->data;
    size_signature_size = tensor->dims_signature->size;
  } else {
    size_signature_data = tensor->dims->data;
    size_signature_size = tensor->dims->size;
  }
  PyObject* np_array =
      PyArrayFromIntVector(size_signature_data, size_signature_size);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::TensorSparsityParameters(int i) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_19(mht_19_v, 663, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorSparsityParameters");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);
  const TfLiteTensor* tensor = interpreter_->tensor(i);
  if (tensor->sparsity == nullptr) {
    return PyDict_New();
  }

  return PyDictFromSparsityParam(*tensor->sparsity);
}

PyObject* InterpreterWrapper::TensorQuantization(int i) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_20(mht_20_v, 677, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorQuantization");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);
  const TfLiteTensor* tensor = interpreter_->tensor(i);
  return PyTupleFromQuantizationParam(tensor->params);
}

PyObject* InterpreterWrapper::TensorQuantizationParameters(int i) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_21(mht_21_v, 687, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::TensorQuantizationParameters");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_TENSOR_BOUNDS_CHECK(i);
  const TfLiteTensor* tensor = interpreter_->tensor(i);
  const TfLiteQuantization quantization = tensor->quantization;
  float* scales_data = nullptr;
  int32_t* zero_points_data = nullptr;
  int32_t scales_size = 0;
  int32_t zero_points_size = 0;
  int32_t quantized_dimension = 0;
  if (quantization.type == kTfLiteAffineQuantization) {
    const TfLiteAffineQuantization* q_params =
        reinterpret_cast<const TfLiteAffineQuantization*>(quantization.params);
    if (q_params->scale) {
      scales_data = q_params->scale->data;
      scales_size = q_params->scale->size;
    }
    if (q_params->zero_point) {
      zero_points_data = q_params->zero_point->data;
      zero_points_size = q_params->zero_point->size;
    }
    quantized_dimension = q_params->quantized_dimension;
  }
  PyObject* scales_array = PyArrayFromFloatVector(scales_data, scales_size);
  PyObject* zero_points_array =
      PyArrayFromIntVector(zero_points_data, zero_points_size);

  PyObject* result = PyTuple_New(3);
  PyTuple_SET_ITEM(result, 0, scales_array);
  PyTuple_SET_ITEM(result, 1, zero_points_array);
  PyTuple_SET_ITEM(result, 2, PyLong_FromLong(quantized_dimension));
  return result;
}

PyObject* InterpreterWrapper::SetTensor(int i, PyObject* value,
                                        int subgraph_index) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_22(mht_22_v, 725, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::SetTensor");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_SUBGRAPH_BOUNDS_CHECK(subgraph_index);
  TFLITE_PY_SUBGRAPH_TENSOR_BOUNDS_CHECK(i, subgraph_index);

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  TfLiteTensor* tensor = interpreter_->subgraph(subgraph_index)->tensor(i);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor:"
                 " Got value of type %s"
                 " but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), i, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension mismatch."
                 " Got %d"
                 " but expected %d for input %d.",
                 PyArray_NDIM(array), tensor->dims->size, i);
    return nullptr;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Dimension mismatch."
                   " Got %ld"
                   " but expected %d for dimension %d of input %d.",
                   PyArray_SHAPE(array)[j], tensor->dims->data[j], j, i);
      return nullptr;
    }
  }

  if (tensor->type != kTfLiteString) {
    // Only allow empty tensors.
    if (tensor->data.raw == nullptr && tensor->bytes) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor:"
                   " Tensor is unallocated. Try calling allocate_tensors()"
                   " first");
      return nullptr;
    }

    size_t size = PyArray_NBYTES(array);
    if (size != tensor->bytes) {
      PyErr_Format(PyExc_ValueError,
                   "numpy array had %zu bytes but expected %zu bytes.", size,
                   tensor->bytes);
      return nullptr;
    }
    memcpy(tensor->data.raw, PyArray_DATA(array), size);
  } else {
    DynamicBuffer dynamic_buffer;
    if (!python_utils::FillStringBufferWithPyArray(value, &dynamic_buffer)) {
      return nullptr;
    }
    dynamic_buffer.WriteToTensor(tensor, nullptr);
  }
  Py_RETURN_NONE;
}

int InterpreterWrapper::NumNodes() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_23(mht_23_v, 802, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::NumNodes");

  if (!interpreter_) {
    return 0;
  }
  return interpreter_->nodes_size();
}

PyObject* InterpreterWrapper::NodeInputs(int i) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_24(mht_24_v, 812, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::NodeInputs");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_NODES_BOUNDS_CHECK(i);

  const TfLiteNode* node = &(interpreter_->node_and_registration(i)->first);
  PyObject* inputs =
      PyArrayFromIntVector(node->inputs->data, node->inputs->size);
  return inputs;
}

PyObject* InterpreterWrapper::NodeOutputs(int i) const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_25(mht_25_v, 825, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::NodeOutputs");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_NODES_BOUNDS_CHECK(i);

  const TfLiteNode* node = &(interpreter_->node_and_registration(i)->first);
  PyObject* outputs =
      PyArrayFromIntVector(node->outputs->data, node->outputs->size);
  return outputs;
}

std::string InterpreterWrapper::NodeName(int i) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_26(mht_26_v, 838, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::NodeName");

  if (!interpreter_ || i >= interpreter_->nodes_size() || i < 0) {
    return "";
  }
  // Get op name from registration
  const TfLiteRegistration* node_registration =
      &(interpreter_->node_and_registration(i)->second);
  int32_t op_code = node_registration->builtin_code;
  std::string op_name;
  if (op_code == tflite::BuiltinOperator_CUSTOM) {
    const char* custom_name = node_registration->custom_name;
    op_name = custom_name ? custom_name : "UnknownCustomOp";
  } else {
    op_name = tflite::EnumNamesBuiltinOperator()[op_code];
  }
  std::string op_name_str(op_name);
  return op_name_str;
}

namespace {

// Checks to see if a tensor access can succeed (returns nullptr on error).
// Otherwise returns Py_None.
PyObject* CheckGetTensorArgs(Interpreter* interpreter_, int tensor_index,
                             TfLiteTensor** tensor, int* type_num,
                             int subgraph_index) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_27(mht_27_v, 866, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "CheckGetTensorArgs");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_SUBGRAPH_BOUNDS_CHECK(subgraph_index);
  TFLITE_PY_SUBGRAPH_TENSOR_BOUNDS_CHECK(tensor_index, subgraph_index);

  *tensor = interpreter_->subgraph(subgraph_index)->tensor(tensor_index);
  // Invalid size only when bytes are 0 but pointer is allocated.
  if ((*tensor)->bytes == 0 && (*tensor)->data.raw) {
    PyErr_SetString(PyExc_ValueError, "Invalid tensor size.");
    return nullptr;
  }

  *type_num = python_utils::TfLiteTypeToPyArrayType((*tensor)->type);
  if (*type_num == -1) {
    PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
    return nullptr;
  }

  // Tensor data can't be null if size is > 0. 0 bytes is valid if tensor
  // is empty.
  if (!(*tensor)->data.raw && (*tensor)->bytes) {
    PyErr_SetString(PyExc_ValueError,
                    "Tensor data is null."
                    " Run allocate_tensors() first");
    return nullptr;
  }

  Py_RETURN_NONE;
}

}  // namespace

PyObject* InterpreterWrapper::GetSignatureDefs() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_28(mht_28_v, 901, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::GetSignatureDefs");

  PyObject* result = PyDict_New();
  for (const auto& sig_key : interpreter_->signature_keys()) {
    PyObject* signature_def = PyDict_New();
    PyObject* inputs = PyDict_New();
    PyObject* outputs = PyDict_New();
    const auto& signature_def_inputs =
        interpreter_->signature_inputs(sig_key->c_str());
    const auto& signature_def_outputs =
        interpreter_->signature_outputs(sig_key->c_str());
    for (const auto& input : signature_def_inputs) {
      PyDict_SetItemString(inputs, input.first.c_str(),
                           PyLong_FromLong(input.second));
    }
    for (const auto& output : signature_def_outputs) {
      PyDict_SetItemString(outputs, output.first.c_str(),
                           PyLong_FromLong(output.second));
    }

    PyDict_SetItemString(signature_def, "inputs", inputs);
    PyDict_SetItemString(signature_def, "outputs", outputs);
    PyDict_SetItemString(result, sig_key->c_str(), signature_def);
  }
  return result;
}

PyObject* InterpreterWrapper::GetSubgraphIndexFromSignature(
    const char* signature_key) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("signature_key: \"" + (signature_key == nullptr ? std::string("nullptr") : std::string((char*)signature_key)) + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_29(mht_29_v, 932, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::GetSubgraphIndexFromSignature");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  int32_t subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key);

  if (subgraph_index < 0) {
    PyErr_SetString(PyExc_ValueError, "No matching signature.");
    return nullptr;
  }
  return PyLong_FromLong(static_cast<int64_t>(subgraph_index));
}

PyObject* InterpreterWrapper::GetTensor(int i, int subgraph_index) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_30(mht_30_v, 948, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::GetTensor");

  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;

  PyObject* check_result = CheckGetTensorArgs(interpreter_.get(), i, &tensor,
                                              &type_num, subgraph_index);
  if (check_result == nullptr) return check_result;
  Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  if (tensor->type != kTfLiteString && tensor->type != kTfLiteResource &&
      tensor->type != kTfLiteVariant) {
    // Make a buffer copy but we must tell Numpy It owns that data or else
    // it will leak.
    void* data = malloc(tensor->bytes);
    if (!data) {
      PyErr_SetString(PyExc_ValueError, "Malloc to copy tensor failed.");
      return nullptr;
    }
    memcpy(data, tensor->data.raw, tensor->bytes);
    PyObject* np_array;
    if (tensor->sparsity == nullptr) {
      np_array =
          PyArray_SimpleNewFromData(dims.size(), dims.data(), type_num, data);
    } else {
      std::vector<npy_intp> sparse_buffer_dims(1);
      size_t size_of_type;
      if (GetSizeOfType(nullptr, tensor->type, &size_of_type) != kTfLiteOk) {
        PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
        free(data);
        return nullptr;
      }
      sparse_buffer_dims[0] = tensor->bytes / size_of_type;
      np_array = PyArray_SimpleNewFromData(
          sparse_buffer_dims.size(), sparse_buffer_dims.data(), type_num, data);
    }
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(np_array),
                        NPY_ARRAY_OWNDATA);
    return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
  } else {
    // Create a C-order array so the data is contiguous in memory.
    const int32_t kCOrder = 0;
    PyObject* py_object =
        PyArray_EMPTY(dims.size(), dims.data(), NPY_OBJECT, kCOrder);

    if (py_object == nullptr) {
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate PyArray.");
      return nullptr;
    }

    PyArrayObject* py_array = reinterpret_cast<PyArrayObject*>(py_object);
    PyObject** data = reinterpret_cast<PyObject**>(PyArray_DATA(py_array));
    auto num_strings = GetStringCount(tensor);
    for (int j = 0; j < num_strings; ++j) {
      auto ref = GetString(tensor, j);

      PyObject* bytes = PyBytes_FromStringAndSize(ref.str, ref.len);
      if (bytes == nullptr) {
        Py_DECREF(py_object);
        PyErr_Format(PyExc_ValueError,
                     "Could not create PyBytes from string %d of input %d.", j,
                     i);
        return nullptr;
      }
      // PyArray_EMPTY produces an array full of Py_None, which we must decref.
      Py_DECREF(data[j]);
      data[j] = bytes;
    }
    return py_object;
  }
}

PyObject* InterpreterWrapper::tensor(PyObject* base_object, int tensor_index,
                                     int subgraph_index) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_31(mht_31_v, 1026, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::tensor");

  // Sanity check accessor
  TfLiteTensor* tensor = nullptr;
  int type_num = 0;

  PyObject* check_result = CheckGetTensorArgs(
      interpreter_.get(), tensor_index, &tensor, &type_num, subgraph_index);
  if (check_result == nullptr) return check_result;
  Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  PyArrayObject* np_array =
      reinterpret_cast<PyArrayObject*>(PyArray_SimpleNewFromData(
          dims.size(), dims.data(), type_num, tensor->data.raw));
  Py_INCREF(base_object);  // SetBaseObject steals, so we need to add.
  PyArray_SetBaseObject(np_array, base_object);
  return PyArray_Return(np_array);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromFile(
    const char* model_path, int op_resolver_id,
    const std::vector<std::string>& registerers_by_name,
    const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
    std::string* error_msg, bool preserve_all_tensors) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("model_path: \"" + (model_path == nullptr ? std::string("nullptr") : std::string((char*)model_path)) + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_32(mht_32_v, 1054, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::CreateWrapperCPPFromFile");

  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  std::unique_ptr<InterpreterWrapper::Model> model =
      Model::BuildFromFile(model_path, error_reporter.get());
  return CreateInterpreterWrapper(std::move(model), op_resolver_id,
                                  std::move(error_reporter),
                                  registerers_by_name, registerers_by_func,
                                  error_msg, preserve_all_tensors);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromFile(
    const char* model_path, int op_resolver_id,
    const std::vector<std::string>& registerers, std::string* error_msg,
    bool preserve_all_tensors) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("model_path: \"" + (model_path == nullptr ? std::string("nullptr") : std::string((char*)model_path)) + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_33(mht_33_v, 1071, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::CreateWrapperCPPFromFile");

  return CreateWrapperCPPFromFile(model_path, op_resolver_id, registerers,
                                  {} /*registerers_by_func*/, error_msg,
                                  preserve_all_tensors);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, int op_resolver_id,
    const std::vector<std::string>& registerers_by_name,
    const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
    std::string* error_msg, bool preserve_all_tensors) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_34(mht_34_v, 1084, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::CreateWrapperCPPFromBuffer");

  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<InterpreterWrapper::Model> model =
      Model::BuildFromBuffer(buf, length, error_reporter.get());
  return CreateInterpreterWrapper(std::move(model), op_resolver_id,
                                  std::move(error_reporter),
                                  registerers_by_name, registerers_by_func,
                                  error_msg, preserve_all_tensors);
}

InterpreterWrapper* InterpreterWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, int op_resolver_id,
    const std::vector<std::string>& registerers, std::string* error_msg,
    bool preserve_all_tensors) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_35(mht_35_v, 1106, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::CreateWrapperCPPFromBuffer");

  return CreateWrapperCPPFromBuffer(data, op_resolver_id, registerers, {},
                                    error_msg, preserve_all_tensors);
}

PyObject* InterpreterWrapper::ResetVariableTensors() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_36(mht_36_v, 1114, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::ResetVariableTensors");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::SetNumThreads(int num_threads) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_37(mht_37_v, 1123, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::SetNumThreads");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  interpreter_->SetNumThreads(num_threads);
  Py_RETURN_NONE;
}

PyObject* InterpreterWrapper::ModifyGraphWithDelegate(
    TfLiteDelegate* delegate) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSinterpreter_wrapperPSinterpreter_wrapperDTcc mht_38(mht_38_v, 1133, "", "./tensorflow/lite/python/interpreter_wrapper/interpreter_wrapper.cc", "InterpreterWrapper::ModifyGraphWithDelegate");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->ModifyGraphWithDelegate(delegate));
  Py_RETURN_NONE;
}

}  // namespace interpreter_wrapper
}  // namespace tflite
