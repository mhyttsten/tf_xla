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
class MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/python/optimize/calibration_wrapper.h"

#include <memory>
#include <sstream>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/python/interpreter_wrapper/numpy.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_error_reporter.h"
#include "tensorflow/lite/python/interpreter_wrapper/python_utils.h"
#include "tensorflow/lite/shared_library.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"
#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#define TFLITE_PY_CHECK(x)               \
  if ((x) != kTfLiteOk) {                \
    return error_reporter_->exception(); \
  }

#define TFLITE_PY_ENSURE_VALID_INTERPRETER()                               \
  if (!interpreter_) {                                                     \
    PyErr_SetString(PyExc_ValueError, "Interpreter was not initialized."); \
    return nullptr;                                                        \
  }

namespace tflite {
namespace calibration_wrapper {

namespace {

using python_utils::PyDecrefDeleter;

std::unique_ptr<tflite::ModelT> CreateMutableModel(const tflite::Model& model) {
  auto copied_model = absl::make_unique<tflite::ModelT>();
  model.UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

bool NoOpModel(const tflite::FlatBufferModel& model) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_0(mht_0_v, 230, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "NoOpModel");

  return model->subgraphs()->size() == 1 &&
         (!model->subgraphs()->begin()->operators() ||
          model->subgraphs()->begin()->operators()->size() == 0);
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "TfLiteTypeToSchemaType");

  switch (type) {
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(b/129336260): No schema type for none.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteFloat64:
      return TensorType_FLOAT64;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt32:
      return TensorType_UINT32;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteUInt64:
      return TensorType_UINT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteUInt16:
      return TensorType_UINT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
    case kTfLiteComplex128:
      return TensorType_COMPLEX128;
    case kTfLiteResource:
      return TensorType_RESOURCE;
    case kTfLiteVariant:
      return TensorType_VARIANT;
  }
  // No default to get compiler error when new type is introduced.
}

bool RegisterCustomOpByName(const char* registerer_name,
                            tflite::MutableOpResolver* resolver) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("registerer_name: \"" + (registerer_name == nullptr ? std::string("nullptr") : std::string((char*)registerer_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_2(mht_2_v, 286, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "RegisterCustomOpByName");

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
    PyErr_Format(PyExc_ValueError,
                 "Looking up symbol '%s' failed with error '%s'.",
                 registerer_name, SharedLibrary::GetError());
    return false;
  }

  // Call the registerer with the resolver.
  registerer(resolver);
  return true;
}

// Returns the dimension from the stored list in the PyObject. If the given
// PyObject is not a list, it will return absl::optional and set the Python
// error message to notify users.
absl::optional<std::vector<int>> ConvertInputShapeToVector(
    PyObject* input_shapes, size_t index) {
  PyObject* shape = PyList_GetItem(input_shapes, index);
  if (!shape || !PyList_Check(shape)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid %ld input shape: expected to be a list.", index);
    return absl::nullopt;
  }
  size_t size = PyList_Size(shape);
  std::vector<int> dims(size);
  for (size_t dim_index = 0; dim_index < size; ++dim_index) {
    PyObject* dim = PyList_GetItem(shape, dim_index);
    dims[dim_index] = PyLong_AsLong(dim);
  }
  return dims;
}

}  // namespace

PyObject* AddIntermediateTensors(PyObject* data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_3(mht_3_v, 335, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "AddIntermediateTensors");

  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ::tflite::python::ImportNumpy();

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    PyErr_Format(PyExc_ValueError, "Invalid model");
    return nullptr;
  }
  flatbuffers::FlatBufferBuilder builder;
  auto tflite_model = CreateMutableModel(*model->GetModel());
  if (optimize::AddIntermediateTensorsToFusedOp(&builder, tflite_model.get()) !=
      kTfLiteOk) {
    error_reporter->exception();
    return nullptr;
  }

  if (builder.GetSize()) {
    return python_utils::ConvertToPyString(
        reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
        builder.GetSize());
  } else {
    // When AddIntermediateTensorsToFusedOp early returns, return the model as
    // it is.
    return python_utils::ConvertToPyString(buf, length);
  }
}

CalibrationWrapper::CalibrationWrapper(
    std::unique_ptr<tflite::Interpreter> interpreter,
    std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> resolver,
    std::unique_ptr<tflite::interpreter_wrapper::PythonErrorReporter>
        error_reporter,
    std::unique_ptr<tflite::FlatBufferModel> model,
    std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader,
    std::unique_ptr<std::string> model_str)
    : interpreter_(std::move(interpreter)),
      error_reporter_(std::move(error_reporter)),
      resolver_(std::move(resolver)),
      model_(std::move(model)),
      reader_(std::move(reader)),
      model_str_(std::move(model_str)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_4(mht_4_v, 387, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::CalibrationWrapper");
}

CalibrationWrapper::~CalibrationWrapper() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_5(mht_5_v, 392, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::~CalibrationWrapper");
}

PyObject* CalibrationWrapper::Prepare() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_6(mht_6_v, 397, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::Prepare");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Prepare(std::string signature_key) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("signature_key: \"" + signature_key + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_7(mht_7_v, 408, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::Prepare");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  SignatureRunner* runner =
      interpreter_->GetSignatureRunner(signature_key.c_str());
  if (runner == nullptr) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  TFLITE_PY_CHECK(runner->AllocateTensors());
  TFLITE_PY_CHECK(interpreter_->ResetVariableTensors());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Prepare(PyObject* input_shapes,
                                      std::string signature_key) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("signature_key: \"" + signature_key + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_8(mht_8_v, 427, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::Prepare");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_shapes)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected shapes to be a list.");
    return nullptr;
  }
  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  auto* subgraph = interpreter_->subgraph(subgraph_index);

  const size_t inputs_size = PyList_Size(input_shapes);
  if (inputs_size != subgraph->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected %ld items got %ld items.",
                 subgraph->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    absl::optional<std::vector<int>> dims =
        ConvertInputShapeToVector(input_shapes, i);
    if (!dims.has_value()) {
      return nullptr;
    }
    int input_tensor_idx = subgraph->inputs()[i];
    if (subgraph->ResizeInputTensor(input_tensor_idx, *dims) != kTfLiteOk) {
      PyErr_Format(PyExc_ValueError, "Failed to resize %ld input tensor.", i);
      return nullptr;
    }
  }

  return Prepare(signature_key);
}

PyObject* CalibrationWrapper::Prepare(PyObject* input_shapes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_9(mht_9_v, 470, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::Prepare");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_shapes)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected shapes to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_shapes);
  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input shapes: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    absl::optional<std::vector<int>> dims =
        ConvertInputShapeToVector(input_shapes, i);
    if (!dims.has_value()) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (interpreter_->ResizeInputTensor(input_tensor_idx, *dims) != kTfLiteOk) {
      PyErr_Format(PyExc_ValueError, "Failed to resize %ld input tensor.", i);
      return nullptr;
    }
  }

  return Prepare();
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value,
                                         std::string signature_key) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("signature_key: \"" + signature_key + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_10(mht_10_v, 507, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::FeedTensor");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }
  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  const size_t inputs_size = PyList_Size(input_value);

  auto* subgraph = interpreter_->subgraph(subgraph_index);
  if (inputs_size != subgraph->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 subgraph->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = subgraph->inputs()[i];
    if (!SetTensor(input_tensor_idx, input, signature_key)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(subgraph->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::FeedTensor(PyObject* input_value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_11(mht_11_v, 549, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::FeedTensor");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  if (!PyList_Check(input_value)) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input type: expected input to be a list.");
    return nullptr;
  }

  const size_t inputs_size = PyList_Size(input_value);

  if (inputs_size != interpreter_->inputs().size()) {
    PyErr_Format(PyExc_ValueError,
                 "Invalid input size: expected %ld items got %ld items.",
                 interpreter_->inputs().size(), inputs_size);
    return nullptr;
  }

  for (size_t i = 0; i < inputs_size; ++i) {
    PyObject* input = PyList_GetItem(input_value, i);
    if (!input) {
      return nullptr;
    }
    int input_tensor_idx = interpreter_->inputs()[i];
    if (!SetTensor(input_tensor_idx, input)) {
      return nullptr;
    }
  }

  TFLITE_PY_CHECK(interpreter_->Invoke());
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value,
                                        std::string signature_key) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("signature_key: \"" + signature_key + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_12(mht_12_v, 586, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::SetTensor");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();
  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  const int subgraph_index =
      interpreter_->GetSubgraphIndexFromSignature(signature_key.c_str());
  if (subgraph_index == -1) {
    PyErr_Format(PyExc_ValueError, "Invalid signature key: %s",
                 signature_key.c_str());
    return nullptr;
  }
  auto* subgraph = interpreter_->subgraph(subgraph_index);
  const TfLiteTensor* tensor = subgraph->tensor(index);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: "
                 "Got value of type %s "
                 "but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension count mismatch, expected %d "
                 "but found %d",
                 tensor->dims->size, PyArray_NDIM(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_NDIM(array));
  bool has_unknown_dims = false;
  for (int j = 0; j < PyArray_NDIM(array); ++j) {
    // Ensure the calibration data input shape is the same as the model input
    // shape unless the dimension is unknown.
    if (tensor->dims_signature != nullptr &&
        tensor->dims_signature->size == tensor->dims->size &&
        tensor->dims_signature->data[j] == -1) {
      has_unknown_dims = true;
    } else if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Size mismatch, expected %d for dim "
                   "%d but found %ld",
                   tensor->dims->data[j], j, PyArray_SHAPE(array)[j]);
      return nullptr;
    }
    dims[j] = PyArray_SHAPE(array)[j];
  }

  // Resize the input tensor if there are unknown dimensions.
  if (has_unknown_dims) {
    // Does strict checking on the `ResizeInputTensor` call.
    TFLITE_PY_CHECK(subgraph->ResizeInputTensorStrict(index, dims));
    TFLITE_PY_CHECK(subgraph->AllocateTensors());
  }

  // Re-read the updated tensor after the allocation is done.
  tensor = subgraph->tensor(index);

  size_t size = PyArray_NBYTES(array);

  if (tensor->type == kTfLiteString) {
    tflite::DynamicBuffer buffer;
    buffer.AddString(reinterpret_cast<const char*>(PyArray_BYTES(array)), size);
    buffer.WriteToTensor(subgraph->tensor(index), /*new_shape=*/nullptr);
    Py_RETURN_NONE;
  }

  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::SetTensor(int index, PyObject* value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_13(mht_13_v, 677, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::SetTensor");

  TFLITE_PY_ENSURE_VALID_INTERPRETER();

  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(
      PyArray_FromAny(value, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    PyErr_SetString(PyExc_ValueError,
                    "Failed to convert value into readable tensor.");
    return nullptr;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());
  const TfLiteTensor* tensor = interpreter_->tensor(index);

  if (python_utils::TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: "
                 "Got value of type %s "
                 "but expected type %s for input %d, name: %s ",
                 TfLiteTypeGetName(python_utils::TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return nullptr;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(
        PyExc_ValueError,
        "Cannot set tensor: Dimension count mismatch, expected %d but found %d",
        tensor->dims->size, PyArray_NDIM(array));
    return nullptr;
  }

  std::vector<int> dims(PyArray_NDIM(array));
  bool has_unknown_dims = false;
  for (int j = 0; j < PyArray_NDIM(array); ++j) {
    // Ensure the calibration data input shape is the same as the model input
    // shape unless the dimension is unknown.
    if (tensor->dims_signature != nullptr &&
        tensor->dims_signature->size == tensor->dims->size &&
        tensor->dims_signature->data[j] == -1) {
      has_unknown_dims = true;
    } else if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Size mismatch, expected %d for dim "
                   "%d but found %ld",
                   tensor->dims->data[j], j, PyArray_SHAPE(array)[j]);
      return nullptr;
    }
    dims[j] = PyArray_SHAPE(array)[j];
  }

  // Resize the input tensor if there are unknown dimensions.
  if (has_unknown_dims) {
    // Does strict checking on the `ResizeInputTensor` call.
    TFLITE_PY_CHECK(interpreter_->ResizeInputTensorStrict(index, dims));
    TFLITE_PY_CHECK(interpreter_->AllocateTensors());
  }

  // Re-read the updated tensor after the allocation is done.
  tensor = interpreter_->tensor(index);

  size_t size = PyArray_NBYTES(array);

  if (tensor->type == kTfLiteString) {
    tflite::DynamicBuffer buffer;
    buffer.AddString(reinterpret_cast<const char*>(PyArray_BYTES(array)), size);
    buffer.WriteToTensor(interpreter_->tensor(index), /*new_shape=*/nullptr);
    Py_RETURN_NONE;
  }

  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return nullptr;
  }
  memcpy(tensor->data.raw, PyArray_DATA(array), size);
  Py_RETURN_NONE;
}

PyObject* CalibrationWrapper::Calibrate() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_14(mht_14_v, 760, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::Calibrate");

  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto loc = tflite::Model::Pack(builder, tflite_model.get());
  tflite::FinishModelBuffer(builder, loc);
  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            int activations_py_type,
                                            int bias_py_type) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_15(mht_15_v, 778, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::QuantizeModel");

  return QuantizeModel(input_py_type, output_py_type, allow_float,
                       activations_py_type, bias_py_type,
                       /*disable_per_channel=*/false);
}

PyObject* CalibrationWrapper::QuantizeModel(
    int input_py_type, int output_py_type, bool allow_float,
    int activations_py_type, int bias_py_type, bool disable_per_channel) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_16(mht_16_v, 789, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::QuantizeModel");

  if (NoOpModel(*model_)) {
    return python_utils::ConvertToPyString(model_str_->data(),
                                           model_str_->size());
  }

  TfLiteType input_type = python_utils::TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = python_utils::TfLiteTypeFromPyType(output_py_type);
  TfLiteType activations_type =
      python_utils::TfLiteTypeFromPyType(activations_py_type);
  TfLiteType bias_type = python_utils::TfLiteTypeFromPyType(bias_py_type);

  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = kTfLiteOk;

  status = tflite::optimize::QuantizeModelAllOperators(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float,
      TfLiteTypeToSchemaType(activations_type),
      TfLiteTypeToSchemaType(bias_type), disable_per_channel,
      error_reporter_.get());

  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

PyObject* CalibrationWrapper::QuantizeModel(int input_py_type,
                                            int output_py_type,
                                            bool allow_float,
                                            const char* operator_output_name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("operator_output_name: \"" + (operator_output_name == nullptr ? std::string("nullptr") : std::string((char*)operator_output_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_17(mht_17_v, 835, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::QuantizeModel");

  string op_name = std::string(operator_output_name);

  TfLiteType input_type = python_utils::TfLiteTypeFromPyType(input_py_type);
  TfLiteType output_type = python_utils::TfLiteTypeFromPyType(output_py_type);
  if (input_type == kTfLiteNoType || output_type == kTfLiteNoType) {
    PyErr_SetString(PyExc_ValueError,
                    "Input/output type cannot be kTfLiteNoType");
    return nullptr;
  }
  auto tflite_model = CreateMutableModel(*model_->GetModel());
  reader_->AddCalibrationToModel(tflite_model.get(), /*update=*/false);
  flatbuffers::FlatBufferBuilder builder;
  auto status = tflite::optimize::QuantizeModel(
      &builder, tflite_model.get(), TfLiteTypeToSchemaType(input_type),
      TfLiteTypeToSchemaType(output_type), allow_float, {op_name},
      /*activations_type=*/TensorType_INT8, /*bias_type=*/TensorType_INT32,
      error_reporter_.get());
  if (status != kTfLiteOk) {
    error_reporter_->exception();
    return nullptr;
  }

  return python_utils::ConvertToPyString(
      reinterpret_cast<const char*>(builder.GetCurrentBufferPointer()),
      builder.GetSize());
}

/*static*/ CalibrationWrapper* CalibrationWrapper::CreateWrapperCPPFromBuffer(
    PyObject* data, const std::vector<std::string>& registerers_by_name,
    const std::vector<std::function<void(uintptr_t)>>& registerers_by_func,
    std::string* error_msg) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSpythonPSoptimizePScalibration_wrapperDTcc mht_18(mht_18_v, 869, "", "./tensorflow/lite/python/optimize/calibration_wrapper.cc", "CalibrationWrapper::CreateWrapperCPPFromBuffer");

  using tflite::interpreter_wrapper::PythonErrorReporter;
  char* buf = nullptr;
  Py_ssize_t length;
  std::unique_ptr<PythonErrorReporter> error_reporter(new PythonErrorReporter);
  ::tflite::python::ImportNumpy();

  if (python_utils::ConvertFromPyString(data, &buf, &length) == -1) {
    *error_msg = "Failed to convert from python string";
    return nullptr;
  }
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buf, length,
                                               error_reporter.get());
  if (!model) {
    *error_msg = "Invalid model";
    return nullptr;
  }
  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  for (const auto& registerer : registerers_by_name) {
    if (!RegisterCustomOpByName(registerer.c_str(), resolver.get())) {
      *error_msg =
          absl::StrFormat("Looking up symbol '%s' failed with error '%s'.",
                          registerer.c_str(), SharedLibrary::GetError());
      return nullptr;
    }
  }
  for (const auto& registerer : registerers_by_func) {
    registerer(reinterpret_cast<uintptr_t>(resolver.get()));
  }
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::optimize::calibration::CalibrationReader> reader;
  auto status = tflite::optimize::calibration::BuildLoggingInterpreter(
      *model, *resolver, &interpreter, &reader);
  if (status != kTfLiteOk) {
    *error_msg = error_reporter->message();
    return nullptr;
  }

  auto model_str = std::make_unique<std::string>(buf, length);
  // If we are not going to use this string during quantization, reset the
  // pointer and release the memory.
  if (!NoOpModel(*model)) {
    model_str.reset();
  }

  auto wrapper = new CalibrationWrapper(
      std::move(interpreter), std::move(resolver), std::move(error_reporter),
      std::move(model), std::move(reader), std::move(model_str));
  return wrapper;
}

}  // namespace calibration_wrapper
}  // namespace tflite
