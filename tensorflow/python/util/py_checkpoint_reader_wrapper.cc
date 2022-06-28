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
class MHTracer_DTPStensorflowPSpythonPSutilPSpy_checkpoint_reader_wrapperDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSutilPSpy_checkpoint_reader_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPSpy_checkpoint_reader_wrapperDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace py = pybind11;

// TODO(amitpatankar): Move the custom type casters to separate common header
// only libraries.

namespace pybind11 {
namespace detail {

/* This is a custom type caster for the TensorShape object. For more
 * documentation please refer to this link:
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html#custom-type-casters
 * The PyCheckpointReader methods sometimes return the `TensorShape` object
 * and the `DataType` object as outputs. This custom type caster helps Python
 * handle it's conversion from C++ to Python. Since we do not accept these
 * classes as arguments from Python, it is not necessary to define the `load`
 * function to cast the object from Python to a C++ object.
 */

template <>
struct type_caster<tensorflow::TensorShape> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::TensorShape, _("tensorflow::TensorShape"));

  static handle cast(const tensorflow::TensorShape& src,
                     return_value_policy unused_policy, handle unused_handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSpy_checkpoint_reader_wrapperDTcc mht_0(mht_0_v, 229, "", "./tensorflow/python/util/py_checkpoint_reader_wrapper.cc", "cast");

    // TODO(amitpatankar): Simplify handling TensorShape as output later.
    size_t dims = src.dims();
    tensorflow::Safe_PyObjectPtr value(PyList_New(dims));
    for (size_t i = 0; i < dims; ++i) {
#if PY_MAJOR_VERSION >= 3
      tensorflow::Safe_PyObjectPtr dim_value(
          tensorflow::make_safe(PyLong_FromLong(src.dim_size(i))));
#else
      tensorflow::Safe_PyObjectPtr dim_value(
          tensorflow::make_safe(PyInt_FromLong(src.dim_size(i))));
#endif
      PyList_SET_ITEM(value.get(), i, dim_value.release());
    }

    return value.release();
  }
};

template <>
struct type_caster<tensorflow::DataType> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::DataType, _("tensorflow::DataType"));

  static handle cast(const tensorflow::DataType& src,
                     return_value_policy unused_policy, handle unused_handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSpy_checkpoint_reader_wrapperDTcc mht_1(mht_1_v, 257, "", "./tensorflow/python/util/py_checkpoint_reader_wrapper.cc", "cast");

#if PY_MAJOR_VERSION >= 3
    tensorflow::Safe_PyObjectPtr value(
        tensorflow::make_safe(PyLong_FromLong(src)));
#else
    tensorflow::Safe_PyObjectPtr value(
        tensorflow::make_safe(PyInt_FromLong(src)));
#endif
    return value.release();
  }
};

}  // namespace detail
}  // namespace pybind11

namespace tensorflow {

static py::object CheckpointReader_GetTensor(
    tensorflow::checkpoint::CheckpointReader* reader, const string& name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSpythonPSutilPSpy_checkpoint_reader_wrapperDTcc mht_2(mht_2_v, 279, "", "./tensorflow/python/util/py_checkpoint_reader_wrapper.cc", "CheckpointReader_GetTensor");

  Safe_TF_StatusPtr status = make_safe(TF_NewStatus());
  PyObject* py_obj = Py_None;
  std::unique_ptr<tensorflow::Tensor> tensor;
  reader->GetTensor(name, &tensor, status.get());

  // Error handling if unable to get Tensor.
  tensorflow::MaybeRaiseFromTFStatus(status.get());

  tensorflow::MaybeRaiseFromStatus(
      tensorflow::TensorToNdarray(*tensor, &py_obj));

  return tensorflow::PyoOrThrow(
      PyArray_Return(reinterpret_cast<PyArrayObject*>(py_obj)));
}

}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_checkpoint_reader, m) {
  // Initialization code to use numpy types in the type casters.
  import_array1();
  py::class_<tensorflow::checkpoint::CheckpointReader> checkpoint_reader_class(
      m, "CheckpointReader");
  checkpoint_reader_class
      .def(py::init([](const std::string& filename) {
        tensorflow::Safe_TF_StatusPtr status =
            tensorflow::make_safe(TF_NewStatus());
        // pybind11 support smart pointers and will own freeing the memory when
        // complete.
        // https://pybind11.readthedocs.io/en/master/advanced/smart_ptrs.html#std-unique-ptr
        auto checkpoint =
            std::make_unique<tensorflow::checkpoint::CheckpointReader>(
                filename, status.get());
        tensorflow::MaybeRaiseFromTFStatus(status.get());
        return checkpoint;
      }))
      .def("debug_string",
           [](tensorflow::checkpoint::CheckpointReader& self) {
             return py::bytes(self.DebugString());
           })
      .def("get_variable_to_shape_map",
           &tensorflow::checkpoint::CheckpointReader::GetVariableToShapeMap)
      .def("_GetVariableToDataTypeMap",
           &tensorflow::checkpoint::CheckpointReader::GetVariableToDataTypeMap)
      .def("_HasTensor", &tensorflow::checkpoint::CheckpointReader::HasTensor)
      .def_static("CheckpointReader_GetTensor",
                  &tensorflow::CheckpointReader_GetTensor);
};
