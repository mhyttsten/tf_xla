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
class MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc() {
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
#include "tensorflow/python/framework/python_api_parameter_converter.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/framework/op_def_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

#if PY_MAJOR_VERSION < 3
// Python 2.x:
#define PY_INT_AS_LONG(x) (PyInt_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyString_InternFromString(x))
#else
// Python 3.x:
#define PY_INT_AS_LONG(x) (PyLong_AsLong(x))
#define PY_STRING_INTERN_FROM_STRING(x) (PyUnicode_InternFromString(x))
#endif

// Evaluate `condition`, and if it returns false then return false.
#define RETURN_IF_FALSE(condition)  \
  do {                              \
    if (!(condition)) return false; \
  } while (0)

#define PyList_ITEMS(o) (((PyListObject*)(o))->ob_item)

namespace tensorflow {

using InferredAttributes = PythonAPIInfo::InferredAttributes;
using ParamIndex = PythonAPIInfo::ParamIndex;
using Attribute = PythonAPIInfo::Attribute;
using InputWithFixedDType = PythonAPIInfo::InputWithFixedDType;
using InputsWithTypeAttr = PythonAPIInfo::InputsWithTypeAttr;
using InputsWithTypeListAttr = PythonAPIInfo::InputsWithTypeListAttr;

namespace {

// Returns `dtype._type_enum`.
Safe_PyObjectPtr GetAttr_TypeEnum(PyObject* dtype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_0(mht_0_v, 224, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "GetAttr_TypeEnum");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("_type_enum");
  return Safe_PyObjectPtr(PyObject_GetAttr(dtype, attr));
}

// Returns `tensor.dtype`.
Safe_PyObjectPtr GetAttr_DType(PyObject* tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_1(mht_1_v, 233, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "GetAttr_DType");

  static PyObject* attr = PY_STRING_INTERN_FROM_STRING("dtype");
  return Safe_PyObjectPtr(PyObject_GetAttr(tensor, attr));
}

// Raises a TypeError with a message constructed by applying StrCat to the
// specified strings.  If an exception has already been set when this function
// is called, then add its message as a suffix to the message string.
template <typename... Args>
void RaiseTypeError(Args... args) {
  string message = absl::StrCat(args...);
  if (!PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError, message.c_str());
  } else {
    PyObject* exc_type;
    PyObject* exc_value;
    PyObject* exc_traceback;
    PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
    PyErr_Format(PyExc_TypeError, "%s: %S", message.c_str(), exc_value);
    Py_XDECREF(exc_type);
    Py_XDECREF(exc_value);
    Py_XDECREF(exc_traceback);
  }
}

// Returns the DataType for a `tf.dtypes.DType` object (or DT_INVALID if it
// is not a valid DType object).
ABSL_MUST_USE_RESULT
DataType DataTypeFromPyDType(PyObject* dtype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_2(mht_2_v, 264, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "DataTypeFromPyDType");

  if (!dtype) {
    return DT_INVALID;
  }
  Safe_PyObjectPtr enum_field = GetAttr_TypeEnum(dtype);
  if (!enum_field) {
    return DT_INVALID;
  }
  DataType result = static_cast<DataType>(PY_INT_AS_LONG(enum_field.get()));
  return result;
}

// Update `dtype` with an inferred dtype from `value`.  In particular, if
// `dtype == DT_INVALID` and `value` is a `Tensor`, then set `dtype` to
// `value.dtype`.  (If `dtype` is not `DT_INVALID`, or `value` is not a
// tensor, then do nothing.)  Returns false on exception.
ABSL_MUST_USE_RESULT
bool InferDType(PyObject* value, DataType& dtype) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_3(mht_3_v, 284, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "InferDType");

  if (dtype != DT_INVALID) return true;  // Already have dtype.

  if (EagerTensor_CheckExact(value)) {
    dtype = PyEagerTensor_Dtype(value);
    return true;
  }

  if (swig::IsTensor(value)) {
    Safe_PyObjectPtr py_dtype = GetAttr_DType(value);
    if (!py_dtype) return false;
    dtype = DataTypeFromPyDType(py_dtype.get());  // set output parameter
    return true;
  }
  return true;
}

// Returns true if `dtype` is in `ok_dtypes`, or `ok_dtypes` is null or empty.
ABSL_MUST_USE_RESULT
bool IsOkDType(DataType dtype, const std::vector<DataType>* ok_dtypes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_4(mht_4_v, 306, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "IsOkDType");

  return (ok_dtypes == nullptr || ok_dtypes->empty() ||
          std::find(ok_dtypes->begin(), ok_dtypes->end(), dtype) !=
              ok_dtypes->end());
}

// Formatter for DataTypes for absl::StrJoin.
struct DataTypeFormatter {
  void operator()(std::string* out, DataType dtype) const {
    out->append(DataType_Name(dtype));
  }
};

// Converts `src` to a tensor using `tensor_converter.Convert`.  If `src` is
// replaced by a new value then decref the replaced value.  If an error
// occurs, then re-raise it as a TypeError with a prefix indicating the API
// name and the parameter name.
//
// Args:
//   src: The value that should be converted (in-place).
//   dtype: The dtype to convert `src` to, or DT_INVALID for unconstraned.
//     If DT_INVALID, then `dtype` will be set to the actual dtype of the
//     converted value.
//   tensor_converter: Class used to convert python values to tensors.
//   api_info: Information about the API we're converting this value for
//     (for error messages).
//   param_index: Index of the parameter we're converting (for error messages).
//   ok_dtypes: List of valid dtypes for conversion (optional).
//   default_dtype: Default dtype -- used if converting the value to a tensor
//     with unconstrained dtype returns a value not in ok_dtypes.
ABSL_MUST_USE_RESULT
bool ConvertToTensorInPlace(PyObject*& src, DataType& dtype,
                            const PythonTensorConverter& tensor_converter,
                            const PythonAPIInfo& api_info, int param_index,
                            const std::vector<DataType>* ok_dtypes = nullptr,
                            DataType default_dtype = DT_INVALID) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_5(mht_5_v, 344, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "ConvertToTensorInPlace");

  bool inferred_dtype = (dtype == DT_INVALID);
  Safe_PyObjectPtr converted = tensor_converter.Convert(src, dtype);
  if (!converted) {
    RaiseTypeError(api_info.api_name(), " argument ",
                   api_info.param_names()[param_index]);
    return false;
  }

  if (inferred_dtype && !IsOkDType(dtype, ok_dtypes)) {
    // Converting `src` to a tensor gave us a disallowed dtype; try again
    // with `default_dtype`.
    if (default_dtype == DT_INVALID) {
      RaiseTypeError(api_info.api_name(), " argument ",
                     api_info.param_names()[param_index], ": Expected one of {",
                     absl::StrJoin(*ok_dtypes, ", ", DataTypeFormatter()),
                     "}, but got ", DataType_Name(dtype));
      return false;
    } else {
      dtype = default_dtype;
      converted = tensor_converter.Convert(src, dtype);
      if (!converted) {
        RaiseTypeError(api_info.api_name(), " argument ",
                       api_info.param_names()[param_index]);
        return false;
      }
    }
  }

  Py_DECREF(src);
  src = converted.release();
  return true;
}

// Converts the specified attribute parameter to the expected type.  Modifies
// `params` in-place.  Returns true on success, or sets an exception and
// returns false on failure.
ABSL_MUST_USE_RESULT
bool ConvertAttribute(const Attribute& attr, const PythonAPIInfo& api_info,
                      absl::Span<PyObject*> params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_6(mht_6_v, 386, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "ConvertAttribute");

  if (attr.index == -1) return true;  // Inferred attribute.
  PyObject* src = params[attr.index];
  Safe_PyObjectPtr converted = ConvertPyObjectToAttributeType(src, attr.type);
  if (!converted) {
    RaiseTypeError(api_info.api_name(), " argument ",
                   api_info.param_names()[attr.index]);
    return false;
  }
  if (converted.get() != src) {
    Py_DECREF(src);
    params[attr.index] = converted.release();
  }
  return true;
}

// Converts the specified fixed-dtype input parameter to a Tensor with the
// expected dtype.  Modifies `params` in-place.  Returns true on success, or
// sets an exception and returns false on failure.
ABSL_MUST_USE_RESULT
bool ConvertInputWithFixedDType(const InputWithFixedDType& input,
                                const PythonTensorConverter& tensor_converter,
                                const PythonAPIInfo& api_info,
                                absl::Span<PyObject*> params) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_7(mht_7_v, 412, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "ConvertInputWithFixedDType");

  DataType dtype = input.dtype;
  PyObject*& src = params[input.index];
  if (!input.is_list) {
    RETURN_IF_FALSE(ConvertToTensorInPlace(src, dtype, tensor_converter,
                                           api_info, input.index));
  } else {
    DCHECK(PyList_CheckExact(src));
    PyObject** items = PyList_ITEMS(src);
    Py_ssize_t len = PyList_GET_SIZE(src);
    for (Py_ssize_t i = 0; i < len; ++i) {
      RETURN_IF_FALSE(ConvertToTensorInPlace(items[i], dtype, tensor_converter,
                                             api_info, input.index));
    }
  }
  return true;
}

// Infers a consistent dtype for the specified collection of homogeneous-dtype
// input parameters, and converts those parameters to Tensors (or lists of
// Tensors) with that dtype. Modifies `params` in-place, and updates
// `inferred_attrs` with the inferred dtype (if it's not null).  Returns true
// on success, or sets an exception and returns false on failure.
ABSL_MUST_USE_RESULT
bool ConvertInputsWithTypeAttr(const InputsWithTypeAttr& input,
                               const PythonTensorConverter& tensor_converter,
                               const PythonAPIInfo& api_info,
                               absl::Span<PyObject*> params,
                               InferredAttributes* inferred_attrs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_8(mht_8_v, 443, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "ConvertInputsWithTypeAttr");

  DataType dtype = DT_INVALID;
  if (input.type_attr->index != -1) {
    // explicit type attribute
    PyObject* py_dtype = params[input.type_attr->index];
    dtype = DataTypeFromPyDType(py_dtype);
  } else {
    // implicit type attribute: infer the dtype.
    // First, check the single-tensor inputs.
    for (ParamIndex index : input.tensor_params) {
      RETURN_IF_FALSE(InferDType(params[index], dtype));
      if (dtype != DT_INVALID) break;
    }
    // Next, check the list-of-tensor inputs.
    if (dtype == DT_INVALID) {
      for (ParamIndex index : input.tensor_list_params) {
        PyObject* tensor_list = params[index];
        DCHECK(PyList_CheckExact(tensor_list));
        Py_ssize_t num_tensors = PyList_GET_SIZE(tensor_list);
        PyObject** tensors = PyList_ITEMS(tensor_list);
        for (Py_ssize_t i = 0; i < num_tensors; ++i) {
          RETURN_IF_FALSE(InferDType(tensors[i], dtype));
          if (dtype != DT_INVALID) break;
        }
        if (dtype != DT_INVALID) break;
      }
    }
  }

  // Convert the single-tensor inputs to tensors.
  for (ParamIndex index : input.tensor_params) {
    RETURN_IF_FALSE(
        ConvertToTensorInPlace(params[index], dtype, tensor_converter, api_info,
                               index, &input.ok_dtypes, input.default_dtype));
  }

  // Convert the list-of-tensor inputs to tensors.
  for (ParamIndex index : input.tensor_list_params) {
    PyObject* tensor_list = params[index];
    DCHECK(PyList_CheckExact(tensor_list));
    Py_ssize_t num_tensors = PyList_GET_SIZE(tensor_list);
    PyObject** items = PyList_ITEMS(tensor_list);
    for (Py_ssize_t i = 0; i < num_tensors; ++i) {
      RETURN_IF_FALSE(ConvertToTensorInPlace(items[i], dtype, tensor_converter,
                                             api_info, index, &input.ok_dtypes,
                                             input.default_dtype));
    }
  }

  if (inferred_attrs) {
    if (dtype == DT_INVALID) {
      dtype = input.default_dtype;
    }
    // TODO(b/164980194) Should we raise an exception here if we didn't manage
    // to infer a dtype?  (I.e., if there were no single-tensor inputs and all
    // list-of-tensor inputs were empty, and there's no default dtype.)
    int inferred_index = input.type_attr->inferred_index;
    if (inferred_index != -1) {
      inferred_attrs->types[inferred_index] = dtype;
    }
  }

  return true;
}

// Infers a consistent list of dtypes for the specified collection of
// heterogeneous-dtype input parameters, and converts those parameters to lists
// of Tensors with those dtypes. Modifies `params` in-place, and updates
// `inferred_attrs` with the inferred dtypes (if it's not null).  Returns true
// on success, or sets an exception and returns false on failure.
ABSL_MUST_USE_RESULT
bool ConvertInputsWithTypeListAttr(
    const InputsWithTypeListAttr& input,
    const PythonTensorConverter& tensor_converter,
    const PythonAPIInfo& api_info, absl::Span<PyObject*> params,
    InferredAttributes* inferred_attrs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_9(mht_9_v, 521, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "ConvertInputsWithTypeListAttr");

  DCHECK(!input.tensor_list_params.empty());

  // Get the number of tensors from the first input list; and check that the
  // remaining lists have the same size.
  DCHECK(PyList_CheckExact(params[input.tensor_list_params[0]]));
  Py_ssize_t num_tensors = PyList_GET_SIZE(params[input.tensor_list_params[0]]);
  for (int i = 1; i < input.tensor_list_params.size(); ++i) {
    DCHECK(PyList_CheckExact(params[input.tensor_list_params[i]]));
    if (num_tensors != PyList_GET_SIZE(params[input.tensor_list_params[i]])) {
      RaiseTypeError(api_info.api_name(), " expected parameters ",
                     api_info.param_names()[0], " and ",
                     api_info.param_names()[i],
                     " to be lists of the same length.");
      return false;
    }
  }

  // Get the list of dtypes.
  std::vector<DataType> dtypes(num_tensors, DT_INVALID);
  if (input.type_list_attr->index != -1) {
    // Dtypes are specified by an explicit attribute.
    PyObject* py_dtypes = params[input.type_list_attr->index];
    if (PyList_GET_SIZE(py_dtypes) != num_tensors) {
      RaiseTypeError(api_info.api_name(), " expected parameters ",
                     api_info.param_names()[0], " and ",
                     api_info.param_names()[input.type_list_attr->index],
                     "to be lists of the same length.");
      return false;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(py_dtypes); ++i) {
      dtypes[i] = DataTypeFromPyDType(PyList_GetItem(py_dtypes, i));
    }
  } else {
    // Dtypes are implicit: infer them.
    for (Py_ssize_t i = 0; i < num_tensors; ++i) {
      for (ParamIndex index : input.tensor_list_params) {
        PyObject* tensor_list = params[index];
        DCHECK(PyList_CheckExact(tensor_list));
        PyObject* item = PyList_GET_ITEM(tensor_list, i);
        RETURN_IF_FALSE(InferDType(item, dtypes[i]));
        if (dtypes[i] != DT_INVALID) break;
      }
    }
  }

  // Convert tensors.
  for (ParamIndex index : input.tensor_list_params) {
    PyObject* tensor_list = params[index];
    PyObject** items = PyList_ITEMS(tensor_list);
    for (Py_ssize_t i = 0; i < num_tensors; ++i) {
      DataType default_dtype = i < input.default_dtypes.size()
                                   ? input.default_dtypes[i]
                                   : DT_INVALID;
      RETURN_IF_FALSE(ConvertToTensorInPlace(items[i], dtypes[i],
                                             tensor_converter, api_info, index,
                                             &input.ok_dtypes, default_dtype));
    }
  }

  if (inferred_attrs) {
    int inferred_index = input.type_list_attr->inferred_index;
    if (inferred_index != -1) {
      inferred_attrs->type_lists[inferred_index].swap(dtypes);
    }
  }

  return true;
}

// Infers length attributes for Tensor-list parameters from their values, and
// updates `inferred_length_attrs` with the inferred length.  Sets an exception
// if multiple Tensor-list parameters have the same length attribute but
// different lengths. Returns true on success, or sets an exception and returns
// false on failure.
ABSL_MUST_USE_RESULT
bool InferLengthAttributes(const absl::Span<PyObject*> params,
                           const PythonAPIInfo& api_info,
                           std::vector<int64_t>& inferred_length_attrs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_10(mht_10_v, 602, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "InferLengthAttributes");

  for (int i = 0; i < api_info.inputs_with_number_attrs().size(); ++i) {
    const auto& inputs = api_info.inputs_with_number_attrs()[i];
    DCHECK(!inputs.tensor_list_params.empty());

    // Use the first tensor_list parameter to infer the length attribute.
    PyObject* tensors = params[inputs.tensor_list_params[0]];
    DCHECK(PyList_CheckExact(tensors));
    int inferred_length = PyList_GET_SIZE(tensors);

    // Check that any other tensor_list parameters have matching length.
    for (int j = 1; j < inputs.tensor_list_params.size(); ++j) {
      int num_tensors = PyList_GET_SIZE(params[inputs.tensor_list_params[j]]);
      if (num_tensors != inferred_length) {
        RaiseTypeError(api_info.api_name(), " expected parameters ",
                       api_info.param_names()[inputs.tensor_list_params[0]],
                       " and ",
                       api_info.param_names()[inputs.tensor_list_params[j]],
                       " to be lists with the same length.");
      }
    }

    int inferred_index = inputs.number_attr->inferred_index;
    if (inferred_index != -1) {
      inferred_length_attrs[inferred_index] = inferred_length;
    }
  }
  return true;
}

}  // namespace

bool ConvertPythonAPIParameters(const PythonAPIInfo& api_info,
                                const PythonTensorConverter& tensor_converter,
                                absl::Span<PyObject*> params,
                                InferredAttributes* inferred_attrs) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_11(mht_11_v, 640, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "ConvertPythonAPIParameters");

  // Make room for inferred attributes.
  if (inferred_attrs) {
    inferred_attrs->types.resize(api_info.inferred_type_attrs().size());
    inferred_attrs->type_lists.resize(
        api_info.inferred_type_list_attrs().size());
    inferred_attrs->lengths.resize(api_info.inferred_length_attrs().size());
  }

  for (const auto& attr : api_info.attributes()) {
    RETURN_IF_FALSE(ConvertAttribute(attr, api_info, params));
  }

  for (const auto& input : api_info.inputs_with_fixed_dtype()) {
    RETURN_IF_FALSE(
        ConvertInputWithFixedDType(input, tensor_converter, api_info, params));
  }

  for (int i = 0; i < api_info.inputs_with_type_attrs().size(); ++i) {
    RETURN_IF_FALSE(ConvertInputsWithTypeAttr(
        api_info.inputs_with_type_attrs()[i], tensor_converter, api_info,
        params, inferred_attrs));
  }

  for (int i = 0; i < api_info.inputs_with_type_list_attrs().size(); ++i) {
    RETURN_IF_FALSE(ConvertInputsWithTypeListAttr(
        api_info.inputs_with_type_list_attrs()[i], tensor_converter, api_info,
        params, inferred_attrs));
  }

  if (inferred_attrs) {
    RETURN_IF_FALSE(
        InferLengthAttributes(params, api_info, inferred_attrs->lengths));
  }

  return true;
}

bool CopyPythonAPITensorLists(const PythonAPIInfo& api_info,
                              absl::Span<PyObject*> params) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSframeworkPSpython_api_parameter_converterDTcc mht_12(mht_12_v, 682, "", "./tensorflow/python/framework/python_api_parameter_converter.cc", "CopyPythonAPITensorLists");

  for (const auto& input : api_info.inputs()) {
    if (input.is_list) {
      PyObject* src = params[input.index];
      PyObject* copy = PySequence_List(src);
      if (!copy) {
        RaiseTypeError(api_info.api_name(), " expected a list of Tensors for '",
                       api_info.param_names()[input.index], "'; got ",
                       src->ob_type->tp_name, ".");
        return false;
      }
      Py_DECREF(params[input.index]);
      params[input.index] = copy;
    }
  }
  return true;
}

}  // namespace tensorflow
