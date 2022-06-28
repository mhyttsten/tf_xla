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
class MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/lib/core/ndarray_tensor.h"

#include <cstring>
#include <optional>

#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/bfloat16.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/numpy.h"

namespace tensorflow {
namespace {

char const* numpy_type_name(int numpy_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_0(mht_0_v, 203, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "numpy_type_name");

  switch (numpy_type) {
#define TYPE_CASE(s) \
  case s:            \
    return #s

    TYPE_CASE(NPY_BOOL);
    TYPE_CASE(NPY_BYTE);
    TYPE_CASE(NPY_UBYTE);
    TYPE_CASE(NPY_SHORT);
    TYPE_CASE(NPY_USHORT);
    TYPE_CASE(NPY_INT);
    TYPE_CASE(NPY_UINT);
    TYPE_CASE(NPY_LONG);
    TYPE_CASE(NPY_ULONG);
    TYPE_CASE(NPY_LONGLONG);
    TYPE_CASE(NPY_ULONGLONG);
    TYPE_CASE(NPY_FLOAT);
    TYPE_CASE(NPY_DOUBLE);
    TYPE_CASE(NPY_LONGDOUBLE);
    TYPE_CASE(NPY_CFLOAT);
    TYPE_CASE(NPY_CDOUBLE);
    TYPE_CASE(NPY_CLONGDOUBLE);
    TYPE_CASE(NPY_OBJECT);
    TYPE_CASE(NPY_STRING);
    TYPE_CASE(NPY_UNICODE);
    TYPE_CASE(NPY_VOID);
    TYPE_CASE(NPY_DATETIME);
    TYPE_CASE(NPY_TIMEDELTA);
    TYPE_CASE(NPY_HALF);
    TYPE_CASE(NPY_NTYPES);
    TYPE_CASE(NPY_NOTYPE);
    TYPE_CASE(NPY_CHAR);
    TYPE_CASE(NPY_USERDEF);
    default:
      return "not a numpy type";
  }
}

Status PyArrayDescr_to_TF_DataType(PyArray_Descr* descr,
                                   TF_DataType* out_tf_datatype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_1(mht_1_v, 246, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "PyArrayDescr_to_TF_DataType");

  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;

  // Return an error if the fields attribute is null.
  // Occurs with an improper conversion attempt to resource.
  if (descr->fields == nullptr) {
    return errors::Internal("Unexpected numpy data type");
  }

  if (PyDict_Next(descr->fields, &pos, &key, &value)) {
    // In Python 3, the keys of numpy custom struct types are unicode, unlike
    // Python 2, where the keys are bytes.
    const char* key_string =
        PyBytes_Check(key) ? PyBytes_AsString(key)
                           : PyBytes_AsString(PyUnicode_AsASCIIString(key));
    if (!key_string) {
      return errors::Internal("Corrupt numpy type descriptor");
    }
    tensorflow::string key = key_string;
    // The typenames here should match the field names in the custom struct
    // types constructed in test_util.py.
    // TODO(mrry,keveman): Investigate Numpy type registration to replace this
    // hard-coding of names.
    if (key == "quint8") {
      *out_tf_datatype = TF_QUINT8;
    } else if (key == "qint8") {
      *out_tf_datatype = TF_QINT8;
    } else if (key == "qint16") {
      *out_tf_datatype = TF_QINT16;
    } else if (key == "quint16") {
      *out_tf_datatype = TF_QUINT16;
    } else if (key == "qint32") {
      *out_tf_datatype = TF_QINT32;
    } else if (key == "resource") {
      *out_tf_datatype = TF_RESOURCE;
    } else {
      return errors::Internal("Unsupported numpy data type");
    }
    return Status::OK();
  }
  return errors::Internal("Unsupported numpy data type");
}

Status PyArray_TYPE_to_TF_DataType(PyArrayObject* array,
                                   TF_DataType* out_tf_datatype) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_2(mht_2_v, 295, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "PyArray_TYPE_to_TF_DataType");

  int pyarray_type = PyArray_TYPE(array);
  PyArray_Descr* descr = PyArray_DESCR(array);
  switch (pyarray_type) {
    case NPY_FLOAT16:
      *out_tf_datatype = TF_HALF;
      break;
    case NPY_FLOAT32:
      *out_tf_datatype = TF_FLOAT;
      break;
    case NPY_FLOAT64:
      *out_tf_datatype = TF_DOUBLE;
      break;
    case NPY_INT32:
      *out_tf_datatype = TF_INT32;
      break;
    case NPY_UINT8:
      *out_tf_datatype = TF_UINT8;
      break;
    case NPY_UINT16:
      *out_tf_datatype = TF_UINT16;
      break;
    case NPY_UINT32:
      *out_tf_datatype = TF_UINT32;
      break;
    case NPY_UINT64:
      *out_tf_datatype = TF_UINT64;
      break;
    case NPY_INT8:
      *out_tf_datatype = TF_INT8;
      break;
    case NPY_INT16:
      *out_tf_datatype = TF_INT16;
      break;
    case NPY_INT64:
      *out_tf_datatype = TF_INT64;
      break;
    case NPY_BOOL:
      *out_tf_datatype = TF_BOOL;
      break;
    case NPY_COMPLEX64:
      *out_tf_datatype = TF_COMPLEX64;
      break;
    case NPY_COMPLEX128:
      *out_tf_datatype = TF_COMPLEX128;
      break;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      *out_tf_datatype = TF_STRING;
      break;
    case NPY_VOID:
      // Quantized types are currently represented as custom struct types.
      // PyArray_TYPE returns NPY_VOID for structs, and we should look into
      // descr to derive the actual type.
      // Direct feeds of certain types of ResourceHandles are represented as a
      // custom struct type.
      return PyArrayDescr_to_TF_DataType(descr, out_tf_datatype);
    default:
      if (pyarray_type == Bfloat16NumpyType()) {
        *out_tf_datatype = TF_BFLOAT16;
        break;
      } else if (pyarray_type == NPY_ULONGLONG) {
        // NPY_ULONGLONG is equivalent to NPY_UINT64, while their enum values
        // might be different on certain platforms.
        *out_tf_datatype = TF_UINT64;
        break;
      } else if (pyarray_type == NPY_LONGLONG) {
        // NPY_LONGLONG is equivalent to NPY_INT64, while their enum values
        // might be different on certain platforms.
        *out_tf_datatype = TF_INT64;
        break;
      } else if (pyarray_type == NPY_INT) {
        // NPY_INT is equivalent to NPY_INT32, while their enum values might be
        // different on certain platforms.
        *out_tf_datatype = TF_INT32;
        break;
      } else if (pyarray_type == NPY_UINT) {
        // NPY_UINT is equivalent to NPY_UINT32, while their enum values might
        // be different on certain platforms.
        *out_tf_datatype = TF_UINT32;
        break;
      }
      return errors::Internal("Unsupported numpy type: ",
                              numpy_type_name(pyarray_type));
  }
  return Status::OK();
}

Status PyObjectToString(PyObject* obj, const char** ptr, Py_ssize_t* len,
                        PyObject** ptr_owner) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_3(mht_3_v, 388, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "PyObjectToString");

  *ptr_owner = nullptr;
  if (PyBytes_Check(obj)) {
    char* buf;
    if (PyBytes_AsStringAndSize(obj, &buf, len) != 0) {
      return errors::Internal("Unable to get element as bytes.");
    }
    *ptr = buf;
    return Status::OK();
  } else if (PyUnicode_Check(obj)) {
#if (PY_MAJOR_VERSION > 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >= 3))
    *ptr = PyUnicode_AsUTF8AndSize(obj, len);
    if (*ptr != nullptr) return Status::OK();
#else
    PyObject* utemp = PyUnicode_AsUTF8String(obj);
    char* buf;
    if (utemp != nullptr && PyBytes_AsStringAndSize(utemp, &buf, len) != -1) {
      *ptr = buf;
      *ptr_owner = utemp;
      return Status::OK();
    }
    Py_XDECREF(utemp);
#endif
    return errors::Internal("Unable to convert element to UTF-8");
  } else {
    return errors::Internal("Unsupported object type ", obj->ob_type->tp_name);
  }
}

// Iterate over the string array 'array', extract the ptr and len of each string
// element and call f(ptr, len).
template <typename F>
Status PyBytesArrayMap(PyArrayObject* array, F f) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_4(mht_4_v, 423, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "PyBytesArrayMap");

  Safe_PyObjectPtr iter = tensorflow::make_safe(
      PyArray_IterNew(reinterpret_cast<PyObject*>(array)));
  while (PyArray_ITER_NOTDONE(iter.get())) {
    auto item = tensorflow::make_safe(PyArray_GETITEM(
        array, static_cast<char*>(PyArray_ITER_DATA(iter.get()))));
    if (!item) {
      return errors::Internal("Unable to get element from the feed - no item.");
    }
    Py_ssize_t len;
    const char* ptr;
    PyObject* ptr_owner = nullptr;
    TF_RETURN_IF_ERROR(PyObjectToString(item.get(), &ptr, &len, &ptr_owner));
    f(ptr, len);
    Py_XDECREF(ptr_owner);
    PyArray_ITER_NEXT(iter.get());
  }
  return Status::OK();
}

// Encode the strings in 'array' into a contiguous buffer and return the base of
// the buffer. The caller takes ownership of the buffer.
Status EncodePyBytesArray(PyArrayObject* array, int64_t nelems, size_t* size,
                          void** buffer) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_5(mht_5_v, 449, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "EncodePyBytesArray");

  // Encode all strings.
  *size = nelems * sizeof(tensorflow::tstring);
  std::unique_ptr<tensorflow::tstring[]> base_ptr(
      new tensorflow::tstring[nelems]);
  tensorflow::tstring* dst = base_ptr.get();

  TF_RETURN_IF_ERROR(
      PyBytesArrayMap(array, [&dst](const char* ptr, Py_ssize_t len) {
        dst->assign(ptr, len);
        dst++;
      }));
  *buffer = base_ptr.release();
  return Status::OK();
}

Status CopyTF_TensorStringsToPyArray(const TF_Tensor* src, uint64 nelems,
                                     PyArrayObject* dst) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_6(mht_6_v, 469, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "CopyTF_TensorStringsToPyArray");

  const void* tensor_data = TF_TensorData(src);
  DCHECK(tensor_data != nullptr);
  DCHECK_EQ(TF_STRING, TF_TensorType(src));

  const tstring* tstr = static_cast<const tstring*>(tensor_data);

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  auto iter = make_safe(PyArray_IterNew(reinterpret_cast<PyObject*>(dst)));
  for (int64_t i = 0; i < static_cast<int64_t>(nelems); ++i) {
    const tstring& tstr_i = tstr[i];
    auto py_string =
        make_safe(PyBytes_FromStringAndSize(tstr_i.data(), tstr_i.size()));
    if (py_string == nullptr) {
      return errors::Internal(
          "failed to create a python byte array when converting element #", i,
          " of a TF_STRING tensor to a numpy ndarray");
    }

    if (PyArray_SETITEM(dst, static_cast<char*>(PyArray_ITER_DATA(iter.get())),
                        py_string.get()) != 0) {
      return errors::Internal("Error settings element #", i,
                              " in the numpy ndarray");
    }
    PyArray_ITER_NEXT(iter.get());
  }
  return Status::OK();
}

// Determine the dimensions of a numpy ndarray to be created to represent an
// output Tensor.
Status GetPyArrayDimensionsForTensor(const TF_Tensor* tensor,
                                     gtl::InlinedVector<npy_intp, 4>* dims,
                                     int64_t* nelems) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_7(mht_7_v, 506, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "GetPyArrayDimensionsForTensor");

  dims->clear();
  const int ndims = TF_NumDims(tensor);
  if (TF_TensorType(tensor) == TF_RESOURCE) {
    if (ndims != 0) {
      return errors::InvalidArgument(
          "Fetching of non-scalar resource tensors is not supported.");
    }
    dims->push_back(TF_TensorByteSize(tensor));
    *nelems = dims->back();
  } else {
    *nelems = 1;
    for (int i = 0; i < ndims; ++i) {
      dims->push_back(TF_Dim(tensor, i));
      *nelems *= dims->back();
    }
  }
  return Status::OK();
}

// Determine the type description (PyArray_Descr) of a numpy ndarray to be
// created to represent an output Tensor.
Status GetPyArrayDescrForTensor(const TF_Tensor* tensor,
                                PyArray_Descr** descr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_8(mht_8_v, 532, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "GetPyArrayDescrForTensor");

  if (TF_TensorType(tensor) == TF_RESOURCE) {
    PyObject* field = PyTuple_New(3);
#if PY_MAJOR_VERSION < 3
    PyTuple_SetItem(field, 0, PyBytes_FromString("resource"));
#else
    PyTuple_SetItem(field, 0, PyUnicode_FromString("resource"));
#endif
    PyTuple_SetItem(field, 1, PyArray_TypeObjectFromType(NPY_UBYTE));
    PyTuple_SetItem(field, 2, PyLong_FromLong(1));
    PyObject* fields = PyList_New(1);
    PyList_SetItem(fields, 0, field);
    int convert_result = PyArray_DescrConverter(fields, descr);
    Py_CLEAR(fields);
    if (convert_result != 1) {
      return errors::Internal("Failed to create numpy array description for ",
                              "TF_RESOURCE-type tensor");
    }
  } else {
    int type_num = -1;
    TF_RETURN_IF_ERROR(
        TF_DataType_to_PyArray_TYPE(TF_TensorType(tensor), &type_num));
    *descr = PyArray_DescrFromType(type_num);
  }

  return Status::OK();
}

inline void FastMemcpy(void* dst, const void* src, size_t size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_9(mht_9_v, 563, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "FastMemcpy");

  // clang-format off
  switch (size) {
    // Most compilers will generate inline code for fixed sizes,
    // which is significantly faster for small copies.
    case  1: memcpy(dst, src, 1); break;
    case  2: memcpy(dst, src, 2); break;
    case  3: memcpy(dst, src, 3); break;
    case  4: memcpy(dst, src, 4); break;
    case  5: memcpy(dst, src, 5); break;
    case  6: memcpy(dst, src, 6); break;
    case  7: memcpy(dst, src, 7); break;
    case  8: memcpy(dst, src, 8); break;
    case  9: memcpy(dst, src, 9); break;
    case 10: memcpy(dst, src, 10); break;
    case 11: memcpy(dst, src, 11); break;
    case 12: memcpy(dst, src, 12); break;
    case 13: memcpy(dst, src, 13); break;
    case 14: memcpy(dst, src, 14); break;
    case 15: memcpy(dst, src, 15); break;
    case 16: memcpy(dst, src, 16); break;
#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_POSIX) && \
    !defined(IS_MOBILE_PLATFORM)
    // On Linux, memmove appears to be faster than memcpy for
    // large sizes, strangely enough.
    default: memmove(dst, src, size); break;
#else
    default: memcpy(dst, src, size); break;
#endif
  }
  // clang-format on
}

}  // namespace

// TODO(slebedev): revise TF_TensorToPyArray usages and switch to the
// aliased version where appropriate.
Status TF_TensorToMaybeAliasedPyArray(Safe_TF_TensorPtr tensor,
                                      PyObject** out_ndarray) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_10(mht_10_v, 604, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "TF_TensorToMaybeAliasedPyArray");

  auto dtype = TF_TensorType(tensor.get());
  if (dtype == TF_STRING || dtype == TF_RESOURCE) {
    return TF_TensorToPyArray(std::move(tensor), out_ndarray);
  }

  TF_Tensor* moved = tensor.release();
  int64_t nelems = -1;
  gtl::InlinedVector<npy_intp, 4> dims;
  TF_RETURN_IF_ERROR(GetPyArrayDimensionsForTensor(moved, &dims, &nelems));
  return ArrayFromMemory(
      dims.size(), dims.data(), TF_TensorData(moved),
      static_cast<DataType>(dtype), [moved] { TF_DeleteTensor(moved); },
      out_ndarray);
}

// Converts the given TF_Tensor to a numpy ndarray.
// If the returned status is OK, the caller becomes the owner of *out_array.
Status TF_TensorToPyArray(Safe_TF_TensorPtr tensor, PyObject** out_ndarray) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_11(mht_11_v, 625, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "TF_TensorToPyArray");

  // A fetched operation will correspond to a null tensor, and a None
  // in Python.
  if (tensor == nullptr) {
    Py_INCREF(Py_None);
    *out_ndarray = Py_None;
    return Status::OK();
  }
  int64_t nelems = -1;
  gtl::InlinedVector<npy_intp, 4> dims;
  TF_RETURN_IF_ERROR(
      GetPyArrayDimensionsForTensor(tensor.get(), &dims, &nelems));

  // If the type is neither string nor resource we can reuse the Tensor memory.
  TF_Tensor* original = tensor.get();
  TF_Tensor* moved = TF_TensorMaybeMove(tensor.release());
  if (moved != nullptr) {
    if (ArrayFromMemory(
            dims.size(), dims.data(), TF_TensorData(moved),
            static_cast<DataType>(TF_TensorType(moved)),
            [moved] { TF_DeleteTensor(moved); }, out_ndarray)
            .ok()) {
      return Status::OK();
    }
  }
  tensor.reset(original);

  // Copy the TF_TensorData into a newly-created ndarray and return it.
  PyArray_Descr* descr = nullptr;
  TF_RETURN_IF_ERROR(GetPyArrayDescrForTensor(tensor.get(), &descr));
  Safe_PyObjectPtr safe_out_array =
      tensorflow::make_safe(PyArray_Empty(dims.size(), dims.data(), descr, 0));
  if (!safe_out_array) {
    return errors::Internal("Could not allocate ndarray");
  }
  PyArrayObject* py_array =
      reinterpret_cast<PyArrayObject*>(safe_out_array.get());
  if (TF_TensorType(tensor.get()) == TF_STRING) {
    Status s = CopyTF_TensorStringsToPyArray(tensor.get(), nelems, py_array);
    if (!s.ok()) {
      return s;
    }
  } else if (static_cast<size_t>(PyArray_NBYTES(py_array)) !=
             TF_TensorByteSize(tensor.get())) {
    return errors::Internal("ndarray was ", PyArray_NBYTES(py_array),
                            " bytes but TF_Tensor was ",
                            TF_TensorByteSize(tensor.get()), " bytes");
  } else {
    FastMemcpy(PyArray_DATA(py_array), TF_TensorData(tensor.get()),
               PyArray_NBYTES(py_array));
  }

  *out_ndarray = safe_out_array.release();
  return Status::OK();
}

Status NdarrayToTensor(TFE_Context* ctx, PyObject* ndarray,
                       Safe_TF_TensorPtr* ret) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_12(mht_12_v, 685, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "NdarrayToTensor");

  DCHECK(ret != nullptr);

  // Make sure we dereference this array object in case of error, etc.
  Safe_PyObjectPtr array_safe(make_safe(
      PyArray_FromAny(ndarray, nullptr, 0, 0, NPY_ARRAY_CARRAY_RO, nullptr)));
  if (!array_safe) return errors::InvalidArgument("Not a ndarray.");
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  // Convert numpy dtype to TensorFlow dtype.
  TF_DataType dtype = TF_FLOAT;
  TF_RETURN_IF_ERROR(PyArray_TYPE_to_TF_DataType(array, &dtype));

  int64_t nelems = 1;
  gtl::InlinedVector<int64_t, 4> dims;
  for (int i = 0; i < PyArray_NDIM(array); ++i) {
    dims.push_back(PyArray_SHAPE(array)[i]);
    nelems *= dims[i];
  }

  // Create a TF_Tensor based on the fed data. In the case of non-string data
  // type, this steals a reference to array, which will be relinquished when
  // the underlying buffer is deallocated. For string, a new temporary buffer
  // is allocated into which the strings are encoded.
  if (dtype == TF_RESOURCE) {
    size_t size = PyArray_NBYTES(array);
    array_safe.release();

    if (ctx) {
      *ret = make_safe(new TF_Tensor{tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), {}, 0, PyArray_DATA(array),
          size, &DelayedNumpyDecref, array)});
    } else {
      *ret = make_safe(TF_NewTensor(dtype, {}, 0, PyArray_DATA(array), size,
                                    &DelayedNumpyDecref, array));
    }

  } else if (dtype != TF_STRING) {
    size_t size = PyArray_NBYTES(array);
    array_safe.release();
    if (ctx) {
      *ret = make_safe(new TF_Tensor{tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), dims.data(), dims.size(),
          PyArray_DATA(array), size, &DelayedNumpyDecref, array)});
    } else {
      *ret = make_safe(TF_NewTensor(dtype, dims.data(), dims.size(),
                                    PyArray_DATA(array), size,
                                    &DelayedNumpyDecref, array));
    }

  } else {
    size_t size = 0;
    void* encoded = nullptr;
    TF_RETURN_IF_ERROR(EncodePyBytesArray(array, nelems, &size, &encoded));
    if (ctx) {
      *ret = make_safe(new TF_Tensor{tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), dims.data(), dims.size(),
          encoded, size,
          [](void* data, size_t len, void* arg) {
            delete[] reinterpret_cast<tensorflow::tstring*>(data);
          },
          nullptr)});
    } else {
      *ret = make_safe(TF_NewTensor(
          dtype, dims.data(), dims.size(), encoded, size,
          [](void* data, size_t len, void* arg) {
            delete[] reinterpret_cast<tensorflow::tstring*>(data);
          },
          nullptr));
    }
  }

  return Status::OK();
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src, Status* status);

Status NdarrayToTensor(PyObject* obj, Tensor* ret) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_13(mht_13_v, 766, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "NdarrayToTensor");

  Safe_TF_TensorPtr tf_tensor = make_safe(static_cast<TF_Tensor*>(nullptr));
  Status s = NdarrayToTensor(nullptr /*ctx*/, obj, &tf_tensor);
  if (!s.ok()) {
    return s;
  }
  return TF_TensorToTensor(tf_tensor.get(), ret);
}

Status TensorToNdarray(const Tensor& t, PyObject** ret) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensorDTcc mht_14(mht_14_v, 778, "", "./tensorflow/python/lib/core/ndarray_tensor.cc", "TensorToNdarray");

  Status status;
  Safe_TF_TensorPtr tf_tensor = make_safe(TF_TensorFromTensor(t, &status));
  if (!status.ok()) {
    return status;
  }
  return TF_TensorToMaybeAliasedPyArray(std::move(tf_tensor), ret);
}

}  // namespace tensorflow
