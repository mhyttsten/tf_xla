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
class MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc() {
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

// Must be included first.
#include "tensorflow/python/lib/core/numpy.h"

#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/bfloat16.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"

namespace tensorflow {

// Mutex used to serialize accesses to cached vector of pointers to python
// arrays to be dereferenced.
static mutex* DelayedDecrefLock() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_0(mht_0_v, 200, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "DelayedDecrefLock");

  static mutex* decref_lock = new mutex;
  return decref_lock;
}

// Caches pointers to numpy arrays which need to be dereferenced.
static std::vector<void*>* DecrefCache() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_1(mht_1_v, 209, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "DecrefCache");

  static std::vector<void*>* decref_cache = new std::vector<void*>;
  return decref_cache;
}

// Destructor passed to TF_NewTensor when it reuses a numpy buffer. Stores a
// pointer to the pyobj in a buffer to be dereferenced later when we're actually
// holding the GIL.
void DelayedNumpyDecref(void* data, size_t len, void* obj) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_2(mht_2_v, 220, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "DelayedNumpyDecref");

  mutex_lock ml(*DelayedDecrefLock());
  DecrefCache()->push_back(obj);
}

// Actually dereferences cached numpy arrays. REQUIRES being called while
// holding the GIL.
void ClearDecrefCache() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_3(mht_3_v, 230, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "ClearDecrefCache");

  std::vector<void*> cache_copy;
  {
    mutex_lock ml(*DelayedDecrefLock());
    cache_copy.swap(*DecrefCache());
  }
  for (void* obj : cache_copy) {
    Py_DECREF(reinterpret_cast<PyObject*>(obj));
  }
}

// Structure which keeps a reference to a Tensor alive while numpy has a pointer
// to it.
struct TensorReleaser {
  // Python macro to include standard members.
  PyObject_HEAD

      // Destructor responsible for releasing the memory.
      std::function<void()>* destructor;
};

extern PyTypeObject TensorReleaserType;

static void TensorReleaser_dealloc(PyObject* pself) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_4(mht_4_v, 256, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "TensorReleaser_dealloc");

  TensorReleaser* self = reinterpret_cast<TensorReleaser*>(pself);
  (*self->destructor)();
  delete self->destructor;
  TensorReleaserType.tp_free(pself);
}

// clang-format off
PyTypeObject TensorReleaserType = {
    PyVarObject_HEAD_INIT(nullptr, 0) /* head init */
    "tensorflow_wrapper",             /* tp_name */
    sizeof(TensorReleaser),           /* tp_basicsize */
    0,                                /* tp_itemsize */
    /* methods */
    TensorReleaser_dealloc,      /* tp_dealloc */
#if PY_VERSION_HEX < 0x03080000
    nullptr,                     /* tp_print */
#else
    0,                           /* tp_vectorcall_offset */
#endif
    nullptr,                     /* tp_getattr */
    nullptr,                     /* tp_setattr */
    nullptr,                     /* tp_compare */
    nullptr,                     /* tp_repr */
    nullptr,                     /* tp_as_number */
    nullptr,                     /* tp_as_sequence */
    nullptr,                     /* tp_as_mapping */
    nullptr,                     /* tp_hash */
    nullptr,                     /* tp_call */
    nullptr,                     /* tp_str */
    nullptr,                     /* tp_getattro */
    nullptr,                     /* tp_setattro */
    nullptr,                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,          /* tp_flags */
    "Wrapped TensorFlow Tensor", /* tp_doc */
    nullptr,                     /* tp_traverse */
    nullptr,                     /* tp_clear */
    nullptr,                     /* tp_richcompare */
};
// clang-format on

Status TF_DataType_to_PyArray_TYPE(TF_DataType tf_datatype,
                                   int* out_pyarray_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_5(mht_5_v, 301, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "TF_DataType_to_PyArray_TYPE");

  switch (tf_datatype) {
    case TF_HALF:
      *out_pyarray_type = NPY_FLOAT16;
      break;
    case TF_FLOAT:
      *out_pyarray_type = NPY_FLOAT32;
      break;
    case TF_DOUBLE:
      *out_pyarray_type = NPY_FLOAT64;
      break;
    case TF_INT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_UINT32:
      *out_pyarray_type = NPY_UINT32;
      break;
    case TF_UINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_UINT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    case TF_INT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_INT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_INT64:
      *out_pyarray_type = NPY_INT64;
      break;
    case TF_UINT64:
      *out_pyarray_type = NPY_UINT64;
      break;
    case TF_BOOL:
      *out_pyarray_type = NPY_BOOL;
      break;
    case TF_COMPLEX64:
      *out_pyarray_type = NPY_COMPLEX64;
      break;
    case TF_COMPLEX128:
      *out_pyarray_type = NPY_COMPLEX128;
      break;
    case TF_STRING:
      *out_pyarray_type = NPY_OBJECT;
      break;
    case TF_RESOURCE:
      *out_pyarray_type = NPY_VOID;
      break;
    // TODO(keveman): These should be changed to NPY_VOID, and the type used for
    // the resulting numpy array should be the custom struct types that we
    // expect for quantized types.
    case TF_QINT8:
      *out_pyarray_type = NPY_INT8;
      break;
    case TF_QUINT8:
      *out_pyarray_type = NPY_UINT8;
      break;
    case TF_QINT16:
      *out_pyarray_type = NPY_INT16;
      break;
    case TF_QUINT16:
      *out_pyarray_type = NPY_UINT16;
      break;
    case TF_QINT32:
      *out_pyarray_type = NPY_INT32;
      break;
    case TF_BFLOAT16:
      *out_pyarray_type = Bfloat16NumpyType();
      break;
    default:
      return errors::Internal("Tensorflow type ", tf_datatype,
                              " not convertible to numpy dtype.");
  }
  return Status::OK();
}

Status ArrayFromMemory(int dim_size, npy_intp* dims, void* data, DataType dtype,
                       std::function<void()> destructor, PyObject** result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSndarray_tensor_bridgeDTcc mht_6(mht_6_v, 383, "", "./tensorflow/python/lib/core/ndarray_tensor_bridge.cc", "ArrayFromMemory");

  if (dtype == DT_STRING || dtype == DT_RESOURCE) {
    return errors::FailedPrecondition(
        "Cannot convert string or resource Tensors.");
  }

  int type_num = -1;
  Status s =
      TF_DataType_to_PyArray_TYPE(static_cast<TF_DataType>(dtype), &type_num);
  if (!s.ok()) {
    return s;
  }

  auto* np_array = reinterpret_cast<PyArrayObject*>(
      PyArray_SimpleNewFromData(dim_size, dims, type_num, data));
  PyArray_CLEARFLAGS(np_array, NPY_ARRAY_OWNDATA);
  if (PyType_Ready(&TensorReleaserType) == -1) {
    return errors::Unknown("Python type initialization failed.");
  }
  auto* releaser = reinterpret_cast<TensorReleaser*>(
      TensorReleaserType.tp_alloc(&TensorReleaserType, 0));
  releaser->destructor = new std::function<void()>(std::move(destructor));
  if (PyArray_SetBaseObject(np_array, reinterpret_cast<PyObject*>(releaser)) ==
      -1) {
    Py_DECREF(releaser);
    return errors::Unknown("Python array refused to use memory.");
  }
  *result = reinterpret_cast<PyObject*>(np_array);
  return Status::OK();
}

}  // namespace tensorflow
