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

#ifndef TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_
#define TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_
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
class MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh {
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
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh() {
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


// Place `<locale>` before <Python.h> to avoid build failure in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// Wrapper-class allowing to use Python hashing/comparison functions
// for PyObject*.
//
// Note that unlike Safe_PyObjectPtr this class does not steal a
// reference to a Python object. The caller is responsible for doing
// Py_INCREF/Py_DECREF.
struct PyObjectPtr {
  template <typename H>
  friend H AbslHashValue(H h, const PyObjectPtr& obj) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh mht_0(mht_0_v, 212, "", "./tensorflow/python/eager/pywrap_tensor_conversion.h", "AbslHashValue");

    return H::combine(std::move(h), PyObject_Hash(obj.ptr));
  }

  explicit PyObjectPtr(PyObject* ptr) : ptr(ptr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh mht_1(mht_1_v, 219, "", "./tensorflow/python/eager/pywrap_tensor_conversion.h", "PyObjectPtr");
}

  explicit inline operator PyObject*() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh mht_2(mht_2_v, 224, "", "./tensorflow/python/eager/pywrap_tensor_conversion.h", "*");
 return ptr; }

  inline bool operator==(const PyObjectPtr& other) const {
    // We require exact type equality to account for 0 == 0.0 == False.
    if (Py_TYPE(ptr) != Py_TYPE(other.ptr)) {
      return false;
    }

    bool result = PyObject_RichCompareBool(ptr, other.ptr, Py_EQ) > 0;
    CHECK(!PyErr_Occurred());
    return result;
  }

 private:
  PyObject* ptr;
};

// Cache mapping PyObject* to the corresponding on-device TFE_TensorHandles.
// Used to speed up ConvertToEagerTensor for scalars.
// TODO(slebedev): move ConvertToEagerTensor here.
struct TFE_TensorHandleCache {
  static TFE_TensorHandleCache* Get();

  TFE_TensorHandleCache() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh mht_3(mht_3_v, 250, "", "./tensorflow/python/eager/pywrap_tensor_conversion.h", "TFE_TensorHandleCache");
 cache.reserve(64); }
  ~TFE_TensorHandleCache() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh mht_4(mht_4_v, 254, "", "./tensorflow/python/eager/pywrap_tensor_conversion.h", "~TFE_TensorHandleCache");
 DecrefUnrefAll(); }

  TFE_TensorHandle* Lookup(PyObject* value, tensorflow::DataType dtype,
                           TFE_Context* ctx,
                           absl::string_view device_name) const;

  void Insert(PyObject* value, tensorflow::DataType dtype, TFE_Context* ctx,
              absl::string_view device_name, TFE_TensorHandle* h);

  void Clear();

 private:
  // TODO(kkb): Instead of `TFE_Context*` key, ideally Python's context object
  // should have TFE_TensorHandleCache instance. Migrate once we Python context
  // object is backed by C++ data structure. b/169790439
  using Key = std::tuple<PyObjectPtr, tensorflow::DataType, TFE_Context*,
                         absl::string_view>;

  void DecrefUnrefAll() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_tensor_conversionDTh mht_5(mht_5_v, 275, "", "./tensorflow/python/eager/pywrap_tensor_conversion.h", "DecrefUnrefAll");

    for (const auto& p : cache) {
      Py_DECREF(static_cast<PyObject*>(std::get<0>(p.first)));
      TFE_DeleteTensorHandle(p.second);
    }
  }

  // Not guarded by a mutex because the code is only used while the
  // GIL is held.
  absl::flat_hash_map<Key, TFE_TensorHandle*> cache;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_
