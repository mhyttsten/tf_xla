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
class MHTracer_DTPStensorflowPSpythonPSutilPSnestDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSutilPSnestDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPSnestDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/util/nest.h"

#include <utility>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace tensorflow {

namespace {

// Gets a string representation of the input object.
//
// Args:
//   o: a python object.
//   length: If set to negative, the whole string is returned. Otherwise, the
//       string gets clipped to 'length' in size.
//
// Returns:
//   A string representation.
std::string PyObject_ToString(PyObject* o, int length = -1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSnestDTcc mht_0(mht_0_v, 207, "", "./tensorflow/python/util/nest.cc", "PyObject_ToString");

  auto str_o = make_safe(PyObject_Str(o));
  std::string str = PyUnicode_AsUTF8(str_o.get());
  if (length < 0 || str.size() <= length) {
    return str;
  }
  tensorflow::StringPiece str_piece(str);
  return tensorflow::strings::StrCat(str_piece.substr(length), "...");
}

// Gets a list of keys from a dict or mapping type object.
//
// Args:
//   o: a dictionary or mapping type object.
//
// Returns:
//   A new reference to a list.
//
// Raises:
//   TypeError: if `o` is not a dict or mapping type object.
PyObject* GetKeysFromDictOrMapping(PyObject* o) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSnestDTcc mht_1(mht_1_v, 230, "", "./tensorflow/python/util/nest.cc", "GetKeysFromDictOrMapping");

  if (PyDict_Check(o)) {
    return PyDict_Keys(o);
  } else if (PyMapping_Check(o)) {
    return PyMapping_Keys(o);
  } else {
    auto* o_type = Py_TYPE(o);
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a type compatible with dict or mapping, got '",
            o_type->tp_name, "'")
            .c_str());
    return nullptr;
  }
}

}  // namespace

PyObject* FlattenDictItems(PyObject* dict) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSnestDTcc mht_2(mht_2_v, 252, "", "./tensorflow/python/util/nest.cc", "FlattenDictItems");

  if (!PyDict_Check(dict) && !swig::IsMapping(dict)) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "FlattenDictItems: 'dict' must be a dictionary or ",
                        "collection.Mapping type object, instead of '",
                        Py_TYPE(dict)->tp_name, "'.")
                        .c_str());
    return nullptr;
  }
  PyObject* flat_dictionary = PyDict_New();
  auto keys = make_safe(GetKeysFromDictOrMapping(dict));
  for (size_t i = 0; i < PyList_Size(keys.get()); ++i) {
    auto* key = PyList_GetItem(keys.get(), i);
    // We use a general approach in case 'dict' is a PyMapping type,
    // but not a PyDict type.
    auto* value = PyObject_GetItem(dict, key);
    if (swig::IsNested(key)) {
      // The dict might contain list - list pairs.
      auto flat_keys = make_safe(swig::Flatten(key, false));
      auto flat_values = make_safe(swig::Flatten(value, false));
      size_t flat_keys_sz = PyList_Size(flat_keys.get());
      size_t flat_values_sz = PyList_Size(flat_values.get());
      if (flat_keys_sz != flat_values_sz) {
        PyErr_SetString(
            PyExc_ValueError,
            tensorflow::strings::StrCat(
                "Could not flatten dictionary. Key had ", flat_keys_sz,
                " elements, but value had ", flat_values_sz,
                " elements. Key: ", PyObject_ToString(flat_keys.get()),
                ", value: ", PyObject_ToString(flat_values.get()), ".")
                .c_str());
        Py_DecRef(flat_dictionary);
        return nullptr;
      }
      for (size_t i = 0; i < flat_keys_sz; ++i) {
        auto* flat_key = PyList_GetItem(flat_keys.get(), i);
        auto* flat_value = PyList_GetItem(flat_values.get(), i);
        if (PyDict_GetItem(flat_dictionary, flat_key) != nullptr) {
          PyErr_SetString(
              PyExc_ValueError,
              tensorflow::strings::StrCat(
                  "Cannot flatten dict because this key is not unique: ",
                  PyObject_ToString(flat_key))
                  .c_str());
          Py_DecRef(flat_dictionary);
          return nullptr;
        }
        PyDict_SetItem(flat_dictionary, flat_key, flat_value);
      }
    } else {
      if (PyDict_GetItem(flat_dictionary, key) != nullptr) {
        PyErr_SetString(
            PyExc_ValueError,
            tensorflow::strings::StrCat(
                "Cannot flatten dict because this key is not unique: ",
                PyObject_ToString(key))
                .c_str());
        Py_DecRef(flat_dictionary);
        return nullptr;
      }
      PyDict_SetItem(flat_dictionary, key, value);
    }
    // Manually decrease because PyObject_GetItem() returns a new reference.
    Py_DECREF(value);
  }
  return flat_dictionary;
}

}  // namespace tensorflow
