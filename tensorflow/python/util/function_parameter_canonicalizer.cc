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
class MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc() {
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

#include "tensorflow/python/util/function_parameter_canonicalizer.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace {
inline const char* PyUnicodeAsUtf8Compat(PyObject* obj) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_0(mht_0_v, 194, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "PyUnicodeAsUtf8Compat");

#if PY_MAJOR_VERSION < 3
  return PyString_AS_STRING(obj);
#else
  return PyUnicode_AsUTF8(obj);
#endif
}

inline PyObject* PyUnicodeInternFromStringCompat(const char* str) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("str: \"" + (str == nullptr ? std::string("nullptr") : std::string((char*)str)) + "\"");
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_1(mht_1_v, 206, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "PyUnicodeInternFromStringCompat");

#if PY_MAJOR_VERSION < 3
  return PyString_InternFromString(str);
#else
  return PyUnicode_InternFromString(str);
#endif
}

inline void PyUnicodeInternInPlaceCompat(PyObject** obj) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_2(mht_2_v, 217, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "PyUnicodeInternInPlaceCompat");

#if PY_MAJOR_VERSION < 3
  PyString_InternInPlace(obj);
#else
  PyUnicode_InternInPlace(obj);
#endif
}

}  // namespace

namespace tensorflow {

FunctionParameterCanonicalizer::FunctionParameterCanonicalizer(
    absl::Span<const char*> arg_names, absl::Span<PyObject*> defaults)
    : positional_args_size_(arg_names.size() - defaults.size()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_3(mht_3_v, 234, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "FunctionParameterCanonicalizer::FunctionParameterCanonicalizer");

  DCheckPyGilState();
  DCHECK_GE(positional_args_size_, 0);

  interned_arg_names_.reserve(arg_names.size());
  for (const char* obj : arg_names)
    interned_arg_names_.emplace_back(PyUnicodeInternFromStringCompat(obj));

  DCHECK(AreInternedArgNamesUnique());

  for (PyObject* obj : defaults) Py_INCREF(obj);
  defaults_ = std::vector<Safe_PyObjectPtr>(defaults.begin(), defaults.end());
}

bool FunctionParameterCanonicalizer::Canonicalize(
    PyObject* args, PyObject* kwargs, absl::Span<PyObject*> result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_4(mht_4_v, 252, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "FunctionParameterCanonicalizer::Canonicalize");

  // TODO(kkb): Closely follow `Python/ceval.c`'s logic and error handling.

  DCheckPyGilState();
  DCHECK(PyTuple_CheckExact(args));
  DCHECK(kwargs == nullptr || PyDict_CheckExact(kwargs));
  DCHECK_EQ(result.size(), interned_arg_names_.size());

  const int args_size = Py_SIZE(args);
  int remaining_positional_args_count = positional_args_size_ - args_size;

  // Check if the number of input arguments are too many.
  if (TF_PREDICT_FALSE(args_size > interned_arg_names_.size())) {
    PyErr_SetString(
        PyExc_TypeError,
        absl::StrCat("Too many arguments were given. Expected ",
                     interned_arg_names_.size(), " but got ", args_size, ".")
            .c_str());
    return false;
  }

  // Fill positional arguments.
  for (int i = 0; i < args_size; ++i) result[i] = PyTuple_GET_ITEM(args, i);

  // Fill default arguments.
  for (int i = std::max(positional_args_size_, args_size);
       i < interned_arg_names_.size(); ++i)
    result[i] = defaults_[i - positional_args_size_].get();

  // Fill keyword arguments.
  if (kwargs != nullptr) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      std::size_t index = InternedArgNameLinearSearch(key);

      // Check if key object(argument name) was found in the pre-built intern
      // string table.
      if (TF_PREDICT_FALSE(index == interned_arg_names_.size())) {
        // `key` might not be an interend string, so get the interned string
        // and try again.  Note: we need to call INCREF before we use
        // InternInPlace, to prevent the key in the dictionary from being
        // prematurely deleted in the case where InternInPlace switches `key`
        // to point at a new object.  We call DECREF(key) once we're done
        // (which might decref the original key *or* the interned version).
        Py_INCREF(key);
        PyUnicodeInternInPlaceCompat(&key);
        index = InternedArgNameLinearSearch(key);
        Py_DECREF(key);

        // Stil not found, then return an error.
        if (TF_PREDICT_FALSE(index == interned_arg_names_.size())) {
          PyErr_Format(PyExc_TypeError,
                       "Got an unexpected keyword argument '%s'",
                       PyUnicodeAsUtf8Compat(key));
          return false;
        }
      }

      // Check if the keyword argument overlaps with positional arguments.
      if (TF_PREDICT_FALSE(index < args_size)) {
        PyErr_Format(PyExc_TypeError, "Got multiple values for argument '%s'",
                     PyUnicodeAsUtf8Compat(key));
        return false;
      }

      if (TF_PREDICT_FALSE(index < positional_args_size_))
        --remaining_positional_args_count;

      result[index] = value;
    }
  }

  // Check if all the arguments are filled.
  // Example failure, not enough number of arguments passed: `matmul(x)`
  if (TF_PREDICT_FALSE(remaining_positional_args_count > 0)) {
    // TODO(kkb): Report what arguments are missing.
    PyErr_SetString(PyExc_TypeError, "Missing required positional argument");
    return false;
  }

  return true;
}

ABSL_MUST_USE_RESULT
ABSL_ATTRIBUTE_HOT
inline std::size_t FunctionParameterCanonicalizer::InternedArgNameLinearSearch(
    PyObject* name) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_5(mht_5_v, 342, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "FunctionParameterCanonicalizer::InternedArgNameLinearSearch");

  std::size_t result = interned_arg_names_.size();

  for (std::size_t i = 0; i < interned_arg_names_.size(); ++i)
    if (TF_PREDICT_FALSE(name == interned_arg_names_[i].get())) return i;

  return result;
}

bool FunctionParameterCanonicalizer::AreInternedArgNamesUnique() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSutilPSfunction_parameter_canonicalizerDTcc mht_6(mht_6_v, 354, "", "./tensorflow/python/util/function_parameter_canonicalizer.cc", "FunctionParameterCanonicalizer::AreInternedArgNamesUnique");

  absl::flat_hash_set<PyObject*> interned_arg_names_set;
  for (const Safe_PyObjectPtr& obj : interned_arg_names_)
    interned_arg_names_set.emplace(obj.get());

  return interned_arg_names_set.size() == interned_arg_names_.size();
}
}  // namespace tensorflow
