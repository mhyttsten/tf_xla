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

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
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
class MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh {
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
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh() {
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


#include <Python.h>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"

namespace tensorflow {

namespace internal {

inline PyObject* CodeToPyExc(const int code) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_0(mht_0_v, 203, "", "./tensorflow/python/lib/core/pybind11_status.h", "CodeToPyExc");

  switch (code) {
    case error::Code::INVALID_ARGUMENT:
      return PyExc_ValueError;
    case error::Code::OUT_OF_RANGE:
      return PyExc_IndexError;
    case error::Code::UNIMPLEMENTED:
      return PyExc_NotImplementedError;
    default:
      return PyExc_RuntimeError;
  }
}

inline PyObject* StatusToPyExc(const Status& status) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_1(mht_1_v, 219, "", "./tensorflow/python/lib/core/pybind11_status.h", "StatusToPyExc");

  return CodeToPyExc(status.code());
}

inline PyObject* TFStatusToPyExc(const TF_Status* status) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_2(mht_2_v, 226, "", "./tensorflow/python/lib/core/pybind11_status.h", "TFStatusToPyExc");

  return CodeToPyExc(TF_GetCode(status));
}

inline pybind11::dict StatusPayloadToDict(const Status& status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_3(mht_3_v, 233, "", "./tensorflow/python/lib/core/pybind11_status.h", "StatusPayloadToDict");

  pybind11::dict dict;
  const auto& payloads = errors::GetPayloads(status);
  for (auto& pair : payloads) {
    dict[PyBytes_FromString(pair.first.c_str())] =
        PyBytes_FromString(pair.second.c_str());
  }
  return dict;
}

inline pybind11::dict TFStatusPayloadToDict(TF_Status* status) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_4(mht_4_v, 246, "", "./tensorflow/python/lib/core/pybind11_status.h", "TFStatusPayloadToDict");

  return StatusPayloadToDict(status->status);
}

}  // namespace internal

inline void MaybeRaiseFromStatus(const Status& status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_5(mht_5_v, 255, "", "./tensorflow/python/lib/core/pybind11_status.h", "MaybeRaiseFromStatus");

  if (!status.ok()) {
    PyErr_SetString(internal::StatusToPyExc(status),
                    status.error_message().c_str());
    throw pybind11::error_already_set();
  }
}

inline void SetRegisteredErrFromStatus(const tensorflow::Status& status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_6(mht_6_v, 266, "", "./tensorflow/python/lib/core/pybind11_status.h", "SetRegisteredErrFromStatus");

  PyErr_SetObject(PyExceptionRegistry::Lookup(status.code()),
                  pybind11::make_tuple(pybind11::none(), pybind11::none(),
                                       status.error_message(),
                                       internal::StatusPayloadToDict(status))
                      .ptr());
}

inline void SetRegisteredErrFromTFStatus(TF_Status* status) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_7(mht_7_v, 277, "", "./tensorflow/python/lib/core/pybind11_status.h", "SetRegisteredErrFromTFStatus");

  PyErr_SetObject(PyExceptionRegistry::Lookup(TF_GetCode(status)),
                  pybind11::make_tuple(pybind11::none(), pybind11::none(),
                                       TF_Message(status),
                                       internal::TFStatusPayloadToDict(status))
                      .ptr());
}

inline void MaybeRaiseRegisteredFromStatus(const tensorflow::Status& status) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_8(mht_8_v, 288, "", "./tensorflow/python/lib/core/pybind11_status.h", "MaybeRaiseRegisteredFromStatus");

  if (!status.ok()) {
    SetRegisteredErrFromStatus(status);
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseRegisteredFromStatusWithGIL(
    const tensorflow::Status& status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_9(mht_9_v, 299, "", "./tensorflow/python/lib/core/pybind11_status.h", "MaybeRaiseRegisteredFromStatusWithGIL");

  if (!status.ok()) {
    // Acquire GIL for throwing exception.
    pybind11::gil_scoped_acquire acquire;
    SetRegisteredErrFromStatus(status);
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseFromTFStatus(TF_Status* status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_10(mht_10_v, 311, "", "./tensorflow/python/lib/core/pybind11_status.h", "MaybeRaiseFromTFStatus");

  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    PyErr_SetString(internal::TFStatusToPyExc(status), TF_Message(status));
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseRegisteredFromTFStatus(TF_Status* status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_11(mht_11_v, 322, "", "./tensorflow/python/lib/core/pybind11_status.h", "MaybeRaiseRegisteredFromTFStatus");

  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    SetRegisteredErrFromTFStatus(status);
    throw pybind11::error_already_set();
  }
}

inline void MaybeRaiseRegisteredFromTFStatusWithGIL(TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_12(mht_12_v, 333, "", "./tensorflow/python/lib/core/pybind11_status.h", "MaybeRaiseRegisteredFromTFStatusWithGIL");

  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    // Acquire GIL for throwing exception.
    pybind11::gil_scoped_acquire acquire;
    SetRegisteredErrFromTFStatus(status);
    throw pybind11::error_already_set();
  }
}

}  // namespace tensorflow

namespace pybind11 {
namespace detail {

// Convert tensorflow::Status
//
// Raise an exception if a given status is not OK, otherwise return None.
//
// The correspondence between status codes and exception classes is given
// by PyExceptionRegistry. Note that the registry should be initialized
// in order to be used, see PyExceptionRegistry::Init.
template <>
struct type_caster<tensorflow::Status> {
 public:
  PYBIND11_TYPE_CASTER(tensorflow::Status, _("Status"));
  static handle cast(tensorflow::Status status, return_value_policy, handle) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_13(mht_13_v, 362, "", "./tensorflow/python/lib/core/pybind11_status.h", "cast");

    tensorflow::MaybeRaiseFromStatus(status);
    return none().inc_ref();
  }
};

// Convert tensorflow::StatusOr
//
// Uses the same logic as the Abseil implementation: raise an exception if the
// status is not OK, otherwise return its payload.
template <typename PayloadType>
struct type_caster<tensorflow::StatusOr<PayloadType>> {
 public:
  using PayloadCaster = make_caster<PayloadType>;
  using StatusCaster = make_caster<tensorflow::Status>;
  static constexpr auto name = PayloadCaster::name;

  static handle cast(const tensorflow::StatusOr<PayloadType>* src,
                     return_value_policy policy, handle parent) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_14(mht_14_v, 383, "", "./tensorflow/python/lib/core/pybind11_status.h", "cast");

    if (!src) return none().release();
    return cast_impl(*src, policy, parent);
  }

  static handle cast(const tensorflow::StatusOr<PayloadType>& src,
                     return_value_policy policy, handle parent) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_15(mht_15_v, 392, "", "./tensorflow/python/lib/core/pybind11_status.h", "cast");

    return cast_impl(src, policy, parent);
  }

  static handle cast(tensorflow::StatusOr<PayloadType>&& src,
                     return_value_policy policy, handle parent) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_16(mht_16_v, 400, "", "./tensorflow/python/lib/core/pybind11_status.h", "cast");

    return cast_impl(std::move(src), policy, parent);
  }

 private:
  template <typename CType>
  static handle cast_impl(CType&& src, return_value_policy policy,
                          handle parent) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSpythonPSlibPScorePSpybind11_statusDTh mht_17(mht_17_v, 410, "", "./tensorflow/python/lib/core/pybind11_status.h", "cast_impl");

    if (src.ok()) {
      // Convert and return the payload.
      return PayloadCaster::cast(std::forward<CType>(src).ValueOrDie(), policy,
                                 parent);
    } else {
      // Convert and return the error.
      return StatusCaster::cast(std::forward<CType>(src).status(),
                                return_value_policy::move, parent);
    }
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_STATUS_H_
