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

// Helpers for converting Python values into buffers.

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_VALUES_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_VALUES_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh() {
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


#include <memory>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"

namespace xla {

struct DevicePutResult {
  explicit DevicePutResult(PjRtBuffer* b, bool weak_type,
                           pybind11::object owning_pybuffer)
      : buffer(b), weak_type(weak_type), owning_pybuffer(owning_pybuffer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh mht_0(mht_0_v, 202, "", "./tensorflow/compiler/xla/python/py_values.h", "DevicePutResult");
}
  explicit DevicePutResult(std::unique_ptr<PjRtBuffer> new_buffer,
                           bool weak_type)
      : buffer(new_buffer.get()),
        weak_type(weak_type),
        owned_buffer(std::move(new_buffer)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/python/py_values.h", "DevicePutResult");
}

  // Points to the on-device buffer. Not owned.
  PjRtBuffer* buffer;
  bool weak_type;

  // One of owned_buffer or owning_pybuffer is valid. If owned_buffer is
  // non-null, it holds ownership of the buffer. Otherwise owning_pybuffer is
  // the PyBuffer object that owns the buffer.
  std::unique_ptr<PjRtBuffer> owned_buffer;
  pybind11::object owning_pybuffer;
};

// Copies a buffer-like object to be on device.
//
// If `arg` is not convertible to a `PjRtBuffer` from C++, an error will be
// returned; float0s are not supported yet.
// If the value is known to be a PyBuffer object, py_buffer can be passed as
// an optimization to avoid a Python->C++ cast.
//
// May throw exceptions from pybind11 in addition to failing via an error
// Status. (We could catch these if needed, but there seems little point.)
struct DevicePutOptions {
  bool squash_64bit_types = false;
  bool allow_zero_copy = true;
};
StatusOr<DevicePutResult> DevicePut(pybind11::handle arg, PjRtDevice* to_device,
                                    const DevicePutOptions& options);

// Returns `true` if `arg` is a JAX float0 array.
bool IsFloat0(pybind11::array arg);

// Describes the abstract shape and dtype of an argument.
struct PyArgSignature {
  PyArgSignature(PrimitiveType dtype, absl::Span<const int64_t> shape,
                 bool weak_type)
      : dtype(dtype), shape(shape.begin(), shape.end()), weak_type(weak_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh mht_2(mht_2_v, 249, "", "./tensorflow/compiler/xla/python/py_values.h", "PyArgSignature");
}
  // This is the XLA dtype of the object.
  const PrimitiveType dtype;
  const absl::InlinedVector<int64_t, 4> shape;
  // JAX arguments can be of weak type, if and only if they are Python scalars
  // or `DeviceArray` values such that `aval.weak_type` is true.
  const bool weak_type;
  bool operator==(const PyArgSignature& other) const {
    return std::tie(dtype, weak_type, shape) ==
           std::tie(other.dtype, other.weak_type, other.shape);
  }
  bool operator!=(const PyArgSignature& other) const {
    return !(*this == other);
  }
  std::string DebugString() const;
};

// Returns the PyArgSignature associated with an argument. Returns an error if
// the argument is not supported.
StatusOr<PyArgSignature> PyArgSignatureOfValue(pybind11::handle arg,
                                               bool jax_enable_x64);

template <typename H>
H AbslHashValue(H h, const xla::PyArgSignature& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_valuesDTh mht_3(mht_3_v, 275, "", "./tensorflow/compiler/xla/python/py_values.h", "AbslHashValue");

  h = H::combine(std::move(h), s.dtype);
  h = H::combine_contiguous(std::move(h), s.shape.data(), s.shape.size());
  return h;
}
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_VALUES_H_
