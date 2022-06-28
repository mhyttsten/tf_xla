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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh() {
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
#include <stdexcept>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Python wrapper around PjRtBuffer. We use a wrapper class:
// a) to keep the PjRtClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
//
// A `PyBuffer` can be used from Python without being wrapped in a Python
// `DeviceArray` object.
class PyBuffer {
 public:
  // pybind11::object typed subclass for PyBuffer objects.
  class pyobject : public pybind11::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    pybind11::object, PyBuffer::IsPyBuffer);
    pyobject() = default;
    PyBuffer* buf() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/python/py_buffer.h", "buf");
 return PyBuffer::AsPyBufferUnchecked(*this); }
  };
  using object = pyobject;

  static object Make(std::shared_ptr<PyClient> client,
                     std::shared_ptr<PjRtBuffer> buffer,
                     std::shared_ptr<Traceback> traceback);

  // Returns true if `h` is a PyBuffer.
  static bool IsPyBuffer(pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Does not do any checking.
  static PyBuffer* AsPyBufferUnchecked(pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Returns an error status if
  // !IsPyBuffer(handle)
  static StatusOr<PyBuffer*> AsPyBuffer(pybind11::handle handle);

  // Gets a Python handle to an existing PyBuffer. Assumes the PyObject was
  // allocated on the Python heap, which is the case if Make() was used.
  pybind11::handle AsHandle();

  ~PyBuffer();

  std::shared_ptr<PyClient> client() const { return client_; }
  PjRtBuffer* buffer() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_1(mht_1_v, 244, "", "./tensorflow/compiler/xla/python/py_buffer.h", "buffer");
 return buffer_.get(); }
  std::shared_ptr<PjRtBuffer> shared_ptr_buffer() const { return buffer_; }

  ClientAndPtr<PjRtDevice> device() const;
  absl::string_view platform_name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/python/py_buffer.h", "platform_name");

    return buffer_->client()->platform_name();
  }
  bool is_deleted() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_3(mht_3_v, 257, "", "./tensorflow/compiler/xla/python/py_buffer.h", "is_deleted");
 return buffer_->IsDeleted(); }

  StatusOr<pybind11::object> CopyToDevice(
      const ClientAndPtr<PjRtDevice>& dst_device) const;

  StatusOr<size_t> OnDeviceSizeInBytes() {
    return buffer_->GetOnDeviceSizeInBytes();
  }

  void Delete() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_4(mht_4_v, 269, "", "./tensorflow/compiler/xla/python/py_buffer.h", "Delete");

    buffer_->Delete();
    host_value_ = nullptr;
  }

  // Makes a copy of this PyBuffer object that shares the underlying PjRtBuffer.
  // This is useful because we may wish to change JAX metadata (e.g., the sticky
  // device) without copying the buffer.
  object Clone() const;

  // Returns xla::InvalidArgument if the buffer has been deleted.
  // See `PjRtFuture` for the semantics of `IsReady` and `IsKnownReady`.
  StatusOr<bool> IsReady() {
    if (buffer_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return buffer_->GetReadyFuture().IsReady();
  }
  StatusOr<bool> IsKnownReady() {
    if (buffer_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return buffer_->GetReadyFuture().IsKnownReady();
  }

  // Returns xla::InvalidArgument if the buffer has been deleted.
  Status BlockHostUntilReady();
  Status CopyToHostAsync();

  const Shape& shape() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_5(mht_5_v, 301, "", "./tensorflow/compiler/xla/python/py_buffer.h", "shape");
 return buffer_->on_device_shape(); }

  StatusOr<std::uintptr_t> UnsafeBufferPointer() const;

  // Implementation of the CUDA array interface for sharing GPU buffers with
  // other Python libraries.
  StatusOr<pybind11::dict> CudaArrayInterface();

  const std::shared_ptr<Traceback>& traceback() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_6(mht_6_v, 312, "", "./tensorflow/compiler/xla/python/py_buffer.h", "traceback");
 return traceback_; }

  // Returns the size (i.e. number of elements) of the (host) numpy array.
  StatusOr<int64_t> size();

  // Returns the number of dimensions of the (host) numpy array.
  int ndim() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_7(mht_7_v, 321, "", "./tensorflow/compiler/xla/python/py_buffer.h", "ndim");
 return buffer()->on_device_shape().dimensions_size(); }

  pybind11::tuple python_shape() const;
  pybind11::dtype python_dtype() const;

  // Representing the logical view of the underlying dynamic shapes.
  StatusOr<const Shape*> xla_dynamic_shape();

  Status set_sticky_device(PjRtDevice* sticky_device) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_8(mht_8_v, 332, "", "./tensorflow/compiler/xla/python/py_buffer.h", "set_sticky_device");

    TF_RET_CHECK(sticky_device == nullptr ||
                 sticky_device == buffer_->device());
    sticky_device_ = sticky_device;
    return Status::OK();
  }
  PjRtDevice* sticky_device() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_9(mht_9_v, 341, "", "./tensorflow/compiler/xla/python/py_buffer.h", "sticky_device");
 return sticky_device_; }

  void set_weak_type(absl::optional<bool> weak_type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_10(mht_10_v, 346, "", "./tensorflow/compiler/xla/python/py_buffer.h", "set_weak_type");
 weak_type_ = weak_type; }
  absl::optional<bool> weak_type() const { return weak_type_; }

  StatusOr<pybind11::object> AsNumPyArray(pybind11::handle this_obj);

  void SetAval(pybind11::object aval) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_11(mht_11_v, 354, "", "./tensorflow/compiler/xla/python/py_buffer.h", "SetAval");
 aval_ = aval; }
  pybind11::object GetAval() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_12(mht_12_v, 358, "", "./tensorflow/compiler/xla/python/py_buffer.h", "GetAval");
 return aval_; }

  static Status RegisterTypes(pybind11::module& m);
  static PyObject* base_type() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_13(mht_13_v, 364, "", "./tensorflow/compiler/xla/python/py_buffer.h", "base_type");
 return base_type_; }
  static PyObject* type() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTh mht_14(mht_14_v, 368, "", "./tensorflow/compiler/xla/python/py_buffer.h", "type");
 return type_; }

 private:
  // PyBuffer objects must not be allocated directly since they must always live
  // on the Python heap. Use Make() instead.
  PyBuffer(std::shared_ptr<PyClient> client, std::shared_ptr<PjRtBuffer> buffer,
           std::shared_ptr<Traceback> traceback);

  static PyObject* base_type_;
  static PyObject* type_;

  friend class PyClient;

  struct HostValue {
    absl::Notification ready;
    Status status;
    std::shared_ptr<xla::Literal> value;
  };
  std::shared_ptr<PyClient> client_;
  std::shared_ptr<PjRtBuffer> buffer_;
  std::shared_ptr<Traceback> traceback_;
  std::shared_ptr<HostValue> host_value_;  // Protected by the GIL.

  // JAX uses this field to record whether a buffer is committed to a particular
  // device by the user (https://github.com/google/jax/pull/1916).
  PjRtDevice* sticky_device_ = nullptr;

  // TODO(phawkins): consider not keeping an explicit aval on C++ buffer
  // objects.
  pybind11::object aval_ = pybind11::none();

  // An optional weak type. If absent, the JAX jit code computes the weak_type
  // from the aval_.weak_type attribute. This is a backwards compatibility
  // measure for older Python code that does not set weak_type explicitly.
  // TODO(phawkins): drop support for older jax Python versions and make
  // weak_type mandatory.
  absl::optional<bool> weak_type_ = absl::nullopt;

  absl::optional<Shape> dynamic_shape_ = absl::nullopt;
  // Doubly-linked list of all PyBuffers known to the client. Protected by the
  // GIL. Since multiple PyBuffers may share the same PjRtBuffer, there may be
  // duplicate PjRtBuffers in this list.
  PyBuffer* next_;
  PyBuffer* prev_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
