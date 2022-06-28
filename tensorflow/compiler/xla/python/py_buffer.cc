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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc() {
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

#include "tensorflow/compiler/xla/python/py_buffer.h"

#include <functional>
#include <string>
#include <type_traits>

#include "absl/base/casts.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace py = pybind11;

namespace {

// Representation of a DeviceArrayBase as a Python object. Since
// a DeviceArrayBase has no fields, this is just a PyObject.
struct PyBufferBasePyObject {
  PyObject_HEAD;
};
static_assert(std::is_standard_layout<PyBufferBasePyObject>::value,
              "PyBufferBasePyObject must be standard layout");

// Representation of a DeviceArray as a Python object.
struct PyBufferPyObject {
  PyBufferBasePyObject base;
  PyBuffer buffer;
  // Used by the Python interpreter to maintain a list of weak references to
  // this object.
  PyObject* weakrefs;
};
static_assert(std::is_standard_layout<PyBufferPyObject>::value,
              "PyBufferPyObject must be standard layout");

PyObject* PyBuffer_tp_new(PyTypeObject* subtype, PyObject* args,
                          PyObject* kwds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_0(mht_0_v, 229, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer_tp_new");

  PyBufferPyObject* self =
      reinterpret_cast<PyBufferPyObject*>(subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->weakrefs = nullptr;
  return reinterpret_cast<PyObject*>(self);
}

void PyBuffer_tp_dealloc(PyObject* self) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_1(mht_1_v, 240, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer_tp_dealloc");

  PyTypeObject* tp = Py_TYPE(self);
  PyBufferPyObject* o = reinterpret_cast<PyBufferPyObject*>(self);
  if (o->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  o->buffer.~PyBuffer();
  tp->tp_free(self);
  Py_DECREF(tp);
}

}  // namespace

/*static*/ PyBuffer::object PyBuffer::Make(
    std::shared_ptr<PyClient> client, std::shared_ptr<PjRtBuffer> buffer,
    std::shared_ptr<Traceback> traceback) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_2(mht_2_v, 258, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::Make");

  py::object obj = py::reinterpret_steal<py::object>(PyBuffer_tp_new(
      reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr));
  PyBufferPyObject* buf = reinterpret_cast<PyBufferPyObject*>(obj.ptr());
  new (&buf->buffer)
      PyBuffer(std::move(client), std::move(buffer), std::move(traceback));
  return py::reinterpret_borrow<PyBuffer::object>(obj);
}

bool PyBuffer::IsPyBuffer(py::handle handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_3(mht_3_v, 270, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::IsPyBuffer");

  return handle.get_type() == PyBuffer::type();
}

/*static*/ PyBuffer* PyBuffer::AsPyBufferUnchecked(pybind11::handle handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::AsPyBufferUnchecked");

  return &(reinterpret_cast<PyBufferPyObject*>(handle.ptr())->buffer);
}

/*static*/ StatusOr<PyBuffer*> PyBuffer::AsPyBuffer(pybind11::handle handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_5(mht_5_v, 284, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::AsPyBuffer");

  if (!IsPyBuffer(handle)) {
    return InvalidArgument("Expected a DeviceArray");
  }
  return AsPyBufferUnchecked(handle);
}

py::handle PyBuffer::AsHandle() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_6(mht_6_v, 294, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::AsHandle");

  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(PyBufferPyObject, buffer));
}

PyBuffer::PyBuffer(std::shared_ptr<PyClient> client,
                   std::shared_ptr<PjRtBuffer> buffer,
                   std::shared_ptr<Traceback> traceback)
    : client_(std::move(client)),
      buffer_(std::move(buffer)),
      traceback_(std::move(traceback)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_7(mht_7_v, 307, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::PyBuffer");

  CHECK(PyGILState_Check());
  next_ = client_->buffers_[buffer_->device()->id()];
  client_->buffers_[buffer_->device()->id()] = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyBuffer::~PyBuffer() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_8(mht_8_v, 320, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::~PyBuffer");

  CHECK(PyGILState_Check());
  if (client_->buffers_[device()->id()] == this) {
    client_->buffers_[device()->id()] = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

StatusOr<int64_t> PyBuffer::size() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_9(mht_9_v, 336, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::size");

  Shape max_buffer_shape = buffer()->on_device_shape();
  if (max_buffer_shape.is_dynamic()) {
    TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());
    return ShapeUtil::ElementsIn(*dynamic_shape);
  }
  return ShapeUtil::ElementsIn(max_buffer_shape);
}

StatusOr<const Shape*> PyBuffer::xla_dynamic_shape() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_10(mht_10_v, 348, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::xla_dynamic_shape");

  CHECK(PyGILState_Check());
  if (buffer_->on_device_shape().is_static()) {
    return &buffer_->on_device_shape();
  }
  // Python buffer protocol references shape data by pointer, therefore we must
  // store a valid copy of the shape.
  if (!dynamic_shape_) {
    Shape dynamic_shape;
    {
      py::gil_scoped_release gil_release;
      TF_ASSIGN_OR_RETURN(dynamic_shape, buffer_->logical_on_device_shape());
    }
    dynamic_shape_ = dynamic_shape;
  }
  return &dynamic_shape_.value();
}

pybind11::tuple PyBuffer::python_shape() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_11(mht_11_v, 369, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::python_shape");

  return SpanToTuple(buffer()->on_device_shape().dimensions());
}

pybind11::dtype PyBuffer::python_dtype() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_12(mht_12_v, 376, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::python_dtype");

  PrimitiveType primitive = buffer()->on_device_shape().element_type();
  return PrimitiveTypeToDtype(primitive).ValueOrDie();
}

ClientAndPtr<PjRtDevice> PyBuffer::device() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_13(mht_13_v, 384, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::device");

  return WrapWithClient(client_, buffer_->device());
}

PyBuffer::object PyBuffer::Clone() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_14(mht_14_v, 391, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::Clone");

  auto buffer = Make(client_, buffer_, traceback_);
  buffer.buf()->sticky_device_ = sticky_device_;
  buffer.buf()->aval_ = aval_;
  return buffer;
}

StatusOr<py::object> PyBuffer::CopyToDevice(
    const ClientAndPtr<PjRtDevice>& dst_device) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_15(mht_15_v, 402, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::CopyToDevice");

  CHECK(dst_device.get() != nullptr);
  auto transfer_guard_formatter = [this, &dst_device] {
    auto shape = py::cast<std::string>(py::str(python_shape()));
    auto dtype = py::cast<std::string>(py::str(python_dtype()));
    return absl::StrCat("shape=", shape, ", dtype=", dtype,
                        ", device=", device()->DebugString(),
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));

  GlobalPyRefManager()->CollectGarbage();
  std::unique_ptr<PjRtBuffer> out;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(out, buffer_->CopyToDevice(dst_device.get()));
  }
  auto traceback = Traceback::Get();
  return Make(dst_device.client, std::move(out), std::move(traceback));
}

Status PyBuffer::BlockHostUntilReady() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_16(mht_16_v, 427, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::BlockHostUntilReady");

  GlobalPyRefManager()->CollectGarbage();
  py::gil_scoped_release gil_release;
  return buffer_->BlockHostUntilReady();
}

Status PyBuffer::CopyToHostAsync() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_17(mht_17_v, 436, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::CopyToHostAsync");

  if (!buffer_->IsOnCpu() && !host_value_) {
    auto transfer_guard_formatter = [this] {
      auto shape = py::cast<std::string>(py::str(python_shape()));
      auto dtype = py::cast<std::string>(py::str(python_dtype()));
      return absl::StrCat("shape=", shape, ", dtype=", dtype,
                          ", device=", device()->DebugString());
    };
    TF_RETURN_IF_ERROR(
        jax::ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

    std::shared_ptr<HostValue> host_value = std::make_shared<HostValue>();
    host_value_ = host_value;
    // TODO(b/182461453): This is a blocking call. If we further implemented
    // populating dynamic shape metadata while fetching the literal, we wouldn't
    // need this static approach.
    TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());

    py::gil_scoped_release gil;
    host_value->value = std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(*dynamic_shape));
    Literal* literal = host_value->value.get();
    buffer_->ToLiteral(literal,
                       [host_value{std::move(host_value)}](Status status) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_18(mht_18_v, 462, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "lambda");

                         host_value->status = std::move(status);
                         host_value->ready.Notify();
                       });
  }
  return Status::OK();
}

StatusOr<pybind11::object> PyBuffer::AsNumPyArray(py::handle this_obj) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_19(mht_19_v, 473, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::AsNumPyArray");

  if (buffer_->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  TF_RET_CHECK(buffer_->on_device_shape().IsArray());
  // On CPU, we can return the value in a zero-copy way.
  if (buffer_->IsOnCpu()) {
    TF_ASSIGN_OR_RETURN(const auto* shape, xla_dynamic_shape());
    TF_ASSIGN_OR_RETURN(py::dtype dtype,
                        PrimitiveTypeToDtype(shape->element_type()));
    // Objects that must be kept alive while the array is alive.
    struct Hold {
      py::object buffer;
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
    };
    auto hold = std::make_unique<Hold>();
    TF_ASSIGN_OR_RETURN(hold->external_reference_hold,
                        buffer_->AcquireExternalReference());
    hold->buffer = py::reinterpret_borrow<py::object>(this_obj);
    void* data = hold->external_reference_hold->OpaqueDeviceMemoryDataPointer();
    py::capsule hold_capsule(hold.release(),
                             [](void* h) { delete static_cast<Hold*>(h); });
    py::array array(dtype, shape->dimensions(), ByteStridesForShape(*shape),
                    data, hold_capsule);
    array.attr("flags").attr("writeable") = Py_False;
    {
      py::gil_scoped_release gil;
      TF_RETURN_IF_ERROR(buffer_->BlockHostUntilReady());
    }
    return array;
  }

  TF_RETURN_IF_ERROR(CopyToHostAsync());
  if (!host_value_->ready.HasBeenNotified()) {
    py::gil_scoped_release gil;
    host_value_->ready.WaitForNotification();
  }
  TF_RETURN_IF_ERROR(host_value_->status);
  TF_ASSIGN_OR_RETURN(py::object array, LiteralToPython(host_value_->value));
  array.attr("flags").attr("writeable") = Py_False;
  return array;
}

StatusOr<std::uintptr_t> PyBuffer::UnsafeBufferPointer() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_20(mht_20_v, 519, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::UnsafeBufferPointer");

  return client_->pjrt_client()->UnsafeBufferPointer(buffer_.get());
}

StatusOr<py::dict> PyBuffer::CudaArrayInterface() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_21(mht_21_v, 526, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::CudaArrayInterface");

  // TODO(zhangqiaorjc): Differentiate between NVidia and other GPUs.
  if (buffer_->client()->platform_id() != GpuId()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (!buffer_->on_device_shape().IsArray()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for array buffers.");
  }
  if (buffer_->on_device_shape().element_type() == BF16) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for bfloat16 buffers.");
  }
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      buffer_->on_device_shape().layout()));

  py::dict result;
  TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());
  result["shape"] = SpanToTuple(dynamic_shape->dimensions());
  TF_ASSIGN_OR_RETURN(py::str typestr,
                      TypeDescriptorForPrimitiveType(
                          buffer_->on_device_shape().element_type()));
  result["typestr"] = std::move(typestr);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      buffer_->AcquireExternalReference());
  const void* root_ptr =
      external_reference_hold->OpaqueDeviceMemoryDataPointer();
  py::tuple data(2);
  data[0] = py::int_(absl::bit_cast<std::uintptr_t>(root_ptr));
  data[1] = py::bool_(true);  // read-only
  result["data"] = std::move(data);
  result["version"] = py::int_(2);
  return result;
}

// PEP 3118 buffer protocol implementation.

namespace {

// Extra data to be kept alive by the consumer of the buffer protocol.
struct ExtraBufferInfo {
  explicit ExtraBufferInfo(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold)
      : external_reference_hold(std::move(external_reference_hold)) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_22(mht_22_v, 574, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "ExtraBufferInfo");
}

  std::string format;
  std::vector<Py_ssize_t> strides;
  // We keep an external reference hold to the PjRtBuffer. This prevents a
  // use-after-free in the event that Delete() is called on a buffer with an
  // live buffer protocol view. It does however mean that Delete() sometimes
  // won't actually delete immediately.
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
};

int PyBuffer_bf_getbuffer(PyObject* exporter, Py_buffer* view, int flags) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_23(mht_23_v, 588, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer_bf_getbuffer");

  Status status = [&]() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_24(mht_24_v, 592, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "lambda");

    TF_ASSIGN_OR_RETURN(PyBuffer * py_buffer, PyBuffer::AsPyBuffer(exporter));
    PjRtBuffer& buffer = *py_buffer->buffer();
    TF_ASSIGN_OR_RETURN(const auto* shape, py_buffer->xla_dynamic_shape());
    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

    if (!buffer.IsOnCpu()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }
    if (!buffer.on_device_shape().IsArray()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for array buffers.");
    }
    // If we allowed exports of formatted BF16 buffers, consumers would get
    // confused about the type because there is no way to describe BF16 to
    // Python.
    if (buffer.on_device_shape().element_type() == BF16 &&
        ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)) {
      return InvalidArgument(
          "bfloat16 buffer format not supported by Python buffer protocol.");
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
      return InvalidArgument("XLA buffers are read-only.");
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
        buffer.AcquireExternalReference());
    if (buffer.IsDeleted()) {
      return InvalidArgument("Deleted buffer used in buffer protocol.");
    }

    if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS ||
         (flags & PyBUF_STRIDES) == PyBUF_ND) &&
        !LayoutUtil::IsMonotonicWithDim0Major(shape->layout())) {
      return InvalidArgument("Buffer is not in C-contiguous layout.");
    } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape->layout())) {
      return InvalidArgument("Buffer is not in F-contiguous layout.");
    } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Major(shape->layout()) &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape->layout())) {
      return InvalidArgument("Buffer is not in contiguous layout.");
    }
    std::memset(view, 0, sizeof(Py_buffer));
    const void* root_ptr =
        external_reference_hold->OpaqueDeviceMemoryDataPointer();
    view->buf = const_cast<void*>(root_ptr);
    auto extra =
        absl::make_unique<ExtraBufferInfo>(std::move(external_reference_hold));
    view->itemsize = ShapeUtil::ByteSizeOfPrimitiveType(shape->element_type());
    view->len = ShapeUtil::ByteSizeOf(*shape);
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
      TF_ASSIGN_OR_RETURN(extra->format, FormatDescriptorForPrimitiveType(
                                             shape->element_type()));
      view->format = const_cast<char*>(extra->format.c_str());
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
      view->ndim = shape->dimensions_size();
      static_assert(sizeof(int64_t) == sizeof(Py_ssize_t),
                    "Py_ssize_t must be 64 bits");
      if (view->ndim != 0) {
        view->shape = reinterpret_cast<Py_ssize_t*>(
            const_cast<int64_t*>(shape->dimensions().data()));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          extra->strides = ByteStridesForShape(*shape);
          view->strides = extra->strides.data();
        }
      }
    }
    TF_RETURN_IF_ERROR(buffer.BlockHostUntilReady());
    view->internal = extra.release();
    return Status::OK();
  }();
  if (!status.ok()) {
    // numpy.asarray(...) silents the PyExc_BufferError. Adding a log here helps
    // debugging when the error really occurs.
    VLOG(1) << "Buffer Protocol Error: " << status;
    PyErr_SetString(PyExc_BufferError, status.ToString().c_str());
    return -1;
  }
  view->obj = exporter;
  Py_INCREF(view->obj);
  return 0;
}

void PyBuffer_bf_releasebuffer(PyObject*, Py_buffer* buffer) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_25(mht_25_v, 684, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer_bf_releasebuffer");

  auto extra = static_cast<ExtraBufferInfo*>(buffer->internal);
  delete extra;
}

PyBufferProcs PyBuffer_tp_as_buffer = []() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_26(mht_26_v, 692, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "lambda");

  PyBufferProcs procs;
  procs.bf_getbuffer = &PyBuffer_bf_getbuffer;
  procs.bf_releasebuffer = &PyBuffer_bf_releasebuffer;
  return procs;
}();

}  // namespace

PyObject* PyBuffer::base_type_ = nullptr;
PyObject* PyBuffer::type_ = nullptr;

Status PyBuffer::RegisterTypes(py::module& m) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_27(mht_27_v, 707, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "PyBuffer::RegisterTypes");

  // We do not use pybind11::class_ to build Python wrapper objects because
  // creation, destruction, and casting of buffer objects is performance
  // critical. By using hand-written Python classes, we can avoid extra C heap
  // allocations, and we can avoid pybind11's slow cast<>() implementation
  // during jit dispatch.

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  {
    py::str name = py::str("DeviceArrayBase");
    py::str qualname = py::str("DeviceArrayBase");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called. Otherwise the GC might see a half-constructed
    // type object.
    if (!heap_type) {
      return Internal("Unable to create heap type object");
    }
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "DeviceArrayBase";
    type->tp_basicsize = sizeof(PyBufferBasePyObject);
    type->tp_flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE;
    TF_RET_CHECK(PyType_Ready(type) == 0);
    base_type_ = reinterpret_cast<PyObject*>(type);
  }
  py::object base_type = py::reinterpret_borrow<py::object>(base_type_);
  base_type.attr("__module__") = m.attr("__name__");
  m.attr("DeviceArrayBase") = base_type;

  {
    py::tuple bases = py::make_tuple(base_type);
    py::str name = py::str("DeviceArray");
    py::str qualname = py::str("DeviceArray");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called below. Otherwise the GC might see a
    // half-constructed type object.
    if (!heap_type) {
      return Internal("Unable to create heap type object");
    }
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "DeviceArray";
    type->tp_basicsize = sizeof(PyBufferPyObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    type->tp_bases = bases.release().ptr();
    type->tp_dealloc = PyBuffer_tp_dealloc;
    type->tp_new = PyBuffer_tp_new;
    // Supported protocols
    type->tp_as_number = &heap_type->as_number;
    type->tp_as_sequence = &heap_type->as_sequence;
    type->tp_as_mapping = &heap_type->as_mapping;
    type->tp_as_buffer = &PyBuffer_tp_as_buffer;

    // Allow weak references to DeviceArray objects.
    type->tp_weaklistoffset = offsetof(PyBufferPyObject, weakrefs);

    TF_RET_CHECK(PyType_Ready(type) == 0);
    type_ = reinterpret_cast<PyObject*>(type);
  }
  py::object type = py::reinterpret_borrow<py::object>(type_);
  m.attr("DeviceArray") = type;
  m.attr("PyLocalBuffer") = type;
  m.attr("Buffer") = type;

  // Add methods and properties to the class. We use pybind11 and add methods
  // dynamically mostly because this is easy to write and allows us to use
  // pybind11's casting logic. This is most likely slightly slower than
  // hand-writing bindings, but most of these methods are not performance
  // critical.
  using jax::property;
  using jax::property_readonly;
  type.attr("__array_priority__") =
      property_readonly([](py::object self) -> int { return 100; });
  type.attr("_device") = property(
      [](PyBuffer::object self) -> ClientAndPtr<PjRtDevice> {
        return WrapWithClient(self.buf()->client(),
                              self.buf()->sticky_device());
      },
      [](PyBuffer::object self, PjRtDevice* sticky_device) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_28(mht_28_v, 796, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "lambda");

        return self.buf()->set_sticky_device(sticky_device);
      });
  type.attr("aval") = property(
      [](PyBuffer::object self) -> py::object { return self.buf()->GetAval(); },
      [](PyBuffer::object self, py::object aval) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_29(mht_29_v, 804, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "lambda");

        return self.buf()->SetAval(std::move(aval));
      });
  type.attr("weak_type") = property(
      [](PyBuffer::object self) -> absl::optional<bool> {
        return self.buf()->weak_type();
      },
      [](PyBuffer::object self, absl::optional<bool> weak_type) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_bufferDTcc mht_30(mht_30_v, 814, "", "./tensorflow/compiler/xla/python/py_buffer.cc", "lambda");

        return self.buf()->set_weak_type(weak_type);
      });
  type.attr("device_buffer") =
      property_readonly([](py::object self) { return self; });
  type.attr(
      "shape") = property_readonly([](PyBuffer::object self) -> py::tuple {
    return SpanToTuple(self.buf()->buffer()->on_device_shape().dimensions());
  });
  type.attr("dtype") = property_readonly([](PyBuffer::object self) {
    PrimitiveType primitive =
        self.buf()->buffer()->on_device_shape().element_type();
    return PrimitiveTypeToDtype(primitive).ValueOrDie();
  });
  type.attr("size") =
      property_readonly([](PyBuffer::object self) -> StatusOr<int64_t> {
        return self.buf()->size();
      });
  type.attr("ndim") = property_readonly(
      [](PyBuffer::object self) -> int { return self.buf()->ndim(); });
  type.attr("_value") = property_readonly(
      [](PyBuffer::object self) -> StatusOr<pybind11::object> {
        GlobalPyRefManager()->CollectGarbage();
        return self.buf()->AsNumPyArray(self);
      });
  type.attr("copy_to_device") = py::cpp_function(
      [](PyBuffer::object self, const ClientAndPtr<PjRtDevice>& dst_device) {
        return self.buf()->CopyToDevice(dst_device);
      },
      py::is_method(type));
  type.attr("on_device_size_in_bytes") = py::cpp_function(
      [](PyBuffer::object self) -> StatusOr<size_t> {
        return self.buf()->OnDeviceSizeInBytes();
      },
      py::is_method(type));
  type.attr("delete") = py::cpp_function(
      [](PyBuffer::object self) { self.buf()->Delete(); }, py::is_method(type));
  type.attr("block_host_until_ready") = py::cpp_function(
      [](PyBuffer::object self) {
        // TODO(phawkins): remove 3 months after the release of jaxlib >= 0.3.2.
        PythonDeprecationWarning(
            "block_host_until_ready() on a JAX array object is deprecated, use "
            "block_until_ready() instead.");
        return self.buf()->BlockHostUntilReady();
      },
      py::is_method(type));
  type.attr("is_ready") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->IsReady(); },
      py::is_method(type));
  type.attr("is_known_ready") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->IsKnownReady(); },
      py::is_method(type));
  type.attr("block_until_ready") = py::cpp_function(
      [](PyBuffer::object self) -> StatusOr<PyBuffer::object> {
        TF_RETURN_IF_ERROR(self.buf()->BlockHostUntilReady());
        return std::move(self);
      },
      py::is_method(type));
  type.attr("copy_to_host_async") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->CopyToHostAsync(); },
      py::is_method(type));
  type.attr("to_py") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->AsNumPyArray(self); },
      py::is_method(type));
  type.attr("xla_shape") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->shape(); },
      py::is_method(type));
  type.attr("xla_dynamic_shape") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->xla_dynamic_shape(); },
      py::is_method(type));
  type.attr("client") = property_readonly(
      [](PyBuffer::object self) { return self.buf()->client(); });
  type.attr("device") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->device(); },
      py::is_method(type));
  type.attr("platform") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->platform_name(); },
      py::is_method(type));
  type.attr("is_deleted") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->is_deleted(); },
      py::is_method(type));
  type.attr("unsafe_buffer_pointer") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->UnsafeBufferPointer(); },
      py::is_method(type));
  type.attr("__cuda_array_interface__") = property_readonly(
      [](PyBuffer::object self) { return self.buf()->CudaArrayInterface(); });
  type.attr("traceback") = property_readonly(
      [](PyBuffer::object self) { return self.buf()->traceback(); });
  type.attr("clone") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->Clone(); },
      py::is_method(type));
  type.attr("__module__") = m.attr("__name__");
  return Status::OK();
}

}  // namespace xla
