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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc() {
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

#include "tensorflow/compiler/xla/python/sharded_device_array.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/core/platform/statusor.h"

namespace jax {

namespace py = pybind11;

namespace {

struct ShardedDeviceArrayBaseObject {
  PyObject_HEAD;
};
static_assert(std::is_standard_layout<ShardedDeviceArrayBaseObject>::value,
              "ShardedDeviceArrayBaseObject must be standard layout");

struct ShardedDeviceArrayObject {
  ShardedDeviceArrayBaseObject base;
  ShardedDeviceArray sda;
  // Used by the Python interpreter to maintain a list of weak references to
  // this object.
  PyObject* weakrefs;
};
static_assert(std::is_standard_layout<ShardedDeviceArrayObject>::value,
              "ShardedDeviceArrayObject must be standard layout");

PyObject* sharded_device_array_tp_new(PyTypeObject* subtype, PyObject* args,
                                      PyObject* kwds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "sharded_device_array_tp_new");

  ShardedDeviceArrayObject* self = reinterpret_cast<ShardedDeviceArrayObject*>(
      subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->weakrefs = nullptr;
  return reinterpret_cast<PyObject*>(self);
}

void sharded_device_array_tp_dealloc(PyObject* self) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "sharded_device_array_tp_dealloc");

  PyTypeObject* tp = Py_TYPE(self);
  ShardedDeviceArrayObject* o =
      reinterpret_cast<ShardedDeviceArrayObject*>(self);
  if (o->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  o->sda.~ShardedDeviceArray();
  tp->tp_free(self);
  Py_DECREF(tp);
}

}  // namespace

void ShardedDeviceArray::Delete() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_2(mht_2_v, 249, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::Delete");

  // If already deleted, do nothing.
  if (is_deleted_) {
    return;
  }
  for (xla::PjRtBuffer* pjrt_buffer : GetPjRtBuffers().ConsumeValueOrDie()) {
    pjrt_buffer->Delete();
  }
  device_buffers_ = absl::nullopt;
  cpp_device_buffers_ = absl::nullopt;
  npy_value_ = absl::nullopt;
  is_deleted_ = true;
}

xla::StatusOr<absl::Span<xla::PjRtBuffer* const>>
ShardedDeviceArray::GetPjRtBuffers() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_3(mht_3_v, 267, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::GetPjRtBuffers");

  if (cpp_device_buffers_.has_value()) {
    return absl::MakeConstSpan(cpp_device_buffers_.value());
  }

  if (!device_buffers_.has_value()) {
    return xla::InvalidArgument("ShardedDeviceArray has been deleted.");
  }
  const int num_devices = device_buffers_->size();
  std::vector<xla::PjRtBuffer*> cpp_device_buffers;
  cpp_device_buffers.reserve(num_devices);
  int i = 0;
  for (auto& handle : device_buffers_.value()) {
    // Note that invariants guarantee the cast should never fail.
    TF_ASSIGN_OR_RETURN(xla::PyBuffer * pybuffer,
                        xla::PyBuffer::AsPyBuffer(handle));
    cpp_device_buffers.push_back(pybuffer->buffer());
    i += 1;
  }
  cpp_device_buffers_ = std::move(cpp_device_buffers);
  return absl::MakeConstSpan(cpp_device_buffers_.value());
}

PyObject* ShardedDeviceArray::base_type_ = nullptr;
PyObject* ShardedDeviceArray::type_ = nullptr;

/*static*/ ShardedDeviceArray::object ShardedDeviceArray::Make(
    py::object aval, ShardingSpec sharding_spec, py::list device_buffers,
    py::object indices, bool weak_type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_4(mht_4_v, 298, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::Make");

  py::object obj =
      py::reinterpret_steal<py::object>(sharded_device_array_tp_new(
          reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr));
  ShardedDeviceArrayObject* sda =
      reinterpret_cast<ShardedDeviceArrayObject*>(obj.ptr());
  new (&sda->sda)
      ShardedDeviceArray(aval, std::move(sharding_spec),
                         std::move(device_buffers), indices, weak_type);
  return py::reinterpret_borrow<ShardedDeviceArray::object>(obj);
}

bool ShardedDeviceArray::IsShardedDeviceArray(py::handle handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_5(mht_5_v, 313, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::IsShardedDeviceArray");

  return handle.get_type() == ShardedDeviceArray::type();
}

/*static*/ ShardedDeviceArray*
ShardedDeviceArray::AsShardedDeviceArrayUnchecked(py::handle handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_6(mht_6_v, 321, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::AsShardedDeviceArrayUnchecked");

  return &(reinterpret_cast<ShardedDeviceArrayObject*>(handle.ptr())->sda);
}

/*static*/ xla::StatusOr<ShardedDeviceArray*>
ShardedDeviceArray::AsShardedDeviceArray(py::handle handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_7(mht_7_v, 329, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::AsShardedDeviceArray");

  if (!IsShardedDeviceArray(handle)) {
    return xla::InvalidArgument("Expected a ShardedDeviceArray");
  }
  return AsShardedDeviceArrayUnchecked(handle);
}

py::handle ShardedDeviceArray::AsHandle() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_8(mht_8_v, 339, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::AsHandle");

  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(ShardedDeviceArrayObject, sda));
}

/*static*/ xla::Status ShardedDeviceArray::RegisterTypes(py::module& m) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_9(mht_9_v, 347, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "ShardedDeviceArray::RegisterTypes");

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  // Similar to py_buffer.cc
  {
    py::str name = py::str("ShardedDeviceArrayBase");
    py::str qualname = py::str("ShardedDeviceArrayBase");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called. Otherwise the GC might see a half-constructed
    // type object.
    if (!heap_type) {
      return xla::Internal("Unable to create heap type object");
    }
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "ShardedDeviceArrayBase";
    type->tp_basicsize = sizeof(ShardedDeviceArrayBaseObject);
    type->tp_flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE;
    TF_RET_CHECK(PyType_Ready(type) == 0);
    base_type_ = reinterpret_cast<PyObject*>(type);
  }
  py::object base_type = py::reinterpret_borrow<py::object>(base_type_);
  base_type.attr("__module__") = m.attr("__name__");
  m.attr("ShardedDeviceArrayBase") = base_type;

  {
    py::tuple bases = py::make_tuple(base_type);
    py::str name = py::str("ShardedDeviceArray");
    py::str qualname = py::str("ShardedDeviceArray");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called below. Otherwise the GC might see a
    // half-constructed type object.
    if (!heap_type) {
      return xla::Internal("Unable to create heap type object");
    }
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "ShardedDeviceArray";
    type->tp_basicsize = sizeof(ShardedDeviceArrayObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    type->tp_bases = bases.release().ptr();
    type->tp_dealloc = sharded_device_array_tp_dealloc;
    type->tp_new = sharded_device_array_tp_new;
    // Supported protocols
    type->tp_as_number = &heap_type->as_number;
    type->tp_as_sequence = &heap_type->as_sequence;
    type->tp_as_mapping = &heap_type->as_mapping;
    type->tp_as_buffer = nullptr;

    // Allow weak references to DeviceArray objects.
    type->tp_weaklistoffset = offsetof(ShardedDeviceArrayObject, weakrefs);

    TF_RET_CHECK(PyType_Ready(type) == 0);
    type_ = reinterpret_cast<PyObject*>(type);
  }
  py::object type = py::reinterpret_borrow<py::object>(type_);
  type.attr("__module__") = m.attr("__name__");
  m.attr("ShardedDeviceArray") = type;

  type.attr("make") = def_static([](py::object aval, ShardingSpec sharding_spec,
                                    py::list device_buffers, py::object indices,
                                    bool weak_type) {
    return ShardedDeviceArray::Make(aval, sharding_spec, device_buffers,
                                    indices, weak_type);
  });
  type.attr("aval") =
      property_readonly([](ShardedDeviceArray::object self) -> py::object {
        return self.sda()->aval();
      });
  type.attr("indices") =
      property_readonly([](ShardedDeviceArray::object self) -> py::object {
        return self.sda()->indices();
      });
  type.attr("sharding_spec") =
      property_readonly([](ShardedDeviceArray::object self) {
        return self.sda()->GetShardingSpec();
      });
  type.attr("device_buffers") =
      property_readonly([](ShardedDeviceArray::object self) {
        return self.sda()->device_buffers();
      });
  type.attr("_npy_value") = property(
      [](ShardedDeviceArray::object self) { return self.sda()->npy_value(); },
      [](ShardedDeviceArray::object self, py::object npy_value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_10(mht_10_v, 440, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "lambda");

        return self.sda()->set_npy_value(npy_value);
      });
  type.attr("_one_replica_buffer_indices") = property(
      [](ShardedDeviceArray::object self) {
        return self.sda()->one_replica_buffer_indices();
      },
      [](ShardedDeviceArray::object self, py::object obj) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSsharded_device_arrayDTcc mht_11(mht_11_v, 450, "", "./tensorflow/compiler/xla/python/sharded_device_array.cc", "lambda");

        return self.sda()->set_one_replica_buffer_indices(obj);
      });
  type.attr("shape") = property_readonly([](ShardedDeviceArray::object self) {
    return self.sda()->aval().attr("shape");
  });
  type.attr("dtype") = property_readonly([](ShardedDeviceArray::object self) {
    return self.sda()->aval().attr("dtype");
  });
  type.attr("size") = property_readonly([](ShardedDeviceArray::object self) {
    py::tuple shape = py::cast<py::tuple>(self.sda()->aval().attr("shape"));
    int64_t size = 1;
    for (auto dim : shape) {
      size *= py::cast<int64_t>(dim);
    }
    return size;
  });
  type.attr("ndim") = property_readonly([](ShardedDeviceArray::object self) {
    return py::len(self.sda()->aval().attr("shape"));
  });

  type.attr("delete") = py::cpp_function(
      [](ShardedDeviceArray::object self) { self.sda()->Delete(); },
      py::is_method(type));
  type.attr("is_deleted") = py::cpp_function(
      [](ShardedDeviceArray::object self) { return self.sda()->is_deleted(); },
      py::is_method(type));

  return xla::Status::OK();
}

}  // namespace jax
