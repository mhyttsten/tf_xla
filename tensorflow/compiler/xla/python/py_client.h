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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh() {
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
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class PyBuffer;
class PyClient;
class PyExecutable;

// Custom holder types.
//
// We must keep the PyClient object alive as long as any of the runtime
// objects are alive. Since we don't have a lot of control over Python
// destructor ordering, we keep the PyClient object as a std::shared_ptr<>,
// and ensure that each Python runtime object holds a reference to the
// PyClient. An alternative design would be to keep a single global
// singleton PyClient, although this seems less flexible, especially for
// writing tests.
//
// To maintain PyClient references, we define pybind11 holder classes that
// are custom smart pointers that also keep a reference to a PyClient.
// pybind11 has a `keep_alive` feature that has a similar goal, but it doesn't
// seem sufficiently flexible to describe ownership relationships in cases where
// the ownership doesn't pertain to a direct argument or return value of a
// function. Another alternative to the holder classes would be to create proxy
// objects that contain both a reference and a runtime class; holder classes
// seem less tedious to define.

// A pair of a PyClient reference and an unowned pointer to T.
template <typename T>
struct ClientAndPtr {
  ClientAndPtr() = default;
  // pybind11 requires that we define a constructor that takes a raw pointer,
  // but it should be unreachable.
  explicit ClientAndPtr(T*) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_0(mht_0_v, 231, "", "./tensorflow/compiler/xla/python/py_client.h", "ClientAndPtr");

    LOG(FATAL) << "ClientAndPtr should constructed via WrapWithClient.";
  }

  ClientAndPtr(const ClientAndPtr&) = default;
  ClientAndPtr(ClientAndPtr&&) = default;
  ClientAndPtr& operator=(const ClientAndPtr&) = default;
  ClientAndPtr& operator=(ClientAndPtr&&) = default;

  std::shared_ptr<PyClient> client;
  T* contents;

  T* get() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_1(mht_1_v, 246, "", "./tensorflow/compiler/xla/python/py_client.h", "get");
 return contents; }
  T* operator->() const { return contents; }
  T& operator*() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/python/py_client.h", "*");
 return *contents; }
};

// By defining a templated helper function, we can use return type deduction
// and avoid specifying types at the caller.
template <typename T>
ClientAndPtr<T> WrapWithClient(std::shared_ptr<PyClient> client, T* contents) {
  ClientAndPtr<T> result;
  result.client = std::move(client);
  result.contents = contents;
  return result;
}

// Python wrapper around PjRtClient.
// We use a wrapper class to add Python-specific functionality.
class PyClient : public std::enable_shared_from_this<PyClient> {
 public:
  explicit PyClient(std::unique_ptr<PjRtClient> pjrt_client);
  explicit PyClient(std::shared_ptr<PjRtClient> pjrt_client);
  virtual ~PyClient();

  PjRtClient* pjrt_client() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_3(mht_3_v, 275, "", "./tensorflow/compiler/xla/python/py_client.h", "pjrt_client");
 return pjrt_client_.get(); }
  std::shared_ptr<PjRtClient> shared_pjrt_client() { return pjrt_client_; }

  absl::string_view platform_name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_4(mht_4_v, 281, "", "./tensorflow/compiler/xla/python/py_client.h", "platform_name");

    return pjrt_client_->platform_name();
  }
  absl::string_view platform_version() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_5(mht_5_v, 287, "", "./tensorflow/compiler/xla/python/py_client.h", "platform_version");

    return pjrt_client_->platform_version();
  }
  absl::string_view runtime_type() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_6(mht_6_v, 293, "", "./tensorflow/compiler/xla/python/py_client.h", "runtime_type");

    return PjRtRuntimeTypeString(pjrt_client_->runtime_type());
  }
  int addressable_device_count() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_7(mht_7_v, 299, "", "./tensorflow/compiler/xla/python/py_client.h", "addressable_device_count");

    return pjrt_client_->addressable_device_count();
  }
  int device_count() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_8(mht_8_v, 305, "", "./tensorflow/compiler/xla/python/py_client.h", "device_count");
 return pjrt_client_->device_count(); }
  int process_index() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_clientDTh mht_9(mht_9_v, 309, "", "./tensorflow/compiler/xla/python/py_client.h", "process_index");
 return pjrt_client_->process_index(); }

  std::vector<ClientAndPtr<PjRtDevice>> Devices();
  std::vector<ClientAndPtr<PjRtDevice>> LocalDevices();

  // Returns a vector of live PyBuffer objects. PyBuffer objects may share
  // PjRtBuffers, so there may be duplicates of the same underlying device
  // buffer.
  std::vector<pybind11::object> LiveBuffers();
  std::vector<pybind11::object> LiveBuffersOnDevice(PjRtDevice* device);

  // Returns a vector of live PyExecutable objects.
  // note: must return std::shared_ptr instead of raw ptrs
  // https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#std-shared-ptr
  std::vector<std::shared_ptr<PyExecutable>> LiveExecutables();

  // TODO(zhangqiaorjc): Remove when we have transparent defragmentation.
  Status Defragment();

  StatusOr<std::vector<std::vector<ClientAndPtr<PjRtDevice>>>>
  GetDefaultDeviceAssignment(int num_replicas, int num_partitions);

  // TODO(skye): delete after all callers can handle 2D output
  StatusOr<std::vector<ClientAndPtr<PjRtDevice>>> GetDefaultDeviceAssignment1D(
      int num_replicas);

  StatusOr<ChannelHandle> CreateChannelHandle() {
    return pjrt_client_->CreateChannelHandle();
  }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() {
    return pjrt_client_->CreateDeviceToHostChannelHandle();
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() {
    return pjrt_client_->CreateHostToDeviceChannelHandle();
  }

  StatusOr<pybind11::object> BufferFromPyval(
      pybind11::handle argument, PjRtDevice* device, bool force_copy,
      PjRtClient::HostBufferSemantics host_buffer_semantics);

  StatusOr<std::shared_ptr<PyExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options);
  StatusOr<std::shared_ptr<PyExecutable>> CompileMlir(std::string mlir_module,
                                                      CompileOptions options);

  StatusOr<pybind11::bytes> SerializeExecutable(
      const PyExecutable& executable) const;
  StatusOr<std::shared_ptr<PyExecutable>> DeserializeExecutable(
      const std::string& serialized, CompileOptions options);

  // TODO(skyewm): remove when jax stop providing hlo_module
  StatusOr<std::shared_ptr<PyExecutable>> DeserializeExecutable(
      const std::string& serialized, std::shared_ptr<HloModule> hlo_module,
      CompileOptions options) {
    return DeserializeExecutable(serialized, options);
  }

  StatusOr<pybind11::bytes> HeapProfile();

  // Adds code to `builder` to call Python host function `callable` with
  // `operands`, returning a result of `result_shape`. If desired, the operand
  // layouts can be constrained by `operand_layouts`. Returns a pair of the
  // output XlaOp, together with an object that must be kept alive as long as
  // the Python callback may be called. Typically the callback may be kept
  // alive by attaching it to the executable built from this computation.
  //
  // Callable receives as arguments NumPy arrays for arguments with array types,
  // and None for Token argument. The callable must return a tuple of either
  // arrays or None values.
  //
  // This is a method of PyClient since different platforms may implement this
  // functionality in different ways.
  StatusOr<std::pair<XlaOp, pybind11::object>> EmitPythonCallback(
      pybind11::function callable, XlaBuilder& builder,
      absl::Span<XlaOp const> operands, absl::Span<Shape const> result_shapes,
      absl::optional<std::vector<Shape>> operand_layouts, bool has_side_effect);

 private:
  friend class PyBuffer;
  friend class PyExecutable;

  std::shared_ptr<PjRtClient> pjrt_client_;

  // Pointers to intrusive doubly-linked lists of buffers and executables, used
  // to iterate over all known objects when heap profiling. The list structure
  // is protected by the GIL.

  // buffers_ is a per-device list, indexed by device->id().
  std::vector<PyBuffer*> buffers_;
  PyExecutable* executables_ = nullptr;
};

}  // namespace xla

PYBIND11_DECLARE_HOLDER_TYPE(T, xla::ClientAndPtr<T>);

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_CLIENT_H_
