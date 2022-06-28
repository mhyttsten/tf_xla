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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh() {
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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Python wrapper around PjRtExecutable. We use a wrapper class:
// a) to keep the PyClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
class PyExecutable : public std::enable_shared_from_this<PyExecutable> {
 public:
  PyExecutable(std::shared_ptr<PyClient> client,
               std::unique_ptr<PjRtExecutable> executable,
               std::shared_ptr<Traceback> traceback,
               absl::optional<std::string> fingerprint);
  ~PyExecutable();

  std::shared_ptr<PyClient> client() const { return client_; }
  std::shared_ptr<PjRtExecutable> executable() const { return executable_; }

  absl::Span<const PjRtExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const {
    return executable_->addressable_device_logical_ids();
  }

  std::vector<ClientAndPtr<PjRtDevice>> AddressableDevices() const;

  int64_t SizeOfGeneratedCodeInBytes() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/python/py_executable.h", "SizeOfGeneratedCodeInBytes");

    return executable_->SizeOfGeneratedCodeInBytes();
  }

  StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const {
    return executable_->GetCompiledMemoryStats();
  }

  void Delete() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_1(mht_1_v, 235, "", "./tensorflow/compiler/xla/python/py_executable.h", "Delete");
 return executable_->Delete(); }

  bool is_deleted() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_2(mht_2_v, 240, "", "./tensorflow/compiler/xla/python/py_executable.h", "is_deleted");
 return executable_->IsDeleted(); }

  StatusOr<std::vector<PyBuffer::object>> Execute(
      absl::Span<PyBuffer::object const> args);

  // Takes args indexed by argid then deviceid, transposes them, and passes to
  // PjRtExecutable::Execute. The result is similarly transposed back into the
  // argid,deviceid format.
  // args is [num_args x num_devices].
  StatusOr<std::vector<std::vector<PyBuffer::object>>>
  ExecuteShardedOnLocalDevices(
      absl::Span<const std::vector<PyBuffer::object>> args);

  StatusOr<std::vector<std::shared_ptr<HloModule>>> HloModules() const;

  Traceback* traceback() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_3(mht_3_v, 258, "", "./tensorflow/compiler/xla/python/py_executable.h", "traceback");
 return traceback_.get(); }

  const PjRtExecutable& pjrt_executable() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_4(mht_4_v, 263, "", "./tensorflow/compiler/xla/python/py_executable.h", "pjrt_executable");
 return *executable_; }

  PjRtExecutable* mutable_pjrt_executable() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_5(mht_5_v, 268, "", "./tensorflow/compiler/xla/python/py_executable.h", "mutable_pjrt_executable");
 return executable_.get(); }
  const ExecuteOptions& options() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_6(mht_6_v, 272, "", "./tensorflow/compiler/xla/python/py_executable.h", "options");
 return options_; }
  const absl::optional<std::string>& fingerprint() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTh mht_7(mht_7_v, 276, "", "./tensorflow/compiler/xla/python/py_executable.h", "fingerprint");

    return fingerprint_;
  }

  // Keep `obj` alive as long as PyExecutable.
  void KeepAlive(pybind11::object obj);

 private:
  friend class PyClient;

  std::shared_ptr<PyClient> client_;
  std::shared_ptr<PjRtExecutable> executable_;
  std::shared_ptr<Traceback> traceback_;

  // Identical executables (i.e. representing the same program) will have the
  // same fingerprint. nullopt on platforms or executables where fingerprints
  // aren't implemented.
  absl::optional<std::string> fingerprint_;

  // The options to pass to `executable_.Execute`.
  ExecuteOptions options_;

  // Python objects to keep alive as requested by user.
  std::vector<pybind11::object> keepalives_;

  // Doubly-linked list of all executables known to the client. Protected by the
  // GIL.
  PyExecutable* next_;
  PyExecutable* prev_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_
