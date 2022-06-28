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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc() {
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

#include "tensorflow/compiler/xla/python/py_executable.h"

#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace xla {

namespace py = pybind11;

PyExecutable::PyExecutable(std::shared_ptr<PyClient> client,
                           std::unique_ptr<PjRtExecutable> executable,
                           std::shared_ptr<Traceback> traceback,
                           absl::optional<std::string> fingerprint)
    : client_(std::move(client)),
      executable_(std::move(executable)),
      traceback_(std::move(traceback)),
      fingerprint_(std::move(fingerprint)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::PyExecutable");

  CHECK(PyGILState_Check());
  next_ = client_->executables_;
  client_->executables_ = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
  options_.untuple_result = true;
  if (fingerprint_) {
    options_.launch_id = tensorflow::Fingerprint32(*fingerprint_);
    VLOG(1) << "Fingerprint for executable " << executable_->name() << ": "
            << *fingerprint_;
  }
}

PyExecutable::~PyExecutable() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::~PyExecutable");

  CHECK(PyGILState_Check());
  if (client_->executables_ == this) {
    client_->executables_ = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

std::vector<ClientAndPtr<PjRtDevice>> PyExecutable::AddressableDevices() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::AddressableDevices");

  std::vector<ClientAndPtr<PjRtDevice>> devices;
  devices.reserve(executable_->addressable_devices().size());
  for (PjRtDevice* device : executable_->addressable_devices()) {
    devices.push_back(WrapWithClient(client_, device));
  }
  return devices;
}

StatusOr<std::vector<PyBuffer::object>> PyExecutable::Execute(
    absl::Span<PyBuffer::object const> args) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_3(mht_3_v, 252, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::Execute");

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  {
    py::gil_scoped_release gil_release;
    std::vector<PjRtBuffer*> arg_buffers(args.size());
    absl::c_transform(
        args, arg_buffers.begin(),
        [](const PyBuffer::object& buf) { return buf.buf()->buffer(); });
    TF_ASSIGN_OR_RETURN(output_buffers,
                        executable_->Execute({arg_buffers}, options_));
  }
  auto traceback = Traceback::Get();
  std::vector<PyBuffer::object> outputs;
  outputs.reserve(output_buffers[0].size());
  for (auto& buffer : output_buffers[0]) {
    outputs.push_back(PyBuffer::Make(client_, std::move(buffer), traceback));
  }
  return outputs;
}

StatusOr<std::vector<std::vector<PyBuffer::object>>>
PyExecutable::ExecuteShardedOnLocalDevices(
    absl::Span<const std::vector<PyBuffer::object>> args) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::ExecuteShardedOnLocalDevices");

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  int num_computations = executable_->addressable_devices().size();
  {
    py::gil_scoped_release gil_release;
    for (const auto& arg : args) {
      if (arg.size() != num_computations) {
        return xla::InvalidArgument(
            "Expected args to execute_sharded_on_local_devices to have %d "
            "shards, got: [%s]",
            num_computations,
            absl::StrJoin(
                args, ", ",
                [](std::string* out, const std::vector<PyBuffer::object>& arg) {
                  out->append(std::to_string(arg.size()));
                }));
      }
    }
    std::vector<std::vector<PjRtBuffer*>> arg_buffers(num_computations);
    const int num_args = args.size();
    for (int computation = 0; computation < num_computations; ++computation) {
      arg_buffers[computation].resize(num_args);
      absl::c_transform(args, arg_buffers[computation].begin(),
                        [&](const std::vector<PyBuffer::object>& arg) {
                          return arg[computation].buf()->buffer();
                        });
    }
    TF_ASSIGN_OR_RETURN(output_buffers,
                        executable_->Execute(arg_buffers, options_));
  }
  auto traceback = Traceback::Get();
  int num_output_buffers = output_buffers[0].size();
  std::vector<std::vector<PyBuffer::object>> outputs;
  outputs.resize(num_output_buffers);
  for (int buffer_id = 0; buffer_id < num_output_buffers; ++buffer_id) {
    outputs[buffer_id].reserve(num_computations);
    for (int computation = 0; computation < num_computations; ++computation) {
      outputs[buffer_id].push_back(PyBuffer::Make(
          client_, std::move(output_buffers[computation][buffer_id]),
          traceback));
    }
  }
  return outputs;
}

StatusOr<std::vector<std::shared_ptr<HloModule>>> PyExecutable::HloModules()
    const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_5(mht_5_v, 326, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::HloModules");

  return executable_->GetHloModules();
}

void PyExecutable::KeepAlive(py::object obj) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPSpy_executableDTcc mht_6(mht_6_v, 333, "", "./tensorflow/compiler/xla/python/py_executable.cc", "PyExecutable::KeepAlive");

  keepalives_.push_back(std::move(obj));
}

}  // namespace xla
