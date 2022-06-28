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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This files implements the configuration management for transfer guards.
// C++ backends responsible for enforcing transfer guard levels.

#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"

#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/types/optional.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"

namespace jax {

namespace py = ::pybind11;

namespace {

// Protected by the GIL.
TransferGuardState& global_state = *new TransferGuardState();

ABSL_CONST_INIT thread_local TransferGuardState thread_local_state;

// The default transfer guard level.
constexpr TransferGuardLevel kDefaultGuardLevel = TransferGuardLevel::kAllow;

// Returns the transfer guard action for a transfer.
TransferGuardAction GetTransferGuardAction(TransferGuardLevel guard_level,
                                           bool explicit_transfer) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_0(mht_0_v, 217, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "GetTransferGuardAction");

  switch (guard_level) {
    case TransferGuardLevel::kAllow:
      return TransferGuardAction::kAllow;
    case TransferGuardLevel::kLog:
      if (explicit_transfer) {
        return TransferGuardAction::kAllow;
      } else {
        return TransferGuardAction::kLog;
      }
    case TransferGuardLevel::kDisallow:
      if (explicit_transfer) {
        return TransferGuardAction::kAllow;
      } else {
        return TransferGuardAction::kDisallow;
      }
    case TransferGuardLevel::kLogExplicit:
      return TransferGuardAction::kLog;
    case TransferGuardLevel::kDisallowExplicit:
      return TransferGuardAction::kDisallow;
    default:
      // Unreachable; gracefully handle the unexpected guard level and prevent a
      // compiler warning.
      return TransferGuardAction::kDisallow;
  }
}

// Returns the transfer guard action for a host-to-device transfer.
// REQUIRES: Python GIL.
TransferGuardAction GetTransferGuardActionForHostToDevice() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "GetTransferGuardActionForHostToDevice");

  return GetTransferGuardAction(
      thread_local_state.host_to_device.value_or(
          global_state.host_to_device.value_or(kDefaultGuardLevel)),
      thread_local_state.explicit_device_put);
}

// Returns the transfer guard action for a device-to-device transfer.
// REQUIRES: Python GIL.
TransferGuardAction GetTransferGuardActionForDeviceToDevice() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_2(mht_2_v, 261, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "GetTransferGuardActionForDeviceToDevice");

  return GetTransferGuardAction(
      thread_local_state.device_to_device.value_or(
          global_state.device_to_device.value_or(kDefaultGuardLevel)),
      thread_local_state.explicit_device_put);
}

// Returns the transfer guard action for a device-to-host transfer.
// REQUIRES: Python GIL.
TransferGuardAction GetTransferGuardActionForDeviceToHost() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_3(mht_3_v, 273, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "GetTransferGuardActionForDeviceToHost");

  return GetTransferGuardAction(
      thread_local_state.device_to_host.value_or(
          global_state.device_to_host.value_or(kDefaultGuardLevel)),
      thread_local_state.explicit_device_get);
}

}  // namespace

xla::Status ApplyTransferGuardToHostToDevice(
    absl::FunctionRef<std::string()> formatter) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_4(mht_4_v, 286, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "ApplyTransferGuardToHostToDevice");

  switch (GetTransferGuardActionForHostToDevice()) {
    case TransferGuardAction::kAllow:
      break;
    case TransferGuardAction::kLog:
      LOG(WARNING) << "host-to-device transfer: " << formatter();
      break;
    case TransferGuardAction::kDisallow:
      return xla::InvalidArgument("Disallowed host-to-device transfer: %s",
                                  formatter());
  }
  return xla::Status::OK();
}

xla::Status ApplyTransferGuardToDeviceToDevice(
    absl::FunctionRef<std::string()> formatter) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_5(mht_5_v, 304, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "ApplyTransferGuardToDeviceToDevice");

  switch (GetTransferGuardActionForDeviceToDevice()) {
    case TransferGuardAction::kAllow:
      break;
    case TransferGuardAction::kLog:
      LOG(WARNING) << "device-to-device transfer: " << formatter();
      break;
    case TransferGuardAction::kDisallow:
      return xla::InvalidArgument("Disallowed device-to-device transfer: %s",
                                  formatter());
  }
  return xla::Status::OK();
}

xla::Status ApplyTransferGuardToDeviceToHost(
    absl::FunctionRef<std::string()> formatter) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_6(mht_6_v, 322, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "ApplyTransferGuardToDeviceToHost");

  switch (GetTransferGuardActionForDeviceToHost()) {
    case TransferGuardAction::kAllow:
      break;
    case TransferGuardAction::kLog:
      LOG(WARNING) << "device-to-host transfer: " << formatter();
      break;
    case TransferGuardAction::kDisallow:
      return xla::InvalidArgument("Disallowed device-to-host transfer: %s",
                                  formatter());
  }
  return xla::Status::OK();
}

void BuildTransferGuardSubmodule(py::module& m) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStransfer_guard_libDTcc mht_7(mht_7_v, 339, "", "./tensorflow/compiler/xla/python/transfer_guard_lib.cc", "BuildTransferGuardSubmodule");

  py::module tglib = m.def_submodule("transfer_guard_lib",
                                     "Jax transfer guard support library");

  py::enum_<TransferGuardLevel> tglevel(tglib, "TransferGuardLevel");
  tglevel.value("ALLOW", TransferGuardLevel::kAllow);
  tglevel.value("LOG", TransferGuardLevel::kLog);
  tglevel.value("DISALLOW", TransferGuardLevel::kDisallow);
  tglevel.value("LOG_EXPLICIT", TransferGuardLevel::kLogExplicit);
  tglevel.value("DISALLOW_EXPLICIT", TransferGuardLevel::kDisallowExplicit);

  py::class_<TransferGuardState> tgstate(tglib, "TransferGuardState");
  tgstate.def_readwrite("host_to_device", &TransferGuardState::host_to_device);
  tgstate.def_readwrite("device_to_device",
                        &TransferGuardState::device_to_device);
  tgstate.def_readwrite("device_to_host", &TransferGuardState::device_to_host);
  tgstate.def_readwrite("explicit_device_put",
                        &TransferGuardState::explicit_device_put);
  tgstate.def_readwrite("explicit_device_get",
                        &TransferGuardState::explicit_device_get);

  tglib.def(
      "global_state", [&]() { return &global_state; },
      py::return_value_policy::reference);
  tglib.def(
      "thread_local_state", [&]() { return &thread_local_state; },
      py::return_value_policy::reference);
}

}  // namespace jax
