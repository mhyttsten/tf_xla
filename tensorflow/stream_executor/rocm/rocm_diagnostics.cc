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
class MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"

#include <dirent.h>
#include <limits.h>
#include <link.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/numbers.h"
#include "tensorflow/stream_executor/lib/process_state.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"

namespace stream_executor {
namespace rocm {

string DriverVersionToString(DriverVersion version) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "DriverVersionToString");

  return absl::StrFormat("%d.%d.%d", std::get<0>(version), std::get<1>(version),
                         std::get<2>(version));
}

string DriverVersionStatusToString(port::StatusOr<DriverVersion> version) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "DriverVersionStatusToString");

  if (!version.ok()) {
    return version.status().ToString();
  }

  return DriverVersionToString(version.ValueOrDie());
}

port::StatusOr<DriverVersion> StringToDriverVersion(const string& value) {
  std::vector<string> pieces = absl::StrSplit(value, '.');
  if (pieces.size() != 2 && pieces.size() != 3) {
    return port::Status{port::error::INVALID_ARGUMENT,
                        absl::StrFormat("expected %%d.%%d or %%d.%%d.%%d form "
                                        "for driver version; got \"%s\"",
                                        value.c_str())};
  }

  int major;
  int minor;
  int patch = 0;
  if (!port::safe_strto32(pieces[0], &major)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse major version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[0].c_str(), value.c_str())};
  }
  if (!port::safe_strto32(pieces[1], &minor)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse minor version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[1].c_str(), value.c_str())};
  }
  if (pieces.size() == 3 && !port::safe_strto32(pieces[2], &patch)) {
    return port::Status{
        port::error::INVALID_ARGUMENT,
        absl::StrFormat("could not parse patch version number \"%s\" as an "
                        "integer from string \"%s\"",
                        pieces[2].c_str(), value.c_str())};
  }

  DriverVersion result{major, minor, patch};
  VLOG(2) << "version string \"" << value << "\" made value "
          << DriverVersionToString(result);
  return result;
}

}  // namespace rocm
}  // namespace stream_executor

namespace stream_executor {
namespace gpu {

// -- class Diagnostician

string Diagnostician::GetDevNodePath(int dev_node_ordinal) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_2(mht_2_v, 282, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::GetDevNodePath");

  return absl::StrCat("/dev/kfd", dev_node_ordinal);
}

void Diagnostician::LogDiagnosticInformation() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_3(mht_3_v, 289, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::LogDiagnosticInformation");

  LOG(INFO) << "retrieving ROCM diagnostic information for host: "
            << port::Hostname();

  LogDriverVersionInformation();
}

/* static */ void Diagnostician::LogDriverVersionInformation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_4(mht_4_v, 299, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::LogDriverVersionInformation");

  LOG(INFO) << "hostname: " << port::Hostname();
  if (VLOG_IS_ON(1)) {
    const char* value = getenv("LD_LIBRARY_PATH");
    string library_path = value == nullptr ? "" : value;
    VLOG(1) << "LD_LIBRARY_PATH is: \"" << library_path << "\"";

    std::vector<string> pieces = absl::StrSplit(library_path, ':');
    for (const auto& piece : pieces) {
      if (piece.empty()) {
        continue;
      }
      DIR* dir = opendir(piece.c_str());
      if (dir == nullptr) {
        VLOG(1) << "could not open \"" << piece << "\"";
        continue;
      }
      while (dirent* entity = readdir(dir)) {
        VLOG(1) << piece << " :: " << entity->d_name;
      }
      closedir(dir);
    }
  }
  port::StatusOr<DriverVersion> dso_version = FindDsoVersion();
  LOG(INFO) << "librocm reported version is: "
            << rocm::DriverVersionStatusToString(dso_version);

  port::StatusOr<DriverVersion> kernel_version = FindKernelDriverVersion();
  LOG(INFO) << "kernel reported version is: "
            << rocm::DriverVersionStatusToString(kernel_version);

  if (kernel_version.ok() && dso_version.ok()) {
    WarnOnDsoKernelMismatch(dso_version, kernel_version);
  }
}

// Iterates through loaded DSOs with DlIteratePhdrCallback to find the
// driver-interfacing DSO version number. Returns it as a string.
port::StatusOr<DriverVersion> Diagnostician::FindDsoVersion() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_5(mht_5_v, 340, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::FindDsoVersion");

  port::StatusOr<DriverVersion> result{port::Status{
      port::error::NOT_FOUND,
      "was unable to find librocm.so DSO loaded into this program"}};

  // Callback used when iterating through DSOs. Looks for the driver-interfacing
  // DSO and yields its version number into the callback data, when found.
  auto iterate_phdr = [](struct dl_phdr_info* info, size_t size,
                         void* data) -> int {
    if (strstr(info->dlpi_name, "librocm.so.1")) {
      VLOG(1) << "found DLL info with name: " << info->dlpi_name;
      char resolved_path[PATH_MAX] = {0};
      if (realpath(info->dlpi_name, resolved_path) == nullptr) {
        return 0;
      }
      VLOG(1) << "found DLL info with resolved path: " << resolved_path;
      const char* slash = rindex(resolved_path, '/');
      if (slash == nullptr) {
        return 0;
      }
      const char* so_suffix = ".so.";
      const char* dot = strstr(slash, so_suffix);
      if (dot == nullptr) {
        return 0;
      }
      string dso_version = dot + strlen(so_suffix);
      // TODO(b/22689637): Eliminate the explicit namespace if possible.
      auto stripped_dso_version = absl::StripSuffix(dso_version, ".ld64");
      auto result = static_cast<port::StatusOr<DriverVersion>*>(data);
      *result = rocm::StringToDriverVersion(string(stripped_dso_version));
      return 1;
    }
    return 0;
  };

  dl_iterate_phdr(iterate_phdr, &result);

  return result;
}

port::StatusOr<DriverVersion> Diagnostician::FindKernelModuleVersion(
    const string& driver_version_file_contents) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("driver_version_file_contents: \"" + driver_version_file_contents + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_6(mht_6_v, 385, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::FindKernelModuleVersion");

  static const char* kDriverFilePrelude = "Kernel Module  ";
  size_t offset = driver_version_file_contents.find(kDriverFilePrelude);
  if (offset == string::npos) {
    return port::Status{
        port::error::NOT_FOUND,
        absl::StrCat("could not find kernel module information in "
                     "driver version file contents: \"",
                     driver_version_file_contents, "\"")};
  }

  string version_and_rest = driver_version_file_contents.substr(
      offset + strlen(kDriverFilePrelude), string::npos);
  size_t space_index = version_and_rest.find(" ");
  auto kernel_version = version_and_rest.substr(0, space_index);
  // TODO(b/22689637): Eliminate the explicit namespace if possible.
  auto stripped_kernel_version = absl::StripSuffix(kernel_version, ".ld64");
  return rocm::StringToDriverVersion(string(stripped_kernel_version));
}

void Diagnostician::WarnOnDsoKernelMismatch(
    port::StatusOr<DriverVersion> dso_version,
    port::StatusOr<DriverVersion> kernel_version) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_7(mht_7_v, 410, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::WarnOnDsoKernelMismatch");

  if (kernel_version.ok() && dso_version.ok() &&
      dso_version.ValueOrDie() == kernel_version.ValueOrDie()) {
    LOG(INFO) << "kernel version seems to match DSO: "
              << rocm::DriverVersionToString(kernel_version.ValueOrDie());
  } else {
    LOG(ERROR) << "kernel version "
               << rocm::DriverVersionStatusToString(kernel_version)
               << " does not match DSO version "
               << rocm::DriverVersionStatusToString(dso_version)
               << " -- cannot find working devices in this configuration";
  }
}

port::StatusOr<DriverVersion> Diagnostician::FindKernelDriverVersion() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSrocmPSrocm_diagnosticsDTcc mht_8(mht_8_v, 427, "", "./tensorflow/stream_executor/rocm/rocm_diagnostics.cc", "Diagnostician::FindKernelDriverVersion");

  auto status = port::Status{port::error::UNIMPLEMENTED,
                             "kernel reported driver version not implemented"};
  return status;
}

}  // namespace gpu
}  // namespace stream_executor
