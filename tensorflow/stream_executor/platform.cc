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
class MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/platform.h"

#include "tensorflow/stream_executor/platform/port.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

std::string PlatformKindString(PlatformKind kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_0(mht_0_v, 196, "", "./tensorflow/stream_executor/platform.cc", "PlatformKindString");

  switch (kind) {
    case PlatformKind::kCuda:
      return "CUDA";
    case PlatformKind::kROCm:
      return "ROCm";
    case PlatformKind::kOpenCL:
      return "OpenCL";
    case PlatformKind::kHost:
      return "Host";
    case PlatformKind::kMock:
      return "Mock";
    default:
      return absl::StrCat("InvalidPlatformKind(", static_cast<int>(kind), ")");
  }
}

PlatformKind PlatformKindFromString(std::string kind) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("kind: \"" + kind + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_1(mht_1_v, 217, "", "./tensorflow/stream_executor/platform.cc", "PlatformKindFromString");

  for (int i = 0; i < static_cast<int>(PlatformKind::kSize); ++i) {
    if (kind == PlatformKindString(static_cast<PlatformKind>(i))) {
      return static_cast<PlatformKind>(i);
    }
  }

  return PlatformKind::kInvalid;
}

bool PlatformIsRunnable(PlatformKind kind) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_2(mht_2_v, 230, "", "./tensorflow/stream_executor/platform.cc", "PlatformIsRunnable");

  switch (kind) {
    case PlatformKind::kCuda:
    case PlatformKind::kROCm:
    case PlatformKind::kOpenCL:
    case PlatformKind::kHost:
      return true;
    default:
      return false;
  }
}

bool PlatformIsRunnableOnDevice(PlatformKind kind) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_3(mht_3_v, 245, "", "./tensorflow/stream_executor/platform.cc", "PlatformIsRunnableOnDevice");

  switch (kind) {
    case PlatformKind::kCuda:
    case PlatformKind::kROCm:
    case PlatformKind::kOpenCL:
      return true;
    default:
      return false;
  }
}

void CheckPlatformKindIsValid(PlatformKind kind) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_4(mht_4_v, 259, "", "./tensorflow/stream_executor/platform.cc", "CheckPlatformKindIsValid");

  CHECK(static_cast<int>(PlatformKind::kCuda) <= static_cast<int>(kind) &&
        static_cast<int>(kind) <= static_cast<int>(PlatformKind::kMock))
      << "invalid GPU executor kind: " << PlatformKindString(kind);
}

StreamExecutorConfig::StreamExecutorConfig()
    : ordinal(-1), device_options(DeviceOptions::Default()) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_5(mht_5_v, 269, "", "./tensorflow/stream_executor/platform.cc", "StreamExecutorConfig::StreamExecutorConfig");
}

StreamExecutorConfig::StreamExecutorConfig(int ordinal_in)
    : ordinal(ordinal_in), device_options(DeviceOptions::Default()) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_6(mht_6_v, 275, "", "./tensorflow/stream_executor/platform.cc", "StreamExecutorConfig::StreamExecutorConfig");
}

Platform::~Platform() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_7(mht_7_v, 280, "", "./tensorflow/stream_executor/platform.cc", "Platform::~Platform");
}

bool Platform::Initialized() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_8(mht_8_v, 285, "", "./tensorflow/stream_executor/platform.cc", "Platform::Initialized");
 return true; }

port::Status Platform::Initialize(
    const std::map<std::string, std::string> &platform_options) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_9(mht_9_v, 291, "", "./tensorflow/stream_executor/platform.cc", "Platform::Initialize");

  if (!platform_options.empty()) {
    return port::Status(port::error::UNIMPLEMENTED,
                        "this platform does not support custom initialization");
  }
  return port::Status::OK();
}

port::Status Platform::ForceExecutorShutdown() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_10(mht_10_v, 302, "", "./tensorflow/stream_executor/platform.cc", "Platform::ForceExecutorShutdown");

  return port::Status(port::error::UNIMPLEMENTED,
                      "executor shutdown is not supported on this platform");
}

std::unique_ptr<Platform::PeerAccessMap> Platform::GetPeerAccessMap() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_11(mht_11_v, 310, "", "./tensorflow/stream_executor/platform.cc", "Platform::GetPeerAccessMap");

  auto *map = new PeerAccessMap;

  int device_count = VisibleDeviceCount();
  for (int i = 0; i < device_count; ++i) {
    for (int j = 0; j < device_count; ++j) {
      StreamExecutor *from = ExecutorForDevice(i).ValueOrDie();
      StreamExecutor *to = ExecutorForDevice(j).ValueOrDie();
      (*map)[{i, j}] = from->CanEnablePeerAccessTo(to);
    }
  }

  return std::unique_ptr<Platform::PeerAccessMap>{map};
}

port::Status Platform::EnablePeerAccess() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSplatformDTcc mht_12(mht_12_v, 328, "", "./tensorflow/stream_executor/platform.cc", "Platform::EnablePeerAccess");

  auto peer_access_map = GetPeerAccessMap();
  for (const auto &access : *peer_access_map) {
    auto devices = access.first;
    if (access.second) {
      StreamExecutor *from = ExecutorForDevice(devices.first).ValueOrDie();
      StreamExecutor *to = ExecutorForDevice(devices.second).ValueOrDie();
      auto status = from->EnablePeerAccessTo(to);
      if (!status.ok()) {
        return status;
      }
    } else {
      LOG(INFO) << "cannot enable peer access from device ordinal "
                << devices.first << " to device ordinal " << devices.second;
    }
  }
  return port::Status::OK();
}

}  // namespace stream_executor
