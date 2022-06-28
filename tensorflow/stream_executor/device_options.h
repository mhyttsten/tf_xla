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

// Contains device-level options that can be specified at a platform level.
// Example usage:
//    auto device_options = DeviceOptions::Default();

#ifndef TENSORFLOW_STREAM_EXECUTOR_DEVICE_OPTIONS_H_
#define TENSORFLOW_STREAM_EXECUTOR_DEVICE_OPTIONS_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh() {
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


#include <map>

#include "absl/strings/str_join.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

// Indicates a set of options for a device's usage, which generally must be
// provided at StreamExecutor device-initialization time.
//
// These are intended to be useful-but-not-mandatorily-supported options for
// using devices on the underlying platform. Presently, if the option requested
// is not available on the target platform, a warning will be emitted.
struct DeviceOptions {
 public:
  // When it is observed that more memory has to be allocated for thread stacks,
  // this flag prevents it from ever being deallocated. Potentially saves
  // thrashing the thread stack memory allocation, but at the potential cost of
  // some memory space.
  static constexpr unsigned kDoNotReclaimStackAllocation = 0x1;

  // The following options refer to synchronization options when
  // using SynchronizeStream or SynchronizeContext.

  // Synchronize with spinlocks.
  static constexpr unsigned kScheduleSpin = 0x02;
  // Synchronize with spinlocks that also call CPU yield instructions.
  static constexpr unsigned kScheduleYield = 0x04;
  // Synchronize with a "synchronization primitive" (e.g. mutex).
  static constexpr unsigned kScheduleBlockingSync = 0x08;

  static constexpr unsigned kMask = 0xf;  // Mask of all available flags.

  // Constructs an or-d together set of device options.
  explicit DeviceOptions(unsigned flags) : flags_(flags) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh mht_0(mht_0_v, 227, "", "./tensorflow/stream_executor/device_options.h", "DeviceOptions");

    CHECK((flags & kMask) == flags);
  }

  // Factory for the default set of device options.
  static DeviceOptions Default() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh mht_1(mht_1_v, 235, "", "./tensorflow/stream_executor/device_options.h", "Default");
 return DeviceOptions(0); }

  unsigned flags() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh mht_2(mht_2_v, 240, "", "./tensorflow/stream_executor/device_options.h", "flags");
 return flags_; }

  bool operator==(const DeviceOptions& other) const {
    return flags_ == other.flags_ &&
           non_portable_tags == other.non_portable_tags;
  }

  bool operator!=(const DeviceOptions& other) const {
    return !(*this == other);
  }

  std::string ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_optionsDTh mht_3(mht_3_v, 254, "", "./tensorflow/stream_executor/device_options.h", "ToString");

    std::vector<std::string> flags_on;
    if (flags_ & kDoNotReclaimStackAllocation) {
      flags_on.push_back("kDoNotReclaimStackAllocation");
    }
    if (flags_ & kScheduleSpin) {
      flags_on.push_back("kScheduleSpin");
    }
    if (flags_ & kScheduleYield) {
      flags_on.push_back("kScheduleYield");
    }
    if (flags_ & kScheduleBlockingSync) {
      flags_on.push_back("kScheduleBlockingSync");
    }
    return flags_on.empty() ? "none" : absl::StrJoin(flags_on, "|");
  }

  // Platform-specific device options. Expressed as key-value pairs to avoid
  // DeviceOptions subclass proliferation.
  std::map<std::string, std::string> non_portable_tags;

 private:
  unsigned flags_;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DEVICE_OPTIONS_H_
