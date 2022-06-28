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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/renamed_device.h"

#include "absl/memory/memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {

/* static */
std::unique_ptr<Device> RenamedDevice::NewRenamedDevice(
    const string& new_base, Device* underlying, bool owns_underlying,
    bool isolate_session_state,
    thread::ThreadPoolInterface* underlying_threadpool) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("new_base: \"" + new_base + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/common_runtime/renamed_device.cc", "RenamedDevice::NewRenamedDevice");

  DeviceNameUtils::ParsedName parsed_name;
  CHECK(DeviceNameUtils::ParseFullName(new_base, &parsed_name));
  DeviceNameUtils::ParsedName underlying_parsed_name =
      underlying->parsed_name();
  CHECK(underlying_parsed_name.has_type);
  CHECK(underlying_parsed_name.has_id);
  parsed_name.type = underlying_parsed_name.type;
  parsed_name.id = underlying_parsed_name.id;
  string name = DeviceNameUtils::FullName(parsed_name.job, parsed_name.replica,
                                          parsed_name.task, parsed_name.type,
                                          parsed_name.id);
  DeviceAttributes attributes(underlying->attributes());
  attributes.set_name(name);
  // Call absl::WrapUnique to access private constructor.
  return absl::WrapUnique(
      new RenamedDevice(underlying, attributes, owns_underlying,
                        isolate_session_state, underlying_threadpool));
}

RenamedDevice::RenamedDevice(Device* underlying,
                             const DeviceAttributes& attributes,
                             bool owns_underlying_device,
                             bool isolate_session_state,
                             thread::ThreadPoolInterface* underlying_threadpool)
    : Device(underlying->env(), attributes),
      underlying_device_(underlying),
      owns_underlying_device_(owns_underlying_device),
      isolate_session_state_(isolate_session_state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/common_runtime/renamed_device.cc", "RenamedDevice::RenamedDevice");

  if (underlying_threadpool != nullptr) {
    underlying_threadpool_.reset(new thread::ThreadPool(underlying_threadpool));
    eigen_worker_threads_.workers = underlying_threadpool_.get();
    eigen_worker_threads_.num_threads = underlying_threadpool->NumThreads();
    set_tensorflow_cpu_worker_threads(&eigen_worker_threads_);
    set_tensorflow_device_thread_pool(underlying_threadpool_.get());

    Eigen::ThreadPoolDevice eigen_threadpool_device(
        underlying_threadpool, underlying_threadpool->NumThreads());
    set_eigen_cpu_device(&eigen_threadpool_device);
  }
}

RenamedDevice::~RenamedDevice() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSrenamed_deviceDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/common_runtime/renamed_device.cc", "RenamedDevice::~RenamedDevice");

  if (owns_underlying_device_) {
    delete underlying_device_;
  }
}

}  // namespace tensorflow
