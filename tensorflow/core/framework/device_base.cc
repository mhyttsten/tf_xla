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
class MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc() {
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

#include "tensorflow/core/framework/device_base.h"

#include <algorithm>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/notification.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

DeviceBase::~DeviceBase() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/framework/device_base.cc", "DeviceBase::~DeviceBase");

  for (auto& temp : eigen_cpu_devices_) {
    delete temp;
  }
  eigen_cpu_devices_.clear();
}

Status DeviceContext::CopyDeviceTensorToCPUSync(const Tensor* device_tensor,
                                                StringPiece tensor_name,
                                                Device* device,
                                                Tensor* cpu_tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/framework/device_base.cc", "DeviceContext::CopyDeviceTensorToCPUSync");

  absl::Notification n;
  Status status;
  CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor,
                        [&](const Status& s) {
                          status = s;
                          n.Notify();
                        });
  n.WaitForNotification();
  return status;
}

Status DeviceContext::CopyCPUTensorToDeviceSync(const Tensor* cpu_tensor,
                                                Device* device,
                                                Tensor* device_tensor) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/framework/device_base.cc", "DeviceContext::CopyCPUTensorToDeviceSync");

  absl::Notification n;
  Status status;
  CopyCPUTensorToDevice(cpu_tensor, device, device_tensor,
                        [&](const Status& s) {
                          status = s;
                          n.Notify();
                        });
  n.WaitForNotification();
  return status;
}

const DeviceAttributes& DeviceBase::attributes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/framework/device_base.cc", "DeviceBase::attributes");

  LOG(FATAL) << "DeviceBase does not implement attributes()";  // Crash OK
  std::abort();
}

const string& DeviceBase::name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/framework/device_base.cc", "DeviceBase::name");

  LOG(FATAL) << "DeviceBase does not implement name()";  // Crash OK
  std::abort();
}

const DeviceNameUtils::ParsedName& DeviceBase::parsed_name() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_5(mht_5_v, 260, "", "./tensorflow/core/framework/device_base.cc", "DeviceBase::parsed_name");

  LOG(FATAL) << "DeviceBase does not implement parsed_name()";  // Crash OK
  std::abort();
}

void DeviceBase::set_eigen_cpu_device(Eigen::ThreadPoolDevice* d) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_6(mht_6_v, 268, "", "./tensorflow/core/framework/device_base.cc", "DeviceBase::set_eigen_cpu_device");

  // Eigen::ThreadPoolDevice is a very cheap struct (two pointers and
  // an int).  Therefore, we can afford a pre-allocated array of
  // Eigen::ThreadPoolDevice.  Here, we ensure that
  // Eigen::ThreadPoolDevices in eigen_cpu_devices_ has increasingly
  // larger numThreads.
  for (int i = 1; i <= d->numThreads(); ++i) {
    eigen_cpu_devices_.push_back(new Eigen::ThreadPoolDevice(
        d->getPool(), i /* numThreads() */, d->allocator()));
  }
}

const Eigen::ThreadPoolDevice* DeviceBase::eigen_cpu_device() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_7(mht_7_v, 283, "", "./tensorflow/core/framework/device_base.cc", "DeviceBase::eigen_cpu_device");

  // Based on GetPerThreadMaxParallelism(), we return a different
  // pre-allocated Eigen::ThreadPoolDevice. All these ThreadPoolDevice
  // use the same underlying threadpool. But they use different
  // nominal numThreads() hoping that the user of the returned
  // Eigen::ThreadPoolDevice may not aggressively occupy all the
  // threads in the underlying threadpool.
  const int parallelism = std::max<int>(
      1,
      std::min<int>(GetPerThreadMaxParallelism(), eigen_cpu_devices_.size()));
  return eigen_cpu_devices_[parallelism - 1];
}

namespace {

absl::flat_hash_set<std::string>* GetSymbolicDeviceList() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_8(mht_8_v, 301, "", "./tensorflow/core/framework/device_base.cc", "GetSymbolicDeviceList");

  static absl::flat_hash_set<std::string>* symbolic_device_list =
      new absl::flat_hash_set<std::string>();
  return symbolic_device_list;
}

}  // namespace

void AddSymbolicExecutionDevice(const absl::string_view device_name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_9(mht_9_v, 313, "", "./tensorflow/core/framework/device_base.cc", "AddSymbolicExecutionDevice");

  GetSymbolicDeviceList()->insert(std::string(device_name));
}

bool IsSymbolicExecutionDevice(const absl::string_view device_name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("device_name: \"" + std::string(device_name.data(), device_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSdevice_baseDTcc mht_10(mht_10_v, 321, "", "./tensorflow/core/framework/device_base.cc", "IsSymbolicExecutionDevice");

  absl::flat_hash_set<std::string>* symbolic_devices = GetSymbolicDeviceList();
  if (symbolic_devices->contains(device_name)) {
    return true;
  } else {
    return false;
  }
}

}  // namespace tensorflow
