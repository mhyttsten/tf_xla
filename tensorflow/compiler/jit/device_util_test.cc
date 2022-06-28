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
class MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_util.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Status PickDeviceHelper(bool allow_mixing_unknown_and_cpu,
                        absl::Span<const absl::string_view> device_names,
                        string* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/jit/device_util_test.cc", "PickDeviceHelper");

  jit::DeviceInfoCache cache;
  jit::DeviceSet device_set;
  for (absl::string_view name : device_names) {
    TF_ASSIGN_OR_RETURN(jit::DeviceId device_id, cache.GetIdFor(name));
    device_set.Insert(device_id);
  }

  TF_ASSIGN_OR_RETURN(
      jit::DeviceId result_id,
      PickDeviceForXla(cache, device_set, allow_mixing_unknown_and_cpu));
  *result = string(cache.GetNameFor(result_id));
  return Status::OK();
}

void CheckPickDeviceResult(absl::string_view expected_result,
                           bool allow_mixing_unknown_and_cpu,
                           absl::Span<const absl::string_view> inputs) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("expected_result: \"" + std::string(expected_result.data(), expected_result.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/compiler/jit/device_util_test.cc", "CheckPickDeviceResult");

  string result;
  TF_ASSERT_OK(PickDeviceHelper(allow_mixing_unknown_and_cpu, inputs, &result))
      << "inputs = [" << absl::StrJoin(inputs, ", ")
      << "], allow_mixing_unknown_and_cpu=" << allow_mixing_unknown_and_cpu
      << ", expected_result=" << expected_result;
  EXPECT_EQ(result, expected_result);
}

void CheckPickDeviceHasError(bool allow_mixing_unknown_and_cpu,
                             absl::Span<const absl::string_view> inputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/jit/device_util_test.cc", "CheckPickDeviceHasError");

  string result;
  EXPECT_FALSE(
      PickDeviceHelper(allow_mixing_unknown_and_cpu, inputs, &result).ok());
}

const char* kCPU0 = "/job:localhost/replica:0/task:0/device:CPU:0";
const char* kGPU0 = "/job:localhost/replica:0/task:0/device:GPU:0";
const char* kXPU0 = "/job:localhost/replica:0/task:0/device:XPU:0";
const char* kYPU0 = "/job:localhost/replica:0/task:0/device:YPU:0";

const char* kCPU1 = "/job:localhost/replica:0/task:0/device:CPU:1";
const char* kGPU1 = "/job:localhost/replica:0/task:0/device:GPU:1";
const char* kXPU1 = "/job:localhost/replica:0/task:0/device:XPU:1";

const char* kCPU0Partial = "/device:CPU:0";
const char* kGPU0Partial = "/device:GPU:0";
const char* kXPU0Partial = "/device:XPU:0";

TEST(PickDeviceForXla, UniqueDevice) {
  CheckPickDeviceResult(kGPU0, false, {kGPU0, kGPU0});
}

TEST(PickDeviceForXla, MoreSpecificDevice) {
  CheckPickDeviceResult(kCPU0, false, {kCPU0, kCPU0Partial});
  CheckPickDeviceResult(kGPU0, false, {kGPU0, kGPU0Partial});
  // Unknown devices do not support merging of full and partial specifications.
  CheckPickDeviceHasError(false, {kXPU1, kXPU0Partial});
}

TEST(PickDeviceForXla, DeviceOrder) {
  CheckPickDeviceResult(kGPU0, false, {kGPU0, kCPU0});
  CheckPickDeviceResult(kGPU0, false, {kCPU0, kGPU0});
  CheckPickDeviceResult(kXPU0, true, {kXPU0, kCPU0});
}

TEST(PickDeviceForXla, MultipleUnknownDevices) {
  CheckPickDeviceHasError(false, {kXPU0, kYPU0});
}

TEST(PickDeviceForXla, GpuAndUnknown) {
  CheckPickDeviceHasError(false, {kGPU0, kXPU1});
}

TEST(PickDeviceForXla, UnknownAndCpu) {
  CheckPickDeviceHasError(false, {kXPU0, kCPU1});
}

TEST(PickDeviceForXla, MultipleDevicesOfSameType) {
  CheckPickDeviceHasError(true, {kCPU0, kCPU1});
  CheckPickDeviceHasError(false, {kCPU0, kCPU1});
  CheckPickDeviceHasError(false, {kGPU0, kGPU1});
  CheckPickDeviceHasError(false, {kXPU0, kXPU1});
  CheckPickDeviceHasError(false, {kCPU0, kCPU1, kGPU0});
}

void SimpleRoundTripTestForDeviceSet(int num_devices) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSjitPSdevice_util_testDTcc mht_3(mht_3_v, 288, "", "./tensorflow/compiler/jit/device_util_test.cc", "SimpleRoundTripTestForDeviceSet");

  jit::DeviceSet device_set;
  jit::DeviceInfoCache device_info_cache;

  std::vector<string> expected_devices, actual_devices;

  for (int i = 0; i < num_devices; i++) {
    string device_name =
        absl::StrCat("/job:localhost/replica:0/task:0/device:XPU:", i);
    TF_ASSERT_OK_AND_ASSIGN(jit::DeviceId device_id,
                            device_info_cache.GetIdFor(device_name));
    device_set.Insert(device_id);
    expected_devices.push_back(device_name);
  }

  device_set.ForEach([&](jit::DeviceId device_id) {
    actual_devices.push_back(string(device_info_cache.GetNameFor(device_id)));
    return true;
  });

  EXPECT_EQ(expected_devices, actual_devices);
}

TEST(DeviceSetTest, SimpleRoundTrip_One) { SimpleRoundTripTestForDeviceSet(1); }

TEST(DeviceSetTest, SimpleRoundTrip_Small) {
  SimpleRoundTripTestForDeviceSet(8);
}

TEST(DeviceSetTest, SimpleRoundTrip_Large) {
  SimpleRoundTripTestForDeviceSet(800);
}

}  // namespace
}  // namespace tensorflow
