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
class MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc() {
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
#include "tensorflow/lite/nnapi/nnapi_handler.h"

#include <cstdio>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

// static
const char NnApiHandler::kNnapiReferenceDeviceName[] = "nnapi-reference";
// static
const int NnApiHandler::kNnapiReferenceDevice = 1;
// static
const int NnApiHandler::kNnapiDevice = 2;

char* NnApiHandler::nnapi_device_name_ = nullptr;
int NnApiHandler::nnapi_device_feature_level_;

const NnApi* NnApiPassthroughInstance() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiPassthroughInstance");

  static const NnApi orig_nnapi_copy = *NnApiImplementation();
  return &orig_nnapi_copy;
}

// static
NnApiHandler* NnApiHandler::Instance() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiHandler::Instance");

  // Ensuring that the original copy of nnapi is saved before we return
  // access to NnApiHandler
  NnApiPassthroughInstance();
  static NnApiHandler handler{const_cast<NnApi*>(NnApiImplementation())};
  return &handler;
}

void NnApiHandler::Reset() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiHandler::Reset");

  // Restores global NNAPI to original value
  *nnapi_ = *NnApiPassthroughInstance();
}

void NnApiHandler::SetAndroidSdkVersion(int version,
                                        bool set_unsupported_ops_to_null) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_3(mht_3_v, 232, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiHandler::SetAndroidSdkVersion");

  nnapi_->android_sdk_version = version;
  nnapi_->nnapi_runtime_feature_level = version;

  if (!set_unsupported_ops_to_null) {
    return;
  }

  if (version < 29) {
    nnapi_->ANeuralNetworks_getDeviceCount = nullptr;
    nnapi_->ANeuralNetworks_getDevice = nullptr;
    nnapi_->ANeuralNetworksDevice_getName = nullptr;
    nnapi_->ANeuralNetworksDevice_getVersion = nullptr;
    nnapi_->ANeuralNetworksDevice_getFeatureLevel = nullptr;
    nnapi_->ANeuralNetworksDevice_getType = nullptr;
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices = nullptr;
    nnapi_->ANeuralNetworksCompilation_createForDevices = nullptr;
    nnapi_->ANeuralNetworksCompilation_setCaching = nullptr;
    nnapi_->ANeuralNetworksExecution_compute = nullptr;
    nnapi_->ANeuralNetworksExecution_getOutputOperandRank = nullptr;
    nnapi_->ANeuralNetworksExecution_getOutputOperandDimensions = nullptr;
    nnapi_->ANeuralNetworksBurst_create = nullptr;
    nnapi_->ANeuralNetworksBurst_free = nullptr;
    nnapi_->ANeuralNetworksExecution_burstCompute = nullptr;
    nnapi_->ANeuralNetworksMemory_createFromAHardwareBuffer = nullptr;
    nnapi_->ANeuralNetworksExecution_setMeasureTiming = nullptr;
    nnapi_->ANeuralNetworksExecution_getDuration = nullptr;
    nnapi_->ANeuralNetworksDevice_getExtensionSupport = nullptr;
    nnapi_->ANeuralNetworksModel_getExtensionOperandType = nullptr;
    nnapi_->ANeuralNetworksModel_getExtensionOperationType = nullptr;
    nnapi_->ANeuralNetworksModel_setOperandExtensionData = nullptr;
  }
  if (version < 28) {
    nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16 = nullptr;
  }
}

void NnApiHandler::SetDeviceName(const std::string& name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_4(mht_4_v, 273, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiHandler::SetDeviceName");

  delete[] nnapi_device_name_;
  nnapi_device_name_ = new char[name.size() + 1];
  std::strcpy(nnapi_device_name_, name.c_str());  // NOLINT
}

void NnApiHandler::GetDeviceNameReturnsName(const std::string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_5(mht_5_v, 283, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiHandler::GetDeviceNameReturnsName");

  NnApiHandler::SetDeviceName(name);
  GetDeviceNameReturns<0>();
}

void NnApiHandler::SetNnapiSupportedDevice(const std::string& name,
                                           int feature_level) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTcc mht_6(mht_6_v, 293, "", "./tensorflow/lite/nnapi/nnapi_handler.cc", "NnApiHandler::SetNnapiSupportedDevice");

  NnApiHandler::SetDeviceName(name);
  nnapi_device_feature_level_ = feature_level;

  GetDeviceCountReturnsCount<2>();
  nnapi_->ANeuralNetworks_getDevice =
      [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
    if (devIndex > 1) {
      return ANEURALNETWORKS_BAD_DATA;
    }

    if (devIndex == 1) {
      *device =
          reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice);
    } else {
      *device = reinterpret_cast<ANeuralNetworksDevice*>(
          NnApiHandler::kNnapiReferenceDevice);
    }
    return ANEURALNETWORKS_NO_ERROR;
  };
  nnapi_->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    if (device ==
        reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice)) {
      *name = NnApiHandler::nnapi_device_name_;
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (device == reinterpret_cast<ANeuralNetworksDevice*>(
                      NnApiHandler::kNnapiReferenceDevice)) {
      *name = NnApiHandler::kNnapiReferenceDeviceName;
      return ANEURALNETWORKS_NO_ERROR;
    }

    return ANEURALNETWORKS_BAD_DATA;
  };
  nnapi_->ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice* device, int64_t* featureLevel) -> int {
    if (device ==
        reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice)) {
      *featureLevel = NnApiHandler::nnapi_device_feature_level_;
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (device == reinterpret_cast<ANeuralNetworksDevice*>(
                      NnApiHandler::kNnapiReferenceDevice)) {
      *featureLevel = 1000;
      return ANEURALNETWORKS_NO_ERROR;
    }

    return ANEURALNETWORKS_BAD_DATA;
  };
}

}  // namespace nnapi
}  // namespace tflite
