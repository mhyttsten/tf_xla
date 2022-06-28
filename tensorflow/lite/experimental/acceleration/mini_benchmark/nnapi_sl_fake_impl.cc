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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <stdlib.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

namespace {

std::string GetTempDir() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "GetTempDir");

  const char* temp_dir = getenv("TEST_TMPDIR");
  if (temp_dir == nullptr || temp_dir[0] == '\0') {
#ifdef __ANDROID__
    return "/data/local/tmp";
#else
    return "/tmp";
#endif
  } else {
    return temp_dir;
  }
}

std::string CallCountFilePath() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_1(mht_1_v, 216, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "CallCountFilePath");

  return GetTempDir() + "/nnapi_sl_fake_impl.out";
}
// We write a . in the trace file to allow a caller to count the number of
// calls to NNAPI SL.
void TraceCall() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_2(mht_2_v, 224, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "TraceCall");

  std::ofstream trace_file(CallCountFilePath().c_str(), std::ofstream::app);
  if (trace_file) {
    std::cerr << "Tracing call\n";
    trace_file << '.';
    if (!trace_file) {
      std::cerr << "Error writing to '" << CallCountFilePath() << "'\n";
    }
  } else {
    std::cerr << "FAKE_NNAPI_SL: UNABLE TO TRACE CALL\n";
  }
}

template <int return_value, typename... Types>
int TraceCallAndReturn(Types... args) {
  TraceCall();
  return return_value;
}

template <typename... Types>
void JustTraceCall(Types... args) {
  TraceCall();
}

const uint32_t kDefaultMemoryPaddingAndAlignment = 64;

NnApiSLDriverImplFL5 GetNnApiSlDriverImpl() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_3(mht_3_v, 253, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "GetNnApiSlDriverImpl");

  NnApiSLDriverImplFL5 sl_driver_impl;

  sl_driver_impl.base = {ANEURALNETWORKS_FEATURE_LEVEL_5};
  sl_driver_impl.ANeuralNetworksBurst_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksBurst_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksCompilation_createForDevices =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_finish =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_free = JustTraceCall;
  sl_driver_impl
      .ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    TraceCall();
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl
      .ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    TraceCall();
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    TraceCall();
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    TraceCall();
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksCompilation_setCaching =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_setPreference =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_setPriority =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksCompilation_setTimeout =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksDevice_getExtensionSupport =
      [](const ANeuralNetworksDevice* device, const char* extensionName,
         bool* isExtensionSupported) -> int {
    *isExtensionSupported = false;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice* device, int64_t* featureLevel) -> int {
    TraceCall();
    *featureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    TraceCall();
    *name = "mockDevice";
    return ANEURALNETWORKS_BAD_DATA;
  };
  sl_driver_impl.ANeuralNetworksDevice_getType =
      [](const ANeuralNetworksDevice* device, int32_t* type) -> int {
    *type = ANEURALNETWORKS_DEVICE_CPU;
    return ANEURALNETWORKS_BAD_DATA;
  };
  sl_driver_impl.ANeuralNetworksDevice_getVersion =
      [](const ANeuralNetworksDevice* device, const char** version) -> int {
    TraceCall();
    *version = "mock";
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksDevice_wait =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksEvent_createFromSyncFenceFd =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksEvent_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksEvent_getSyncFenceFd =
      TraceCallAndReturn<ANEURALNETWORKS_BAD_DATA>;
  sl_driver_impl.ANeuralNetworksEvent_wait =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_burstCompute =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_compute =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_enableInputAndOutputPadding =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksExecution_getDuration =
      [](const ANeuralNetworksExecution* execution, int32_t durationCode,
         uint64_t* duration) -> int {
    TraceCall();
    *duration = UINT64_MAX;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworksExecution_getOutputOperandDimensions =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_getOutputOperandRank =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setInput =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setInputFromMemory =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setLoopTimeout =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setMeasureTiming =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setOutput =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setOutputFromMemory =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setReusable =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_setTimeout =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksExecution_startComputeWithDependencies =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_addInputRole =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_addOutputRole =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_finish =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemoryDesc_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksMemoryDesc_setDimensions =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_copy =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_createFromAHardwareBuffer =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_createFromDesc =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_createFromFd =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksMemory_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksModel_addOperand =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_addOperation =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_create =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_finish =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_free = JustTraceCall;
  sl_driver_impl.ANeuralNetworksModel_getExtensionOperandType =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_getExtensionOperationType =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_getSupportedOperationsForDevices =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_identifyInputsAndOutputs =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandExtensionData =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandValue =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandValueFromMemory =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworksModel_setOperandValueFromModel =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworks_getDefaultLoopTimeout = []() -> uint64_t {
    TraceCall();
    return UINT64_MAX;
  };
  sl_driver_impl.ANeuralNetworks_getDevice =
      TraceCallAndReturn<ANEURALNETWORKS_NO_ERROR>;
  sl_driver_impl.ANeuralNetworks_getDeviceCount =
      [](uint32_t* num_devices) -> int {
    TraceCall();
    *num_devices = 0;
    return ANEURALNETWORKS_NO_ERROR;
  };
  sl_driver_impl.ANeuralNetworks_getMaximumLoopTimeout = []() -> uint64_t {
    TraceCall();
    return UINT64_MAX;
  };
  sl_driver_impl.ANeuralNetworks_getRuntimeFeatureLevel = []() -> int64_t {
    TraceCall();
    return ANEURALNETWORKS_FEATURE_LEVEL_5;
  };

  return sl_driver_impl;
}

}  // namespace

extern "C" NnApiSLDriverImpl* ANeuralNetworks_getSLDriverImpl() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_4(mht_4_v, 457, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "ANeuralNetworks_getSLDriverImpl");

  static NnApiSLDriverImplFL5 sl_driver_impl = GetNnApiSlDriverImpl();
  return reinterpret_cast<NnApiSLDriverImpl*>(&sl_driver_impl);
}

namespace tflite {
namespace acceleration {

void InitNnApiSlInvocationStatus() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_5(mht_5_v, 468, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "InitNnApiSlInvocationStatus");
 unlink(CallCountFilePath().c_str()); }

bool WasNnApiSlInvoked() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_6(mht_6_v, 473, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "WasNnApiSlInvoked");

  std::cerr << "Checking if file '" << CallCountFilePath() << "' exists.\n";
  if (FILE* trace_file = fopen(CallCountFilePath().c_str(), "r")) {
    fclose(trace_file);
    return true;
  } else {
    return false;
  }
}

int CountNnApiSlApiCalls() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSmini_benchmarkPSnnapi_sl_fake_implDTcc mht_7(mht_7_v, 486, "", "./tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.cc", "CountNnApiSlApiCalls");

  FILE* trace_file = fopen(CallCountFilePath().c_str(), "r");
  int call_count = 0;
  while (fgetc(trace_file) != EOF) {
    call_count++;
  }
  return call_count;
}

}  // namespace acceleration
}  // namespace tflite
