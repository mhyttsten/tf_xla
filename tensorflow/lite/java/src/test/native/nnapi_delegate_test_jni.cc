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
class MHTracer_DTPStensorflowPSlitePSjavaPSsrcPStestPSnativePSnnapi_delegate_test_jniDTcc {
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
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPStestPSnativePSnnapi_delegate_test_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSjavaPSsrcPStestPSnativePSnnapi_delegate_test_jniDTcc() {
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

#include <jni.h>

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

namespace {

bool g_sl_has_been_called = false;

template <int return_value, typename... Types>
int DoNothingAndReturn(Types... args) {
  g_sl_has_been_called = true;
  return return_value;
}

template <typename... Types>
void DoNothing(Types... args) {
  g_sl_has_been_called = true;
}

const uint32_t kDefaultMemoryPaddingAndAlignment = 64;

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_getMockSlHandle(JNIEnv* env,
                                                                 jclass clazz) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPStestPSnativePSnnapi_delegate_test_jniDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/java/src/test/native/nnapi_delegate_test_jni.cc", "Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_getMockSlHandle");

  g_sl_has_been_called = false;

  NnApiSLDriverImplFL5* supportLibraryImplementation =
      new NnApiSLDriverImplFL5();

  supportLibraryImplementation->base.implFeatureLevel =
      ANEURALNETWORKS_FEATURE_LEVEL_5;

  // Most calls do nothing and return NO_ERROR as result. Errors are returned in
  // get* calls that are not trivial to mock.
  supportLibraryImplementation->ANeuralNetworksBurst_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksBurst_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksCompilation_createForDevices =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_finish =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_free = DoNothing;
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryAlignmentForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    g_sl_has_been_called = true;
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryAlignmentForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* alignment) -> int {
    g_sl_has_been_called = true;
    *alignment = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryPaddingForInput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    g_sl_has_been_called = true;
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksCompilation_getPreferredMemoryPaddingForOutput =
      [](const ANeuralNetworksCompilation* compilation, uint32_t index,
         uint32_t* padding) -> int {
    g_sl_has_been_called = true;
    *padding = kDefaultMemoryPaddingAndAlignment;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksCompilation_setCaching =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_setPreference =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_setPriority =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksCompilation_setTimeout =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksDevice_getExtensionSupport =
      [](const ANeuralNetworksDevice* device, const char* extensionName,
         bool* isExtensionSupported) -> int {
    g_sl_has_been_called = true;
    *isExtensionSupported = false;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getFeatureLevel =
      [](const ANeuralNetworksDevice* device, int64_t* featureLevel) -> int {
    g_sl_has_been_called = true;
    *featureLevel = ANEURALNETWORKS_FEATURE_LEVEL_5;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getName =
      [](const ANeuralNetworksDevice* device, const char** name) -> int {
    g_sl_has_been_called = true;
    *name = "mockDevice";
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getType =
      [](const ANeuralNetworksDevice* device, int32_t* type) -> int {
    g_sl_has_been_called = true;
    *type = ANEURALNETWORKS_DEVICE_CPU;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_getVersion =
      [](const ANeuralNetworksDevice* device, const char** version) -> int {
    g_sl_has_been_called = true;
    *version = "mock";
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation->ANeuralNetworksDevice_wait =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksEvent_createFromSyncFenceFd =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksEvent_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksEvent_getSyncFenceFd =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksEvent_wait =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_burstCompute =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_compute =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksExecution_enableInputAndOutputPadding =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksExecution_getDuration =
      [](const ANeuralNetworksExecution* execution, int32_t durationCode,
         uint64_t* duration) -> int {
    g_sl_has_been_called = true;
    *duration = UINT64_MAX;
    return ANEURALNETWORKS_NO_ERROR;
  };
  supportLibraryImplementation
      ->ANeuralNetworksExecution_getOutputOperandDimensions =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksExecution_getOutputOperandRank =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksExecution_setInput =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setInputFromMemory =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setLoopTimeout =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setMeasureTiming =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setOutput =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setOutputFromMemory =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setReusable =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksExecution_setTimeout =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksExecution_startComputeWithDependencies =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_addInputRole =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_addOutputRole =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_finish =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksMemoryDesc_setDimensions =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_copy =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksMemory_createFromAHardwareBuffer =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_createFromDesc =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_createFromFd =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksMemory_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksModel_addOperand =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_addOperation =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_create =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_finish =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_free = DoNothing;
  supportLibraryImplementation->ANeuralNetworksModel_getExtensionOperandType =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksModel_getExtensionOperationType =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation
      ->ANeuralNetworksModel_getSupportedOperationsForDevices =
      DoNothingAndReturn<ANEURALNETWORKS_BAD_DATA>;
  supportLibraryImplementation->ANeuralNetworksModel_identifyInputsAndOutputs =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandExtensionData =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation
      ->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValue =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValueFromMemory =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworksModel_setOperandValueFromModel =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworks_getDefaultLoopTimeout =
      []() -> uint64_t {
    g_sl_has_been_called = true;
    return UINT64_MAX;
  };
  supportLibraryImplementation->ANeuralNetworks_getDevice =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworks_getDeviceCount =
      DoNothingAndReturn<ANEURALNETWORKS_NO_ERROR>;
  supportLibraryImplementation->ANeuralNetworks_getMaximumLoopTimeout =
      []() -> uint64_t {
    g_sl_has_been_called = true;
    return UINT64_MAX;
  };
  supportLibraryImplementation->ANeuralNetworks_getRuntimeFeatureLevel =
      []() -> int64_t {
    g_sl_has_been_called = true;
    return ANEURALNETWORKS_FEATURE_LEVEL_5;
  };

  return reinterpret_cast<jlong>(supportLibraryImplementation);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_hasNnApiSlBeenCalled(
    JNIEnv* env, jclass clazz) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPStestPSnativePSnnapi_delegate_test_jniDTcc mht_1(mht_1_v, 437, "", "./tensorflow/lite/java/src/test/native/nnapi_delegate_test_jni.cc", "Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_hasNnApiSlBeenCalled");

  return g_sl_has_been_called;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_closeMockSl(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPStestPSnativePSnnapi_delegate_test_jniDTcc mht_2(mht_2_v, 447, "", "./tensorflow/lite/java/src/test/native/nnapi_delegate_test_jni.cc", "Java_org_tensorflow_lite_nnapi_NnApiDelegateTest_closeMockSl");

  delete reinterpret_cast<NnApiSLDriverImplFL5*>(handle);
}

}  // extern "C"
