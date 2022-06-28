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
class MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handler_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handler_testDTcc() {
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

#include <cstdint>
#include <cstdio>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

using testing::Eq;
using testing::Ne;
using testing::NotNull;

void ExpectEquals(const NnApi& left, const NnApi& right);

class NnApiHandlerTest : public ::testing::Test {
 protected:
  ~NnApiHandlerTest() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handler_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/nnapi/nnapi_handler_test.cc", "~NnApiHandlerTest");
 NnApiHandler::Instance()->Reset(); }
};

TEST_F(NnApiHandlerTest, ShouldAlterNnApiInstanceBehaviour) {
  const NnApi* nnapi = NnApiImplementation();

  const auto device_count_stub = [](uint32_t* device_count) -> int {
    *device_count = 999;
    return ANEURALNETWORKS_NO_ERROR;
  };

  NnApiHandler::Instance()->StubGetDeviceCountWith(device_count_stub);

  ASSERT_THAT(nnapi->ANeuralNetworks_getDeviceCount, NotNull());

  uint32_t device_count = 0;
  nnapi->ANeuralNetworks_getDeviceCount(&device_count);
  EXPECT_THAT(device_count, Eq(999));
}

TEST_F(NnApiHandlerTest, ShouldRestoreNnApiToItsOriginalValueWithReset) {
  NnApi nnapi_orig_copy = *NnApiImplementation();

  auto device_count_override = [](uint32_t* device_count) -> int {
    *device_count = 777;
    return ANEURALNETWORKS_NO_ERROR;
  };

  NnApiHandler::Instance()->StubGetDeviceCountWith(device_count_override);

  EXPECT_THAT(nnapi_orig_copy.ANeuralNetworks_getDeviceCount,
              Ne(NnApiImplementation()->ANeuralNetworks_getDeviceCount));

  NnApiHandler::Instance()->Reset();

  ExpectEquals(nnapi_orig_copy, *NnApiImplementation());
}

int (*device_count_ptr)(uint32_t*);
TEST_F(NnApiHandlerTest, ShouldSupportPassthroughCalls) {
  const NnApi* nnapi = NnApiImplementation();
  device_count_ptr = nnapi->ANeuralNetworks_getDeviceCount;

  NnApiHandler::Instance()->StubGetDeviceCountWith(
      [](uint32_t* device_count) -> int {
        return NnApiPassthroughInstance()->ANeuralNetworks_getDeviceCount ==
               device_count_ptr;
      });

  uint32_t device_count = 0;
  EXPECT_THAT(nnapi->ANeuralNetworks_getDeviceCount(&device_count), Eq(1));
}

TEST_F(NnApiHandlerTest, ShouldSetNnApiMembersToNullAsPerSdkVersion_NNAPI11) {
  auto* handler = NnApiHandler::Instance();

  // Setting non null values for nnapi functions
  handler->SetNnapiSupportedDevice("devvice", 1000);
  handler->GetSupportedOperationsForDevicesReturns<1>();
  handler->CompilationCreateForDevicesReturns<1>();
  handler->ExecutionComputeReturns<1>();
  handler->MemoryCreateFromFdReturns<1>();

  handler->SetAndroidSdkVersion(28, /*set_unsupported_ops_to_null=*/true);

  const NnApi* nnapi = NnApiImplementation();

  using ::testing::IsNull;

  EXPECT_THAT(nnapi->ANeuralNetworks_getDeviceCount, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworks_getDevice, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getName, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getVersion, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getFeatureLevel, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_createForDevices, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_setCaching, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_compute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandRank, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandDimensions,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_create, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_free, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_burstCompute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksMemory_createFromAHardwareBuffer, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_setMeasureTiming, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getDuration, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getExtensionSupport, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperandType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperationType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_setOperandExtensionData, IsNull());
}

TEST_F(NnApiHandlerTest, ShouldSetNnApiMembersToNullAsPerSdkVersion_NNAPI10) {
  auto* handler = NnApiHandler::Instance();

  // Setting non null values for nnapi functions
  handler->SetNnapiSupportedDevice("devvice", 1000);
  handler->GetSupportedOperationsForDevicesReturns<1>();
  handler->CompilationCreateForDevicesReturns<1>();
  handler->ExecutionComputeReturns<1>();
  handler->MemoryCreateFromFdReturns<1>();

  handler->SetAndroidSdkVersion(27, /*set_unsupported_ops_to_null=*/true);

  const NnApi* nnapi = NnApiImplementation();

  using ::testing::IsNull;

  EXPECT_THAT(nnapi->ANeuralNetworks_getDeviceCount, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworks_getDevice, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getName, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getVersion, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getFeatureLevel, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getSupportedOperationsForDevices,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_createForDevices, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksCompilation_setCaching, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_compute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandRank, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getOutputOperandDimensions,
              IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_create, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksBurst_free, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_burstCompute, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksMemory_createFromAHardwareBuffer, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_setMeasureTiming, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksExecution_getDuration, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksDevice_getExtensionSupport, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperandType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_getExtensionOperationType, IsNull());
  EXPECT_THAT(nnapi->ANeuralNetworksModel_setOperandExtensionData, IsNull());

  EXPECT_THAT(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16,
              IsNull());
}

void ExpectEquals(const NnApi& left, const NnApi& right) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handler_testDTcc mht_1(mht_1_v, 347, "", "./tensorflow/lite/nnapi/nnapi_handler_test.cc", "ExpectEquals");

#define EXPECT_NNAPI_MEMBER_EQ(name) EXPECT_EQ(left.name, right.name)

  EXPECT_NNAPI_MEMBER_EQ(nnapi_exists);
  EXPECT_NNAPI_MEMBER_EQ(android_sdk_version);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksMemory_createFromFd);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksMemory_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_finish);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_addOperand);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_setOperandValue);
  EXPECT_NNAPI_MEMBER_EQ(
      ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_setOperandValueFromMemory);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_addOperation);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_identifyInputsAndOutputs);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_relaxComputationFloat32toFloat16);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_setPreference);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_finish);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setInput);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setInputFromMemory);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setOutput);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setOutputFromMemory);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_startCompute);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksEvent_wait);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksEvent_free);
  EXPECT_NNAPI_MEMBER_EQ(ASharedMemory_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworks_getDeviceCount);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworks_getDevice);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getName);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getVersion);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getFeatureLevel);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksDevice_getType);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksModel_getSupportedOperationsForDevices);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_createForDevices);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksCompilation_setCaching);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_compute);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_getOutputOperandRank);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_getOutputOperandDimensions);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksBurst_create);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksBurst_free);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_burstCompute);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksMemory_createFromAHardwareBuffer);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_setMeasureTiming);
  EXPECT_NNAPI_MEMBER_EQ(ANeuralNetworksExecution_getDuration);

#undef EXPECT_NNAPI_MEMBER_EQ
}

}  // namespace nnapi
}  // namespace tflite
