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
#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh() {
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


// Cannot mock the delegate when using the disabled version
// (see the condition in the BUILD file).
#ifndef NNAPI_DELEGATE_DISABLED

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <memory>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_handler.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace delegate {
namespace nnapi {

class NnApiMock : public ::tflite::nnapi::NnApiHandler {
 public:
  explicit NnApiMock(NnApi* nnapi, int android_sdk_version = 29)
      : ::tflite::nnapi::NnApiHandler(nnapi) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_0(mht_0_v, 211, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "NnApiMock");

    nnapi_->nnapi_exists = true;
    nnapi_->android_sdk_version = android_sdk_version;
    nnapi_->nnapi_runtime_feature_level = android_sdk_version;

    nnapi_->ANeuralNetworksCompilation_free =
        [](ANeuralNetworksCompilation* compilation) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_1(mht_1_v, 220, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "lambda");
};
    nnapi_->ANeuralNetworksMemory_free = [](ANeuralNetworksMemory* memory) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_2(mht_2_v, 224, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "lambda");
};
    nnapi_->ANeuralNetworksModel_free = [](ANeuralNetworksModel* model) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_3(mht_3_v, 228, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "lambda");
};
    nnapi_->ANeuralNetworksExecution_free =
        [](ANeuralNetworksExecution* execution) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_4(mht_4_v, 233, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "lambda");
};
    nnapi_->ASharedMemory_create = [](const char* name, size_t size) -> int {
      return open("/dev/zero", O_RDWR);
    };
    nnapi_->ANeuralNetworksEvent_free = [](ANeuralNetworksEvent* event) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_5(mht_5_v, 240, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "lambda");
};

    ModelCreateReturns<ANEURALNETWORKS_NO_ERROR>();
    AddOperandReturns<ANEURALNETWORKS_NO_ERROR>();
    SetOperandValueReturns<ANEURALNETWORKS_NO_ERROR>();
    AddOperationReturns<ANEURALNETWORKS_NO_ERROR>();
    IdentifyInputAndOutputsReturns<ANEURALNETWORKS_NO_ERROR>();
    RelaxComputationFloatReturns<ANEURALNETWORKS_NO_ERROR>();
    ModelFinishReturns<ANEURALNETWORKS_NO_ERROR>();
    MemoryCreateFromFdReturns<ANEURALNETWORKS_NO_ERROR>();
    CompilationCreateReturns<ANEURALNETWORKS_NO_ERROR>();
    CompilationCreateForDevicesReturns<ANEURALNETWORKS_NO_ERROR>();
    CompilationFinishReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionCreateReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionSetInputFromMemoryReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionSetOutputFromMemoryReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionComputeReturns<ANEURALNETWORKS_NO_ERROR>();
    ExecutionStartComputeReturns<ANEURALNETWORKS_NO_ERROR>();
    EventWaitReturns<ANEURALNETWORKS_NO_ERROR>();
    SetPriorityReturns<ANEURALNETWORKS_NO_ERROR>();
    SetOperandSymmPerChannelQuantParamsReturns<ANEURALNETWORKS_NO_ERROR>();
    SetNnapiSupportedDevice("test-device", android_sdk_version);
  }

  ~NnApiMock() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_6(mht_6_v, 267, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "~NnApiMock");
 Reset(); }
};

class NnApiDelegateMockTest : public ::testing::Test {
 protected:
  void SetUp() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSnnapi_delegate_mock_testDTh mht_7(mht_7_v, 275, "", "./tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h", "SetUp");

    nnapi_ = *NnApiImplementation();
    nnapi_mock_ = absl::make_unique<NnApiMock>(&nnapi_);
  }

  std::unique_ptr<NnApiMock> nnapi_mock_;

 private:
  NnApi nnapi_;
};

}  // namespace nnapi
}  // namespace delegate
}  // namespace tflite

#endif  // #ifndef NNAPI_DELEGATE_DISABLED

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_MOCK_TEST_H_
