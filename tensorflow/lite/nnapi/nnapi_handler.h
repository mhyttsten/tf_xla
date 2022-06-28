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
#ifndef TENSORFLOW_LITE_NNAPI_NNAPI_HANDLER_H_
#define TENSORFLOW_LITE_NNAPI_NNAPI_HANDLER_H_
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
class MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh {
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
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh() {
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


#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/nnapi/NeuralNetworksTypes.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

// Offers an interface to alter the behaviour of the NNAPI instance.
// As for NNAPI, it is designed to be a singleton.
// It allows to change the behaviour of some of the methods with some stub
// implementation and then to reset the behavior to the original one using
// Reset().
//
class NnApiHandler {
 public:
  // No destructor defined to allow this class to be used as singleton.

  // Factory method, only one instance per process/jni library.
  static NnApiHandler* Instance();

  // Makes the current object a transparent proxy again, resetting any
  // applied changes to its methods.
  void Reset();

  // Using templates in the ...Returns methods because the functions need to be
  // stateless and the template generated code is more readable than using a
  // file-local variable in the method implementation to store the configured
  // result.

  template <int Value>
  void GetDeviceCountReturns() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_0(mht_0_v, 217, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "GetDeviceCountReturns");

    nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
      *numDevices = 1;
      return Value;
    };
  }

  template <int DeviceCount>
  void GetDeviceCountReturnsCount() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_1(mht_1_v, 228, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "GetDeviceCountReturnsCount");

    nnapi_->ANeuralNetworks_getDeviceCount = [](uint32_t* numDevices) -> int {
      *numDevices = DeviceCount;
      return ANEURALNETWORKS_NO_ERROR;
    };
  }

  void StubGetDeviceCountWith(int(stub)(uint32_t*)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_2(mht_2_v, 238, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubGetDeviceCountWith");

    nnapi_->ANeuralNetworks_getDeviceCount = stub;
  }

  template <int Value>
  void GetDeviceReturns() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_3(mht_3_v, 246, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "GetDeviceReturns");

    nnapi_->ANeuralNetworks_getDevice =
        [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
      *device =
          reinterpret_cast<ANeuralNetworksDevice*>(NnApiHandler::kNnapiDevice);
      return Value;
    };
  }

  void StubGetDeviceWith(int(stub)(uint32_t, ANeuralNetworksDevice**)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_4(mht_4_v, 258, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubGetDeviceWith");

    nnapi_->ANeuralNetworks_getDevice = stub;
  }

  template <int Value>
  void GetDeviceNameReturns() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_5(mht_5_v, 266, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "GetDeviceNameReturns");

    nnapi_->ANeuralNetworksDevice_getName =
        [](const ANeuralNetworksDevice* device, const char** name) -> int {
      *name = NnApiHandler::nnapi_device_name_;
      return Value;
    };
  }

  void GetDeviceNameReturnsName(const std::string& name);

  void StubGetDeviceNameWith(int(stub)(const ANeuralNetworksDevice*,
                                       const char**)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_6(mht_6_v, 280, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubGetDeviceNameWith");

    nnapi_->ANeuralNetworksDevice_getName = stub;
  }

  // Configure all the functions related to device browsing to support
  // a device with the given name and the cpu fallback nnapi-reference.
  // The extra device will return support the specified feature level
  void SetNnapiSupportedDevice(const std::string& name, int feature_level = 29);

  template <int Value>
  void ModelCreateReturns() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_7(mht_7_v, 293, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ModelCreateReturns");

    nnapi_->ANeuralNetworksModel_create = [](ANeuralNetworksModel** model) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_8(mht_8_v, 297, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

      *model = reinterpret_cast<ANeuralNetworksModel*>(1);
      return Value;
    };
  }

  void StubModelCreateWith(int(stub)(ANeuralNetworksModel** model)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_9(mht_9_v, 306, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubModelCreateWith");

    nnapi_->ANeuralNetworksModel_create = stub;
  }

  template <int Value>
  void AddOperandReturns() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_10(mht_10_v, 314, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "AddOperandReturns");

    nnapi_->ANeuralNetworksModel_addOperand =
        [](ANeuralNetworksModel* model,
           const ANeuralNetworksOperandType* type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_11(mht_11_v, 320, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  void StubAddOperandWith(int(stub)(ANeuralNetworksModel* model,
                                    const ANeuralNetworksOperandType* type)) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_12(mht_12_v, 327, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubAddOperandWith");

    nnapi_->ANeuralNetworksModel_addOperand = stub;
  }

  template <int Value>
  void SetOperandValueReturns() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_13(mht_13_v, 335, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "SetOperandValueReturns");

    nnapi_->ANeuralNetworksModel_setOperandValue =
        [](ANeuralNetworksModel* model, int32_t index, const void* buffer,
           size_t length) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_14(mht_14_v, 341, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  template <int Value>
  void AddOperationReturns() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_15(mht_15_v, 348, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "AddOperationReturns");

    nnapi_->ANeuralNetworksModel_addOperation =
        [](ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
           uint32_t inputCount, const uint32_t* inputs, uint32_t outputCount,
           const uint32_t* outputs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_16(mht_16_v, 355, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  void StubAddOperationWith(
      int(stub)(ANeuralNetworksModel* model, ANeuralNetworksOperationType type,
                uint32_t inputCount, const uint32_t* inputs,
                uint32_t outputCount, const uint32_t* outputs)) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_17(mht_17_v, 364, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubAddOperationWith");

    nnapi_->ANeuralNetworksModel_addOperation = stub;
  }

  template <int Value>
  void IdentifyInputAndOutputsReturns() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_18(mht_18_v, 372, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "IdentifyInputAndOutputsReturns");

    nnapi_->ANeuralNetworksModel_identifyInputsAndOutputs =
        [](ANeuralNetworksModel* model, uint32_t inputCount,
           const uint32_t* inputs, uint32_t outputCount,
           const uint32_t* outputs) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_19(mht_19_v, 379, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  template <int Value>
  void RelaxComputationFloatReturns() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_20(mht_20_v, 386, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "RelaxComputationFloatReturns");

    nnapi_->ANeuralNetworksModel_relaxComputationFloat32toFloat16 =
        [](ANeuralNetworksModel* model, bool allow) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_21(mht_21_v, 391, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  template <int Value>
  void ModelFinishReturns() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_22(mht_22_v, 398, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ModelFinishReturns");

    nnapi_->ANeuralNetworksModel_finish = [](ANeuralNetworksModel* model) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_23(mht_23_v, 402, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

      return Value;
    };
  }

  template <int Value>
  void MemoryCreateFromFdReturns() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_24(mht_24_v, 411, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "MemoryCreateFromFdReturns");

    nnapi_->ANeuralNetworksMemory_createFromFd =
        [](size_t size, int protect, int fd, size_t offset,
           ANeuralNetworksMemory** memory) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_25(mht_25_v, 417, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

          *memory = reinterpret_cast<ANeuralNetworksMemory*>(2);
          return Value;
        };
  }

  template <int Value>
  void CompilationCreateReturns() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_26(mht_26_v, 427, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "CompilationCreateReturns");

    nnapi_->ANeuralNetworksCompilation_create =
        [](ANeuralNetworksModel* model,
           ANeuralNetworksCompilation** compilation) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_27(mht_27_v, 433, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

          *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(3);
          return Value;
        };
  }

  template <int Value>
  void CompilationCreateForDevicesReturns() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_28(mht_28_v, 443, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "CompilationCreateForDevicesReturns");

    nnapi_->ANeuralNetworksCompilation_createForDevices =
        [](ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           ANeuralNetworksCompilation** compilation) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_29(mht_29_v, 450, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

          *compilation = reinterpret_cast<ANeuralNetworksCompilation*>(3);
          return Value;
        };
  }

  void StubCompilationCreateForDevicesWith(int(stub)(
      ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
      uint32_t numDevices, ANeuralNetworksCompilation** compilation)) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_30(mht_30_v, 461, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubCompilationCreateForDevicesWith");

    nnapi_->ANeuralNetworksCompilation_createForDevices = stub;
  }

  template <int Value>
  void CompilationFinishReturns() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_31(mht_31_v, 469, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "CompilationFinishReturns");

    nnapi_->ANeuralNetworksCompilation_finish =
        [](ANeuralNetworksCompilation* compilation) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_32(mht_32_v, 474, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  template <int Value>
  void ExecutionCreateReturns() {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_33(mht_33_v, 481, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ExecutionCreateReturns");

    nnapi_->ANeuralNetworksExecution_create =
        [](ANeuralNetworksCompilation* compilation,
           ANeuralNetworksExecution** execution) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_34(mht_34_v, 487, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

          if (compilation == nullptr) return 1;
          *execution = reinterpret_cast<ANeuralNetworksExecution*>(4);
          return Value;
        };
  }
  template <int Value>
  void ExecutionSetInputFromMemoryReturns() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_35(mht_35_v, 497, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ExecutionSetInputFromMemoryReturns");

    nnapi_->ANeuralNetworksExecution_setInputFromMemory =
        [](ANeuralNetworksExecution* execution, int32_t index,
           const ANeuralNetworksOperandType* type,
           const ANeuralNetworksMemory* memory, size_t offset,
           size_t length) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_36(mht_36_v, 505, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }
  template <int Value>
  void ExecutionSetOutputFromMemoryReturns() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_37(mht_37_v, 511, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ExecutionSetOutputFromMemoryReturns");

    nnapi_->ANeuralNetworksExecution_setOutputFromMemory =
        [](ANeuralNetworksExecution* execution, int32_t index,
           const ANeuralNetworksOperandType* type,
           const ANeuralNetworksMemory* memory, size_t offset,
           size_t length) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_38(mht_38_v, 519, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  template <int Value>
  void ExecutionComputeReturns() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_39(mht_39_v, 526, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ExecutionComputeReturns");

    nnapi_->ANeuralNetworksExecution_compute =
        [](ANeuralNetworksExecution* execution) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_40(mht_40_v, 531, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  template <int Value>
  void GetSupportedOperationsForDevicesReturns() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_41(mht_41_v, 538, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "GetSupportedOperationsForDevicesReturns");

    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
        [](const ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           bool* supportedOps) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_42(mht_42_v, 545, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");
 return Value; };
  }

  void StubGetSupportedOperationsForDevicesWith(
      int(stub)(const ANeuralNetworksModel* model,
                const ANeuralNetworksDevice* const* devices,
                uint32_t numDevices, bool* supportedOps)) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_43(mht_43_v, 554, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "StubGetSupportedOperationsForDevicesWith");

    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices = stub;
  }

  template <int Value>
  void ExecutionStartComputeReturns() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_44(mht_44_v, 562, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "ExecutionStartComputeReturns");

    nnapi_->ANeuralNetworksExecution_startCompute =
        [](ANeuralNetworksExecution* execution, ANeuralNetworksEvent** event) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_45(mht_45_v, 567, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

          *event = reinterpret_cast<ANeuralNetworksEvent*>(1);
          return Value;
        };
  }

  template <int Value>
  void EventWaitReturns() {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_46(mht_46_v, 577, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "EventWaitReturns");

    nnapi_->ANeuralNetworksEvent_wait = [](ANeuralNetworksEvent* event) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_47(mht_47_v, 581, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

      return Value;
    };
  }

  template <int Value>
  void SetPriorityReturns() {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_48(mht_48_v, 590, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "SetPriorityReturns");

    nnapi_->ANeuralNetworksCompilation_setPriority =
        [](ANeuralNetworksCompilation* compilation, int priority) -> int {
      return Value;
    };
  }

  template <int Value>
  void SetOperandSymmPerChannelQuantParamsReturns() {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_49(mht_49_v, 601, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "SetOperandSymmPerChannelQuantParamsReturns");

    nnapi_->ANeuralNetworksModel_setOperandSymmPerChannelQuantParams =
        [](ANeuralNetworksModel* model, int32_t index,
           const ANeuralNetworksSymmPerChannelQuantParams* channelQuant) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_50(mht_50_v, 607, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "lambda");

          return Value;
        };
  }

  /*
   * Sets the SDK Version in the nnapi structure.
   * If set_unsupported_ops_to_null is set to true, all the functions not
   * available at the given sdk level will be set to null too.
   */
  void SetAndroidSdkVersion(int version,
                            bool set_unsupported_ops_to_null = false);

  const NnApi* GetNnApi() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_51(mht_51_v, 623, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "GetNnApi");
 return nnapi_; }

 protected:
  explicit NnApiHandler(NnApi* nnapi) : nnapi_(nnapi) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPSlitePSnnapiPSnnapi_handlerDTh mht_52(mht_52_v, 629, "", "./tensorflow/lite/nnapi/nnapi_handler.h", "NnApiHandler");
 DCHECK(nnapi); }

  NnApi* nnapi_;

  static const char kNnapiReferenceDeviceName[];
  static const int kNnapiReferenceDevice;
  static const int kNnapiDevice;

  static void SetDeviceName(const std::string& name);

 private:
  static char* nnapi_device_name_;
  static int nnapi_device_feature_level_;
};

// Returns a pointer to an unaltered instance of NNAPI. Is intended
// to be used by stub methods when wanting to pass-through to original
// implementation for example:
//
// NnApiTestUtility()->StubGetDeviceWith(
//  [](uint32_t devIndex, ANeuralNetworksDevice** device) -> int {
//        static int count = 0;
//        if (count++ < 1) {
//          NnApiPassthroughInstance()->ANeuralNetworks_getDevice(
//                devIndex, device);
//        } else {
//            return ANEURALNETWORKS_BAD_DATA;
//        }
//   });
const NnApi* NnApiPassthroughInstance();

// Returns an instance of NnApiProxy that can be used to alter
// the behaviour of the TFLite wide instance of NnApi.
NnApiHandler* NnApiProxyInstance();

}  // namespace nnapi
}  // namespace tflite

#endif  // TENSORFLOW_LITE_NNAPI_NNAPI_HANDLER_H_
