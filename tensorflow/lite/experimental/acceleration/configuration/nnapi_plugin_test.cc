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
class MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_kernel.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_mock_test.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/delegate_registry.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"

// Tests for checking that the NNAPI Delegate plugin correctly handles all the
// options from the flatbuffer.
//
// Checking done at NNAPI call level, as that is where we have a mockable
// layer.
namespace tflite {
namespace {

using delegate::nnapi::NnApiMock;

class SingleAddOpModel : public tflite::SingleOpModel {
 public:
  // Note the caller owns the memory of the passed-in 'delegate'.
  void Build(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "Build");

    int input = AddInput({tflite::TensorType_FLOAT32, {1, 2, 2}});
    int constant = AddConstInput({tflite::TensorType_FLOAT32, {1, 2, 2}},
                                 {1.0f, 1.0f, 1.0f, 1.0f});
    AddOutput({tflite::TensorType_FLOAT32, {}});

    SetBuiltinOp(tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
                 tflite::CreateAddOptions(builder_).Union());

    SetDelegate(delegate);
    // Set 'apply_delegate' to false to manually apply the delegate later and
    // check its return status.
    BuildInterpreter({GetShape(input), GetShape(constant)},
                     /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/true);
  }

  tflite::Interpreter* Interpreter() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "Interpreter");
 return interpreter_.get(); }
};

class NNAPIPluginTest : public ::testing::Test {
 protected:
  NNAPIPluginTest() : delegate_(nullptr, [](TfLiteDelegate*) {}) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_2(mht_2_v, 240, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "NNAPIPluginTest");
}
  void SetUp() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_3(mht_3_v, 244, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "SetUp");

    nnapi_ = const_cast<NnApi*>(NnApiImplementation());
    nnapi_mock_ = absl::make_unique<NnApiMock>(nnapi_);
    nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
        [](const ANeuralNetworksModel* model,
           const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
           bool* supportedOps) -> int {
      supportedOps[0] = true;
      return 0;
    };
  }

  template <NNAPIExecutionPreference input, int output>
  void CheckExecutionPreference() {
    // Note - this uses a template since the NNAPI functions are C function
    // pointers rather than lambdas so can't capture variables.
    nnapi_->ANeuralNetworksCompilation_setPreference =
        [](ANeuralNetworksCompilation* compilation, int32_t preference) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_4(mht_4_v, 264, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "lambda");

          return preference - output;
        };
    CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0, input));
    // Since delegation succeeds, the model becomes immutable and hence can't
    // reuse it.
    SingleAddOpModel model;
    model.Build(delegate_.get());
    EXPECT_EQ(model.ApplyDelegate(), kTfLiteOk)
        << " given input: " << input << " expected output: " << output;
  }

  template <NNAPIExecutionPriority input, int output>
  void CheckExecutionPriority() {
    // Note - this uses a template since the NNAPI functions are C function
    // pointers rather than lambdas so can't capture variables.
    nnapi_->ANeuralNetworksCompilation_setPriority =
        [](ANeuralNetworksCompilation* compilation, int32_t priority) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_5(mht_5_v, 284, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "lambda");

          return priority - output;
        };
    CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0,
                                       NNAPIExecutionPreference_UNDEFINED, 0, 0,
                                       /*allow CPU=*/true, input));
    // Since delegation succeeds, the model becomes immutable and hence can't
    // reuse it.
    SingleAddOpModel model;
    model.Build(delegate_.get());
    EXPECT_EQ(model.ApplyDelegate(), kTfLiteOk)
        << " given input: " << input << " expected output: " << output;
  }

  void CreateDelegate(flatbuffers::Offset<NNAPISettings> nnapi_settings) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_6(mht_6_v, 301, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "CreateDelegate");

    tflite_settings_ = flatbuffers::GetTemporaryPointer(
        fbb_,
        CreateTFLiteSettings(fbb_, tflite::Delegate_NNAPI, nnapi_settings));

    plugin_ = delegates::DelegatePluginRegistry::CreateByName(
        "NnapiPlugin", *tflite_settings_);
    delegate_ = plugin_->Create();
  }

  TfLiteStatus ApplyDelegate() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_7(mht_7_v, 314, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "ApplyDelegate");

    model_.Build(delegate_.get());
    return model_.ApplyDelegate();
  }

  NnApi* nnapi_;
  std::unique_ptr<NnApiMock> nnapi_mock_;
  SingleAddOpModel model_;
  flatbuffers::FlatBufferBuilder fbb_;
  const TFLiteSettings* tflite_settings_ = nullptr;
  delegates::TfLiteDelegatePtr delegate_;
  std::unique_ptr<delegates::DelegatePluginInterface> plugin_;
};

TEST_F(NNAPIPluginTest, PassesAcceleratorNameFailure) {
  // Fails with non-existent "foo".
  CreateDelegate(CreateNNAPISettings(fbb_, fbb_.CreateString("foo")));
  EXPECT_EQ(kTfLiteDelegateError, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesAcceleratorNameSuccess) {
  // Succeeds with "test-device" supported by the mock.
  CreateDelegate(CreateNNAPISettings(fbb_, fbb_.CreateString("test-device")));
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesExecutionPreference) {
  CheckExecutionPreference<NNAPIExecutionPreference_UNDEFINED,
                           StatefulNnApiDelegate::Options::kUndefined>();
  CheckExecutionPreference<NNAPIExecutionPreference_NNAPI_LOW_POWER,
                           StatefulNnApiDelegate::Options::kLowPower>();
  CheckExecutionPreference<NNAPIExecutionPreference_NNAPI_FAST_SINGLE_ANSWER,
                           StatefulNnApiDelegate::Options::kFastSingleAnswer>();
  CheckExecutionPreference<NNAPIExecutionPreference_NNAPI_SUSTAINED_SPEED,
                           StatefulNnApiDelegate::Options::kSustainedSpeed>();
}

TEST_F(NNAPIPluginTest, PassesExecutionPriority) {
  nnapi_->android_sdk_version =
      tflite::delegate::nnapi::kMinSdkVersionForNNAPI13;
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED,
                         ANEURALNETWORKS_PRIORITY_DEFAULT>();
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_LOW,
                         ANEURALNETWORKS_PRIORITY_LOW>();
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_MEDIUM,
                         ANEURALNETWORKS_PRIORITY_MEDIUM>();
  CheckExecutionPriority<NNAPIExecutionPriority_NNAPI_PRIORITY_HIGH,
                         ANEURALNETWORKS_PRIORITY_HIGH>();
}

TEST_F(NNAPIPluginTest, PassesCachingParameters) {
  nnapi_->ANeuralNetworksCompilation_setCaching =
      [](ANeuralNetworksCompilation* compilation, const char* cacheDir,
         const uint8_t* token) -> int {
    if (std::string(cacheDir) != "d") return 1;
    // Token is hashed with other bits, just check that it's not empty.
    if (std::string(reinterpret_cast<const char*>(token)).empty()) return 2;
    return 0;
  };
  CreateDelegate(CreateNNAPISettings(fbb_, 0, fbb_.CreateString("d"),
                                     fbb_.CreateString("t")));
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesFalseNNAPICpuFlag) {
  CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0,
                                     NNAPIExecutionPreference_UNDEFINED, 0, 0,
                                     /* allow CPU */ false));
  nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    supportedOps[0] = true;
    // Since no CPU, should only pass one device.
    return numDevices - 1;
  };
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

TEST_F(NNAPIPluginTest, PassesTrueNNAPICpuFlag) {
  CreateDelegate(CreateNNAPISettings(fbb_, 0, 0, 0,
                                     NNAPIExecutionPreference_UNDEFINED, 0, 0,
                                     /* allow CPU */ true));
  nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    supportedOps[0] = true;
    // With CPU allowed, should pass two devices.
    return numDevices - 2;
  };
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
}

/*
 * Building a model with three operations that can be used to create multiple
 * delegated partitions.
 *
 *  input1 ---
 *            | -  ADD -- ROUND --
 *            |                   | - ADD -- output1
 *  input2 ---                    |
 *                                |
 *  input3 -----------------------
 */
class MultiplePartitionsModel : public tflite::MultiOpModel {
 public:
  // Note the caller owns the memory of the passed-in 'delegate'.
  void Build(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_8(mht_8_v, 425, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "Build");

    const tflite::TensorData tensors_data = {tflite::TensorType_FLOAT32,
                                             {1, 2, 2}};
    int input1 = AddInput(tensors_data);
    int input2 = AddInput(tensors_data);
    int input3 = AddInput(tensors_data);
    int add_out = AddInnerTensor<float>(tensors_data);
    int round_out = AddInnerTensor<float>(tensors_data);
    int output = AddOutput(tensors_data);

    AddBuiltinOp(
        tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union(),
        {input1, input2}, {add_out});

    AddBuiltinOp(tflite::BuiltinOperator_ROUND, tflite::BuiltinOptions_NONE,
                 /*builtin_options=*/0, {add_out}, {round_out});

    AddBuiltinOp(
        tflite::BuiltinOperator_ADD, tflite::BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union(),
        {round_out, input3}, {output});

    SetDelegate(delegate);
    // Set 'apply_delegate' to false to manually apply the delegate later and
    // check its return status.
    BuildInterpreter({GetShape(input1), GetShape(input2), GetShape(input3)},
                     /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false,
                     /*allocate_and_delegate=*/true);
  }

  tflite::Interpreter* Interpreter() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_9(mht_9_v, 461, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "Interpreter");
 return interpreter_.get(); }
};

class NNAPIMultiOpPluginTest : public ::testing::Test {
 protected:
  NNAPIMultiOpPluginTest() : delegate_(nullptr, [](TfLiteDelegate*) {}) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_10(mht_10_v, 469, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "NNAPIMultiOpPluginTest");
}
  void SetUp() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_11(mht_11_v, 473, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "SetUp");

    nnapi_ = const_cast<NnApi*>(NnApiImplementation());
    nnapi_mock_ = absl::make_unique<NnApiMock>(nnapi_);
  }

  void CreateDelegate(flatbuffers::Offset<NNAPISettings> nnapi_settings,
                      int max_delegated_partitions) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_12(mht_12_v, 482, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "CreateDelegate");

    tflite_settings_ = flatbuffers::GetTemporaryPointer(
        fbb_,
        CreateTFLiteSettings(fbb_, tflite::Delegate_NNAPI, nnapi_settings,
                             /* gpu_settings */ 0,
                             /* hexagon_settings */ 0,
                             /* xnnpack_settings */ 0,
                             /* coreml_settings */ 0,
                             /* cpu_settings */ 0, max_delegated_partitions));

    plugin_ = delegates::DelegatePluginRegistry::CreateByName(
        "NnapiPlugin", *tflite_settings_);
    delegate_ = plugin_->Create();
  }

  int CountNnApiPartitions() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_13(mht_13_v, 500, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "CountNnApiPartitions");

    return std::count_if(std::begin(model_.Interpreter()->execution_plan()),
                         std::end(model_.Interpreter()->execution_plan()),
                         [this](const int node_index) {
                           return model_.Interpreter()
                                      ->node_and_registration(node_index)
                                      ->first.delegate != nullptr;
                         });
  }

  TfLiteStatus ApplyDelegate() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSexperimentalPSaccelerationPSconfigurationPSnnapi_plugin_testDTcc mht_14(mht_14_v, 513, "", "./tensorflow/lite/experimental/acceleration/configuration/nnapi_plugin_test.cc", "ApplyDelegate");

    model_.Build(delegate_.get());
    return model_.ApplyDelegate();
  }

  NnApi* nnapi_;
  std::unique_ptr<NnApiMock> nnapi_mock_;
  MultiplePartitionsModel model_;
  flatbuffers::FlatBufferBuilder fbb_;
  const TFLiteSettings* tflite_settings_ = nullptr;
  delegates::TfLiteDelegatePtr delegate_;
  std::unique_ptr<delegates::DelegatePluginInterface> plugin_;
};

TEST_F(NNAPIMultiOpPluginTest, PassesMaxDelegatedPartitionsFlag) {
  CreateDelegate(CreateNNAPISettings(
                     fbb_, 0, 0, 0, NNAPIExecutionPreference_UNDEFINED, 0, 0,
                     /* allow CPU */ true,
                     /* execution_priority */
                     tflite::NNAPIExecutionPriority_NNAPI_PRIORITY_UNDEFINED,
                     /* allow_dynamic_dimensions */ false,
                     /* allow_fp16_precision_for_fp32 */ false),
                 /* max_delegated_partitions */ 1);
  nnapi_->ANeuralNetworksModel_getSupportedOperationsForDevices =
      [](const ANeuralNetworksModel* model,
         const ANeuralNetworksDevice* const* devices, uint32_t numDevices,
         bool* supportedOps) -> int {
    supportedOps[0] = true;
    supportedOps[1] = false;
    supportedOps[2] = true;
    return 0;
  };
  EXPECT_EQ(kTfLiteOk, ApplyDelegate());
  EXPECT_EQ(CountNnApiPartitions(), 1);
}

}  // namespace
}  // namespace tflite
