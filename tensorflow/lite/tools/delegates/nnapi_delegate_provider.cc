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
class MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc() {
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
#include <string>
#include <utility>

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

namespace {

using nnapi::NnApiSupportLibrary;

// StatefulNnApiDelegate that holds onto an NnApiSupportLibrary instance
// passed to the constructor for later destruction.
// Note that the support library must outlive the delegate.
class NnApiSupportLibraryDelegate : public StatefulNnApiDelegate {
 public:
  NnApiSupportLibraryDelegate(const NnApiSupportLibrary* nnapi_sl,
                              Options options)
      : StatefulNnApiDelegate(nnapi_sl->getFL5(), options),
        nnapi_sl_(nnapi_sl) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "NnApiSupportLibraryDelegate");
}
  const NnApiSupportLibrary* get_nnapi_sl() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "get_nnapi_sl");
 return nnapi_sl_; }

 private:
  const NnApiSupportLibrary* const nnapi_sl_;
};

}  // namespace

class NnapiDelegateProvider : public DelegateProvider {
 public:
  NnapiDelegateProvider() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_2(mht_2_v, 225, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "NnapiDelegateProvider");

    default_params_.AddParam("use_nnapi", ToolParam::Create<bool>(false));
    default_params_.AddParam("nnapi_execution_preference",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("nnapi_execution_priority",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("nnapi_accelerator_name",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("disable_nnapi_cpu",
                             ToolParam::Create<bool>(true));
    default_params_.AddParam("nnapi_allow_fp16",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("nnapi_allow_dynamic_dimensions",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("nnapi_use_burst_mode",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("nnapi_support_library_path",
                             ToolParam::Create<std::string>(""));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_3(mht_3_v, 256, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "GetName");
 return "NNAPI"; }
};
REGISTER_DELEGATE_PROVIDER(NnapiDelegateProvider);

std::vector<Flag> NnapiDelegateProvider::CreateFlags(ToolParams* params) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_4(mht_4_v, 263, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "NnapiDelegateProvider::CreateFlags");

  std::vector<Flag> flags = {
      CreateFlag<bool>("use_nnapi", params, "use nnapi delegate api"),
      CreateFlag<std::string>("nnapi_execution_preference", params,
                              "execution preference for nnapi delegate. Should "
                              "be one of the following: fast_single_answer, "
                              "sustained_speed, low_power, undefined"),
      CreateFlag<std::string>("nnapi_execution_priority", params,
                              "The model execution priority in nnapi, and it "
                              "should be one of the following: default, low, "
                              "medium and high. This requires Android 11+."),
      CreateFlag<std::string>(
          "nnapi_accelerator_name", params,
          "the name of the nnapi accelerator to use (requires Android Q+)"),
      CreateFlag<bool>("disable_nnapi_cpu", params,
                       "Disable the NNAPI CPU device"),
      CreateFlag<bool>("nnapi_allow_fp16", params,
                       "Allow fp32 computation to be run in fp16"),
      CreateFlag<bool>(
          "nnapi_allow_dynamic_dimensions", params,
          "Whether to allow dynamic dimension sizes without re-compilation. "
          "This requires Android 9+."),
      CreateFlag<bool>(
          "nnapi_use_burst_mode", params,
          "use NNAPI Burst mode if supported. Burst mode allows accelerators "
          "to efficiently manage resources, which would significantly reduce "
          "overhead especially if the same delegate instance is to be used for "
          "multiple inferences."),
      CreateFlag<std::string>(
          "nnapi_support_library_path", params,
          "Path from which NNAPI support library will be loaded to construct "
          "the delegate. In order to use NNAPI delegate with support library, "
          "--nnapi_accelerator_name must be specified and must be equal to one "
          "of the devices provided by the support library."),
  };

  return flags;
}

void NnapiDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_5(mht_5_v, 306, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "NnapiDelegateProvider::LogParams");

  LOG_TOOL_PARAM(params, bool, "use_nnapi", "Use NNAPI", verbose);
  if (!params.Get<bool>("use_nnapi")) return;

  LOG_TOOL_PARAM(params, std::string, "nnapi_execution_preference",
                 "NNAPI execution preference", verbose);
  LOG_TOOL_PARAM(params, std::string, "nnapi_execution_priority",
                 "Model execution priority in nnapi", verbose);
  LOG_TOOL_PARAM(params, std::string, "nnapi_accelerator_name",
                 "NNAPI accelerator name", verbose);

  std::string string_device_names_list =
      nnapi::GetStringDeviceNamesList(NnApiImplementation());
  // Print available devices when possible as it's informative.
  if (!string_device_names_list.empty()) {
    TFLITE_LOG(INFO) << "NNAPI accelerators available: ["
                     << string_device_names_list << "]";
  }

  LOG_TOOL_PARAM(params, bool, "disable_nnapi_cpu", "Disable NNAPI cpu",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "nnapi_allow_fp16", "Allow fp16 in NNAPI",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "nnapi_allow_dynamic_dimensions",
                 "Allow dynamic dimensions in NNAPI", verbose);
  LOG_TOOL_PARAM(params, bool, "nnapi_use_burst_mode",
                 "Use burst mode in NNAPI", verbose);
}

TfLiteDelegatePtr NnapiDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSnnapi_delegate_providerDTcc mht_6(mht_6_v, 339, "", "./tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc", "NnapiDelegateProvider::CreateTfLiteDelegate");

  TfLiteDelegatePtr null_delegate = CreateNullDelegate();
  if (params.Get<bool>("use_nnapi")) {
    StatefulNnApiDelegate::Options options;
    std::string accelerator_name =
        params.Get<std::string>("nnapi_accelerator_name");
    if (!accelerator_name.empty()) {
      options.accelerator_name = accelerator_name.c_str();
    } else {
      options.disallow_nnapi_cpu = params.Get<bool>("disable_nnapi_cpu");
    }

    if (params.Get<bool>("nnapi_allow_fp16")) {
      options.allow_fp16 = true;
    }

    if (params.Get<bool>("nnapi_allow_dynamic_dimensions")) {
      options.allow_dynamic_dimensions = true;
    }

    if (params.Get<bool>("nnapi_use_burst_mode")) {
      options.use_burst_computation = true;
    }

    std::string string_execution_preference =
        params.Get<std::string>("nnapi_execution_preference");
    // Only set execution preference if user explicitly passes one. Otherwise,
    // leave it as whatever NNAPI has as the default.
    if (!string_execution_preference.empty()) {
      tflite::StatefulNnApiDelegate::Options::ExecutionPreference
          execution_preference =
              tflite::StatefulNnApiDelegate::Options::kUndefined;
      if (string_execution_preference == "low_power") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kLowPower;
      } else if (string_execution_preference == "sustained_speed") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
      } else if (string_execution_preference == "fast_single_answer") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kFastSingleAnswer;
      } else if (string_execution_preference == "undefined") {
        execution_preference =
            tflite::StatefulNnApiDelegate::Options::kUndefined;
      } else {
        TFLITE_LOG(WARN) << "The provided value ("
                         << string_execution_preference
                         << ") is not a valid nnapi execution preference.";
      }
      options.execution_preference = execution_preference;
    }

    std::string string_execution_priority =
        params.Get<std::string>("nnapi_execution_priority");
    // Only set execution priority if user explicitly passes one. Otherwise,
    // leave it as whatever NNAPI has as the default.
    if (!string_execution_priority.empty()) {
      int execution_priority = 0;
      if (string_execution_priority == "default") {
        execution_priority = ANEURALNETWORKS_PRIORITY_DEFAULT;
      } else if (string_execution_priority == "low") {
        execution_priority = ANEURALNETWORKS_PRIORITY_LOW;
      } else if (string_execution_priority == "medium") {
        execution_priority = ANEURALNETWORKS_PRIORITY_MEDIUM;
      } else if (string_execution_priority == "high") {
        execution_priority = ANEURALNETWORKS_PRIORITY_HIGH;
      } else {
        TFLITE_LOG(WARN) << "The provided value (" << string_execution_priority
                         << ") is not a valid nnapi execution priority.";
      }
      options.execution_priority = execution_priority;
    }

    int max_delegated_partitions = params.Get<int>("max_delegated_partitions");
    if (max_delegated_partitions >= 0) {
      options.max_number_delegated_partitions = max_delegated_partitions;
    }

    // Serialization.
    std::string serialize_dir =
        params.Get<std::string>("delegate_serialize_dir");
    std::string serialize_token =
        params.Get<std::string>("delegate_serialize_token");
    if (!serialize_dir.empty() && !serialize_token.empty()) {
      options.cache_dir = serialize_dir.c_str();
      options.model_token = serialize_token.c_str();
    }

    if (params.Get<std::string>("nnapi_support_library_path").empty()) {
      const auto* nnapi_impl = NnApiImplementation();
      if (!nnapi_impl->nnapi_exists) {
        TFLITE_LOG(WARN)
            << "NNAPI acceleration is unsupported on this platform.";
        return null_delegate;
      }
      return TfLiteDelegatePtr(
          new StatefulNnApiDelegate(nnapi_impl, options),
          [](TfLiteDelegate* delegate) {
            delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
          });
    } else {
      std::string sl_path =
          params.Get<std::string>("nnapi_support_library_path");
      auto nnapi_impl = nnapi::loadNnApiSupportLibrary(sl_path);
      if (!nnapi_impl) {
        TFLITE_LOG(WARN) << "Couldn't load NNAPI support library from path: "
                         << sl_path;
        return null_delegate;
      }
      return TfLiteDelegatePtr(
          new NnApiSupportLibraryDelegate(nnapi_impl.release(), options),
          [](TfLiteDelegate* delegate) {
            NnApiSupportLibraryDelegate* sl_delegate =
                reinterpret_cast<NnApiSupportLibraryDelegate*>(delegate);
            const NnApiSupportLibrary* sl = sl_delegate->get_nnapi_sl();
            delete sl_delegate;
            delete sl;
          });
    }
  } else if (!params.Get<std::string>("nnapi_accelerator_name").empty()) {
    TFLITE_LOG(WARN)
        << "`--use_nnapi=true` must be set for the provided NNAPI accelerator ("
        << params.Get<std::string>("nnapi_accelerator_name") << ") to be used.";
  } else if (!params.Get<std::string>("nnapi_execution_preference").empty()) {
    TFLITE_LOG(WARN) << "`--use_nnapi=true` must be set for the provided NNAPI "
                        "execution preference ("
                     << params.Get<std::string>("nnapi_execution_preference")
                     << ") to be used.";
  }
  return null_delegate;
}

std::pair<TfLiteDelegatePtr, int>
NnapiDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<bool>("use_nnapi"));
}

}  // namespace tools
}  // namespace tflite
