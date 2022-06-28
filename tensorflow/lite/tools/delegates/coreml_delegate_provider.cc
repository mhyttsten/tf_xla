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
class MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc() {
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

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if defined(__APPLE__)
#include "TargetConditionals.h"
#if (TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR) || \
    (TARGET_OS_OSX && TARGET_CPU_ARM64)
// Only enable coreml delegate when using a real iPhone device or Apple Silicon.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#endif
#endif

namespace tflite {
namespace tools {

class CoreMlDelegateProvider : public DelegateProvider {
 public:
  CoreMlDelegateProvider() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/tools/delegates/coreml_delegate_provider.cc", "CoreMlDelegateProvider");

#if defined(REAL_IPHONE_DEVICE)
    default_params_.AddParam("use_coreml", ToolParam::Create<bool>(false));
    default_params_.AddParam("coreml_version", ToolParam::Create<int>(0));
#endif
  }
  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/tools/delegates/coreml_delegate_provider.cc", "GetName");
 return "COREML"; }
};
REGISTER_DELEGATE_PROVIDER(CoreMlDelegateProvider);

std::vector<Flag> CoreMlDelegateProvider::CreateFlags(
    ToolParams* params) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc mht_2(mht_2_v, 229, "", "./tensorflow/lite/tools/delegates/coreml_delegate_provider.cc", "CoreMlDelegateProvider::CreateFlags");

#if defined(REAL_IPHONE_DEVICE)
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_coreml", params, "use Core ML"),
      CreateFlag<int>("coreml_version", params,
                      "Target Core ML version for model conversion. "
                      "The default value is 0 and it means using the newest "
                      "version that's available on the device."),
  };
  return flags;
#else
  return {};
#endif
}

void CoreMlDelegateProvider::LogParams(const ToolParams& params,
                                       bool verbose) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc mht_3(mht_3_v, 248, "", "./tensorflow/lite/tools/delegates/coreml_delegate_provider.cc", "CoreMlDelegateProvider::LogParams");

#if defined(REAL_IPHONE_DEVICE)
  LOG_TOOL_PARAM(params, bool, "use_coreml", "Use CoreML", verbose);
  LOG_TOOL_PARAM(params, int, "coreml_version", "CoreML version", verbose);
#endif
}

TfLiteDelegatePtr CoreMlDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPScoreml_delegate_providerDTcc mht_4(mht_4_v, 259, "", "./tensorflow/lite/tools/delegates/coreml_delegate_provider.cc", "CoreMlDelegateProvider::CreateTfLiteDelegate");

  TfLiteDelegatePtr delegate = CreateNullDelegate();

#if defined(REAL_IPHONE_DEVICE)
  if (params.Get<bool>("use_coreml")) {
    TfLiteCoreMlDelegateOptions coreml_opts = {
        .enabled_devices = TfLiteCoreMlDelegateAllDevices};
    coreml_opts.coreml_version = params.Get<int>("coreml_version");
    coreml_opts.max_delegated_partitions =
        params.Get<int>("max_delegated_partitions");
    coreml_opts.min_nodes_per_partition =
        params.Get<int>("min_nodes_per_partition");
    delegate = TfLiteDelegatePtr(TfLiteCoreMlDelegateCreate(&coreml_opts),
                                 &TfLiteCoreMlDelegateDelete);
    if (!delegate) {
      TFLITE_LOG(WARN)
          << "CoreML acceleration is unsupported on this platform.";
    }
  }
#endif

  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
CoreMlDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  int rank = 0;
#if defined(REAL_IPHONE_DEVICE)
  rank = params.GetPosition<bool>("use_coreml");
#endif
  return std::make_pair(std::move(ptr), rank);
}

}  // namespace tools
}  // namespace tflite
