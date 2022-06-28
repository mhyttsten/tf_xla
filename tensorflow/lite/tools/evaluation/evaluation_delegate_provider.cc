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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc() {
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

#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"

#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {
namespace {
constexpr char kNnapiDelegate[] = "nnapi";
constexpr char kGpuDelegate[] = "gpu";
constexpr char kHexagonDelegate[] = "hexagon";
constexpr char kXnnpackDelegate[] = "xnnpack";
}  // namespace

TfliteInferenceParams::Delegate ParseStringToDelegateType(
    const std::string& val) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("val: \"" + val + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "ParseStringToDelegateType");

  if (val == kNnapiDelegate) return TfliteInferenceParams::NNAPI;
  if (val == kGpuDelegate) return TfliteInferenceParams::GPU;
  if (val == kHexagonDelegate) return TfliteInferenceParams::HEXAGON;
  if (val == kXnnpackDelegate) return TfliteInferenceParams::XNNPACK;
  return TfliteInferenceParams::NONE;
}

TfLiteDelegatePtr CreateTfLiteDelegate(const TfliteInferenceParams& params,
                                       std::string* error_msg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "CreateTfLiteDelegate");

  const auto type = params.delegate();
  switch (type) {
    case TfliteInferenceParams::NNAPI: {
      auto p = CreateNNAPIDelegate();
      if (!p && error_msg) *error_msg = "NNAPI not supported";
      return p;
    }
    case TfliteInferenceParams::GPU: {
      auto p = CreateGPUDelegate();
      if (!p && error_msg) *error_msg = "GPU delegate not supported.";
      return p;
    }
    case TfliteInferenceParams::HEXAGON: {
      auto p = CreateHexagonDelegate(/*library_directory_path=*/"",
                                     /*profiling=*/false);
      if (!p && error_msg) {
        *error_msg =
            "Hexagon delegate is not supported on the platform or required "
            "libraries are missing.";
      }
      return p;
    }
    case TfliteInferenceParams::XNNPACK: {
      auto p = CreateXNNPACKDelegate(params.num_threads());
      if (!p && error_msg) *error_msg = "XNNPACK delegate not supported.";
      return p;
    }
    case TfliteInferenceParams::NONE:
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
    default:
      if (error_msg) {
        *error_msg = "Creation of delegate type: " +
                     TfliteInferenceParams::Delegate_Name(type) +
                     " not supported yet.";
      }
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
}

DelegateProviders::DelegateProviders()
    : delegate_list_util_(&params_),
      delegates_map_([=]() -> std::unordered_map<std::string, int> {
        std::unordered_map<std::string, int> delegates_map;
        const auto& providers = delegate_list_util_.providers();
        for (int i = 0; i < providers.size(); ++i) {
          delegates_map[providers[i]->GetName()] = i;
        }
        return delegates_map;
      }()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_2(mht_2_v, 264, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "DelegateProviders::DelegateProviders");

  delegate_list_util_.AddAllDelegateParams();
}

std::vector<Flag> DelegateProviders::GetFlags() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_3(mht_3_v, 271, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "DelegateProviders::GetFlags");

  std::vector<Flag> flags;
  delegate_list_util_.AppendCmdlineFlags(flags);
  return flags;
}

bool DelegateProviders::InitFromCmdlineArgs(int* argc, const char** argv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_4(mht_4_v, 280, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "DelegateProviders::InitFromCmdlineArgs");

  std::vector<Flag> flags = GetFlags();
  bool parse_result = Flags::Parse(argc, argv, flags);
  if (!parse_result || params_.Get<bool>("help")) {
    std::string usage = Flags::Usage(argv[0], flags);
    TFLITE_LOG(ERROR) << usage;
    // Returning false intentionally when "--help=true" is specified so that
    // the caller could check the return value to decide stopping the execution.
    parse_result = false;
  }
  return parse_result;
}

TfLiteDelegatePtr DelegateProviders::CreateDelegate(
    const std::string& name) const {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_5(mht_5_v, 298, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "DelegateProviders::CreateDelegate");

  const auto it = delegates_map_.find(name);
  if (it == delegates_map_.end()) {
    return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
  const auto& providers = delegate_list_util_.providers();
  return providers[it->second]->CreateTfLiteDelegate(params_);
}

tools::ToolParams DelegateProviders::GetAllParams(
    const TfliteInferenceParams& params) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTcc mht_6(mht_6_v, 311, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc", "DelegateProviders::GetAllParams");

  tools::ToolParams tool_params;
  tool_params.Merge(params_, /*overwrite*/ false);

  if (params.has_num_threads()) {
    tool_params.Set<int32_t>("num_threads", params.num_threads());
  }

  const auto type = params.delegate();
  switch (type) {
    case TfliteInferenceParams::NNAPI:
      if (tool_params.HasParam("use_nnapi")) {
        tool_params.Set<bool>("use_nnapi", true);
      }
      break;
    case TfliteInferenceParams::GPU:
      if (tool_params.HasParam("use_gpu")) {
        tool_params.Set<bool>("use_gpu", true);
      }
      break;
    case TfliteInferenceParams::HEXAGON:
      if (tool_params.HasParam("use_hexagon")) {
        tool_params.Set<bool>("use_hexagon", true);
      }
      break;
    case TfliteInferenceParams::XNNPACK:
      if (tool_params.HasParam("use_xnnpack")) {
        tool_params.Set<bool>("use_xnnpack", true);
      }
      break;
    default:
      break;
  }
  return tool_params;
}

}  // namespace evaluation
}  // namespace tflite
