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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc() {
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

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "absl/base/call_once.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/errors.h"
#include "third_party/tensorrt/NvInferPlugin.h"
#endif

namespace tensorflow {
namespace tensorrt {

std::tuple<int, int, int> GetLinkedTensorRTVersion() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  return std::tuple<int, int, int>{NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                                   NV_TENSORRT_PATCH};
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

std::tuple<int, int, int> GetLoadedTensorRTVersion() {
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  int ver = getInferLibVersion();
  int major = ver / 1000;
  ver = ver - major * 1000;
  int minor = ver / 100;
  int patch = ver - minor * 100;
  return std::tuple<int, int, int>{major, minor, patch};
#else
  return std::tuple<int, int, int>{0, 0, 0};
#endif
}

}  // namespace tensorrt
}  // namespace tensorflow

#if GOOGLE_CUDA && GOOGLE_TENSORRT
namespace tensorflow {
namespace tensorrt {

Status GetTrtBindingIndex(const char* tensor_name, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tensor_name: \"" + (tensor_name == nullptr ? std::string("nullptr") : std::string((char*)tensor_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc mht_0(mht_0_v, 230, "", "./tensorflow/compiler/tf2tensorrt/common/utils.cc", "GetTrtBindingIndex");

  // If the engine has been built for K profiles, the first getNbBindings() / K
  // bindings are used by profile number 0, the following getNbBindings() / K
  // bindings are used by profile number 1 etc.
  //
  // GetBindingIndex(tensor_name) returns the binding index for the progile 0.
  // We can also consider it as a "binding_index_within_profile".
  *binding_index = cuda_engine->getBindingIndex(tensor_name);
  if (*binding_index == -1) {
    const string msg = absl::StrCat("Input node ", tensor_name, " not found");
    return errors::NotFound(msg);
  }
  int n_profiles = cuda_engine->getNbOptimizationProfiles();
  // If we have more then one optimization profile, then we need to shift the
  // binding index according to the following formula:
  // binding_index_within_engine = binding_index_within_profile +
  //                               profile_index * bindings_per_profile
  const int bindings_per_profile = cuda_engine->getNbBindings() / n_profiles;
  *binding_index = *binding_index + profile_index * bindings_per_profile;
  return Status::OK();
}

Status GetTrtBindingIndex(int network_input_index, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc mht_1(mht_1_v, 257, "", "./tensorflow/compiler/tf2tensorrt/common/utils.cc", "GetTrtBindingIndex");

  const string input_name =
      absl::StrCat(IONamePrefixes::kInputPHName, network_input_index);
  return GetTrtBindingIndex(input_name.c_str(), profile_index, cuda_engine,
                            binding_index);
}

namespace {

void InitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc mht_2(mht_2_v, 269, "", "./tensorflow/compiler/tf2tensorrt/common/utils.cc", "InitializeTrtPlugins");

#if defined(PLATFORM_WINDOWS)
  LOG_WARNING_WITH_PREFIX
      << "Windows support is provided experimentally. No guarantee is made "
         "regarding functionality or engineering support. Use at your own "
         "risk.";
#endif
  LOG(INFO) << "Linked TensorRT version: "
            << absl::StrJoin(GetLinkedTensorRTVersion(), ".");
  LOG(INFO) << "Loaded TensorRT version: "
            << absl::StrJoin(GetLoadedTensorRTVersion(), ".");

  bool plugin_initialized = initLibNvInferPlugins(trt_logger, "");
  if (!plugin_initialized) {
    LOG(ERROR) << "Failed to initialize TensorRT plugins, and conversion may "
                  "fail later.";
  }

  int num_trt_plugins = 0;
  nvinfer1::IPluginCreator* const* trt_plugin_creator_list =
      getPluginRegistry()->getPluginCreatorList(&num_trt_plugins);
  if (!trt_plugin_creator_list) {
    LOG_WARNING_WITH_PREFIX << "Can not find any TensorRT plugins in registry.";
  } else {
    VLOG(1) << "Found the following " << num_trt_plugins
            << " TensorRT plugins in registry:";
    for (int i = 0; i < num_trt_plugins; ++i) {
      if (!trt_plugin_creator_list[i]) {
        LOG_WARNING_WITH_PREFIX
            << "TensorRT plugin at index " << i
            << " is not accessible (null pointer returned by "
               "getPluginCreatorList for this plugin)";
      } else {
        VLOG(1) << "  " << trt_plugin_creator_list[i]->getPluginName();
      }
    }
  }
}

}  // namespace

void MaybeInitializeTrtPlugins(nvinfer1::ILogger* trt_logger) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc mht_3(mht_3_v, 313, "", "./tensorflow/compiler/tf2tensorrt/common/utils.cc", "MaybeInitializeTrtPlugins");

  static absl::once_flag once;
  absl::call_once(once, InitializeTrtPlugins, trt_logger);
}

}  // namespace tensorrt
}  // namespace tensorflow

namespace nvinfer1 {
std::ostream& operator<<(std::ostream& os,
                         const nvinfer1::TensorFormat& format) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc mht_4(mht_4_v, 326, "", "./tensorflow/compiler/tf2tensorrt/common/utils.cc", "operator<<");

  os << "nvinfer1::TensorFormat::";
  switch (format) {
    case nvinfer1::TensorFormat::kLINEAR:
      os << "kLINEAR";
      break;

    case nvinfer1::TensorFormat::kCHW2:
      os << "kCHW2";
      break;

    case nvinfer1::TensorFormat::kHWC8:
      os << "kHWC8";
      break;

    case nvinfer1::TensorFormat::kCHW4:
      os << "kCHW4";
      break;

    case nvinfer1::TensorFormat::kCHW16:
      os << "kCHW16";
      break;

    case nvinfer1::TensorFormat::kCHW32:
      os << "kCHW32";
      break;

#if IS_TRT_VERSION_GE(8, 0, 0, 0)
    case nvinfer1::TensorFormat::kDHWC8:
      os << "kDHWC8";
      break;

    case nvinfer1::TensorFormat::kCDHW32:
      os << "kCDHW32";
      break;

    case nvinfer1::TensorFormat::kHWC:
      os << "kHWC";
      break;

    case nvinfer1::TensorFormat::kDLA_LINEAR:
      os << "kDLA_LINEAR";
      break;

    case nvinfer1::TensorFormat::kDLA_HWC4:
      os << "kDLA_HWC4";
      break;

    case nvinfer1::TensorFormat::kHWC16:
      os << "kHWC16";
      break;
#endif

    default:
      os << "unknown format";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const nvinfer1::DataType& v) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPScommonPSutilsDTcc mht_5(mht_5_v, 388, "", "./tensorflow/compiler/tf2tensorrt/common/utils.cc", "operator<<");

  os << "nvinfer1::DataType::";
  switch (v) {
    case nvinfer1::DataType::kFLOAT:
      os << "kFLOAT";
      break;
    case nvinfer1::DataType::kHALF:
      os << "kHalf";
      break;
    case nvinfer1::DataType::kINT8:
      os << "kINT8";
      break;
    case nvinfer1::DataType::kINT32:
      os << "kINT32";
      break;
    case nvinfer1::DataType::kBOOL:
      os << "kBOOL";
      break;
  }
  return os;
}
}  // namespace nvinfer1

#endif
