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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc() {
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
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate_kernel.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_implementation.h"
#include "tensorflow/lite/delegates/hexagon/utils.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace {
// Should be > 0. > 16 causes problems.
constexpr int kMaxHexagonGraphs = 4;
constexpr int kMaxMaxHexagonGraphs = 16;
constexpr int kMinNodesPerHexagonGraph = 2;

class HexagonDelegate : public SimpleDelegateInterface {
 public:
  explicit HexagonDelegate(const TfLiteHexagonDelegateOptions* params)
      : params_(params != nullptr ? *params
                                  : TfLiteHexagonDelegateOptions({0})) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "HexagonDelegate");

    if (params_.max_delegated_partitions <= 0) {
      params_.max_delegated_partitions = kMaxHexagonGraphs;
    } else if (params_.max_delegated_partitions > kMaxMaxHexagonGraphs) {
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                      "Hexagon delegate: cannot have this many %d partitions, "
                      "and will cap to at most %d partitions.\n",
                      params_.max_delegated_partitions, kMaxMaxHexagonGraphs);
      params_.max_delegated_partitions = kMaxMaxHexagonGraphs;
    }
    if (params_.min_nodes_per_partition <= 0) {
      params_.min_nodes_per_partition = kMinNodesPerHexagonGraph;
    }
  }

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_1(mht_1_v, 229, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "IsNodeSupportedByDelegate");

    return IsNodeSupportedByHexagon(registration, node, context);
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_2(mht_2_v, 236, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "Initialize");
 return kTfLiteOk; }

  const char* Name() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_3(mht_3_v, 241, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "Name");
 return "TfLiteHexagonDelegate"; }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<HexagonDelegateKernel>(params_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_4(mht_4_v, 251, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "DelegateOptions");

    auto options = SimpleDelegateInterface::Options();
    options.max_delegated_partitions = params_.max_delegated_partitions;
    options.min_nodes_per_partition = params_.min_nodes_per_partition;
    return options;
  }

  bool VerifyDelegate() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_5(mht_5_v, 261, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "VerifyDelegate");

    auto* hexagon_nn = HexagonNNImplementation();
    if (hexagon_nn == nullptr) {
      return false;
    }
    if (hexagon_nn->hexagon_nn_version != nullptr &&
        hexagon_nn->hexagon_nn_hexagon_interface_version) {
      int hexagon_nn_version = -1;
      int hexagon_interface_version =
          hexagon_nn->hexagon_nn_hexagon_interface_version();
      if (hexagon_nn->hexagon_nn_version(&hexagon_nn_version) != 0) {
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING,
                        "Failed to fetch Hexagon NN version. This might be "
                        "because you're using incompatible versions of "
                        "libhexagon_interface and libhexagon_nn_skel. "
                        "You must use compatible versions. "
                        "Refer to Tensorflow Lite Hexagon Delegate Guide.");
        return false;
      }
      if (hexagon_nn_version != hexagon_interface_version) {
        TFLITE_LOG_PROD(
            tflite::TFLITE_LOG_WARNING,
            "Incompatible versions between interface library and "
            "libhexagon_skel %d vs %d. You must use compatible versions. "
            "Refer to Tensorflow Lite Hexagon Delegate Guide.",
            hexagon_interface_version, hexagon_nn_version);
        return false;
      }
    }
    return hexagon_nn->hexagon_nn_is_device_supported &&
           hexagon_nn->hexagon_nn_is_device_supported();
  }

 private:
  TfLiteHexagonDelegateOptions params_;
};

}  // namespace
}  // namespace tflite

TfLiteDelegate* TfLiteHexagonDelegateCreate(
    const TfLiteHexagonDelegateOptions* options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_6(mht_6_v, 305, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "TfLiteHexagonDelegateCreate");

  auto hexagon_delegate_interface =
      std::make_unique<tflite::HexagonDelegate>(options);
  if (!hexagon_delegate_interface->VerifyDelegate()) {
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Hexagon Delegate is not supported.\n");
    return nullptr;
  }
  auto* initialized_delegate =
      tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
          std::move(hexagon_delegate_interface));
  if (options->enable_dynamic_batch_size) {
    initialized_delegate->flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
  }
  return initialized_delegate;
}

TfLiteHexagonDelegateOptions TfLiteHexagonDelegateOptionsDefault() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_7(mht_7_v, 325, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "TfLiteHexagonDelegateOptionsDefault");

  TfLiteHexagonDelegateOptions result{0};
  return result;
}

void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_8(mht_8_v, 333, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "TfLiteHexagonDelegateDelete");

  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}

void TfLiteHexagonInit() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_9(mht_9_v, 340, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "TfLiteHexagonInit");
 tflite::HexagonDelegateKernel::InitState(); }

void TfLiteHexagonInitWithPath(const char* lib_directory_path) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("lib_directory_path: \"" + (lib_directory_path == nullptr ? std::string("nullptr") : std::string((char*)lib_directory_path)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_10(mht_10_v, 346, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "TfLiteHexagonInitWithPath");

  if (lib_directory_path != nullptr) {
    std::string env_var_value = lib_directory_path;
    env_var_value += ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH", env_var_value.c_str(), 1 /* overwrite */);
  }
  tflite::HexagonDelegateKernel::InitState();
}
void TfLiteHexagonTearDown() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_delegateDTcc mht_11(mht_11_v, 357, "", "./tensorflow/lite/delegates/hexagon/hexagon_delegate.cc", "TfLiteHexagonTearDown");
 tflite::HexagonDelegateKernel::Teardown(); }
