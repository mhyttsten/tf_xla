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

#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTh() {
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


#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace evaluation {

using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;
class DelegateProviders {
 public:
  DelegateProviders();

  // Returns a list of commandline flags that delegate providers define.
  std::vector<Flag> GetFlags();

  // Initialize delegate-related parameters from commandline arguments and
  // returns true if successful.
  bool InitFromCmdlineArgs(int* argc, const char** argv);

  // Get all parameters from all registered delegate providers.
  const tools::ToolParams& GetAllParams() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSevaluationPSevaluation_delegate_providerDTh mht_0(mht_0_v, 214, "", "./tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h", "GetAllParams");
 return params_; }

  // Get a new set of parameters based on the given TfliteInferenceParams
  // 'params' but considering what have been initialized (i.e. 'params_').
  // Note the same-meaning parameter (e.g. number of TfLite interpreter threads)
  // in TfliteInferenceParams will take precedence over the parameter of the
  // same meaning in 'params_'.
  tools::ToolParams GetAllParams(const TfliteInferenceParams& params) const;

  // Create the a TfLite delegate instance based on the provided delegate
  // 'name'. If the specified one isn't found, an empty TfLiteDelegatePtr is
  // returned.
  TfLiteDelegatePtr CreateDelegate(const std::string& name) const;

  // Create a list of TfLite delegates based on what have been initialized (i.e.
  // 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates()
      const {
    return delegate_list_util_.CreateAllRankedDelegates();
  }

  // Create a list of TfLite delegates based on the given TfliteInferenceParams
  // 'params' but considering what have been initialized (i.e. 'params_').
  std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates(
      const TfliteInferenceParams& params) const {
    auto converted = GetAllParams(params);
    ProvidedDelegateList util(&converted);
    return util.CreateAllRankedDelegates();
  }

 private:
  // Contain delegate-related parameters that are initialized from command-line
  // flags.
  tools::ToolParams params_;

  // A helper to create TfLite delegates.
  ProvidedDelegateList delegate_list_util_;

  // Key is the delegate name, and the value is the index to a TfLite delegate
  // provider in the "delegate_list_util_.providers()" list.
  const std::unordered_map<std::string, int> delegates_map_;
};

// Parse a string 'val' to the corresponding delegate type defined by
// TfliteInferenceParams::Delegate.
TfliteInferenceParams::Delegate ParseStringToDelegateType(
    const std::string& val);

// Create a TfLite delegate based on the given TfliteInferenceParams 'params'.
// If there's an error during the creation, an error message will be recorded to
// 'error_msg' if provided.
TfLiteDelegatePtr CreateTfLiteDelegate(const TfliteInferenceParams& params,
                                       std::string* error_msg = nullptr);
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
