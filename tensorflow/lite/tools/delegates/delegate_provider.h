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

#ifndef TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_
#define TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh() {
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
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tools {

// Same w/ Interpreter::TfLiteDelegatePtr to avoid pulling
// tensorflow/lite/interpreter.h dependency
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

class DelegateProvider {
 public:
  virtual ~DelegateProvider() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_0(mht_0_v, 206, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "~DelegateProvider");
}

  // Create a list of command-line parsable flags based on tool params inside
  // 'params' whose value will be set to the corresponding runtime flag value.
  virtual std::vector<Flag> CreateFlags(ToolParams* params) const = 0;

  // Log tool params. If 'verbose' is set to false, the param is going to be
  // only logged if its value has been set, say via being parsed from
  // commandline flags.
  virtual void LogParams(const ToolParams& params, bool verbose) const = 0;

  // Create a TfLiteDelegate based on tool params.
  virtual TfLiteDelegatePtr CreateTfLiteDelegate(
      const ToolParams& params) const = 0;

  // Similar to the above, create a TfLiteDelegate based on tool params. If the
  // same set of tool params could lead to creating multiple TfLite delegates,
  // also return a relative rank of the delegate that indicates the order of the
  // returned delegate that should be applied to the TfLite runtime.
  virtual std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const = 0;

  virtual std::string GetName() const = 0;

  const ToolParams& DefaultParams() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_1(mht_1_v, 233, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "DefaultParams");
 return default_params_; }

 protected:
  template <typename T>
  Flag CreateFlag(const char* name, ToolParams* params,
                  const std::string& usage) const {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_2_v.push_back("usage: \"" + usage + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_2(mht_2_v, 243, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "CreateFlag");

    return Flag(
        name,
        [params, name](const T& val, int argv_position) {
          params->Set<T>(name, val, argv_position);
        },
        default_params_.Get<T>(name), usage, Flag::kOptional);
  }
  ToolParams default_params_;
};

using DelegateProviderPtr = std::unique_ptr<DelegateProvider>;
using DelegateProviderList = std::vector<DelegateProviderPtr>;

class DelegateProviderRegistrar {
 public:
  template <typename T>
  struct Register {
    Register() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_3(mht_3_v, 264, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "Register");

      auto* const instance = DelegateProviderRegistrar::GetSingleton();
      instance->providers_.emplace_back(DelegateProviderPtr(new T()));
    }
  };

  static const DelegateProviderList& GetProviders() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_4(mht_4_v, 273, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "GetProviders");

    return GetSingleton()->providers_;
  }

 private:
  DelegateProviderRegistrar() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_5(mht_5_v, 281, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "DelegateProviderRegistrar");
}
  DelegateProviderRegistrar(const DelegateProviderRegistrar&) = delete;
  DelegateProviderRegistrar& operator=(const DelegateProviderRegistrar&) =
      delete;

  static DelegateProviderRegistrar* GetSingleton() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_6(mht_6_v, 289, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "GetSingleton");

    static auto* instance = new DelegateProviderRegistrar();
    return instance;
  }
  DelegateProviderList providers_;
};

#define REGISTER_DELEGATE_PROVIDER_VNAME(T) gDelegateProvider_##T##_
#define REGISTER_DELEGATE_PROVIDER(T)                          \
  static tflite::tools::DelegateProviderRegistrar::Register<T> \
      REGISTER_DELEGATE_PROVIDER_VNAME(T);

// Creates a null delegate, useful for cases where no reasonable delegate can be
// created.
TfLiteDelegatePtr CreateNullDelegate();

// A global helper function to get all registered delegate providers.
inline const DelegateProviderList& GetRegisteredDelegateProviders() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_7(mht_7_v, 309, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "GetRegisteredDelegateProviders");

  return DelegateProviderRegistrar::GetProviders();
}

// A helper class to create a list of TfLite delegates based on the provided
// ToolParams and the global DelegateProviderRegistrar.
class ProvidedDelegateList {
 public:
  struct ProvidedDelegate {
    ProvidedDelegate()
        : provider(nullptr), delegate(CreateNullDelegate()), rank(0) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_8(mht_8_v, 322, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "ProvidedDelegate");
}
    const DelegateProvider* provider;
    TfLiteDelegatePtr delegate;
    int rank;
  };

  ProvidedDelegateList() : ProvidedDelegateList(/*params*/ nullptr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_9(mht_9_v, 331, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "ProvidedDelegateList");
}

  // 'params' is the ToolParams instance that this class will operate on,
  // including adding all registered delegate parameters to it etc.
  explicit ProvidedDelegateList(ToolParams* params)
      : providers_(GetRegisteredDelegateProviders()), params_(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_10(mht_10_v, 339, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "ProvidedDelegateList");
}

  const DelegateProviderList& providers() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSdelegatesPSdelegate_providerDTh mht_11(mht_11_v, 344, "", "./tensorflow/lite/tools/delegates/delegate_provider.h", "providers");
 return providers_; }

  // Add all registered delegate params to the contained 'params_'.
  void AddAllDelegateParams() const;

  // Append command-line parsable flags to 'flags' of all registered delegate
  // providers, and associate the flag values at runtime with the contained
  // 'params_'.
  void AppendCmdlineFlags(std::vector<Flag>& flags) const;

  // Removes command-line parsable flag 'name' from 'flags'
  void RemoveCmdlineFlag(std::vector<Flag>& flags,
                         const std::string& name) const;

  // Return a list of TfLite delegates based on the provided 'params', and the
  // list has been already sorted in ascending order according to the rank of
  // the particular parameter that enables the creation of the delegate.
  std::vector<ProvidedDelegate> CreateAllRankedDelegates(
      const ToolParams& params) const;

  // Similar to the above, the list of TfLite delegates are created based on the
  // contained 'params_'.
  std::vector<ProvidedDelegate> CreateAllRankedDelegates() const {
    return CreateAllRankedDelegates(*params_);
  }

 private:
  const DelegateProviderList& providers_;

  // Represent the set of "ToolParam"s that this helper class will operate on.
  ToolParams* const params_;  // Not own the memory.
};
}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_DELEGATES_DELEGATE_PROVIDER_H_
