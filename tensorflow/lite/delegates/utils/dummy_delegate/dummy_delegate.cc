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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc() {
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
#include "tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.h"

#include <utility>

#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace dummy_test {

// Dummy delegate kernel.
class DummyDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit DummyDelegateKernel(const DummyDelegateOptions& options)
      : options_(options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "DummyDelegateKernel");
}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_1(mht_1_v, 203, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "Init");

    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_2(mht_2_v, 210, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "Prepare");

    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_3(mht_3_v, 217, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "Eval");

    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const DummyDelegateOptions options_;
};

// DummyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class DummyDelegate : public SimpleDelegateInterface {
 public:
  explicit DummyDelegate(const DummyDelegateOptions& options)
      : options_(options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_4(mht_4_v, 233, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "DummyDelegate");
}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_5(mht_5_v, 239, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "IsNodeSupportedByDelegate");

    return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_6(mht_6_v, 246, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "Initialize");
 return kTfLiteOk; }

  const char* Name() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_7(mht_7_v, 251, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "Name");

    static constexpr char kName[] = "DummyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<DummyDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_8(mht_8_v, 264, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "DelegateOptions");

    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const DummyDelegateOptions options_;
};

}  // namespace dummy_test
}  // namespace tflite

DummyDelegateOptions TfLiteDummyDelegateOptionsDefault() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_9(mht_9_v, 279, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "TfLiteDummyDelegateOptionsDefault");

  DummyDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this dummy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(const DummyDelegateOptions* options) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_10(mht_10_v, 293, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "TfLiteDummyDelegateCreate");

  std::unique_ptr<tflite::dummy_test::DummyDelegate> dummy(
      new tflite::dummy_test::DummyDelegate(
          options ? *options : TfLiteDummyDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(dummy));
}

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSutilsPSdummy_delegatePSdummy_delegateDTcc mht_11(mht_11_v, 304, "", "./tensorflow/lite/delegates/utils/dummy_delegate/dummy_delegate.cc", "TfLiteDummyDelegateDelete");

  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
