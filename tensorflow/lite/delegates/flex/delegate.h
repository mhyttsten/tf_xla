/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_
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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh() {
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


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {

namespace flex {
namespace testing {
class KernelTest;
}  // namespace testing
}  // namespace flex

// WARNING: This is an experimental interface that is subject to change.
// Delegate that can be used to extract parts of a graph that are designed to be
// executed by TensorFlow's runtime via Eager.
//
// The interpreter must be constructed after the FlexDelegate and destructed
// before the FlexDelegate. This delegate may be used with multiple
// interpreters, but it is *not* thread-safe.
//
// Usage:
//   auto delegate = FlexDelegate::Create();
//   ... build interpreter ...
//
//   if (delegate) {
//     interpreter->ModifyGraphWithDelegate(delegate.get());
//   }
//
//   void* delegate_data = delegate->data_;
//   interpreter->SetCancellationFunction(
//     delegate_data,
//     FlexDelegate::HasCancelled);
//
//   ... run inference ...
//
//    static_cast<FlexDelegate*>(delegate_data)->Cancel();
//
//   ... destroy interpreter ...
//   ... destroy delegate ...
class FlexDelegate : public SimpleDelegateInterface {
 public:
  friend class flex::testing::KernelTest;

  // Creates a delegate that supports TF ops.
  static TfLiteDelegateUniquePtr Create() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh mht_0(mht_0_v, 231, "", "./tensorflow/lite/delegates/flex/delegate.h", "Create");

    return Create(/*base_delegate*/ nullptr);
  }

  ~FlexDelegate() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh mht_1(mht_1_v, 238, "", "./tensorflow/lite/delegates/flex/delegate.h", "~FlexDelegate");
}

  flex::DelegateData* mutable_data() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh mht_2(mht_2_v, 243, "", "./tensorflow/lite/delegates/flex/delegate.h", "mutable_data");
 return &delegate_data_; }

  // This method is thread safe. It does two things:
  //   1. Calls the CancellationManager of the TF eager runtime to support
  //      intra-op cancellation in TF.
  //   2. Uses the CancellationManager to signal TFLite interpreter for inter-op
  //      cancellation.
  // Training is non-recoverable after calling this API.
  void Cancel();

  // The param `data` must be a pointer to a FlexDelegate instance.
  static bool HasCancelled(void* data);

 protected:
  // We sometimes have to create certain stub data to test FlexDelegate. To
  // achieve this, we will make a testing flex delegate class that inherits from
  // FlexDelegate to override certain things for stub data creation. Therefore,
  // this function accepts a FlexDelegate instance to initiliaze it properly for
  // create a testing flex delegate in some cases, and it is only used in
  // testing.
  static TfLiteDelegateUniquePtr Create(
      std::unique_ptr<FlexDelegate> base_delegate);

  FlexDelegate() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh mht_3(mht_3_v, 269, "", "./tensorflow/lite/delegates/flex/delegate.h", "FlexDelegate");
}

  const char* Name() const override;

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override;

  TfLiteStatus Initialize(TfLiteContext* context) override;

  SimpleDelegateInterface::Options DelegateOptions() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPSdelegateDTh mht_4(mht_4_v, 282, "", "./tensorflow/lite/delegates/flex/delegate.h", "DelegateOptions");

    // Use default options.
    return SimpleDelegateInterface::Options();
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override;

  TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                    TfLiteBufferHandle buffer_handle,
                                    TfLiteTensor* output);

  flex::DelegateData delegate_data_;

  // Pointer to the base TfLiteDelegate which is created from the Create call.
  TfLiteDelegate* base_delegate_ = nullptr;

 private:
  // A cancellation manager.
  std::unique_ptr<tensorflow::CancellationManager> cancellation_manager_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_FLEX_DELEGATE_H_
