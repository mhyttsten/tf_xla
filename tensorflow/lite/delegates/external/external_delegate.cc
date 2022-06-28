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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc() {
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
#include "tensorflow/lite/delegates/external/external_delegate.h"

#include <locale>
#include <string>
#include <vector>

#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/shared_library.h"

namespace tflite {
namespace {

// External delegate library construct
struct ExternalLib {
  using CreateDelegatePtr = std::add_pointer<TfLiteDelegate*(
      const char**, const char**, size_t,
      void (*report_error)(const char*))>::type;
  using DestroyDelegatePtr = std::add_pointer<void(TfLiteDelegate*)>::type;
  struct wchar_codecvt : public std::codecvt<wchar_t, char, std::mbstate_t> {};
  std::wstring_convert<wchar_codecvt> converter;

  // Open a given delegate library and load the create/destroy symbols
  bool load(const std::string library) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("library: \"" + library + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "load");

#if defined(_WIN32)
    void* handle = SharedLibrary::LoadLibrary(
        converter.from_bytes(library.c_str()).c_str());
#else
    void* handle = SharedLibrary::LoadLibrary(library.c_str());
#endif  // defined(_WIN32)
    if (handle == nullptr) {
      TFLITE_LOG(TFLITE_LOG_INFO, "Unable to load external delegate from : %s",
                 library.c_str());
    } else {
      create =
          reinterpret_cast<decltype(create)>(SharedLibrary::GetLibrarySymbol(
              handle, "tflite_plugin_create_delegate"));
      destroy =
          reinterpret_cast<decltype(destroy)>(SharedLibrary::GetLibrarySymbol(
              handle, "tflite_plugin_destroy_delegate"));
      return create && destroy;
    }
    return false;
  }

  CreateDelegatePtr create{nullptr};
  DestroyDelegatePtr destroy{nullptr};
};

// An ExternalDelegateWrapper is responsibile to manage a TFLite delegate
// initialized from a shared library. It creates a delegate from the given
// option and storages it to external_delegate_ member variable. On the
// destruction, it conducts necessary clean up process.
class ExternalDelegateWrapper {
 public:
  explicit ExternalDelegateWrapper(
      const TfLiteExternalDelegateOptions* options);
  ~ExternalDelegateWrapper();

  // Return a TfLiteDelegate which is created from
  // tflite_plugin_create_delegate() of an external delegate logic.
  TfLiteDelegate* tflite_external_delegate() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_1(mht_1_v, 248, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "tflite_external_delegate");
 return external_delegate_; }

  // Return a TfLiteDelegate which is convertibile to this class.
  TfLiteDelegate* tflite_wrapper_delegate() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_2(mht_2_v, 254, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "tflite_wrapper_delegate");
 return &wrapper_delegate_; }

 private:
  ExternalLib external_lib_;

  // external delegate instance owned by external delegate logic.
  // It's created by "tflite_plugin_destroy_delegate()" function in the external
  // delegate logic And it should be released by
  // "tflite_plugin_destroy_delegate()" function.
  TfLiteDelegate* external_delegate_;

  // TfLiteDelegate representation of this ExternalDelegateWrapper object.
  TfLiteDelegate wrapper_delegate_;
};

// Converts the given TfLiteDelegate to an ExternalDelegateWrapper instance.
inline ExternalDelegateWrapper* GetExternalDelegateWrapper(
    TfLiteDelegate* delegate) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_3(mht_3_v, 274, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "GetExternalDelegateWrapper");

  return reinterpret_cast<ExternalDelegateWrapper*>(delegate->data_);
}

// Relay Prepare() call to the associated external TfLiteDelegate object.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_4(mht_4_v, 282, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "DelegatePrepare");

  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->Prepare(context, external_delegate);
}

// Relay CopyFromBufferHandle() call to the associated external TfLiteDelegate
// object.
TfLiteStatus DelegateCopyFromBufferHandle(TfLiteContext* context,
                                          struct TfLiteDelegate* delegate,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteTensor* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_5(mht_5_v, 297, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "DelegateCopyFromBufferHandle");

  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->CopyFromBufferHandle(context, delegate,
                                                 buffer_handle, tensor);
}

// Relay CopyToBufferHandle() call to the associated external TfLiteDelegate
// object.
TfLiteStatus DelegateCopyToBufferHandle(TfLiteContext* context,
                                        struct TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_6(mht_6_v, 313, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "DelegateCopyToBufferHandle");

  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->CopyToBufferHandle(context, delegate, buffer_handle,
                                               tensor);
}

// Relay FreeBufferHandle() call to the associated external TfLiteDelegate
// object.
void DelegateFreeBufferHandle(TfLiteContext* context,
                              struct TfLiteDelegate* delegate,
                              TfLiteBufferHandle* handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_7(mht_7_v, 328, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "DelegateFreeBufferHandle");

  auto external_delegate_wrapper = GetExternalDelegateWrapper(delegate);
  TfLiteDelegate* external_delegate =
      external_delegate_wrapper->tflite_external_delegate();
  return external_delegate->FreeBufferHandle(context, delegate, handle);
}

ExternalDelegateWrapper::ExternalDelegateWrapper(
    const TfLiteExternalDelegateOptions* options) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_8(mht_8_v, 339, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "ExternalDelegateWrapper::ExternalDelegateWrapper");

  external_delegate_ = nullptr;
  if (external_lib_.load(options->lib_path)) {
    std::vector<const char*> ckeys, cvalues;
    for (int i = 0; i < options->count; i++) {
      ckeys.push_back(options->keys[i]);
      cvalues.push_back(options->values[i]);
    }

    external_delegate_ = external_lib_.create(ckeys.data(), cvalues.data(),
                                              ckeys.size(), nullptr);
    if (external_delegate_) {
      wrapper_delegate_ = {
          .data_ = reinterpret_cast<void*>(this),
          .Prepare = DelegatePrepare,
          .CopyFromBufferHandle = nullptr,
          .CopyToBufferHandle = nullptr,
          .FreeBufferHandle = nullptr,
          .flags = external_delegate_->flags,
      };
      if (external_delegate_->CopyFromBufferHandle) {
        wrapper_delegate_.CopyFromBufferHandle = DelegateCopyFromBufferHandle;
      }
      if (external_delegate_->CopyToBufferHandle) {
        wrapper_delegate_.CopyToBufferHandle = DelegateCopyToBufferHandle;
      }
      if (external_delegate_->FreeBufferHandle) {
        wrapper_delegate_.FreeBufferHandle = DelegateFreeBufferHandle;
      }
    }
  }
}

ExternalDelegateWrapper::~ExternalDelegateWrapper() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_9(mht_9_v, 375, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "ExternalDelegateWrapper::~ExternalDelegateWrapper");

  if (external_delegate_ != nullptr) {
    external_lib_.destroy(external_delegate_);
  }
}

}  // namespace
}  // namespace tflite

// TfLiteExternalDelegateOptionsInsert adds key/value to the given
// TfLiteExternalDelegateOptions instance.
TfLiteStatus TfLiteExternalDelegateOptionsInsert(
    TfLiteExternalDelegateOptions* options, const char* key,
    const char* value) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("key: \"" + (key == nullptr ? std::string("nullptr") : std::string((char*)key)) + "\"");
   mht_10_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_10(mht_10_v, 393, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "TfLiteExternalDelegateOptionsInsert");

  if (options->count >= kExternalDelegateMaxOptions) {
    return kTfLiteError;
  }
  options->keys[options->count] = key;
  options->values[options->count] = value;
  options->count++;
  return kTfLiteOk;
}

TfLiteExternalDelegateOptions TfLiteExternalDelegateOptionsDefault(
    const char* lib_path) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("lib_path: \"" + (lib_path == nullptr ? std::string("nullptr") : std::string((char*)lib_path)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_11(mht_11_v, 408, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "TfLiteExternalDelegateOptionsDefault");

  // As 'keys' and 'values' don't need to be set here, using designated
  // initializers may cause a compiling error as "non-trivial designated
  // initializers not supported" by some compiler.
  TfLiteExternalDelegateOptions options;
  options.lib_path = lib_path;
  options.count = 0;
  options.insert = TfLiteExternalDelegateOptionsInsert;
  return options;
}

TfLiteDelegate* TfLiteExternalDelegateCreate(
    const TfLiteExternalDelegateOptions* options) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_12(mht_12_v, 423, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "TfLiteExternalDelegateCreate");

  auto* external_delegate_wrapper =
      new tflite::ExternalDelegateWrapper(options);
  if (external_delegate_wrapper) {
    return external_delegate_wrapper->tflite_wrapper_delegate();
  }
  return nullptr;
}

void TfLiteExternalDelegateDelete(TfLiteDelegate* delegate) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSexternalPSexternal_delegateDTcc mht_13(mht_13_v, 435, "", "./tensorflow/lite/delegates/external/external_delegate.cc", "TfLiteExternalDelegateDelete");

  delete tflite::GetExternalDelegateWrapper(delegate);
}
