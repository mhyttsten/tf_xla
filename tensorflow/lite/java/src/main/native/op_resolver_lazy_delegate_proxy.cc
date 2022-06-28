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
class MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc {
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
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#if !TFLITE_DISABLE_SELECT_JAVA_APIS
#include <dlfcn.h>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif
#endif

#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>

#include "tensorflow/lite/core/api/op_resolver_internal.h"
#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/core/shims/cc/model_builder.h"
#if TFLITE_DISABLE_SELECT_JAVA_APIS
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/delegate_plugin.h"
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/xnnpack_plugin.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#else
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif
#include "tensorflow/lite/java/src/main/native/op_resolver_lazy_delegate_proxy.h"

namespace tflite {
namespace jni {

namespace {

#if !TFLITE_DISABLE_SELECT_JAVA_APIS

// Indicates if it is safe to call dlsym to check if a symbol is present
bool IsDlsymSafeForSymbolCheck() {
#ifdef __ANDROID__
  // On Android 4.4 (API 19) and earlier it is not safe to call dlsym to check
  // if a symbol if present, as dlsym will crash if the symbol isn't present.
  // The underlying bug is already fixed in the platform long ago
  // <https://android-review.googlesource.com/c/platform/bionic/+/69033>
  // but as Android 4.4 systems still exist in the wild, we check API version
  // as a work-around.

  char sdk_version_str[PROP_VALUE_MAX + 1];
  std::memset(sdk_version_str, 0, sizeof(sdk_version_str));
  if (!__system_property_get("ro.build.version.sdk", sdk_version_str)) {
    return false;
  }

  char* sdk_version_end = sdk_version_str;
  const auto sdk_version = strtol(sdk_version_str, &sdk_version_end, 10);
  return sdk_version_end != sdk_version_str && sdk_version > 19;
#else
  return true;
#endif
}
#endif

}  // namespace

const TfLiteRegistration* OpResolverLazyDelegateProxy::FindOp(
    tflite::BuiltinOperator op, int version) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc mht_0(mht_0_v, 244, "", "./tensorflow/lite/java/src/main/native/op_resolver_lazy_delegate_proxy.cc", "OpResolverLazyDelegateProxy::FindOp");

  return op_resolver_->FindOp(op, version);
}

const TfLiteRegistration* OpResolverLazyDelegateProxy::FindOp(
    const char* op, int version) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc mht_1(mht_1_v, 253, "", "./tensorflow/lite/java/src/main/native/op_resolver_lazy_delegate_proxy.cc", "OpResolverLazyDelegateProxy::FindOp");

  return op_resolver_->FindOp(op, version);
}

OpResolver::TfLiteDelegateCreators
OpResolverLazyDelegateProxy::GetDelegateCreators() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc mht_2(mht_2_v, 261, "", "./tensorflow/lite/java/src/main/native/op_resolver_lazy_delegate_proxy.cc", "OpResolverLazyDelegateProxy::GetDelegateCreators");

  // Early exit if not using the XNNPack delegate by default
  if (!use_xnnpack_) {
    return OpResolver::TfLiteDelegateCreators();
  }

  return OpResolver::TfLiteDelegateCreators{
      {&OpResolverLazyDelegateProxy::createXNNPackDelegate}};
}

bool OpResolverLazyDelegateProxy::MayContainUserDefinedOps() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSop_resolver_lazy_delegate_proxyDTcc mht_3(mht_3_v, 274, "", "./tensorflow/lite/java/src/main/native/op_resolver_lazy_delegate_proxy.cc", "OpResolverLazyDelegateProxy::MayContainUserDefinedOps");

  return OpResolverInternal::MayContainUserDefinedOps(*op_resolver_);
}

std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
OpResolverLazyDelegateProxy::createXNNPackDelegate(int num_threads) {
  TfLiteDelegate* delegate = nullptr;
  void (*delegate_deleter)(TfLiteDelegate*) = nullptr;
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  // Construct a FlatBuffer containing
  //   TFLiteSettings {
  //     delegate: Delegate.XNNPack
  //     XNNPackSettings {
  //       num_threads: <num_threads>
  //     }
  //   }
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  tflite::XNNPackSettingsBuilder xnnpack_settings_builder(flatbuffer_builder);
  if (num_threads >= 0) {
    xnnpack_settings_builder.add_num_threads(num_threads);
  }
  flatbuffers::Offset<tflite::XNNPackSettings> xnnpack_settings =
      xnnpack_settings_builder.Finish();
  tflite::TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  tflite_settings_builder.add_xnnpack_settings(xnnpack_settings);
  tflite_settings_builder.add_delegate(tflite::Delegate_XNNPACK);
  flatbuffers::Offset<tflite::TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const tflite::TFLiteSettings* tflite_settings_flatbuffer =
      flatbuffers::GetRoot<tflite::TFLiteSettings>(
          flatbuffer_builder.GetBufferPointer());
  // Create an XNNPack delegate plugin using the settings from the flatbuffer.
  const TfLiteOpaqueDelegatePlugin* delegate_plugin =
      TfLiteXnnpackDelegatePluginCApi();
  delegate = reinterpret_cast<TfLiteDelegate*>(
      delegate_plugin->create(tflite_settings_flatbuffer));
  delegate_deleter =
      reinterpret_cast<void (*)(TfLiteDelegate*)>(delegate_plugin->destroy);
#else
  // We use dynamic loading to avoid taking a hard dependency on XNNPack.
  // This allows clients that use trimmed builds to save on binary size.

  if (IsDlsymSafeForSymbolCheck()) {
    auto xnnpack_options_default =
        reinterpret_cast<decltype(TfLiteXNNPackDelegateOptionsDefault)*>(
            dlsym(RTLD_DEFAULT, "TfLiteXNNPackDelegateOptionsDefault"));
    auto xnnpack_create =
        reinterpret_cast<decltype(TfLiteXNNPackDelegateCreate)*>(
            dlsym(RTLD_DEFAULT, "TfLiteXNNPackDelegateCreate"));
    auto xnnpack_delete =
        reinterpret_cast<decltype(TfLiteXNNPackDelegateDelete)*>(
            dlsym(RTLD_DEFAULT, "TfLiteXNNPackDelegateDelete"));

    if (xnnpack_options_default && xnnpack_create && xnnpack_delete) {
      TfLiteXNNPackDelegateOptions options = xnnpack_options_default();
      if (num_threads > 0) {
        options.num_threads = num_threads;
      }
      delegate = xnnpack_create(&options);
      delegate_deleter = xnnpack_delete;
    }
  }
#endif
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      delegate, delegate_deleter);
}

}  // namespace jni
}  // namespace tflite
