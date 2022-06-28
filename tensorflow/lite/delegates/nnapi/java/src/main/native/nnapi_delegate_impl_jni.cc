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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSjavaPSsrcPSmainPSnativePSnnapi_delegate_impl_jniDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSjavaPSsrcPSmainPSnativePSnnapi_delegate_impl_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSjavaPSsrcPSmainPSnativePSnnapi_delegate_impl_jniDTcc() {
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

#include <jni.h>

#include <memory>
#include <type_traits>

#if TFLITE_DISABLE_SELECT_JAVA_APIS
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/nnapi_plugin.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#else
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

#if TFLITE_DISABLE_SELECT_JAVA_APIS
using flatbuffers::FlatBufferBuilder;
using tflite::NNAPISettings;
using tflite::NNAPISettingsBuilder;
using tflite::TFLiteSettings;
using tflite::TFLiteSettingsBuilder;
#else
using tflite::StatefulNnApiDelegate;
#endif

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_createDelegate(
    JNIEnv* env, jclass clazz, jint preference, jstring accelerator_name,
    jstring cache_dir, jstring model_token, jint max_delegated_partitions,
    jboolean override_disallow_cpu, jboolean disallow_cpu_value,
    jboolean allow_fp16, jlong nnapi_support_library_handle) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("accelerator_name: \"" + accelerator_name + "\"");
   mht_0_v.push_back("cache_dir: \"" + cache_dir + "\"");
   mht_0_v.push_back("model_token: \"" + model_token + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSjavaPSsrcPSmainPSnativePSnnapi_delegate_impl_jniDTcc mht_0(mht_0_v, 219, "", "./tensorflow/lite/delegates/nnapi/java/src/main/native/nnapi_delegate_impl_jni.cc", "Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_createDelegate");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  // Use NNAPI Delegate via Delegate Plugin API.
  // This approach would work for the !TFLITE_DISABLE_SELECT_JAVA_APIS case too,
  // but may have a slightly higher overhead due to the need to construct
  // a FlatBuffer for the configuration parameters.

  // Construct a FlatBuffer that contains the following:
  //   TFLiteSettings {
  //     NnapiSettings {
  //       accelerator_name : <accelerator_name>,
  //       cache_directory : <cache_dir>,
  //       model_token : <model_token>,
  //       allow_nnapi_cpu_on_android_10_plus: !<disallow_cpu_value>,
  //       allow_fp16_precision_for_fp32: <allow_fp16>,
  //       support_library_handle: <nnapi_support_library_handle>,
  //     }
  //     max_delegate_partitions: <max_delegated_partitions>
  //   }
  // where the values in angle brackets are the parameters to this function,
  // except that we set the 'allow_nnapi_cpu_on_android_10_plus' field only if
  // <override_disallow_cpu> is true, and that we only set the other fields
  // if they have non-default values.
  FlatBufferBuilder flatbuffer_builder;
  flatbuffers::Offset<flatbuffers::String> accelerator_name_fb_string = 0;
  if (accelerator_name) {
    const char* accelerator_name_c_string =
        env->GetStringUTFChars(accelerator_name, nullptr);
    accelerator_name_fb_string =
        flatbuffer_builder.CreateString(accelerator_name_c_string);
    env->ReleaseStringUTFChars(accelerator_name, accelerator_name_c_string);
  }
  flatbuffers::Offset<flatbuffers::String> cache_directory_fb_string = 0;
  if (cache_dir) {
    const char* cache_directory_c_string =
        env->GetStringUTFChars(cache_dir, nullptr);
    cache_directory_fb_string =
        flatbuffer_builder.CreateString(cache_directory_c_string);
    env->ReleaseStringUTFChars(cache_dir, cache_directory_c_string);
  }
  flatbuffers::Offset<flatbuffers::String> model_token_fb_string = 0;
  if (model_token) {
    const char* model_token_c_string =
        env->GetStringUTFChars(model_token, nullptr);
    model_token_fb_string =
        flatbuffer_builder.CreateString(model_token_c_string);
    env->ReleaseStringUTFChars(model_token, model_token_c_string);
  }
  NNAPISettingsBuilder nnapi_settings_builder(flatbuffer_builder);
  nnapi_settings_builder.add_execution_preference(
      static_cast<tflite::NNAPIExecutionPreference>(preference));
  if (accelerator_name) {
    nnapi_settings_builder.add_accelerator_name(accelerator_name_fb_string);
  }
  if (cache_dir) {
    nnapi_settings_builder.add_cache_directory(cache_directory_fb_string);
  }
  if (model_token) {
    nnapi_settings_builder.add_model_token(model_token_fb_string);
  }
  if (override_disallow_cpu) {
    nnapi_settings_builder.add_allow_nnapi_cpu_on_android_10_plus(
        !disallow_cpu_value);
  }
  if (allow_fp16) {
    nnapi_settings_builder.add_allow_fp16_precision_for_fp32(allow_fp16);
  }
  if (nnapi_support_library_handle) {
    nnapi_settings_builder.add_support_library_handle(
        nnapi_support_library_handle);
  }
  flatbuffers::Offset<NNAPISettings> nnapi_settings =
      nnapi_settings_builder.Finish();
  TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  tflite_settings_builder.add_nnapi_settings(nnapi_settings);
  if (max_delegated_partitions >= 0) {
    tflite_settings_builder.add_max_delegated_partitions(
        max_delegated_partitions);
  }
  flatbuffers::Offset<TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  const TFLiteSettings* settings = flatbuffers::GetRoot<TFLiteSettings>(
      flatbuffer_builder.GetBufferPointer());

  // Construct the delegate using the Delegate Plugin C API,
  // and passing in the flatbuffer settings that we constructed above.
  TfLiteOpaqueDelegate* nnapi_delegate =
      TfLiteNnapiDelegatePluginCApi()->create(settings);
  return reinterpret_cast<jlong>(nnapi_delegate);
#else
  // Use NNAPI Delegate directly.

  // Construct an Options object for the parameter settings.
  StatefulNnApiDelegate::Options options = StatefulNnApiDelegate::Options();
  options.execution_preference =
      (StatefulNnApiDelegate::Options::ExecutionPreference)preference;
  if (accelerator_name) {
    options.accelerator_name =
        env->GetStringUTFChars(accelerator_name, nullptr);
  }
  if (cache_dir) {
    options.cache_dir = env->GetStringUTFChars(cache_dir, nullptr);
  }
  if (model_token) {
    options.model_token = env->GetStringUTFChars(model_token, nullptr);
  }
  if (max_delegated_partitions >= 0) {
    options.max_number_delegated_partitions = max_delegated_partitions;
  }
  if (override_disallow_cpu) {
    options.disallow_nnapi_cpu = disallow_cpu_value;
  }
  if (allow_fp16) {
    options.allow_fp16 = allow_fp16;
  }
  // Construct the delegate, using the options object constructed earlier.
  auto delegate =
      nnapi_support_library_handle
          ? new StatefulNnApiDelegate(reinterpret_cast<NnApiSLDriverImplFL5*>(
                                          nnapi_support_library_handle),
                                      options)
          : new StatefulNnApiDelegate(options);
  // Deallocate temporary strings.
  if (options.accelerator_name) {
    env->ReleaseStringUTFChars(accelerator_name, options.accelerator_name);
  }
  if (options.cache_dir) {
    env->ReleaseStringUTFChars(cache_dir, options.cache_dir);
  }
  if (options.model_token) {
    env->ReleaseStringUTFChars(model_token, options.model_token);
  }
  return reinterpret_cast<jlong>(delegate);
#endif
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_getNnapiErrno(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong delegate) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSjavaPSsrcPSmainPSnativePSnnapi_delegate_impl_jniDTcc mht_1(mht_1_v, 362, "", "./tensorflow/lite/delegates/nnapi/java/src/main/native/nnapi_delegate_impl_jni.cc", "Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_getNnapiErrno");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TfLiteOpaqueDelegate* nnapi_delegate =
      reinterpret_cast<TfLiteOpaqueDelegate*>(delegate);
  return TfLiteNnapiDelegatePluginCApi()->get_delegate_errno(nnapi_delegate);
#else
  StatefulNnApiDelegate* nnapi_delegate =
      reinterpret_cast<StatefulNnApiDelegate*>(delegate);
  return nnapi_delegate->GetNnApiErrno();
#endif
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSnnapiPSjavaPSsrcPSmainPSnativePSnnapi_delegate_impl_jniDTcc mht_2(mht_2_v, 379, "", "./tensorflow/lite/delegates/nnapi/java/src/main/native/nnapi_delegate_impl_jni.cc", "Java_org_tensorflow_lite_nnapi_NnApiDelegateImpl_deleteDelegate");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TfLiteOpaqueDelegate* nnapi_delegate =
      reinterpret_cast<TfLiteOpaqueDelegate*>(delegate);
  TfLiteNnapiDelegatePluginCApi()->destroy(nnapi_delegate);
#else
  delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
#endif
}

}  // extern "C"
