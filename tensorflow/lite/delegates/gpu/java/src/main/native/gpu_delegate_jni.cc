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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc() {
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

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#include "tensorflow/lite/delegates/gpu/gl/request_gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"

extern "C" {

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jboolean precision_loss_allowed,
    jboolean quantized_models_allowed, jint inference_preference,
    jstring serialization_dir, jstring model_token) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("serialization_dir: \"" + serialization_dir + "\"");
   mht_0_v.push_back("model_token: \"" + model_token + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "Java_org_tensorflow_lite_gpu_GpuDelegate_createDelegate");

  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  if (precision_loss_allowed == JNI_TRUE) {
    options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority2 =
        TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
    options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  }
  options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
  if (quantized_models_allowed) {
    options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;
  }
  options.inference_preference = static_cast<int32_t>(inference_preference);
  if (serialization_dir) {
    options.serialization_dir =
        env->GetStringUTFChars(serialization_dir, /*isCopy=*/nullptr);
  }
  if (model_token) {
    options.model_token =
        env->GetStringUTFChars(model_token, /*isCopy=*/nullptr);
  }
  if (options.serialization_dir && options.model_token) {
    options.experimental_flags |=
        TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
  }
  return reinterpret_cast<jlong>(TfLiteGpuDelegateV2Create(&options));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "Java_org_tensorflow_lite_gpu_GpuDelegate_deleteDelegate");

  TfLiteGpuDelegateV2Delete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

namespace {
class CompatibilityListHelper {
 public:
  CompatibilityListHelper()
      : compatibility_list_(
            tflite::acceleration::GPUCompatibilityList::Create()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_2(mht_2_v, 248, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "CompatibilityListHelper");
}
  absl::Status ReadInfo() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_3(mht_3_v, 252, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "ReadInfo");

    auto status = tflite::acceleration::RequestAndroidInfo(&android_info_);
    if (!status.ok()) return status;

    if (android_info_.android_sdk_version < "21") {
      // Weakly linked symbols may not be available on pre-21, and the GPU is
      // not supported anyway so return early.
      return absl::OkStatus();
    }

    std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
    status = tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env);
    if (!status.ok()) return status;

    status = tflite::gpu::gl::RequestGpuInfo(&gpu_info_);
    if (!status.ok()) return status;

    return absl::OkStatus();
  }

  bool IsDelegateSupportedOnThisDevice() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_4(mht_4_v, 275, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "IsDelegateSupportedOnThisDevice");

    return compatibility_list_->Includes(android_info_, gpu_info_);
  }

 private:
  tflite::acceleration::AndroidInfo android_info_;
  tflite::gpu::GpuInfo gpu_info_;
  std::unique_ptr<tflite::acceleration::GPUCompatibilityList>
      compatibility_list_;
};
}  // namespace

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_createCompatibilityList(
    JNIEnv* env, jclass clazz) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_5(mht_5_v, 292, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "Java_org_tensorflow_lite_gpu_CompatibilityList_createCompatibilityList");

  CompatibilityListHelper* compatibility_list = new CompatibilityListHelper;
  auto status = compatibility_list->ReadInfo();
  // Errors in ReadInfo should almost always be failures to construct the OpenGL
  // environment. Treating that as "GPU unsupported" is reasonable, and we can
  // swallow the error.
  status.IgnoreError();
  return reinterpret_cast<jlong>(compatibility_list);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_nativeIsDelegateSupportedOnThisDevice(
    JNIEnv* env, jclass clazz, jlong compatibility_list_handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_6(mht_6_v, 307, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "Java_org_tensorflow_lite_gpu_CompatibilityList_nativeIsDelegateSupportedOnThisDevice");

  CompatibilityListHelper* compatibility_list =
      reinterpret_cast<CompatibilityListHelper*>(compatibility_list_handle);
  return compatibility_list->IsDelegateSupportedOnThisDevice() ? JNI_TRUE
                                                               : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_gpu_CompatibilityList_deleteCompatibilityList(
    JNIEnv* env, jclass clazz, jlong compatibility_list_handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSjavaPSsrcPSmainPSnativePSgpu_delegate_jniDTcc mht_7(mht_7_v, 319, "", "./tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.cc", "Java_org_tensorflow_lite_gpu_CompatibilityList_deleteCompatibilityList");

  CompatibilityListHelper* compatibility_list =
      reinterpret_cast<CompatibilityListHelper*>(compatibility_list_handle);
  delete compatibility_list;
}

}  // extern "C"
