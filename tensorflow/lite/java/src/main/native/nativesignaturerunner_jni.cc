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
class MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc {
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
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/core/shims/cc/interpreter.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"

using tflite::jni::ThrowException;
using tflite_shims::Interpreter;

#ifndef TFLITE_DISABLE_SELECT_JAVA_APIS
namespace tflite {
// A helper class to access private information of SignatureRunner class.
class SignatureRunnerJNIHelper {
 public:
  explicit SignatureRunnerJNIHelper(SignatureRunner* runner)
      : signature_runner_(runner) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_0(mht_0_v, 200, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "SignatureRunnerJNIHelper");
}

  // Gets the subgraph index associated with this SignatureRunner.
  int GetSubgraphIndex() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_1(mht_1_v, 206, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "GetSubgraphIndex");

    if (!signature_runner_) return -1;

    return signature_runner_->signature_def_->subgraph_index;
  }

  // Gets the tensor index of a given input.
  int GetInputTensorIndex(const char* input_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_2(mht_2_v, 217, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "GetInputTensorIndex");

    const auto& it = signature_runner_->signature_def_->inputs.find(input_name);
    if (it == signature_runner_->signature_def_->inputs.end()) {
      return -1;
    }
    return it->second;
  }

  // Gets the tensor index of a given output.
  int GetOutputTensorIndex(const char* output_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_3(mht_3_v, 230, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "GetOutputTensorIndex");

    const auto& it =
        signature_runner_->signature_def_->outputs.find(output_name);
    if (it == signature_runner_->signature_def_->outputs.end()) {
      return -1;
    }
    return it->second;
  }

  // Gets the index of the input specified by `input_name`.
  int GetInputIndex(const char* input_name) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_4(mht_4_v, 244, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "GetInputIndex");

    int input_tensor_index = GetInputTensorIndex(input_name);
    if (input_tensor_index == -1) return -1;

    int count = 0;
    for (int tensor_idx : signature_runner_->subgraph_->inputs()) {
      if (input_tensor_index == tensor_idx) return count;
      ++count;
    }
    return -1;
  }

  // Gets the index of the output specified by `output_name`.
  int GetOutputIndex(const char* output_name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_5(mht_5_v, 261, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "GetOutputIndex");

    int output_tensor_index = GetOutputTensorIndex(output_name);
    if (output_tensor_index == -1) return -1;

    int count = 0;
    for (int tensor_idx : signature_runner_->subgraph_->outputs()) {
      if (output_tensor_index == tensor_idx) return count;
      ++count;
    }
    return -1;
  }

 private:
  SignatureRunner* signature_runner_;
};
}  // namespace tflite

using tflite::SignatureRunner;
using tflite::SignatureRunnerJNIHelper;
using tflite::jni::BufferErrorReporter;
using tflite::jni::CastLongToPointer;
using tflite_shims::Interpreter;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS

extern "C" {
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSignatureRunner(
    JNIEnv* env, jclass clazz, jlong handle, jstring signature_key) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("signature_key: \"" + signature_key + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_6(mht_6_v, 292, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSignatureRunner");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeGetSignatureRunner");
  return -1;
#else
  Interpreter* interpreter = CastLongToPointer<Interpreter>(env, handle);
  if (interpreter == nullptr) return -1;
  const char* signature_key_ptr =
      env->GetStringUTFChars(signature_key, nullptr);

  SignatureRunner* runner = interpreter->GetSignatureRunner(signature_key_ptr);
  if (runner == nullptr) {
    // Release the memory before returning.
    env->ReleaseStringUTFChars(signature_key, signature_key_ptr);
    return -1;
  }

  // Release the memory before returning.
  env->ReleaseStringUTFChars(signature_key, signature_key_ptr);
  return reinterpret_cast<jlong>(runner);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSubgraphIndex(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_7(mht_7_v, 321, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSubgraphIndex");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
             "Not supported: nativeGetSubgraphIndex");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  return SignatureRunnerJNIHelper(runner).GetSubgraphIndex();
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInputNames(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_8(mht_8_v, 338, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInputNames");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeInputNames");
  return nullptr;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return nullptr;
  return tflite::jni::CreateStringArray(runner->input_names(), env);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeOutputNames(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_9(mht_9_v, 355, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeOutputNames");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeOutputNames");
  return nullptr;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return nullptr;
  return tflite::jni::CreateStringArray(runner->output_names(), env);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetInputIndex(
    JNIEnv* env, jclass clazz, jlong handle, jstring input_name) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_10(mht_10_v, 373, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetInputIndex");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeGetInputIndex");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  const char* input_name_ptr = env->GetStringUTFChars(input_name, nullptr);
  int index = SignatureRunnerJNIHelper(runner).GetInputIndex(input_name_ptr);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return index;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetOutputIndex(
    JNIEnv* env, jclass clazz, jlong handle, jstring output_name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("output_name: \"" + output_name + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_11(mht_11_v, 395, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetOutputIndex");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeGetOutputIndex");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  const char* output_name_ptr = env->GetStringUTFChars(output_name, nullptr);
  int index = SignatureRunnerJNIHelper(runner).GetOutputIndex(output_name_ptr);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(output_name, output_name_ptr);
  return index;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeResizeInput(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle,
    jstring input_name, jintArray dims) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_12(mht_12_v, 418, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeResizeInput");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeResizeInput");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return JNI_FALSE;
  // Check whether it is resizing with the same dimensions.
  const char* input_name_ptr = env->GetStringUTFChars(input_name, nullptr);
  TfLiteTensor* target = runner->input_tensor(input_name_ptr);
  if (target == nullptr) {
    // Release the memory before returning.
    env->ReleaseStringUTFChars(input_name, input_name_ptr);
    return JNI_FALSE;
  }
  bool is_changed = tflite::jni::AreDimsDifferent(env, target, dims);
  if (is_changed) {
    TfLiteStatus status = runner->ResizeInputTensor(
        input_name_ptr, tflite::jni::ConvertJIntArrayToVector(env, dims));
    if (status != kTfLiteOk) {
      ThrowException(env, tflite::jni::kIllegalArgumentException,
                     "Internal error: Failed to resize input %s: %s",
                     input_name_ptr, error_reporter->CachedErrorMessage());
      // Release the memory before returning.
      env->ReleaseStringUTFChars(input_name, input_name_ptr);
      return JNI_FALSE;
    }
  }
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return is_changed ? JNI_TRUE : JNI_FALSE;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeAllocateTensors(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_13(mht_13_v, 460, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeAllocateTensors");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeAllocateTensors");
#else

  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return;

  if (runner->AllocateTensors() != kTfLiteOk) {
    ThrowException(
        env, tflite::jni::kIllegalStateException,
        "Internal error: Unexpected failure when preparing tensor allocations:"
        " %s",
        error_reporter->CachedErrorMessage());
  }
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInvoke(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSnativesignaturerunner_jniDTcc mht_14(mht_14_v, 486, "", "./tensorflow/lite/java/src/main/native/nativesignaturerunner_jni.cc", "Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInvoke");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeInvoke");
#else

  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return;

  if (runner->Invoke() != kTfLiteOk) {
    ThrowException(env, tflite::jni::kIllegalStateException,
                   "Internal error: Failed to run on the given Interpreter: %s",
                   error_reporter->CachedErrorMessage());
  }
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}
}  // extern "C"
