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
class MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/java/src/main/native/jni_utils.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensorflow/lite/core/shims/jni/jni_utils.h"

namespace tflite {
namespace jni {

const char kIllegalArgumentException[] = "java/lang/IllegalArgumentException";
const char kIllegalStateException[] = "java/lang/IllegalStateException";
const char kNullPointerException[] = "java/lang/NullPointerException";
const char kUnsupportedOperationException[] =
    "java/lang/UnsupportedOperationException";

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("clazz: \"" + (clazz == nullptr ? std::string("nullptr") : std::string((char*)clazz)) + "\"");
   mht_0_v.push_back("fmt: \"" + (fmt == nullptr ? std::string("nullptr") : std::string((char*)fmt)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "ThrowException");

  va_list args;
  va_start(args, fmt);
  const size_t max_msg_len = 512;
  auto* message = static_cast<char*>(malloc(max_msg_len));
  if (message && (vsnprintf(message, max_msg_len, fmt, args) >= 0)) {
    env->ThrowNew(env->FindClass(clazz), message);
  } else {
    env->ThrowNew(env->FindClass(clazz), "");
  }
  if (message) {
    free(message);
  }
  va_end(args);
}

bool CheckJniInitializedOrThrow(JNIEnv* env) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "CheckJniInitializedOrThrow");

  return TfLiteCheckInitializedOrThrow(env);
}

BufferErrorReporter::BufferErrorReporter(JNIEnv* env, int limit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "BufferErrorReporter::BufferErrorReporter");

  buffer_ = new char[limit];
  if (!buffer_) {
    ThrowException(env, kNullPointerException,
                   "Internal error: Malloc of BufferErrorReporter to hold %d "
                   "char failed.",
                   limit);
    return;
  }
  buffer_[0] = '\0';
  start_idx_ = 0;
  end_idx_ = limit - 1;
}

BufferErrorReporter::~BufferErrorReporter() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_3(mht_3_v, 247, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "BufferErrorReporter::~BufferErrorReporter");
 delete[] buffer_; }

int BufferErrorReporter::Report(const char* format, va_list args) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("format: \"" + (format == nullptr ? std::string("nullptr") : std::string((char*)format)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_4(mht_4_v, 253, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "BufferErrorReporter::Report");

  int size = 0;
  // If an error has already been logged, insert a newline.
  if (start_idx_ > 0 && start_idx_ < end_idx_) {
    buffer_[start_idx_++] = '\n';
    ++size;
  }
  if (start_idx_ < end_idx_) {
    size = vsnprintf(buffer_ + start_idx_, end_idx_ - start_idx_, format, args);
  }
  start_idx_ += size;
  return size;
}

const char* BufferErrorReporter::CachedErrorMessage() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_5(mht_5_v, 270, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "BufferErrorReporter::CachedErrorMessage");
 return buffer_; }

jobjectArray CreateStringArray(const std::vector<const char*>& values,
                               JNIEnv* env) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_6(mht_6_v, 276, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "CreateStringArray");

  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    ThrowException(env, tflite::jni::kUnsupportedOperationException,
                   "Internal error: Can not find java/lang/String class.");
    return nullptr;
  }

  jobjectArray results =
      env->NewObjectArray(values.size(), string_class, env->NewStringUTF(""));
  int i = 0;
  for (const char* value : values) {
    env->SetObjectArrayElement(results, i++, env->NewStringUTF(value));
  }
  return results;
}

bool AreDimsDifferent(JNIEnv* env, TfLiteTensor* tensor, const jintArray dims) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePSjni_utilsDTcc mht_7(mht_7_v, 296, "", "./tensorflow/lite/java/src/main/native/jni_utils.cc", "AreDimsDifferent");

  int num_dims = static_cast<int>(env->GetArrayLength(dims));
  jint* ptr = env->GetIntArrayElements(dims, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Empty dimensions of input array.");
    return true;
  }
  bool is_different = false;
  if (tensor->dims->size != num_dims) {
    is_different = true;
  } else {
    for (int i = 0; i < num_dims; ++i) {
      if (ptr[i] != tensor->dims->data[i]) {
        is_different = true;
        break;
      }
    }
  }
  env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
  return is_different;
}

std::vector<int> ConvertJIntArrayToVector(JNIEnv* env, const jintArray inputs) {
  int size = static_cast<int>(env->GetArrayLength(inputs));
  std::vector<int> outputs(size, 0);
  jint* ptr = env->GetIntArrayElements(inputs, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Array has empty dimensions.");
    return {};
  }
  for (int i = 0; i < size; ++i) {
    outputs[i] = ptr[i];
  }
  env->ReleaseIntArrayElements(inputs, ptr, JNI_ABORT);
  return outputs;
}

}  // namespace jni
}  // namespace tflite
