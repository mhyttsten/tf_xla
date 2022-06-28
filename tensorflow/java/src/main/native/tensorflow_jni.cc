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
class MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/java/src/main/native/tensorflow_jni.h"

#include <limits>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

JNIEXPORT jstring JNICALL Java_org_tensorflow_TensorFlow_version(JNIEnv* env,
                                                                 jclass clazz) {
  return env->NewStringUTF(TF_Version());
}

JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_TensorFlow_registeredOpList(JNIEnv* env, jclass clazz) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc mht_0(mht_0_v, 197, "", "./tensorflow/java/src/main/native/tensorflow_jni.cc", "Java_org_tensorflow_TensorFlow_registeredOpList");

  TF_Buffer* buf = TF_GetAllOpList();
  jint length = static_cast<int>(buf->length);
  jbyteArray ret = env->NewByteArray(length);
  env->SetByteArrayRegion(ret, 0, length, static_cast<const jbyte*>(buf->data));
  TF_DeleteBuffer(buf);
  return ret;
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_TensorFlow_libraryLoad(
    JNIEnv* env, jclass clazz, jstring filename) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc mht_1(mht_1_v, 211, "", "./tensorflow/java/src/main/native/tensorflow_jni.cc", "Java_org_tensorflow_TensorFlow_libraryLoad");

  TF_Status* status = TF_NewStatus();
  const char* cname = env->GetStringUTFChars(filename, nullptr);
  TF_Library* h = TF_LoadLibrary(cname, status);
  throwExceptionIfNotOK(env, status);
  env->ReleaseStringUTFChars(filename, cname);
  TF_DeleteStatus(status);
  return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL Java_org_tensorflow_TensorFlow_libraryDelete(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc mht_2(mht_2_v, 225, "", "./tensorflow/java/src/main/native/tensorflow_jni.cc", "Java_org_tensorflow_TensorFlow_libraryDelete");

  if (handle != 0) {
    TF_DeleteLibraryHandle(reinterpret_cast<TF_Library*>(handle));
  }
}

JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_TensorFlow_libraryOpList(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensorflow_jniDTcc mht_3(mht_3_v, 235, "", "./tensorflow/java/src/main/native/tensorflow_jni.cc", "Java_org_tensorflow_TensorFlow_libraryOpList");

  TF_Buffer buf = TF_GetOpList(reinterpret_cast<TF_Library*>(handle));
  if (buf.length > std::numeric_limits<jint>::max()) {
    throwException(env, kIndexOutOfBoundsException,
                   "Serialized OpList is too large for a byte[] array");
    return nullptr;
  }
  auto ret_len = static_cast<jint>(buf.length);
  jbyteArray ret = env->NewByteArray(ret_len);
  env->SetByteArrayRegion(ret, 0, ret_len, static_cast<const jbyte*>(buf.data));
  return ret;
}
