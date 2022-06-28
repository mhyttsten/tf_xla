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
class MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc() {
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

#include <string.h>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/utils_jni.h"
#include "tensorflow/java/src/main/native/exception_jni.h"
#include "tensorflow/java/src/main/native/session_jni.h"

namespace {
TF_Session* requireHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_0(mht_0_v, 194, "", "./tensorflow/java/src/main/native/session_jni.cc", "requireHandle");

  static_assert(sizeof(jlong) >= sizeof(TF_Session*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(env, kNullPointerException,
                   "close() has been called on the Session");
    return nullptr;
  }
  return reinterpret_cast<TF_Session*>(handle);
}

template <class T>
void resolveHandles(JNIEnv* env, const char* type, jlongArray src_array,
                    T** dst, jint n) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_1(mht_1_v, 211, "", "./tensorflow/java/src/main/native/session_jni.cc", "resolveHandles");

  if (env->ExceptionCheck()) return;
  jint len = env->GetArrayLength(src_array);
  if (len != n) {
    throwException(env, kIllegalArgumentException, "expected %d, got %d %s", n,
                   len, type);
    return;
  }
  jlong* src_start = env->GetLongArrayElements(src_array, nullptr);
  jlong* src = src_start;
  for (int i = 0; i < n; ++i, ++src, ++dst) {
    if (*src == 0) {
      throwException(env, kNullPointerException, "invalid %s (#%d of %d)", type,
                     i, n);
      break;
    }
    *dst = reinterpret_cast<T*>(*src);
  }
  env->ReleaseLongArrayElements(src_array, src_start, JNI_ABORT);
}

void TF_MaybeDeleteBuffer(TF_Buffer* buf) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_2(mht_2_v, 235, "", "./tensorflow/java/src/main/native/session_jni.cc", "TF_MaybeDeleteBuffer");

  if (buf == nullptr) return;
  TF_DeleteBuffer(buf);
}

typedef std::unique_ptr<TF_Buffer, decltype(&TF_MaybeDeleteBuffer)>
    unique_tf_buffer;

unique_tf_buffer MakeUniqueBuffer(TF_Buffer* buf) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_3(mht_3_v, 246, "", "./tensorflow/java/src/main/native/session_jni.cc", "MakeUniqueBuffer");

  return unique_tf_buffer(buf, TF_MaybeDeleteBuffer);
}

}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Session_allocate(
    JNIEnv* env, jclass clazz, jlong graph_handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_4(mht_4_v, 256, "", "./tensorflow/java/src/main/native/session_jni.cc", "Java_org_tensorflow_Session_allocate");

  return Java_org_tensorflow_Session_allocate2(env, clazz, graph_handle,
                                               nullptr, nullptr);
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_Session_allocate2(
    JNIEnv* env, jclass clazz, jlong graph_handle, jstring target,
    jbyteArray config) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_5(mht_5_v, 267, "", "./tensorflow/java/src/main/native/session_jni.cc", "Java_org_tensorflow_Session_allocate2");

  if (graph_handle == 0) {
    throwException(env, kNullPointerException, "Graph has been close()d");
    return 0;
  }
  TF_Graph* graph = reinterpret_cast<TF_Graph*>(graph_handle);
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* opts = TF_NewSessionOptions();
  jbyte* cconfig = nullptr;
  if (config != nullptr) {
    cconfig = env->GetByteArrayElements(config, nullptr);
    TF_SetConfig(opts, cconfig,
                 static_cast<size_t>(env->GetArrayLength(config)), status);
    if (!throwExceptionIfNotOK(env, status)) {
      env->ReleaseByteArrayElements(config, cconfig, JNI_ABORT);
      TF_DeleteSessionOptions(opts);
      TF_DeleteStatus(status);
      return 0;
    }
  }
  const char* ctarget = nullptr;
  if (target != nullptr) {
    ctarget = env->GetStringUTFChars(target, nullptr);
  }
  TF_Session* session = TF_NewSession(graph, opts, status);
  if (config != nullptr) {
    env->ReleaseByteArrayElements(config, cconfig, JNI_ABORT);
  }
  if (target != nullptr) {
    env->ReleaseStringUTFChars(target, ctarget);
  }
  TF_DeleteSessionOptions(opts);
  bool ok = throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);

  return ok ? reinterpret_cast<jlong>(session) : 0;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Session_delete(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_6(mht_6_v, 310, "", "./tensorflow/java/src/main/native/session_jni.cc", "Java_org_tensorflow_Session_delete");

  TF_Session* session = requireHandle(env, handle);
  if (session == nullptr) return;
  TF_Status* status = TF_NewStatus();
  TF_CloseSession(session, status);
  // Result of close is ignored, delete anyway.
  TF_DeleteSession(session, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_Session_run(
    JNIEnv* env, jclass clazz, jlong handle, jbyteArray jrun_options,
    jlongArray input_tensor_handles, jlongArray input_op_handles,
    jintArray input_op_indices, jlongArray output_op_handles,
    jintArray output_op_indices, jlongArray target_op_handles,
    jboolean want_run_metadata, jlongArray output_tensor_handles) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSsession_jniDTcc mht_7(mht_7_v, 329, "", "./tensorflow/java/src/main/native/session_jni.cc", "Java_org_tensorflow_Session_run");

  TF_Session* session = requireHandle(env, handle);
  if (session == nullptr) return nullptr;

  const jint ninputs = env->GetArrayLength(input_tensor_handles);
  const jint noutputs = env->GetArrayLength(output_tensor_handles);
  const jint ntargets = env->GetArrayLength(target_op_handles);

  std::unique_ptr<TF_Output[]> inputs(new TF_Output[ninputs]);
  std::unique_ptr<TF_Tensor* []> input_values(new TF_Tensor*[ninputs]);
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[noutputs]);
  std::unique_ptr<TF_Tensor* []> output_values(new TF_Tensor*[noutputs]);
  std::unique_ptr<TF_Operation* []> targets(new TF_Operation*[ntargets]);
  unique_tf_buffer run_metadata(
      MakeUniqueBuffer(want_run_metadata ? TF_NewBuffer() : nullptr));

  resolveHandles(env, "input Tensors", input_tensor_handles, input_values.get(),
                 ninputs);
  resolveOutputs(env, "input", input_op_handles, input_op_indices, inputs.get(),
                 ninputs);
  resolveOutputs(env, "output", output_op_handles, output_op_indices,
                 outputs.get(), noutputs);
  resolveHandles(env, "target Operations", target_op_handles, targets.get(),
                 ntargets);
  if (env->ExceptionCheck()) return nullptr;

  TF_Status* status = TF_NewStatus();

  unique_tf_buffer run_options(MakeUniqueBuffer(nullptr));
  jbyte* jrun_options_data = nullptr;
  if (jrun_options != nullptr) {
    size_t sz = env->GetArrayLength(jrun_options);
    if (sz > 0) {
      jrun_options_data = env->GetByteArrayElements(jrun_options, nullptr);
      run_options.reset(
          TF_NewBufferFromString(static_cast<void*>(jrun_options_data), sz));
    }
  }

  TF_SessionRun(session, run_options.get(), inputs.get(), input_values.get(),
                static_cast<int>(ninputs), outputs.get(), output_values.get(),
                static_cast<int>(noutputs), targets.get(),
                static_cast<int>(ntargets), run_metadata.get(), status);

  if (jrun_options_data != nullptr) {
    env->ReleaseByteArrayElements(jrun_options, jrun_options_data, JNI_ABORT);
  }

  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  jlong* t = env->GetLongArrayElements(output_tensor_handles, nullptr);
  for (int i = 0; i < noutputs; ++i) {
    t[i] = reinterpret_cast<jlong>(output_values[i]);
  }
  env->ReleaseLongArrayElements(output_tensor_handles, t, 0);

  jbyteArray ret = nullptr;
  if (run_metadata != nullptr) {
    ret = env->NewByteArray(run_metadata->length);
    env->SetByteArrayRegion(ret, 0, run_metadata->length,
                            reinterpret_cast<const jbyte*>(run_metadata->data));
  }
  TF_DeleteStatus(status);
  return ret;
}
