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
class MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc() {
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

#include "tensorflow/java/src/main/native/graph_operation_jni.h"
#include <memory>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_0(mht_0_v, 192, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "requireHandleImpl");

  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(
        env, kNullPointerException,
        "close() has been called on the Graph this Operation was a part of");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

TF_Operation* requireHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_1(mht_1_v, 207, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "requireHandle");

  return requireHandleImpl<TF_Operation>(env, handle);
}

TF_Graph* requireGraphHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_2(mht_2_v, 214, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "requireGraphHandle");

  return requireHandleImpl<TF_Graph>(env, handle);
}
}  // namespace

JNIEXPORT jstring JNICALL Java_org_tensorflow_GraphOperation_name(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_3(mht_3_v, 223, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_name");

  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationName(op));
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_GraphOperation_type(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_4(mht_4_v, 233, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_type");

  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationOpType(op));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_numOutputs(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_5(mht_5_v, 243, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_numOutputs");

  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumOutputs(op);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_outputListLength(
    JNIEnv* env, jclass clazz, jlong handle, jstring name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_6(mht_6_v, 254, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_outputListLength");

  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;

  TF_Status* status = TF_NewStatus();

  const char* cname = env->GetStringUTFChars(name, nullptr);
  int result = TF_OperationOutputListLength(op, cname, status);
  env->ReleaseStringUTFChars(name, cname);

  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  return result;
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_GraphOperation_shape(
    JNIEnv* env, jclass clazz, jlong graph_handle, jlong op_handle,
    jint output_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_7(mht_7_v, 274, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_shape");

  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return nullptr;
  TF_Operation* op = requireHandle(env, op_handle);
  if (op == nullptr) return nullptr;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return nullptr;
  }

  TF_Output output{op, output_index};
  TF_Status* status = TF_NewStatus();
  jsize num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  if (!throwExceptionIfNotOK(env, status) || num_dims < 0) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  static_assert(sizeof(jlong) == sizeof(int64_t),
                "Java long is not compatible with the TensorFlow C API");
  // One might have trivially wanted to do:
  // TF_GraphGetTensorShape(graph, output, static_cast<int64_t*>(dims), ...)
  // but on some platforms this fails with:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long
  // *') is not allowed
  // For now, do the expensive but safe thing of copying.
  std::unique_ptr<int64_t[]> cdims(new int64_t[num_dims]);
  TF_GraphGetTensorShape(graph, output, cdims.get(), static_cast<int>(num_dims),
                         status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  jlongArray ret = env->NewLongArray(num_dims);
  jlong* dims = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = static_cast<jlong>(cdims[i]);
  }
  env->ReleaseLongArrayElements(ret, dims, 0);
  return ret;
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_dtype(
    JNIEnv* env, jclass clazz, jlong graph_handle, jlong op_handle,
    jint output_index) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_8(mht_8_v, 327, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_dtype");

  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = requireHandle(env, op_handle);
  if (op == nullptr) return 0;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return 0;
  }

  return static_cast<jint>(TF_OperationOutputType(TF_Output{op, output_index}));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_inputListLength(
    JNIEnv* env, jclass clazz, jlong handle, jstring name) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_jniDTcc mht_9(mht_9_v, 350, "", "./tensorflow/java/src/main/native/graph_operation_jni.cc", "Java_org_tensorflow_GraphOperation_inputListLength");

  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;

  TF_Status* status = TF_NewStatus();

  const char* cname = env->GetStringUTFChars(name, nullptr);
  int result = TF_OperationInputListLength(op, cname, status);
  env->ReleaseStringUTFChars(name, cname);

  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  return result;
}
