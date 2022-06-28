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
class MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc() {
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

#include "tensorflow/java/src/main/native/graph_operation_builder_jni.h"
#include <cstring>
#include <memory>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {
TF_OperationDescription* requireHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_0(mht_0_v, 192, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "requireHandle");

  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "Operation has already been built");
    return nullptr;
  }
  return reinterpret_cast<TF_OperationDescription*>(handle);
}

bool resolveOutput(JNIEnv* env, jlong op_handle, jint index, TF_Output* out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_1(mht_1_v, 204, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "resolveOutput");

  if (op_handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() was called on the Graph");
    return false;
  }
  out->oper = reinterpret_cast<TF_Operation*>(op_handle);
  out->index = static_cast<int>(index);
  return true;
}

TF_Tensor* requireTensor(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_2(mht_2_v, 218, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "requireTensor");

  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() has been called on the Tensor");
    return nullptr;
  }
  return reinterpret_cast<TF_Tensor*>(handle);
}
}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_GraphOperationBuilder_allocate(
    JNIEnv* env, jclass clazz, jlong graph_handle, jstring type, jstring name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("type: \"" + type + "\"");
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_3(mht_3_v, 234, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_allocate");

  if (graph_handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() has been called on the Graph");
    return 0;
  }
  TF_Graph* graph = reinterpret_cast<TF_Graph*>(graph_handle);
  const char* op_type = env->GetStringUTFChars(type, nullptr);
  const char* op_name = env->GetStringUTFChars(name, nullptr);
  TF_OperationDescription* d = TF_NewOperation(graph, op_type, op_name);
  env->ReleaseStringUTFChars(name, op_name);
  env->ReleaseStringUTFChars(type, op_type);
  static_assert(sizeof(jlong) >= sizeof(TF_OperationDescription*),
                "Cannot represent a C TF_OperationDescription as a Java long");
  return reinterpret_cast<jlong>(d);
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_GraphOperationBuilder_finish(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_4(mht_4_v, 255, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_finish");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  TF_Operation* op = TF_FinishOperation(d, status);
  if (throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return reinterpret_cast<jlong>(op);
  }
  TF_DeleteStatus(status);
  return 0;
}

JNIEXPORT void JNICALL Java_org_tensorflow_GraphOperationBuilder_addInput(
    JNIEnv* env, jclass clazz, jlong handle, jlong op_handle, jint index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_5(mht_5_v, 272, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_addInput");

  TF_Output out;
  if (!resolveOutput(env, op_handle, index, &out)) return;
  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  TF_AddInput(d, out);
}

JNIEXPORT void JNICALL Java_org_tensorflow_GraphOperationBuilder_addInputList(
    JNIEnv* env, jclass clazz, jlong handle, jlongArray op_handles,
    jintArray indices) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_6(mht_6_v, 285, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_addInputList");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  const size_t n = static_cast<size_t>(env->GetArrayLength(op_handles));
  if (env->GetArrayLength(indices) != n) {
    throwException(env, kIllegalArgumentException,
                   "mismatch in number of Operations (%d) and output indices "
                   "(%d) provided",
                   n, env->GetArrayLength(indices));
    return;
  }
  std::unique_ptr<TF_Output[]> o(new TF_Output[n]);
  jlong* oph = env->GetLongArrayElements(op_handles, nullptr);
  jint* idx = env->GetIntArrayElements(indices, nullptr);
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    ok = resolveOutput(env, oph[i], idx[i], &o[i]);
  }
  env->ReleaseIntArrayElements(indices, idx, JNI_ABORT);
  env->ReleaseLongArrayElements(op_handles, oph, JNI_ABORT);
  if (!ok) return;
  TF_AddInputList(d, o.get(), n);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_GraphOperationBuilder_addControlInput(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle,
                                                          jlong op_handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_7(mht_7_v, 316, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_addControlInput");

  if (op_handle == 0) {
    throwException(env, kIllegalStateException,
                   "control input is not valid, "
                   "perhaps the Graph containing it has been closed()?");
    return;
  }
  TF_Operation* control = reinterpret_cast<TF_Operation*>(op_handle);
  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  TF_AddControlInput(d, control);
}

JNIEXPORT void JNICALL Java_org_tensorflow_GraphOperationBuilder_setDevice(
    JNIEnv* env, jclass clazz, jlong handle, jstring device) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_8(mht_8_v, 334, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_setDevice");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  const char* cdevice = env->GetStringUTFChars(device, nullptr);
  TF_SetDevice(d, cdevice);
  env->ReleaseStringUTFChars(device, cdevice);
}

JNIEXPORT void JNICALL Java_org_tensorflow_GraphOperationBuilder_setAttrString(
    JNIEnv* env, jclass clazz, jlong handle, jstring name, jbyteArray value) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_9(mht_9_v, 347, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_setAttrString");

  static_assert(sizeof(jbyte) == 1,
                "Require Java byte to be represented as a single byte");
  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  jbyte* cvalue = env->GetByteArrayElements(value, nullptr);
  TF_SetAttrString(d, cname, cvalue, env->GetArrayLength(value));
  env->ReleaseByteArrayElements(value, cvalue, JNI_ABORT);
  env->ReleaseStringUTFChars(name, cname);
}

#define DEFINE_SET_ATTR_SCALAR(name, jtype, ctype)                    \
  JNIEXPORT void JNICALL                                              \
      Java_org_tensorflow_GraphOperationBuilder_setAttr##name(        \
          JNIEnv* env, jclass clazz, jlong handle, jstring name,      \
          jtype value) {                                              \
    static_assert(                                                    \
        sizeof(ctype) >= sizeof(jtype),                               \
        "Information loss when converting between Java and C types"); \
    TF_OperationDescription* d = requireHandle(env, handle);          \
    if (d == nullptr) return;                                         \
    const char* cname = env->GetStringUTFChars(name, nullptr);        \
    TF_SetAttr##name(d, cname, static_cast<ctype>(value));            \
    env->ReleaseStringUTFChars(name, cname);                          \
  }

#define DEFINE_SET_ATTR_LIST(name, jname, jtype, ctype)              \
  JNIEXPORT void JNICALL                                             \
      Java_org_tensorflow_GraphOperationBuilder_setAttr##name##List( \
          JNIEnv* env, jclass clazz, jlong handle, jstring name,     \
          jtype##Array value) {                                      \
    TF_OperationDescription* d = requireHandle(env, handle);         \
    if (d == nullptr) return;                                        \
    const char* cname = env->GetStringUTFChars(name, nullptr);       \
    /* Make a copy of the array to paper over any differences */     \
    /* in byte representations of the jtype and ctype         */     \
    /* For example, jint vs TF_DataType.                      */     \
    /* If this copy turns out to be a problem in practice     */     \
    /* can avoid it for many types.                           */     \
    const int n = env->GetArrayLength(value);                        \
    std::unique_ptr<ctype[]> cvalue(new ctype[n]);                   \
    jtype* elems = env->Get##jname##ArrayElements(value, nullptr);   \
    for (int i = 0; i < n; ++i) {                                    \
      cvalue[i] = static_cast<ctype>(elems[i]);                      \
    }                                                                \
    TF_SetAttr##name##List(d, cname, cvalue.get(), n);               \
    env->Release##jname##ArrayElements(value, elems, JNI_ABORT);     \
    env->ReleaseStringUTFChars(name, cname);                         \
  }

#define DEFINE_SET_ATTR(name, jname, jtype, ctype) \
  DEFINE_SET_ATTR_SCALAR(name, jtype, ctype)       \
  DEFINE_SET_ATTR_LIST(name, jname, jtype, ctype)

DEFINE_SET_ATTR(Int, Long, jlong, int64_t);
DEFINE_SET_ATTR(Float, Float, jfloat, float);
DEFINE_SET_ATTR(Bool, Boolean, jboolean, unsigned char);
DEFINE_SET_ATTR(Type, Int, jint, TF_DataType);
#undef DEFINE_SET_ATTR
#undef DEFINE_SET_ATTR_LIST
#undef DEFINE_SET_ATTR_SCALAR

JNIEXPORT void JNICALL Java_org_tensorflow_GraphOperationBuilder_setAttrTensor(
    JNIEnv* env, jclass clazz, jlong handle, jstring name,
    jlong tensor_handle) {
  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  TF_Tensor* t = requireTensor(env, tensor_handle);
  if (t == nullptr) return;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TF_SetAttrTensor(d, cname, t, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_GraphOperationBuilder_setAttrTensorList(
    JNIEnv* env, jclass clazz, jlong handle, jstring name,
    jlongArray tensor_handles) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_10(mht_10_v, 432, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_setAttrTensorList");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  const int n = env->GetArrayLength(tensor_handles);
  std::unique_ptr<TF_Tensor*[]> tensors(new TF_Tensor*[n]);
  jlong* jhandles = env->GetLongArrayElements(tensor_handles, nullptr);
  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    tensors[i] = requireTensor(env, jhandles[i]);
    ok = !env->ExceptionCheck();
  }
  env->ReleaseLongArrayElements(tensor_handles, jhandles, JNI_ABORT);
  if (!ok) return;

  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TF_SetAttrTensorList(d, cname, tensors.get(), n, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL Java_org_tensorflow_GraphOperationBuilder_setAttrShape(
    JNIEnv* env, jclass clazz, jlong handle, jstring name, jlongArray shape,
    jint num_dims) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_11(mht_11_v, 460, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_setAttrShape");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  std::unique_ptr<int64_t[]> cvalue;
  // num_dims and env->GetArrayLength(shape) are assumed to be consistent.
  // i.e., either num_dims < 0 or num_dims == env->GetArrayLength(shape).
  if (num_dims > 0) {
    cvalue.reset(new int64_t[num_dims]);
    jlong* elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      cvalue[i] = static_cast<int64_t>(elems[i]);
    }
    env->ReleaseLongArrayElements(shape, elems, JNI_ABORT);
  }
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_SetAttrShape(d, cname, cvalue.get(), static_cast<int>(num_dims));
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_GraphOperationBuilder_setAttrShapeList(
    JNIEnv* env, jclass clazz, jlong handle, jstring name, jlongArray shapes,
    jintArray num_dims) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_12(mht_12_v, 486, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_setAttrShapeList");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  std::unique_ptr<int64_t[]> cshapes;
  std::unique_ptr<int64_t*[]> cdims;
  std::unique_ptr<int[]> cnum_dims;
  const int num_dims_length = env->GetArrayLength(num_dims);
  if (num_dims_length > 0) {
    const int shapes_length = env->GetArrayLength(shapes);
    cshapes.reset(new int64_t[shapes_length]);
    cdims.reset(new int64_t*[num_dims_length]);
    cnum_dims.reset(new int[num_dims_length]);
    jlong* shapes_elems =
        static_cast<jlong*>(env->GetPrimitiveArrayCritical(shapes, nullptr));
    std::memcpy(cshapes.get(), shapes_elems, shapes_length << 3);
    env->ReleasePrimitiveArrayCritical(shapes, shapes_elems, JNI_ABORT);
    int64_t* cshapes_ptr = cshapes.get();
    jint* num_dims_elems =
        static_cast<jint*>(env->GetPrimitiveArrayCritical(num_dims, nullptr));
    for (int i = 0; i < num_dims_length; ++i) {
      cnum_dims[i] = static_cast<int>(num_dims_elems[i]);
      cdims[i] = cshapes_ptr;
      if (cnum_dims[i] > 0) {
        cshapes_ptr += cnum_dims[i];
      }
    }
    env->ReleasePrimitiveArrayCritical(num_dims, num_dims_elems, JNI_ABORT);
  }
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_SetAttrShapeList(d, cname, cdims.get(), cnum_dims.get(), num_dims_length);
  env->ReleaseStringUTFChars(name, cname);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_GraphOperationBuilder_setAttrStringList(
    JNIEnv* env, jclass object, jlong handle, jstring name,
    jobjectArray values) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_operation_builder_jniDTcc mht_13(mht_13_v, 526, "", "./tensorflow/java/src/main/native/graph_operation_builder_jni.cc", "Java_org_tensorflow_GraphOperationBuilder_setAttrStringList");

  TF_OperationDescription* d = requireHandle(env, handle);
  if (d == nullptr) return;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  int num_values = env->GetArrayLength(values);
  static_assert(sizeof(jbyte) == 1,
                "Require Java byte to be represented as a single byte");
  std::unique_ptr<jbyteArray[]> jarrays(new jbyteArray[num_values]);
  std::unique_ptr<jbyte*[]> jvalues(new jbyte*[num_values]);
  std::unique_ptr<void*[]> cvalues(new void*[num_values]);
  std::unique_ptr<size_t[]> lengths(new size_t[num_values]);

  for (int i = 0; i < num_values; ++i) {
    jbyteArray v =
        static_cast<jbyteArray>(env->GetObjectArrayElement(values, i));
    jarrays[i] = v;
    jvalues[i] = env->GetByteArrayElements(v, nullptr);
    cvalues[i] = jvalues[i];
    lengths[i] = static_cast<size_t>(env->GetArrayLength(v));
  }
  TF_SetAttrStringList(d, cname, cvalues.get(), lengths.get(), num_values);
  for (int i = 0; i < num_values; ++i) {
    env->ReleaseByteArrayElements(jarrays[i], jvalues[i], JNI_ABORT);
  }
  env->ReleaseStringUTFChars(name, cname);
}
