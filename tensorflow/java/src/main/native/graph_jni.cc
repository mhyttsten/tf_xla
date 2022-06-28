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
class MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc() {
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

#include "tensorflow/java/src/main/native/graph_jni.h"

#include <limits>
#include <memory>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"
#include "tensorflow/java/src/main/native/utils_jni.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_0(mht_0_v, 195, "", "./tensorflow/java/src/main/native/graph_jni.cc", "requireHandleImpl");

  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() has been called on the Graph");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

TF_Graph* requireHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_1(mht_1_v, 209, "", "./tensorflow/java/src/main/native/graph_jni.cc", "requireHandle");

  return requireHandleImpl<TF_Graph>(env, handle);
}

TF_Operation* requireOperationHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_2(mht_2_v, 216, "", "./tensorflow/java/src/main/native/graph_jni.cc", "requireOperationHandle");

  return requireHandleImpl<TF_Operation>(env, handle);
}
}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Graph_allocate(JNIEnv*, jclass) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_3(mht_3_v, 224, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_allocate");

  return reinterpret_cast<jlong>(TF_NewGraph());
}

JNIEXPORT void JNICALL Java_org_tensorflow_Graph_delete(JNIEnv*, jclass,
                                                        jlong handle) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_4(mht_4_v, 232, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_delete");

  if (handle == 0) return;
  TF_DeleteGraph(reinterpret_cast<TF_Graph*>(handle));
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_Graph_operation(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jstring name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_5(mht_5_v, 244, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_operation");

  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return 0;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Operation* op = TF_GraphOperationByName(g, cname);
  env->ReleaseStringUTFChars(name, cname);
  return reinterpret_cast<jlong>(op);
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Graph_nextOperation(
    JNIEnv* env, jclass clazz, jlong handle, jint position) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_6(mht_6_v, 257, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_nextOperation");

  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return nullptr;

  size_t pos = static_cast<size_t>(position);
  TF_Operation* operation = TF_GraphNextOperation(g, &pos);
  if (operation == nullptr) return nullptr;

  jlong handle_and_position[2];
  handle_and_position[0] = reinterpret_cast<jlong>(operation);
  handle_and_position[1] = static_cast<jlong>(pos);

  jlongArray rhett = env->NewLongArray(2);
  env->SetLongArrayRegion(rhett, 0, 2, handle_and_position);
  return rhett;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Graph_importGraphDef(
    JNIEnv* env, jclass clazz, jlong handle, jbyteArray graph_def,
    jstring prefix) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_7(mht_7_v, 280, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_importGraphDef");

  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return;

  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

  jboolean is_copy;
  const char* cprefix = env->GetStringUTFChars(prefix, &is_copy);
  TF_ImportGraphDefOptionsSetPrefix(opts, cprefix);
  env->ReleaseStringUTFChars(prefix, cprefix);

  static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
  jbyte* bytes = env->GetByteArrayElements(graph_def, &is_copy);
  TF_Buffer* buf =
      TF_NewBufferFromString(bytes, env->GetArrayLength(graph_def));
  TF_Status* status = TF_NewStatus();

  TF_GraphImportGraphDef(g, buf, opts, status);
  throwExceptionIfNotOK(env, status);
  // Continue cleaning up resources even if an exception was thrown.

  TF_DeleteStatus(status);
  TF_DeleteBuffer(buf);
  env->ReleaseByteArrayElements(graph_def, bytes, JNI_ABORT);

  TF_DeleteImportGraphDefOptions(opts);
}

JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_Graph_toGraphDef(JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_8(mht_8_v, 312, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_toGraphDef");

  jbyteArray ret = nullptr;
  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return ret;

  TF_Buffer* buf = TF_NewBuffer();
  TF_Status* status = TF_NewStatus();
  TF_GraphToGraphDef(g, buf, status);
  if (throwExceptionIfNotOK(env, status)) {
    // sizeof(jsize) is less than sizeof(size_t) on some platforms.
    if (buf->length > std::numeric_limits<jint>::max()) {
      throwException(env, kIndexOutOfBoundsException,
                     "GraphDef is too large to serialize into a byte[] array");
    } else {
      static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
      jint ret_len = static_cast<jint>(buf->length);
      ret = env->NewByteArray(ret_len);
      env->SetByteArrayRegion(ret, 0, ret_len,
                              static_cast<const jbyte*>(buf->data));
    }
  }
  TF_DeleteStatus(status);
  TF_DeleteBuffer(buf);
  return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Graph_addGradients(
    JNIEnv* env, jclass clazz, jlong handle, jstring prefix,
    jlongArray y_handles, jintArray y_indices, jlongArray x_handles,
    jintArray x_indices, jlongArray dx_handles, jintArray dx_indices) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_9(mht_9_v, 345, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_addGradients");

  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return nullptr;

  const jint ny = env->GetArrayLength(y_handles);
  const jint nx = env->GetArrayLength(x_handles);

  std::unique_ptr<TF_Output[]> y(new TF_Output[ny]);
  std::unique_ptr<TF_Output[]> x(new TF_Output[nx]);
  std::unique_ptr<TF_Output[]> dx(nullptr);
  std::unique_ptr<TF_Output[]> dy(new TF_Output[nx]);

  resolveOutputs(env, "y", y_handles, y_indices, y.get(), ny);
  resolveOutputs(env, "x", x_handles, x_indices, x.get(), nx);
  if (dx_handles != nullptr) {
    if (env->GetArrayLength(dx_handles) != ny) {
      throwException(env, kIllegalArgumentException,
                     "expected %d, got %d dx handles", ny,
                     env->GetArrayLength(dx_handles));
    }
    dx.reset(new TF_Output[ny]);
    resolveOutputs(env, "dx", dx_handles, dx_indices, dx.get(), ny);
  }
  if (env->ExceptionCheck()) return nullptr;

  const char* cprefix = nullptr;
  if (prefix != nullptr) {
    cprefix = env->GetStringUTFChars(prefix, nullptr);
  }
  TF_Status* status = TF_NewStatus();
  TF_AddGradientsWithPrefix(g, cprefix, y.get(), ny, x.get(), nx, dx.get(),
                            status, dy.get());
  if (prefix != nullptr) {
    env->ReleaseStringUTFChars(prefix, cprefix);
  }
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  // returned array contains both op handles and output indices, in pair
  jlongArray dy_handles_and_indices = env->NewLongArray(nx << 1);
  jlong* dy_elems = env->GetLongArrayElements(dy_handles_and_indices, nullptr);
  for (int i = 0, j = nx; i < nx; ++i, ++j) {
    TF_Output dy_output = dy.get()[i];
    dy_elems[i] = reinterpret_cast<jlong>(dy_output.oper);
    dy_elems[j] = static_cast<jlong>(dy_output.index);
  }
  env->ReleaseLongArrayElements(dy_handles_and_indices, dy_elems, 0);

  return dy_handles_and_indices;
}

// helper function for while loop -- constructs conditional or body subgraph
jlongArray buildSubgraph(JNIEnv* env, jclass clazz, jobject subgraph_builder,
                         TF_Graph* const subgraph,
                         const TF_Output* const inputs,
                         const TF_Output* const outputs, const int ninputs,
                         const int noutputs) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_10(mht_10_v, 407, "", "./tensorflow/java/src/main/native/graph_jni.cc", "buildSubgraph");

  jmethodID build_subgraph_method_id = env->GetStaticMethodID(
      clazz, "buildSubgraph",
      "(Lorg/tensorflow/Graph$WhileSubgraphBuilder;J[J[I[J[I)[J");
  if (build_subgraph_method_id == nullptr) return nullptr;

  jlong subgraph_handle = reinterpret_cast<jlong>(subgraph);

  jlongArray input_handles = env->NewLongArray(ninputs);
  jintArray input_indices = env->NewIntArray(ninputs);
  jlongArray output_handles = env->NewLongArray(noutputs);
  jintArray output_indices = env->NewIntArray(noutputs);

  jlong* input_handles_elems =
      env->GetLongArrayElements(input_handles, nullptr);
  jint* input_indices_elems = env->GetIntArrayElements(input_indices, nullptr);
  jlong* output_handles_elems =
      env->GetLongArrayElements(output_handles, nullptr);
  jint* output_indices_elems =
      env->GetIntArrayElements(output_indices, nullptr);

  for (int i = 0; i < ninputs; ++i) {
    input_handles_elems[i] = reinterpret_cast<jlong>((inputs[i]).oper);
    input_indices_elems[i] = static_cast<jint>((inputs[i]).index);
  }

  for (int i = 0; i < noutputs; ++i) {
    output_handles_elems[i] = reinterpret_cast<jlong>((outputs[i]).oper);
    output_indices_elems[i] = static_cast<jint>((outputs[i]).index);
  }

  env->ReleaseLongArrayElements(input_handles, input_handles_elems, 0);
  env->ReleaseIntArrayElements(input_indices, input_indices_elems, 0);
  env->ReleaseLongArrayElements(output_handles, output_handles_elems, 0);
  env->ReleaseIntArrayElements(output_indices, output_indices_elems, 0);

  // call Java code to construct the subgraph
  jlongArray output_handles_and_indices =
      (jlongArray)env->CallStaticObjectMethod(
          clazz, build_subgraph_method_id, subgraph_builder, subgraph_handle,
          input_handles, input_indices, output_handles, output_indices);

  if (env->ExceptionOccurred()) {
    env->ExceptionDescribe();
    return nullptr;
  }

  // returned array contains both op handles and output indices, in pair
  return output_handles_and_indices;
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Graph_whileLoop(
    JNIEnv* env, jclass clazz, jlong handle, jlongArray input_handles,
    jintArray input_indices, jstring name, jobject cond_graph_builder,
    jobject body_graph_builder) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePSgraph_jniDTcc mht_11(mht_11_v, 465, "", "./tensorflow/java/src/main/native/graph_jni.cc", "Java_org_tensorflow_Graph_whileLoop");

  TF_Graph* g = requireHandle(env, handle);
  TF_Status* status = TF_NewStatus();
  if (g == nullptr) return nullptr;

  int ninputs = env->GetArrayLength(input_handles);

  std::unique_ptr<TF_Output[]> inputs(new TF_Output[ninputs]);
  resolveOutputs(env, "inputs", input_handles, input_indices, inputs.get(),
                 ninputs);
  if (env->ExceptionCheck()) return nullptr;

  // initialize while params
  TF_WhileParams params = TF_NewWhile(g, inputs.get(), ninputs, status);
  throwExceptionIfNotOK(env, status);

  // build conditional subgraph
  jlongArray cond_output_handles_and_indices =
      buildSubgraph(env, clazz, cond_graph_builder, params.cond_graph,
                    params.cond_inputs, &params.cond_output, params.ninputs, 1);

  // build body subgraph
  jlongArray body_output_handles_and_indices = buildSubgraph(
      env, clazz, body_graph_builder, params.body_graph, params.body_inputs,
      params.body_outputs, params.ninputs, params.ninputs);

  if (cond_output_handles_and_indices == nullptr ||
      body_output_handles_and_indices == nullptr)
    return nullptr;

  // set cond_output param to output of the conditional subgraph
  jlong* cond_output_elems =
      env->GetLongArrayElements(cond_output_handles_and_indices, nullptr);
  TF_Operation* cond_output_op =
      requireOperationHandle(env, cond_output_elems[0]);
  params.cond_output = {cond_output_op,
                        static_cast<jint>(cond_output_elems[1])};
  env->ReleaseLongArrayElements(cond_output_handles_and_indices,
                                cond_output_elems, 0);

  // set body_outputs param to outputs of the body subgraph
  jlong* body_output_elems =
      env->GetLongArrayElements(body_output_handles_and_indices, nullptr);
  for (int i = 0, j = ninputs; i < ninputs; ++i, ++j) {
    TF_Operation* body_output_op =
        requireOperationHandle(env, body_output_elems[i]);
    params.body_outputs[i] = {body_output_op,
                              static_cast<jint>(body_output_elems[j])};
  }
  env->ReleaseLongArrayElements(body_output_handles_and_indices,
                                body_output_elems, 0);

  // set loop name param
  params.name = env->GetStringUTFChars(name, nullptr);

  // build the while loop, storing loop outputs in `outputs`
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[ninputs]);
  TF_FinishWhile(&params, status, outputs.get());

  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);

  env->ReleaseStringUTFChars(name, params.name);

  // returned array contains both op handles and output indices, in pair
  jlongArray output_handles_and_indices = env->NewLongArray(ninputs * 2);
  jlong* output_elems =
      env->GetLongArrayElements(output_handles_and_indices, nullptr);
  for (int i = 0, j = ninputs; i < ninputs; ++i, ++j) {
    TF_Output output = outputs.get()[i];
    output_elems[i] = reinterpret_cast<jlong>(output.oper);
    output_elems[j] = static_cast<jlong>(output.index);
  }
  env->ReleaseLongArrayElements(output_handles_and_indices, output_elems, 0);

  return output_handles_and_indices;
}
