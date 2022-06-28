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
class MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc {
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
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc() {
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

#include "tensorflow/java/src/main/native/tensor_jni.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {

TF_Tensor* requireHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_0(mht_0_v, 198, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "requireHandle");

  if (handle == 0) {
    throwException(env, kNullPointerException,
                   "close() was called on the Tensor");
    return nullptr;
  }
  return reinterpret_cast<TF_Tensor*>(handle);
}

size_t elemByteSize(TF_DataType dtype) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_1(mht_1_v, 210, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "elemByteSize");

  // The code in this file makes the assumption that the
  // TensorFlow TF_DataTypes and the Java primitive types
  // have the same byte sizes. Validate that:
  switch (dtype) {
    case TF_BOOL:
    case TF_UINT8:
      static_assert(sizeof(jboolean) == 1,
                    "Java boolean not compatible with TF_BOOL");
      static_assert(sizeof(jbyte) == 1,
                    "Java byte not compatible with TF_UINT8");
      return 1;
    case TF_FLOAT:
    case TF_INT32:
      static_assert(sizeof(jfloat) == 4,
                    "Java float not compatible with TF_FLOAT");
      static_assert(sizeof(jint) == 4, "Java int not compatible with TF_INT32");
      return 4;
    case TF_DOUBLE:
    case TF_INT64:
      static_assert(sizeof(jdouble) == 8,
                    "Java double not compatible with TF_DOUBLE");
      static_assert(sizeof(jlong) == 8,
                    "Java long not compatible with TF_INT64");
      return 8;
    default:
      return 0;
  }
}

// Write a Java scalar object (java.lang.Integer etc.) to a TF_Tensor.
void writeScalar(JNIEnv* env, jobject src, TF_DataType dtype, void* dst,
                 size_t dst_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_2(mht_2_v, 245, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "writeScalar");

  size_t sz = elemByteSize(dtype);
  if (sz != dst_size) {
    throwException(
        env, kIllegalStateException,
        "scalar (%d bytes) not compatible with allocated tensor (%d bytes)", sz,
        dst_size);
    return;
  }
  switch (dtype) {
// env->FindClass and env->GetMethodID are expensive and JNI best practices
// suggest that they should be cached. However, until the creation of scalar
// valued tensors seems to become a noticeable fraction of program execution,
// ignore that cost.
#define CASE(dtype, jtype, method_name, method_signature, call_type)           \
  case dtype: {                                                                \
    jclass clazz = env->FindClass("java/lang/Number");                         \
    jmethodID method = env->GetMethodID(clazz, method_name, method_signature); \
    jtype v = env->Call##call_type##Method(src, method);                       \
    memcpy(dst, &v, sz);                                                       \
    return;                                                                    \
  }
    CASE(TF_FLOAT, jfloat, "floatValue", "()F", Float);
    CASE(TF_DOUBLE, jdouble, "doubleValue", "()D", Double);
    CASE(TF_INT32, jint, "intValue", "()I", Int);
    CASE(TF_INT64, jlong, "longValue", "()J", Long);
    CASE(TF_UINT8, jbyte, "byteValue", "()B", Byte);
#undef CASE
    case TF_BOOL: {
      jclass clazz = env->FindClass("java/lang/Boolean");
      jmethodID method = env->GetMethodID(clazz, "booleanValue", "()Z");
      jboolean v = env->CallBooleanMethod(src, method);
      *(static_cast<unsigned char*>(dst)) = v ? 1 : 0;
      return;
    }
    default:
      throwException(env, kIllegalStateException, "invalid DataType(%d)",
                     dtype);
      return;
  }
}

// Copy a 1-D array of Java primitive types to the tensor buffer dst.
// Returns the number of bytes written to dst.
size_t write1DArray(JNIEnv* env, jarray array, TF_DataType dtype, void* dst,
                    size_t dst_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_3(mht_3_v, 293, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "write1DArray");

  const int nelems = env->GetArrayLength(array);
  jboolean is_copy;
  switch (dtype) {
#define CASE(dtype, jtype, get_type)                                   \
  case dtype: {                                                        \
    jtype##Array a = static_cast<jtype##Array>(array);                 \
    jtype* values = env->Get##get_type##ArrayElements(a, &is_copy);    \
    size_t to_copy = nelems * elemByteSize(dtype);                     \
    if (to_copy > dst_size) {                                          \
      throwException(                                                  \
          env, kIllegalStateException,                                 \
          "cannot write Java array of %d bytes to Tensor of %d bytes", \
          to_copy, dst_size);                                          \
      to_copy = 0;                                                     \
    } else {                                                           \
      memcpy(dst, values, to_copy);                                    \
    }                                                                  \
    env->Release##get_type##ArrayElements(a, values, JNI_ABORT);       \
    return to_copy;                                                    \
  }
    CASE(TF_FLOAT, jfloat, Float);
    CASE(TF_DOUBLE, jdouble, Double);
    CASE(TF_INT32, jint, Int);
    CASE(TF_INT64, jlong, Long);
    CASE(TF_BOOL, jboolean, Boolean);
    CASE(TF_UINT8, jbyte, Byte);
#undef CASE
    default:
      throwException(env, kIllegalStateException, "invalid DataType(%d)",
                     dtype);
      return 0;
  }
}

// Copy the elements of a 1-D array from the tensor buffer src to a 1-D array of
// Java primitive types. Returns the number of bytes read from src.
size_t read1DArray(JNIEnv* env, TF_DataType dtype, const void* src,
                   size_t src_size, jarray dst) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_4(mht_4_v, 334, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "read1DArray");

  const int len = env->GetArrayLength(dst);
  const size_t sz = len * elemByteSize(dtype);
  if (sz > src_size) {
    throwException(
        env, kIllegalStateException,
        "cannot fill a Java array of %d bytes with a Tensor of %d bytes", sz,
        src_size);
    return 0;
  }
  switch (dtype) {
#define CASE(dtype, jtype, primitive_type)                                 \
  case dtype: {                                                            \
    jtype##Array arr = static_cast<jtype##Array>(dst);                     \
    env->Set##primitive_type##ArrayRegion(arr, 0, len,                     \
                                          static_cast<const jtype*>(src)); \
    return sz;                                                             \
  }
    CASE(TF_FLOAT, jfloat, Float);
    CASE(TF_DOUBLE, jdouble, Double);
    CASE(TF_INT32, jint, Int);
    CASE(TF_INT64, jlong, Long);
    CASE(TF_BOOL, jboolean, Boolean);
    CASE(TF_UINT8, jbyte, Byte);
#undef CASE
    default:
      throwException(env, kIllegalStateException, "invalid DataType(%d)",
                     dtype);
  }
  return 0;
}

size_t writeNDArray(JNIEnv* env, jarray src, TF_DataType dtype, int dims_left,
                    char* dst, size_t dst_size) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("dst: \"" + (dst == nullptr ? std::string("nullptr") : std::string((char*)dst)) + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_5(mht_5_v, 371, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "writeNDArray");

  if (dims_left == 1) {
    return write1DArray(env, src, dtype, dst, dst_size);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(src);
    int len = env->GetArrayLength(ndarray);
    size_t sz = 0;
    for (int i = 0; i < len; ++i) {
      jarray row = static_cast<jarray>(env->GetObjectArrayElement(ndarray, i));
      sz +=
          writeNDArray(env, row, dtype, dims_left - 1, dst + sz, dst_size - sz);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return sz;
    }
    return sz;
  }
}

size_t readNDArray(JNIEnv* env, TF_DataType dtype, const char* src,
                   size_t src_size, int dims_left, jarray dst) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_6(mht_6_v, 394, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "readNDArray");

  if (dims_left == 1) {
    return read1DArray(env, dtype, src, src_size, dst);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(dst);
    int len = env->GetArrayLength(ndarray);
    size_t sz = 0;
    for (int i = 0; i < len; ++i) {
      jarray row = static_cast<jarray>(env->GetObjectArrayElement(ndarray, i));
      sz +=
          readNDArray(env, dtype, src + sz, src_size - sz, dims_left - 1, row);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return sz;
    }
    return sz;
  }
}

jbyteArray TF_StringDecodeTojbyteArray(JNIEnv* env, const TF_TString* src) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_7(mht_7_v, 415, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "TF_StringDecodeTojbyteArray");

  const char* dst = TF_TString_GetDataPointer(src);
  size_t dst_len = TF_TString_GetSize(src);

  jbyteArray ret = env->NewByteArray(dst_len);
  jbyte* cpy = env->GetByteArrayElements(ret, nullptr);

  memcpy(cpy, dst, dst_len);
  env->ReleaseByteArrayElements(ret, cpy, 0);
  return ret;
}

class StringTensorWriter {
 public:
  StringTensorWriter(TF_Tensor* t, int num_elements)
      : index_(0), data_(static_cast<TF_TString*>(TF_TensorData(t))) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_8(mht_8_v, 433, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "StringTensorWriter");
}

  void Add(const char* src, size_t len, TF_Status* status) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_9(mht_9_v, 439, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Add");

    if (TF_GetCode(status) != TF_OK) return;
    TF_TString_Init(&data_[index_]);
    TF_TString_Copy(&data_[index_++], src, len);
  }

 private:
  int index_;
  TF_TString* data_;
};

class StringTensorReader {
 public:
  StringTensorReader(const TF_Tensor* t, int num_elements)
      : index_(0), data_(static_cast<const TF_TString*>(TF_TensorData(t))) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_10(mht_10_v, 456, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "StringTensorReader");
}

  jbyteArray Next(JNIEnv* env, TF_Status* status) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_11(mht_11_v, 461, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Next");

    if (TF_GetCode(status) != TF_OK) return nullptr;
    return TF_StringDecodeTojbyteArray(env, &data_[index_++]);
  }

 private:
  int index_;
  const TF_TString* data_;
};

void readNDStringArray(JNIEnv* env, StringTensorReader* reader, int dims_left,
                       jobjectArray dst, TF_Status* status) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_12(mht_12_v, 475, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "readNDStringArray");

  jsize len = env->GetArrayLength(dst);
  if (dims_left == 1) {
    for (jsize i = 0; i < len; ++i) {
      jbyteArray elem = reader->Next(env, status);
      if (TF_GetCode(status) != TF_OK) return;
      env->SetObjectArrayElement(dst, i, elem);
      env->DeleteLocalRef(elem);
    }
    return;
  }
  for (jsize i = 0; i < len; ++i) {
    jobjectArray arr =
        static_cast<jobjectArray>(env->GetObjectArrayElement(dst, i));
    readNDStringArray(env, reader, dims_left - 1, arr, status);
    env->DeleteLocalRef(arr);
    if (TF_GetCode(status) != TF_OK) return;
  }
}
}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Tensor_allocate(JNIEnv* env,
                                                            jclass clazz,
                                                            jint dtype,
                                                            jlongArray shape,
                                                            jlong sizeInBytes) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_13(mht_13_v, 503, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_allocate");

  int num_dims = static_cast<int>(env->GetArrayLength(shape));
  jlong* dims = nullptr;
  if (num_dims > 0) {
    jboolean is_copy;
    dims = env->GetLongArrayElements(shape, &is_copy);
  }
  static_assert(sizeof(jlong) == sizeof(int64_t),
                "Java long is not compatible with the TensorFlow C API");
  // On some platforms "jlong" is a "long" while "int64_t" is a "long long".
  //
  // Thus, static_cast<int64_t*>(dims) will trigger a compiler error:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long
  // *') is not allowed
  //
  // Since this array is typically very small, use the guaranteed safe scheme of
  // creating a copy.
  int64_t* dims_copy = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims_copy[i] = static_cast<int64_t>(dims[i]);
  }
  TF_Tensor* t = TF_AllocateTensor(static_cast<TF_DataType>(dtype), dims_copy,
                                   num_dims, static_cast<size_t>(sizeInBytes));
  delete[] dims_copy;
  if (dims != nullptr) {
    env->ReleaseLongArrayElements(shape, dims, JNI_ABORT);
  }
  if (t == nullptr) {
    throwException(env, kNullPointerException,
                   "unable to allocate memory for the Tensor");
    return 0;
  }
  return reinterpret_cast<jlong>(t);
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_Tensor_allocateScalarBytes(
    JNIEnv* env, jclass clazz, jbyteArray value) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_14(mht_14_v, 542, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_allocateScalarBytes");

  // TF_STRING tensors are encoded with a table of 8-byte offsets followed by
  // TF_StringEncode-encoded bytes.
  size_t src_len = static_cast<int>(env->GetArrayLength(value));
  TF_Tensor* t = TF_AllocateTensor(TF_STRING, nullptr, 0, sizeof(TF_TString));
  TF_TString* dst = static_cast<TF_TString*>(TF_TensorData(t));

  TF_Status* status = TF_NewStatus();
  jbyte* jsrc = env->GetByteArrayElements(value, nullptr);
  // jsrc is an unsigned byte*, TF_StringEncode requires a char*.
  // reinterpret_cast<> for this conversion should be safe.
  TF_TString_Init(&dst[0]);
  TF_TString_Copy(&dst[0], reinterpret_cast<const char*>(jsrc), src_len);

  env->ReleaseByteArrayElements(value, jsrc, JNI_ABORT);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  return reinterpret_cast<jlong>(t);
}

namespace {
void checkForNullEntries(JNIEnv* env, jarray value, int num_dims) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_15(mht_15_v, 569, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "checkForNullEntries");

  jsize len = env->GetArrayLength(value);
  for (jsize i = 0; i < len; ++i) {
    jarray elem = static_cast<jarray>(
        env->GetObjectArrayElement(static_cast<jobjectArray>(value), i));
    if (elem == nullptr) {
      throwException(env, kNullPointerException,
                     "null entries in provided array");
      return;
    }
    env->DeleteLocalRef(elem);
    if (env->ExceptionCheck()) return;
  }
}

void fillNonScalarTF_STRINGTensorData(JNIEnv* env, jarray value, int num_dims,
                                      StringTensorWriter* writer,
                                      TF_Status* status) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_16(mht_16_v, 589, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "fillNonScalarTF_STRINGTensorData");

  if (num_dims == 0) {
    jbyte* jsrc =
        env->GetByteArrayElements(static_cast<jbyteArray>(value), nullptr);
    writer->Add(reinterpret_cast<const char*>(jsrc), env->GetArrayLength(value),
                status);
    env->ReleaseByteArrayElements(static_cast<jbyteArray>(value), jsrc,
                                  JNI_ABORT);
    return;
  }
  jsize len = env->GetArrayLength(value);
  for (jsize i = 0; i < len; ++i) {
    jarray elem = static_cast<jarray>(
        env->GetObjectArrayElement(static_cast<jobjectArray>(value), i));
    fillNonScalarTF_STRINGTensorData(env, elem, num_dims - 1, writer, status);
    env->DeleteLocalRef(elem);
    if (TF_GetCode(status) != TF_OK) return;
  }
}
}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Tensor_allocateNonScalarBytes(
    JNIEnv* env, jclass clazz, jlongArray shape, jobjectArray value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_17(mht_17_v, 614, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_allocateNonScalarBytes");

  // TF_STRING tensors are encoded with a table of 8-byte offsets following by
  // TF_StringEncode-encoded bytes.
  const int num_dims = static_cast<int>(env->GetArrayLength(shape));
  int64_t* dims = new int64_t[num_dims];
  int64_t num_elements = 1;
  {
    jlong* jdims = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = static_cast<int64_t>(jdims[i]);
      num_elements *= dims[i];
    }
    env->ReleaseLongArrayElements(shape, jdims, JNI_ABORT);
  }
  checkForNullEntries(env, value, num_dims);
  if (env->ExceptionCheck()) return 0;
  TF_Tensor* t = TF_AllocateTensor(TF_STRING, dims, num_dims,
                                   sizeof(TF_TString) * num_elements);
  if (t == nullptr) {
    delete[] dims;
    throwException(env, kNullPointerException,
                   "unable to allocate memory for the Tensor");
    return 0;
  }
  TF_Status* status = TF_NewStatus();
  StringTensorWriter writer(t, num_elements);
  fillNonScalarTF_STRINGTensorData(env, value, num_dims, &writer, status);
  delete[] dims;
  jlong ret = 0;
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteTensor(t);
  } else {
    ret = reinterpret_cast<jlong>(t);
  }
  TF_DeleteStatus(status);
  return ret;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Tensor_delete(JNIEnv* env,
                                                         jclass clazz,
                                                         jlong handle) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_18(mht_18_v, 657, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_delete");

  if (handle == 0) return;
  TF_DeleteTensor(reinterpret_cast<TF_Tensor*>(handle));
}

JNIEXPORT jobject JNICALL Java_org_tensorflow_Tensor_buffer(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_19(mht_19_v, 667, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_buffer");

  TF_Tensor* t = requireHandle(env, handle);
  if (t == nullptr) return nullptr;
  void* data = TF_TensorData(t);
  const size_t sz = TF_TensorByteSize(t);

  return env->NewDirectByteBuffer(data, static_cast<jlong>(sz));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_Tensor_dtype(JNIEnv* env,
                                                        jclass clazz,
                                                        jlong handle) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_20(mht_20_v, 681, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_dtype");

  static_assert(sizeof(jint) >= sizeof(TF_DataType),
                "TF_DataType in C cannot be represented as an int in Java");
  TF_Tensor* t = requireHandle(env, handle);
  if (t == nullptr) return 0;
  return static_cast<jint>(TF_TensorType(t));
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Tensor_shape(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_21(mht_21_v, 694, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_shape");

  TF_Tensor* t = requireHandle(env, handle);
  if (t == nullptr) return nullptr;
  static_assert(sizeof(jlong) == sizeof(int64_t),
                "Java long is not compatible with the TensorFlow C API");
  const jsize num_dims = TF_NumDims(t);
  jlongArray ret = env->NewLongArray(num_dims);
  jlong* dims = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = static_cast<jlong>(TF_Dim(t, i));
  }
  env->ReleaseLongArrayElements(ret, dims, 0);
  return ret;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Tensor_setValue(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jobject value) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_22(mht_22_v, 715, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_setValue");

  TF_Tensor* t = requireHandle(env, handle);
  if (t == nullptr) return;
  int num_dims = TF_NumDims(t);
  TF_DataType dtype = TF_TensorType(t);
  void* data = TF_TensorData(t);
  const size_t sz = TF_TensorByteSize(t);
  if (num_dims == 0) {
    writeScalar(env, value, dtype, data, sz);
  } else {
    writeNDArray(env, static_cast<jarray>(value), dtype, num_dims,
                 static_cast<char*>(data), sz);
  }
}

#define DEFINE_GET_SCALAR_METHOD(jtype, dtype, method_suffix)                  \
  JNIEXPORT jtype JNICALL Java_org_tensorflow_Tensor_scalar##method_suffix(    \
      JNIEnv* env, jclass clazz, jlong handle) {                               \
    jtype ret = 0;                                                             \
    TF_Tensor* t = requireHandle(env, handle);                                 \
    if (t == nullptr) return ret;                                              \
    if (TF_NumDims(t) != 0) {                                                  \
      throwException(env, kIllegalStateException, "Tensor is not a scalar");   \
    } else if (TF_TensorType(t) != dtype) {                                    \
      throwException(env, kIllegalStateException, "Tensor is not a %s scalar", \
                     #method_suffix);                                          \
    } else {                                                                   \
      memcpy(&ret, TF_TensorData(t), elemByteSize(dtype));                     \
    }                                                                          \
    return ret;                                                                \
  }
DEFINE_GET_SCALAR_METHOD(jfloat, TF_FLOAT, Float);
DEFINE_GET_SCALAR_METHOD(jdouble, TF_DOUBLE, Double);
DEFINE_GET_SCALAR_METHOD(jint, TF_INT32, Int);
DEFINE_GET_SCALAR_METHOD(jlong, TF_INT64, Long);
DEFINE_GET_SCALAR_METHOD(jboolean, TF_BOOL, Boolean);
#undef DEFINE_GET_SCALAR_METHOD

JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_Tensor_scalarBytes(
    JNIEnv* env, jclass clazz, jlong handle) {
  TF_Tensor* t = requireHandle(env, handle);
  if (t == nullptr) return nullptr;
  if (TF_NumDims(t) != 0) {
    throwException(env, kIllegalStateException, "Tensor is not a scalar");
    return nullptr;
  }
  if (TF_TensorType(t) != TF_STRING) {
    throwException(env, kIllegalArgumentException,
                   "Tensor is not a string/bytes scalar");
    return nullptr;
  }
  const TF_TString* data = static_cast<const TF_TString*>(TF_TensorData(t));
  jbyteArray ret = TF_StringDecodeTojbyteArray(env, &data[0]);
  return ret;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Tensor_readNDArray(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle,
                                                              jobject value) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_23(mht_23_v, 777, "", "./tensorflow/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_Tensor_readNDArray");

  TF_Tensor* t = requireHandle(env, handle);
  if (t == nullptr) return;
  int num_dims = TF_NumDims(t);
  TF_DataType dtype = TF_TensorType(t);
  const void* data = TF_TensorData(t);
  const size_t sz = TF_TensorByteSize(t);
  if (num_dims == 0) {
    throwException(env, kIllegalArgumentException,
                   "copyTo() is not meant for scalar Tensors, use the scalar "
                   "accessor (floatValue(), intValue() etc.) instead");
    return;
  }
  if (dtype == TF_STRING) {
    int64_t num_elements = 1;
    for (int i = 0; i < num_dims; ++i) {
      num_elements *= TF_Dim(t, i);
    }
    StringTensorReader reader(t, num_elements);
    TF_Status* status = TF_NewStatus();
    readNDStringArray(env, &reader, num_dims, static_cast<jobjectArray>(value),
                      status);
    throwExceptionIfNotOK(env, status);
    TF_DeleteStatus(status);
    return;
  }
  readNDArray(env, dtype, static_cast<const char*>(data), sz, num_dims,
              static_cast<jarray>(value));
}
