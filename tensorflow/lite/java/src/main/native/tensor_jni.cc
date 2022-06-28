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
class MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc {
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
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc() {
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

#include <jni.h>

#include <cstring>
#include <memory>
#include <string>

#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/core/shims/cc/interpreter.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/string_util.h"

using tflite::jni::ThrowException;
using tflite_shims::Interpreter;

namespace tflite {
// Convenience handle for obtaining a TfLiteTensor given an interpreter and
// tensor index.
//
// Historically, the Java Tensor class used a TfLiteTensor pointer as its native
// handle. However, this approach isn't generally safe, as the interpreter may
// invalidate all TfLiteTensor* handles during inference or allocation.
class TensorHandleImpl {
 public:
  virtual ~TensorHandleImpl() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "~TensorHandleImpl");
}
  virtual TfLiteTensor* tensor() const = 0;
  virtual int index() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "index");
 return -1; }
};

class InterpreterTensorHandle : public TensorHandleImpl {
 public:
  InterpreterTensorHandle(Interpreter* interpreter, int tensor_index)
      : interpreter_(interpreter), tensor_index_(tensor_index) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "InterpreterTensorHandle");
}

  TfLiteTensor* tensor() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_3(mht_3_v, 228, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "tensor");

    return interpreter_->tensor(tensor_index_);
  }

  int index() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_4(mht_4_v, 235, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "index");
 return tensor_index_; }

 private:
  Interpreter* const interpreter_;
  const int tensor_index_;
};

#if !TFLITE_DISABLE_SELECT_JAVA_APIS
class SignatureRunnerTensorHandle : public TensorHandleImpl {
 public:
  SignatureRunnerTensorHandle(SignatureRunner* runner, const char* name,
                              bool is_input)
      : signature_runner_(runner), name_(name), is_input_(is_input) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_5(mht_5_v, 251, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "SignatureRunnerTensorHandle");
}

  TfLiteTensor* tensor() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_6(mht_6_v, 256, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "tensor");

    if (is_input_) {
      return signature_runner_->input_tensor(name_.c_str());
    }
    return const_cast<TfLiteTensor*>(
        signature_runner_->output_tensor(name_.c_str()));
  }

 private:
  SignatureRunner* signature_runner_;
  std::string name_;
  bool is_input_;
};
#endif

class TensorHandle {
 public:
  TensorHandle(Interpreter* interpreter, int tensor_index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_7(mht_7_v, 276, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "TensorHandle");

    impl_.reset(new InterpreterTensorHandle(interpreter, tensor_index));
  }

#if !TFLITE_DISABLE_SELECT_JAVA_APIS
  TensorHandle(SignatureRunner* runner, const char* name, bool is_input) {
    impl_.reset(new SignatureRunnerTensorHandle(runner, name, is_input));
  }
#endif

  TfLiteTensor* tensor() const { return impl_->tensor(); }
  int index() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_8(mht_8_v, 290, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "index");
 return impl_->index(); }

 private:
  std::unique_ptr<TensorHandleImpl> impl_;
};
}  // namespace tflite

namespace {
using tflite::TensorHandle;

static const char* kByteArrayClassPath = "[B";
static const char* kStringClassPath = "java/lang/String";

TfLiteTensor* GetTensorFromHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_9(mht_9_v, 306, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "GetTensorFromHandle");

  if (handle == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to TfLiteTensor.");
    return nullptr;
  }
  return reinterpret_cast<TensorHandle*>(handle)->tensor();
}

int GetTensorIndexFromHandle(JNIEnv* env, jlong handle) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_10(mht_10_v, 318, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "GetTensorIndexFromHandle");

  if (handle == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to TfLiteTensor.");
    return -1;
  }
  return reinterpret_cast<TensorHandle*>(handle)->index();
}

size_t ElementByteSize(TfLiteType data_type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_11(mht_11_v, 330, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "ElementByteSize");

  // The code in this file makes the assumption that the
  // TensorFlow TF_DataTypes and the Java primitive types
  // have the same byte sizes. Validate that:
  switch (data_type) {
    case kTfLiteFloat32:
      static_assert(sizeof(jfloat) == 4,
                    "Interal error: Java float not compatible with "
                    "kTfLiteFloat");
      return 4;
    case kTfLiteInt32:
      static_assert(sizeof(jint) == 4,
                    "Interal error: Java int not compatible with kTfLiteInt");
      return 4;
    case kTfLiteInt16:
      static_assert(sizeof(jshort) == 2,
                    "Interal error: Java int not compatible with kTfLiteShort");
      return 2;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      static_assert(sizeof(jbyte) == 1,
                    "Interal error: Java byte not compatible with "
                    "kTfLiteUInt8");
      return 1;
    case kTfLiteBool:
      static_assert(sizeof(jboolean) == 1,
                    "Interal error: Java boolean not compatible with "
                    "kTfLiteBool");
      return 1;
    case kTfLiteInt64:
      static_assert(sizeof(jlong) == 8,
                    "Interal error: Java long not compatible with "
                    "kTfLiteInt64");
      return 8;
    default:
      return 0;
  }
}

size_t WriteOneDimensionalArray(JNIEnv* env, jobject object, TfLiteType type,
                                void* dst, size_t dst_size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_12(mht_12_v, 373, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "WriteOneDimensionalArray");

  jarray array = static_cast<jarray>(object);
  const int num_elements = env->GetArrayLength(array);
  size_t to_copy = num_elements * ElementByteSize(type);
  if (to_copy > dst_size) {
    ThrowException(env, tflite::jni::kIllegalStateException,
                   "Internal error: cannot write Java array of %d bytes to "
                   "Tensor of %d bytes",
                   to_copy, dst_size);
    return 0;
  }
  switch (type) {
    case kTfLiteFloat32: {
      jfloatArray float_array = static_cast<jfloatArray>(array);
      jfloat* float_dst = static_cast<jfloat*>(dst);
      env->GetFloatArrayRegion(float_array, 0, num_elements, float_dst);
      return to_copy;
    }
    case kTfLiteInt32: {
      jintArray int_array = static_cast<jintArray>(array);
      jint* int_dst = static_cast<jint*>(dst);
      env->GetIntArrayRegion(int_array, 0, num_elements, int_dst);
      return to_copy;
    }
    case kTfLiteInt16: {
      jshortArray short_array = static_cast<jshortArray>(array);
      jshort* short_dst = static_cast<jshort*>(dst);
      env->GetShortArrayRegion(short_array, 0, num_elements, short_dst);
      return to_copy;
    }
    case kTfLiteInt64: {
      jlongArray long_array = static_cast<jlongArray>(array);
      jlong* long_dst = static_cast<jlong*>(dst);
      env->GetLongArrayRegion(long_array, 0, num_elements, long_dst);
      return to_copy;
    }
    case kTfLiteInt8:
    case kTfLiteUInt8: {
      jbyteArray byte_array = static_cast<jbyteArray>(array);
      jbyte* byte_dst = static_cast<jbyte*>(dst);
      env->GetByteArrayRegion(byte_array, 0, num_elements, byte_dst);
      return to_copy;
    }
    case kTfLiteBool: {
      jbooleanArray bool_array = static_cast<jbooleanArray>(array);
      jboolean* bool_dst = static_cast<jboolean*>(dst);
      env->GetBooleanArrayRegion(bool_array, 0, num_elements, bool_dst);
      return to_copy;
    }
    default: {
      ThrowException(
          env, tflite::jni::kUnsupportedOperationException,
          "DataType error: TensorFlowLite currently supports float "
          "(32 bits), int (32 bits), byte (8 bits), bool (8 bits), and long "
          "(64 bits), support for other types (DataType %d in this "
          "case) will be added in the future",
          kTfLiteFloat32, type);
      return 0;
    }
  }
}

size_t ReadOneDimensionalArray(JNIEnv* env, TfLiteType data_type,
                               const void* src, size_t src_size, jarray dst) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_13(mht_13_v, 439, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "ReadOneDimensionalArray");

  const int len = env->GetArrayLength(dst);
  const size_t size = len * ElementByteSize(data_type);
  if (size > src_size) {
    ThrowException(
        env, tflite::jni::kIllegalStateException,
        "Internal error: cannot fill a Java array of %d bytes with a Tensor of "
        "%d bytes",
        size, src_size);
    return 0;
  }
  switch (data_type) {
    case kTfLiteFloat32: {
      jfloatArray float_array = static_cast<jfloatArray>(dst);
      env->SetFloatArrayRegion(float_array, 0, len,
                               static_cast<const jfloat*>(src));
      return size;
    }
    case kTfLiteInt32: {
      jintArray int_array = static_cast<jintArray>(dst);
      env->SetIntArrayRegion(int_array, 0, len, static_cast<const jint*>(src));
      return size;
    }
    case kTfLiteInt16: {
      jshortArray short_array = static_cast<jshortArray>(dst);
      env->SetShortArrayRegion(short_array, 0, len,
                               static_cast<const jshort*>(src));
      return size;
    }
    case kTfLiteInt64: {
      jlongArray long_array = static_cast<jlongArray>(dst);
      env->SetLongArrayRegion(long_array, 0, len,
                              static_cast<const jlong*>(src));
      return size;
    }
    case kTfLiteInt8:
    case kTfLiteUInt8: {
      jbyteArray byte_array = static_cast<jbyteArray>(dst);
      env->SetByteArrayRegion(byte_array, 0, len,
                              static_cast<const jbyte*>(src));
      return size;
    }
    case kTfLiteBool: {
      jbooleanArray bool_array = static_cast<jbooleanArray>(dst);
      env->SetBooleanArrayRegion(bool_array, 0, len,
                                 static_cast<const jboolean*>(src));
      return size;
    }
    default: {
      ThrowException(env, tflite::jni::kIllegalStateException,
                     "DataType error: invalid DataType(%d)", data_type);
    }
  }
  return 0;
}

size_t ReadMultiDimensionalArray(JNIEnv* env, TfLiteType data_type, char* src,
                                 size_t src_size, int dims_left, jarray dst) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("src: \"" + (src == nullptr ? std::string("nullptr") : std::string((char*)src)) + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_14(mht_14_v, 500, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "ReadMultiDimensionalArray");

  if (dims_left == 1) {
    return ReadOneDimensionalArray(env, data_type, src, src_size, dst);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(dst);
    int len = env->GetArrayLength(ndarray);
    size_t size = 0;
    for (int i = 0; i < len; ++i) {
      jarray row = static_cast<jarray>(env->GetObjectArrayElement(ndarray, i));
      size += ReadMultiDimensionalArray(env, data_type, src + size,
                                        src_size - size, dims_left - 1, row);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return size;
    }
    return size;
  }
}

// Returns the total number of strings read.
int ReadMultiDimensionalStringArray(JNIEnv* env, TfLiteTensor* tensor,
                                    int dims_left, int start_str_index,
                                    jarray dst) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_15(mht_15_v, 524, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "ReadMultiDimensionalStringArray");

  jobjectArray object_array = static_cast<jobjectArray>(dst);
  int len = env->GetArrayLength(object_array);
  int num_strings_read = 0;

  // If dst is a 1-dimensional array, copy the strings into it. Else
  // recursively call ReadMultiDimensionalStringArray over sub-dimensions.
  if (dims_left == 1) {
    for (int i = 0; i < len; ++i) {
      const tflite::StringRef strref =
          tflite::GetString(tensor, start_str_index + num_strings_read);
      // Makes sure the string is null terminated before passing to
      // NewStringUTF.
      std::string str(strref.str, strref.len);
      jstring string_dest = env->NewStringUTF(str.data());
      env->SetObjectArrayElement(object_array, i, string_dest);
      env->DeleteLocalRef(string_dest);
      ++num_strings_read;
    }
  } else {
    for (int i = 0; i < len; ++i) {
      jarray row =
          static_cast<jarray>(env->GetObjectArrayElement(object_array, i));
      num_strings_read += ReadMultiDimensionalStringArray(
          env, tensor, dims_left - 1, start_str_index + num_strings_read, row);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return num_strings_read;
    }
  }

  return num_strings_read;
}

size_t WriteMultiDimensionalArray(JNIEnv* env, jobject src, TfLiteType type,
                                  int dims_left, char** dst, int dst_size) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_16(mht_16_v, 561, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "WriteMultiDimensionalArray");

  if (dims_left <= 1) {
    return WriteOneDimensionalArray(env, src, type, *dst, dst_size);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(src);
    int len = env->GetArrayLength(ndarray);
    size_t sz = 0;
    for (int i = 0; i < len; ++i) {
      jobject row = env->GetObjectArrayElement(ndarray, i);
      char* next_dst = *dst + sz;
      sz += WriteMultiDimensionalArray(env, row, type, dims_left - 1, &next_dst,
                                       dst_size - sz);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return sz;
    }
    return sz;
  }
}

void AddStringDynamicBuffer(JNIEnv* env, jobject src,
                            tflite::DynamicBuffer* dst_buffer) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_17(mht_17_v, 584, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "AddStringDynamicBuffer");

  if (env->IsInstanceOf(src, env->FindClass(kStringClassPath))) {
    jstring str = static_cast<jstring>(src);
    const char* chars = env->GetStringUTFChars(str, nullptr);
    // + 1 for terminating character.
    const int byte_len = env->GetStringUTFLength(str) + 1;
    dst_buffer->AddString(chars, byte_len);
    env->ReleaseStringUTFChars(str, chars);
  }
  if (env->IsInstanceOf(src, env->FindClass(kByteArrayClassPath))) {
    jbyteArray byte_array = static_cast<jbyteArray>(src);
    jsize byte_array_length = env->GetArrayLength(byte_array);
    jbyte* bytes = env->GetByteArrayElements(byte_array, nullptr);
    dst_buffer->AddString(reinterpret_cast<const char*>(bytes),
                          byte_array_length);
    env->ReleaseByteArrayElements(byte_array, bytes, JNI_ABORT);
  }
}

void PopulateStringDynamicBuffer(JNIEnv* env, jobject src,
                                 tflite::DynamicBuffer* dst_buffer,
                                 int dims_left) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_18(mht_18_v, 608, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "PopulateStringDynamicBuffer");

  jobjectArray object_array = static_cast<jobjectArray>(src);
  const int num_elements = env->GetArrayLength(object_array);

  // If src is a 1-dimensional array, add the strings into dst_buffer. Else
  // recursively call populateStringDynamicBuffer over sub-dimensions.
  if (dims_left <= 1) {
    for (int i = 0; i < num_elements; ++i) {
      jobject obj = env->GetObjectArrayElement(object_array, i);
      AddStringDynamicBuffer(env, obj, dst_buffer);
      env->DeleteLocalRef(obj);
    }
  } else {
    for (int i = 0; i < num_elements; ++i) {
      jobject row = env->GetObjectArrayElement(object_array, i);
      PopulateStringDynamicBuffer(env, row, dst_buffer, dims_left - 1);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return;
    }
  }
}

void WriteMultiDimensionalStringArray(JNIEnv* env, jobject src,
                                      TfLiteTensor* tensor) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_19(mht_19_v, 634, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "WriteMultiDimensionalStringArray");

  tflite::DynamicBuffer dst_buffer;
  PopulateStringDynamicBuffer(env, src, &dst_buffer, tensor->dims->size);
  if (!env->ExceptionCheck()) {
    dst_buffer.WriteToTensor(tensor, /*new_shape=*/nullptr);
  }
}

void WriteScalar(JNIEnv* env, jobject src, TfLiteType type, void* dst,
                 int dst_size) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_20(mht_20_v, 646, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "WriteScalar");

  size_t src_size = ElementByteSize(type);
  if (src_size != dst_size) {
    ThrowException(
        env, tflite::jni::kIllegalStateException,
        "Scalar (%d bytes) not compatible with allocated tensor (%d bytes)",
        src_size, dst_size);
    return;
  }
  switch (type) {
// env->FindClass and env->GetMethodID are expensive and JNI best practices
// suggest that they should be cached. However, until the creation of scalar
// valued tensors seems to become a noticeable fraction of program execution,
// ignore that cost.
#define CASE(type, jtype, method_name, method_signature, call_type)            \
  case type: {                                                                 \
    jclass clazz = env->FindClass("java/lang/Number");                         \
    jmethodID method = env->GetMethodID(clazz, method_name, method_signature); \
    jtype v = env->Call##call_type##Method(src, method);                       \
    memcpy(dst, &v, src_size);                                                 \
    return;                                                                    \
  }
    CASE(kTfLiteFloat32, jfloat, "floatValue", "()F", Float);
    CASE(kTfLiteInt32, jint, "intValue", "()I", Int);
    CASE(kTfLiteInt16, jshort, "shortValue", "()S", Short);
    CASE(kTfLiteInt64, jlong, "longValue", "()J", Long);
    CASE(kTfLiteInt8, jbyte, "byteValue", "()B", Byte);
    CASE(kTfLiteUInt8, jbyte, "byteValue", "()B", Byte);
#undef CASE
    case kTfLiteBool: {
      jclass clazz = env->FindClass("java/lang/Boolean");
      jmethodID method = env->GetMethodID(clazz, "booleanValue", "()Z");
      jboolean v = env->CallBooleanMethod(src, method);
      *(static_cast<unsigned char*>(dst)) = v ? 1 : 0;
      return;
    }
    default:
      ThrowException(env, tflite::jni::kIllegalStateException,
                     "Invalid DataType(%d)", type);
      return;
  }
}

void WriteScalarString(JNIEnv* env, jobject src, TfLiteTensor* tensor) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_21(mht_21_v, 692, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "WriteScalarString");

  tflite::DynamicBuffer dst_buffer;
  AddStringDynamicBuffer(env, src, &dst_buffer);
  if (!env->ExceptionCheck()) {
    dst_buffer.WriteToTensor(tensor, /*new_shape=*/nullptr);
  }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_TensorImpl_create(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jint tensor_index) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_22(mht_22_v, 708, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_create");

  Interpreter* interpreter = reinterpret_cast<Interpreter*>(interpreter_handle);
  return reinterpret_cast<jlong>(new TensorHandle(interpreter, tensor_index));
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_TensorImpl_createSignatureInputTensor(
    JNIEnv* env, jclass clazz, jlong signature_runner_handle,
    jstring input_name) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_23(mht_23_v, 720, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_createSignatureInputTensor");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: createSignatureInputTensor");
  return -1;
#else
  tflite::SignatureRunner* runner =
      reinterpret_cast<tflite::SignatureRunner*>(signature_runner_handle);
  if (runner == nullptr) return -1;
  const char* input_name_ptr = env->GetStringUTFChars(input_name, nullptr);
  TensorHandle* handle =
      new TensorHandle(runner, input_name_ptr, /*is_input=*/true);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return reinterpret_cast<jlong>(handle);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_TensorImpl_createSignatureOutputTensor(
    JNIEnv* env, jclass clazz, jlong signature_runner_handle,
    jstring output_name) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("output_name: \"" + output_name + "\"");
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_24(mht_24_v, 745, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_createSignatureOutputTensor");

#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: createSignatureOutputTensor");
  return -1;
#else
  tflite::SignatureRunner* runner =
      reinterpret_cast<tflite::SignatureRunner*>(signature_runner_handle);
  if (runner == nullptr) return -1;
  const char* output_name_ptr = env->GetStringUTFChars(output_name, nullptr);
  TensorHandle* handle =
      new TensorHandle(runner, output_name_ptr, /*is_input=*/false);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(output_name, output_name_ptr);
  return reinterpret_cast<jlong>(handle);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_TensorImpl_delete(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_25(mht_25_v, 767, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_delete");

  delete reinterpret_cast<TensorHandle*>(handle);
}

JNIEXPORT jobject JNICALL Java_org_tensorflow_lite_TensorImpl_buffer(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_26(mht_26_v, 775, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_buffer");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return nullptr;
  if (tensor->data.raw == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Tensor hasn't been allocated.");
    return nullptr;
  }
  return env->NewDirectByteBuffer(static_cast<void*>(tensor->data.raw),
                                  static_cast<jlong>(tensor->bytes));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_TensorImpl_writeDirectBuffer(
    JNIEnv* env, jclass clazz, jlong handle, jobject src) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_27(mht_27_v, 791, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_writeDirectBuffer");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;

  void* src_data_raw = env->GetDirectBufferAddress(src);
  if (!src_data_raw) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Input ByteBuffer is not a direct buffer");
    return;
  }

  if (!tensor->data.data) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Tensor hasn't been allocated.");
    return;
  }

  // Historically, we would simply overwrite the tensor buffer pointer with
  // the direct Buffer address. However, that is generally unsafe, and
  // specifically wrong if the graph happens to have dynamic shapes where
  // arena-allocated input buffers will be refreshed during invocation.
  // TODO(b/156094015): Explore whether this is actually faster than
  // using ByteBuffer.put(ByteBuffer).
  memcpy(tensor->data.data, src_data_raw, tensor->bytes);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_TensorImpl_readMultiDimensionalArray(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle,
                                                              jobject value) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_28(mht_28_v, 824, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_readMultiDimensionalArray");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;
  int num_dims = tensor->dims->size;
  if (num_dims == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Cannot copy empty/scalar Tensors.");
    return;
  }
  if (tensor->type == kTfLiteString) {
    ReadMultiDimensionalStringArray(env, tensor, num_dims, 0,
                                    static_cast<jarray>(value));
  } else {
    ReadMultiDimensionalArray(env, tensor->type, tensor->data.raw,
                              tensor->bytes, num_dims,
                              static_cast<jarray>(value));
  }
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_TensorImpl_writeMultiDimensionalArray(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong handle,
                                                               jobject src) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_29(mht_29_v, 850, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_writeMultiDimensionalArray");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;
  if (tensor->type != kTfLiteString && tensor->data.raw == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Target Tensor hasn't been allocated.");
    return;
  }
  if (tensor->dims->size == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Cannot copy empty/scalar Tensors.");
    return;
  }
  if (tensor->type == kTfLiteString) {
    WriteMultiDimensionalStringArray(env, src, tensor);
  } else {
    WriteMultiDimensionalArray(env, src, tensor->type, tensor->dims->size,
                               &tensor->data.raw, tensor->bytes);
  }
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_TensorImpl_writeScalar(
    JNIEnv* env, jclass clazz, jlong handle, jobject src) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_30(mht_30_v, 875, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_writeScalar");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;
  if ((tensor->type != kTfLiteString) && (tensor->data.raw == nullptr)) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Target Tensor hasn't been allocated.");
    return;
  }
  if ((tensor->dims->size != 0) && (tensor->dims->data[0] != 1)) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Cannot write Java scalar to non-scalar "
                   "Tensor.");
    return;
  }
  if (tensor->type == kTfLiteString) {
    WriteScalarString(env, src, tensor);
  } else {
    WriteScalar(env, src, tensor->type, tensor->data.data, tensor->bytes);
  }
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_TensorImpl_dtype(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_31(mht_31_v, 901, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_dtype");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(tensor->type);
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_lite_TensorImpl_name(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_32(mht_32_v, 911, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_name");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Target Tensor doesn't exist.");
    return nullptr;
  }

  if (tensor->name == nullptr) {
    return env->NewStringUTF("");
  }

  jstring tensor_name = env->NewStringUTF(tensor->name);
  if (tensor_name == nullptr) {
    return env->NewStringUTF("");
  }
  return tensor_name;
}

JNIEXPORT jintArray JNICALL Java_org_tensorflow_lite_TensorImpl_shape(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_33(mht_33_v, 934, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_shape");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return nullptr;
  int num_dims = tensor->dims->size;
  jintArray result = env->NewIntArray(num_dims);
  env->SetIntArrayRegion(result, 0, num_dims, tensor->dims->data);
  return result;
}

JNIEXPORT jintArray JNICALL Java_org_tensorflow_lite_TensorImpl_shapeSignature(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_34(mht_34_v, 947, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_shapeSignature");

  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return nullptr;

  int num_dims = 0;
  int const* data = nullptr;
  if (tensor->dims_signature != nullptr && tensor->dims_signature->size != 0) {
    num_dims = tensor->dims_signature->size;
    data = tensor->dims_signature->data;
  } else {
    num_dims = tensor->dims->size;
    data = tensor->dims->data;
  }
  jintArray result = env->NewIntArray(num_dims);
  env->SetIntArrayRegion(result, 0, num_dims, data);
  return result;
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_TensorImpl_numBytes(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_35(mht_35_v, 969, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_numBytes");

  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(tensor->bytes);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_TensorImpl_hasDelegateBufferHandle(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_36(mht_36_v, 981, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_hasDelegateBufferHandle");

  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return false;
  return tensor->delegate && (tensor->buffer_handle != kTfLiteNullBufferHandle)
             ? JNI_TRUE
             : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_TensorImpl_index(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_37(mht_37_v, 994, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_index");

  return GetTensorIndexFromHandle(env, handle);
}

JNIEXPORT jfloat JNICALL Java_org_tensorflow_lite_TensorImpl_quantizationScale(
    JNIEnv* env, jclass clazz, jlong handle) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_38(mht_38_v, 1002, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_quantizationScale");

  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  return static_cast<jfloat>(tensor ? tensor->params.scale : 0.f);
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_TensorImpl_quantizationZeroPoint(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSlitePSjavaPSsrcPSmainPSnativePStensor_jniDTcc mht_39(mht_39_v, 1013, "", "./tensorflow/lite/java/src/main/native/tensor_jni.cc", "Java_org_tensorflow_lite_TensorImpl_quantizationZeroPoint");

  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  return static_cast<jint>(tensor ? tensor->params.zero_point : 0);
}

}  // extern "C"
