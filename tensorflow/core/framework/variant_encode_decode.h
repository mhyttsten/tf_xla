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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
#define TENSORFLOW_CORE_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh() {
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


#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Type used for tag-dispatch of the Encode/Decode Variant implementations. This
// template can determine whether the first type parameter `T` is one of the
// following:
//
// * A POD type (TypeResolver<T, true>)
// * A tensorflow::Tensor (TypeResolver<T, false, true>)
// * A protocol buffer (TypeResolver<T, false, false, true>)
// * None of the above (TypeResolver<T, false, false, false>)
//
template <typename T, bool = std::is_pod<typename std::decay<T>::type>::value,
          bool = std::is_same<typename std::decay<T>::type,
                              ::tensorflow::Tensor>::value,
          bool = std::is_base_of<protobuf::MessageLite,
                                 typename std::decay<T>::type>::value>
struct TypeResolver {};

// Specialization for POD type
template <typename T>
void EncodeVariantImpl(const T& value, TypeResolver<T, true /* is_pod */>,
                       VariantTensorData* data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/framework/variant_encode_decode.h", "EncodeVariantImpl");

  data->set_metadata(value);
}

// Specialization for tensorflow::Tensor
template <typename T>
void EncodeVariantImpl(const T& value,
                       TypeResolver<T, false /* is_pod */, true /* Tensor */>,
                       VariantTensorData* data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/framework/variant_encode_decode.h", "EncodeVariantImpl");

  data->tensors_.clear();
  data->tensors_.push_back(value);
}

// Specialization for protobuf
template <typename T>
void EncodeVariantImpl(const T& value,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    true /* protobuf */>,
                       VariantTensorData* data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_2(mht_2_v, 245, "", "./tensorflow/core/framework/variant_encode_decode.h", "EncodeVariantImpl");

  value.SerializeToString(&data->metadata_);
}

// Specialization for other types
template <typename T>
void EncodeVariantImpl(const T& value,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       VariantTensorData* data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_3(mht_3_v, 257, "", "./tensorflow/core/framework/variant_encode_decode.h", "EncodeVariantImpl");

  value.Encode(data);
}

// Specialization for POD type
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, true /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_4(mht_4_v, 269, "", "./tensorflow/core/framework/variant_encode_decode.h", "DecodeVariantImpl");

  return data.get_metadata(value);
}

// Specialization for tensorflow::Tensor
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, false /* is_pod */, true /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_5(mht_5_v, 281, "", "./tensorflow/core/framework/variant_encode_decode.h", "DecodeVariantImpl");

  *value = data.tensors(0);
  return true;
}

// Specialization for protobuf
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    true /* protobuf */>,
                       T* value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_6(mht_6_v, 294, "", "./tensorflow/core/framework/variant_encode_decode.h", "DecodeVariantImpl");

  std::string metadata;
  data.get_metadata(&metadata);
  return value->ParseFromString(std::move(metadata));
}

// Specialization for other types
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_7(mht_7_v, 308, "", "./tensorflow/core/framework/variant_encode_decode.h", "DecodeVariantImpl");

  return value->Decode(std::move(data));
}

template <typename C, typename = void>
struct has_type_name : std::false_type {};

template <typename C>
struct has_type_name<
    C, typename std::enable_if<std::is_same<
           decltype(std::declval<C>().TypeName()), string>::value>::type>
    : std::true_type {};

template <typename T, bool = has_type_name<typename std::decay<T>::type>::value,
          bool = std::is_same<typename std::decay<T>::type,
                              ::tensorflow::Tensor>::value,
          bool = std::is_base_of<protobuf::MessageLite,
                                 typename std::decay<T>::type>::value>
struct TypeNameResolver {};

template <typename T>
std::string TypeNameVariantImpl(const T& value,
                                TypeNameResolver<T, true /* has_type_name */>) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_8(mht_8_v, 333, "", "./tensorflow/core/framework/variant_encode_decode.h", "TypeNameVariantImpl");

  return value.TypeName();
}

template <typename T>
std::string TypeNameVariantImpl(
    const T& value,
    TypeNameResolver<T, false /* has_type_name */, true /* Tensor */>) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_9(mht_9_v, 343, "", "./tensorflow/core/framework/variant_encode_decode.h", "TypeNameVariantImpl");

  return "tensorflow::Tensor";
}

template <typename T>
std::string TypeNameVariantImpl(
    const T& value, TypeNameResolver<T, false /* has_type_name */,
                                     false /* Tensor */, true /* protobuf */>) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_10(mht_10_v, 353, "", "./tensorflow/core/framework/variant_encode_decode.h", "TypeNameVariantImpl");

  return value.GetTypeName();
}

template <typename T>
std::string TypeNameVariantImpl(
    const T& value,
    TypeNameResolver<T, false /* has_type_name */, false /* Tensor */,
                     false /* protobuf */>) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_11(mht_11_v, 364, "", "./tensorflow/core/framework/variant_encode_decode.h", "TypeNameVariantImpl");

  return port::MaybeAbiDemangle(TypeIndex::Make<T>().name());
}

template <typename T>
std::string TypeNameVariant(const T& value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_12(mht_12_v, 372, "", "./tensorflow/core/framework/variant_encode_decode.h", "TypeNameVariant");

  return TypeNameVariantImpl(value, TypeNameResolver<T>());
}

template <typename C, typename = void>
struct has_debug_string : std::false_type {};

template <typename C>
struct has_debug_string<
    C, typename std::enable_if<std::is_same<
           decltype(std::declval<C>().DebugString()), string>::value>::type>
    : std::true_type {};

template <typename C, typename = void>
struct can_strcat : std::false_type {};

template <typename C>
struct can_strcat<
    C, typename std::enable_if<std::is_same<
           decltype(strings::StrCat(std::declval<C>())), string>::value>::type>
    : std::true_type {};

template <typename T,
          bool = has_debug_string<typename std::decay<T>::type>::value,
          bool = can_strcat<typename std::decay<T>::type>::value>
struct DebugStringResolver {};

// TODO(ebrevdo): Expand DebugStringResolver to return TypeString if
// there is no StrCat<T>() constructor.
template <typename T>
std::string DebugStringVariantImpl(
    const T& value, DebugStringResolver<T, true /* has_debug_string */>) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_13(mht_13_v, 406, "", "./tensorflow/core/framework/variant_encode_decode.h", "DebugStringVariantImpl");

  return value.DebugString();
}

template <typename T>
std::string DebugStringVariantImpl(
    const T& value, DebugStringResolver<T, false /* has_debug_string */,
                                        true /* can_strcat */>) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_14(mht_14_v, 416, "", "./tensorflow/core/framework/variant_encode_decode.h", "DebugStringVariantImpl");

  return strings::StrCat(value);
}

template <typename T>
std::string DebugStringVariantImpl(
    const T& value, DebugStringResolver<T, false /* has_debug_string */,
                                        false /* can_strcat */>) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_15(mht_15_v, 426, "", "./tensorflow/core/framework/variant_encode_decode.h", "DebugStringVariantImpl");

  return "?";
}

template <typename T>
std::string DebugStringVariant(const T& value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_16(mht_16_v, 434, "", "./tensorflow/core/framework/variant_encode_decode.h", "DebugStringVariant");

  return DebugStringVariantImpl(value, DebugStringResolver<T>());
}

template <typename T>
void EncodeVariant(const T& value, VariantTensorData* data) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_17(mht_17_v, 442, "", "./tensorflow/core/framework/variant_encode_decode.h", "EncodeVariant");

  EncodeVariantImpl(value, TypeResolver<T>(), data);
  data->set_type_name(TypeNameVariant(value));
}

template <typename T>
bool DecodeVariant(VariantTensorData* data, T* value) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_18(mht_18_v, 451, "", "./tensorflow/core/framework/variant_encode_decode.h", "DecodeVariant");

  return DecodeVariantImpl(std::move(*data), TypeResolver<T>(), value);
}

template <typename T>
void EncodeVariant(const T& value, std::string* buf) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_19(mht_19_v, 459, "", "./tensorflow/core/framework/variant_encode_decode.h", "EncodeVariant");

  VariantTensorData data;
  EncodeVariantImpl(value, TypeResolver<T>(), &data);
  data.set_type_name(TypeNameVariant(value));
  DCHECK(buf != nullptr);
  data.SerializeToString(buf);
}

template <typename T>
bool DecodeVariant(std::string* buf, T* value) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSvariant_encode_decodeDTh mht_20(mht_20_v, 471, "", "./tensorflow/core/framework/variant_encode_decode.h", "DecodeVariant");

  VariantTensorData data;
  if (!data.ParseFromString(*buf)) return false;
  if (!DecodeVariantImpl(std::move(data), TypeResolver<T>(), value)) {
    return false;
  }
  return true;
}

// Specializations for VariantTensorDataProto
template <>
std::string TypeNameVariant(const VariantTensorDataProto& value);

template <>
void EncodeVariant(const VariantTensorDataProto& value,
                   VariantTensorData* data);

template <>
bool DecodeVariant(VariantTensorData* data, VariantTensorDataProto* value);

template <>
void EncodeVariant(const VariantTensorDataProto& value, std::string* buf);

template <>
bool DecodeVariant(std::string* buf, VariantTensorDataProto* value);

// Encodes an array of Variant objects in to the given StringListEncoder.
// `variant_array` is assumed to point to an array of `n` Variant objects.
void EncodeVariantList(const Variant* variant_array, int64_t n,
                       std::unique_ptr<port::StringListEncoder> e);

// Decodes an array of Variant objects from the given StringListDecoder.
// `variant_array` is assumed to point to an array of `n` Variant objects.
bool DecodeVariantList(std::unique_ptr<port::StringListDecoder> d,
                       Variant* variant_array, int64_t n);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
