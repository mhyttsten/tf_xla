/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh() {
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


#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensor {

// DeepCopy returns a tensor whose contents are a deep copy of the
// contents of 'other'.  This function is intended only for
// convenience, not speed.
//
// REQUIRES: 'other' must point to data stored in CPU memory.
// REQUIRES: 'other' must be a Tensor of a copy-able type if
//           'other' is not appropriately memory-aligned.
Tensor DeepCopy(const Tensor& other);

// Deep copies input to output.  This function is similar to above, but assumes
// that the memory for the output has already been allocated.
void DeepCopy(const Tensor& input, Tensor* output);

// Concatenates 'tensors' into a single tensor, along their 0th dimension.
//
// REQUIRES: All members of 'tensors' must have the same data type parameter.
// REQUIRES: Each member of 'tensors' must have at least one dimension.
// REQUIRES: Each member of 'tensors' must point to data stored in CPU memory.
// REQUIRES: Each member of 'tensors' must be a Tensor of a copy-able type if it
//           is not appropriately memory-aligned.
Status Concat(const gtl::ArraySlice<Tensor>& tensors,
              Tensor* result) TF_MUST_USE_RESULT;

// Splits 'tensor' into 'sizes.size()' individual tensors, along the 0th
// dimension. The ith output tensor has 0th-dimension size 'sizes[i]'.
//
// REQUIRES: 'tensor' must have at least one dimension.
// REQUIRES: 'tensor.dim_size(0)' must equal the sum of the elements of 'sizes'.
// REQUIRES: 'tensor' must point to data stored in CPU memory.
// REQUIRES: 'tensor' must be a Tensor of a copy-able type if it is not
//           appropriately memory-aligned.
//
// Split() and Concat() are inverse operations.
Status Split(const Tensor& tensor, const gtl::ArraySlice<int64_t>& sizes,
             std::vector<Tensor>* result) TF_MUST_USE_RESULT;

namespace internal {
void SetTensorProtoShape(std::vector<size_t> shape,
                         TensorShapeProto* shape_proto);

template <typename Type>
class TensorProtoFieldHelper : public std::false_type {};

#define DEFINE_PROTO_FIELD_HELPER(TYPE, FIELDNAME)                            \
  template <>                                                                 \
  class TensorProtoFieldHelper<TYPE> : public std::true_type {                \
   public:                                                                    \
    typedef decltype(                                                         \
        std::declval<TensorProto>().FIELDNAME##_val(0)) FieldType;            \
    typedef decltype(                                                         \
        std::declval<TensorProto>().FIELDNAME##_val()) RepeatedFieldType;     \
    typedef decltype(std::declval<TensorProto>().mutable_##FIELDNAME##_val()) \
        MutableRepeatedFieldType;                                             \
    static MutableRepeatedFieldType GetMutableField(TensorProto* proto) {     \
      return proto->mutable_##FIELDNAME##_val();                              \
    }                                                                         \
    static RepeatedFieldType& GetField(const TensorProto& proto) {            \
      return proto.FIELDNAME##_val();                                         \
    }                                                                         \
  }

// The argument pairs in the following macro instantiations encode the
// mapping from C++ type ($1) to repeated field name "$2_val" used for storing
// values in TensorProto. See tensorflow/core/framework/tensor.proto.
DEFINE_PROTO_FIELD_HELPER(float, float);
DEFINE_PROTO_FIELD_HELPER(double, double);
DEFINE_PROTO_FIELD_HELPER(int8, int);
DEFINE_PROTO_FIELD_HELPER(uint8, int);
DEFINE_PROTO_FIELD_HELPER(int16, int);
DEFINE_PROTO_FIELD_HELPER(uint16, int);
DEFINE_PROTO_FIELD_HELPER(int32, int);
DEFINE_PROTO_FIELD_HELPER(uint32, uint32);
DEFINE_PROTO_FIELD_HELPER(int64_t, int64);
DEFINE_PROTO_FIELD_HELPER(uint64, uint64);
DEFINE_PROTO_FIELD_HELPER(bool, bool);
DEFINE_PROTO_FIELD_HELPER(qint8, int);
DEFINE_PROTO_FIELD_HELPER(quint8, int);
DEFINE_PROTO_FIELD_HELPER(qint16, int);
DEFINE_PROTO_FIELD_HELPER(quint16, int);
DEFINE_PROTO_FIELD_HELPER(qint32, int);
DEFINE_PROTO_FIELD_HELPER(Eigen::half, half);
DEFINE_PROTO_FIELD_HELPER(bfloat16, half);
DEFINE_PROTO_FIELD_HELPER(complex64, scomplex);
DEFINE_PROTO_FIELD_HELPER(complex128, dcomplex);

#undef DEFINE_PROTO_HELPER

template <typename T>
struct CopyHelper {
  template <typename SrcIter, typename DstIter>
  static void ToArray(SrcIter begin, SrcIter end, DstIter dst) {
    using SrcType = typename std::iterator_traits<SrcIter>::value_type;
    using DstType = typename std::iterator_traits<DstIter>::value_type;
    std::transform(begin, end, dst, [](const SrcType& x) -> DstType {
      return static_cast<DstType>(x);
    });
  }
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, SrcIter dst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_0(mht_0_v, 299, "", "./tensorflow/core/framework/tensor_util.h", "ToArray");

    std::copy(begin, end, dst);
  }
  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    ToArray(begin, end, dst);
  }
};

// Overloads for Eigen::half and bfloat16 that are 16 bits in size but are
// stored in an int32 field.
template <>
struct CopyHelper<Eigen::half> {
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, Eigen::half* dst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_1(mht_1_v, 316, "", "./tensorflow/core/framework/tensor_util.h", "ToArray");

    std::transform(begin, end, dst, [](int x) -> Eigen::half {
      return Eigen::numext::bit_cast<Eigen::half>(static_cast<uint16>(x));
    });
  }
  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    std::transform(begin, end, dst, [](Eigen::half h) -> int {
      return static_cast<int>(Eigen::numext::bit_cast<uint16>(h));
    });
  }
};

template <>
struct CopyHelper<bfloat16> {
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, bfloat16* dst) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_2(mht_2_v, 335, "", "./tensorflow/core/framework/tensor_util.h", "ToArray");

    std::transform(begin, end, dst, [](int x) -> bfloat16 {
      return Eigen::numext::bit_cast<bfloat16>(static_cast<uint16>(x));
    });
  }
  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    std::transform(begin, end, dst, [](bfloat16 bf16) -> int {
      return static_cast<int>(Eigen::numext::bit_cast<uint16>(bf16));
    });
  }
};

// Overloads for complex types that store real and imaginary parts
// at indices 2*i and 2*i+1 in float or double field.
template <typename RealType>
struct CopyHelper<std::complex<RealType>> {
  template <typename SrcIter>
  static void ToArray(SrcIter begin, SrcIter end, std::complex<RealType>* dst) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_3(mht_3_v, 356, "", "./tensorflow/core/framework/tensor_util.h", "ToArray");

    RealType* real_dst = reinterpret_cast<RealType*>(dst);
    std::copy(begin, end, real_dst);
  }

  template <typename SrcIter, typename DstIter>
  static void FromArray(SrcIter begin, SrcIter end, DstIter dst) {
    size_t n = std::distance(begin, end);
    const RealType* real_begin = reinterpret_cast<const RealType*>(&(*begin));
    std::copy_n(real_begin, 2 * n, dst);
  }
};

// Helper class to extract and insert values into TensorProto represented as
// repeated fields.
template <typename T>
class TensorProtoHelper : public std::true_type {
 public:
  using FieldHelper = TensorProtoFieldHelper<T>;
  using FieldType = typename TensorProtoFieldHelper<T>::FieldType;

  static DataType GetDataType() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_4(mht_4_v, 380, "", "./tensorflow/core/framework/tensor_util.h", "GetDataType");
 return DataTypeToEnum<T>::value; }

  // Returns the number of values of type T encoded in the proto.
  static size_t NumValues(const TensorProto& proto) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_5(mht_5_v, 386, "", "./tensorflow/core/framework/tensor_util.h", "NumValues");

    size_t raw_size = FieldHelper::GetField(proto).size();
    return is_complex<T>::value ? raw_size / 2 : raw_size;
  }

  static void AddValue(const T& value, TensorProto* proto) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_6(mht_6_v, 394, "", "./tensorflow/core/framework/tensor_util.h", "AddValue");

    const T* val_ptr = &value;
    AddValues(val_ptr, val_ptr + 1, proto);
  }

  static T GetValue(size_t index, const TensorProto& proto) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_7(mht_7_v, 402, "", "./tensorflow/core/framework/tensor_util.h", "GetValue");

    const size_t stride = is_complex<T>::value ? 2 : 1;
    T val;
    CopyHelper<T>::ToArray(
        FieldHelper::GetField(proto).begin() + stride * index,
        FieldHelper::GetField(proto).begin() + stride * (index + 1), &val);
    return val;
  }

  template <typename IterType>
  static void AddValues(IterType begin, IterType end, TensorProto* proto) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_8(mht_8_v, 415, "", "./tensorflow/core/framework/tensor_util.h", "AddValues");

    size_t n = std::distance(begin, end);
    FieldType* dst = AppendUninitialized(n, proto);
    CopyHelper<T>::FromArray(begin, end, dst);
  }

  template <typename IterType>
  static void CopyValues(IterType dst, const TensorProto& proto) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_9(mht_9_v, 425, "", "./tensorflow/core/framework/tensor_util.h", "CopyValues");

    CopyHelper<T>::ToArray(FieldHelper::GetField(proto).begin(),
                           FieldHelper::GetField(proto).end(), dst);
  }

  static void Truncate(size_t new_size, TensorProto* proto) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_10(mht_10_v, 433, "", "./tensorflow/core/framework/tensor_util.h", "Truncate");

    if (is_complex<T>::value) new_size *= 2;
    FieldHelper::GetMutableField(proto)->Truncate(new_size);
  }

  static FieldType* AppendUninitialized(size_t n, TensorProto* proto) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_11(mht_11_v, 441, "", "./tensorflow/core/framework/tensor_util.h", "AppendUninitialized");

    if (is_complex<T>::value) n *= 2;
    auto* field = FieldHelper::GetMutableField(proto);
    field->Reserve(field->size() + n);
    return reinterpret_cast<FieldType*>(field->AddNAlreadyReserved(n));
  }
};

// Specialization for string.
template <>
class TensorProtoHelper<string> : public std::true_type {
 public:
  static DataType GetDataType() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_12(mht_12_v, 456, "", "./tensorflow/core/framework/tensor_util.h", "GetDataType");
 return DataType::DT_STRING; }
  static void AddValue(const string& value, TensorProto* proto) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_13(mht_13_v, 461, "", "./tensorflow/core/framework/tensor_util.h", "AddValue");

    *proto->mutable_string_val()->Add() = value;
  }
  template <typename IterType>
  static void AddValues(IterType begin, IterType end, TensorProto* proto) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_14(mht_14_v, 468, "", "./tensorflow/core/framework/tensor_util.h", "AddValues");

    for (IterType it = begin; it != end; ++it) {
      AddValue(*it, proto);
    }
  }
  template <typename IterType>
  static void CopyToTensorContent(IterType begin, IterType end,
                                  TensorProto* proto) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_15(mht_15_v, 478, "", "./tensorflow/core/framework/tensor_util.h", "CopyToTensorContent");

    AddValues(begin, end, proto);
  }
};

}  // namespace internal

// Creates a 'TensorProto' with specified shape and values.
// The dtype and a field to represent data values of the returned 'TensorProto'
// are determined based on type of the 'values' parameter.
template <typename Type>
typename std::enable_if<internal::TensorProtoHelper<Type>::value,
                        TensorProto>::type
CreateTensorProto(const std::vector<Type>& values,
                  const std::vector<size_t>& shape) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_16(mht_16_v, 495, "", "./tensorflow/core/framework/tensor_util.h", "CreateTensorProto");

  TensorProto tensor;
  TensorShapeProto tensor_shape_proto;
  internal::SetTensorProtoShape(shape, &tensor_shape_proto);
  if (TensorShape(tensor_shape_proto).num_elements() != values.size()) {
    LOG(ERROR) << "Shape and number of values (" << values.size()
               << ") are incompatible.";
    return tensor;
  }
  using TypeHelper = internal::TensorProtoHelper<Type>;
  tensor.set_dtype(TypeHelper::GetDataType());
  tensor.mutable_tensor_shape()->Swap(&tensor_shape_proto);
  TypeHelper::AddValues(values.begin(), values.end(), &tensor);
  return tensor;
}

// Converts values in tensor to run-length encoded compressed form.
//
// The elements of a tensor can be stored in a TensorProto in one of the
// following two forms:
// 1. As a raw byte string in the field `tensor_content` containing the
//    serialized in-memory representation of the tensor.
// 2. As values of a repeated field depending on the datatype, e.g. that
//    values of a DT_FLOAT tensor would be stored in the repeated field
//    `float_val`.
// Storage scheme 2 may use a simple form of run-length encoding to compress
// data: If the values contains a tail of identical values, the repeated field
// will be truncated such that the number of values in the repeated field is
// less than the number of elements implied by the field`tensor_shape`. The
// original tensor can be recovered by repeating the final value in the repeated
// field.
//
// The TensorProto will be compressed if a) the tensor contains at least
// min_num_elements elements and b) the compressed tensor proto is would be at
// most the size of the original tensor proto divided by min_compression_ratio.
//
// Returns true if the tensor was compressed.
bool CompressTensorProtoInPlace(int64_t min_num_elements,
                                float min_compression_ratio,
                                TensorProto* tensor);

inline bool CompressTensorProtoInPlace(TensorProto* tensor) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTh mht_17(mht_17_v, 539, "", "./tensorflow/core/framework/tensor_util.h", "CompressTensorProtoInPlace");

  static const int64_t kDefaultMinNumElements = 64;
  static const float kDefaultMinCompressionRatio = 2.0f;
  return CompressTensorProtoInPlace(kDefaultMinNumElements,
                                    kDefaultMinCompressionRatio, tensor);
}

// Make a TensorShape from the contents of shape_t. Shape_t must be a
// 1-dimensional tensor of type int32 or int64.
Status MakeShape(const Tensor& shape_t, TensorShape* out);

}  // namespace tensor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
