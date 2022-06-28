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
class MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc() {
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

#include "tensorflow/core/framework/tensor_util.h"

#include <cmath>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensor {

Tensor DeepCopy(const Tensor& other) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/framework/tensor_util.cc", "DeepCopy");

  Tensor tmp = Tensor(other.dtype(), other.shape());
  DeepCopy(other, &tmp);
  return tmp;
}

void DeepCopy(const Tensor& input, Tensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/framework/tensor_util.cc", "DeepCopy");

  if (DataTypeCanUseMemcpy(input.dtype())) {
    if (input.NumElements() > 0) {
      StringPiece input_data = input.tensor_data();

      // We use StringPiece as a convenient map over the tensor buffer,
      // but we cast the type to get to the underlying buffer to do the
      // copy.
      StringPiece output_data = output->tensor_data();
      memcpy(const_cast<char*>(output_data.data()), input_data.data(),
             input_data.size());
    }
  } else if (input.dtype() == DT_STRING) {
    output->unaligned_flat<tstring>() = input.unaligned_flat<tstring>();
  } else {
    CHECK_EQ(DT_VARIANT, input.dtype());
    output->unaligned_flat<Variant>() = input.unaligned_flat<Variant>();
  }
}

Status Concat(const gtl::ArraySlice<Tensor>& tensors, Tensor* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/framework/tensor_util.cc", "Concat");

  if (tensors.empty()) {
    return errors::InvalidArgument("Cannot concatenate zero tensors");
  }
  int64_t total_dim0_size = 0;
  for (const Tensor& tensor : tensors) {
    if (tensor.dims() == 0) {
      return errors::InvalidArgument(
          "Cannot concatenate a zero-dimensional tensor");
    }
    total_dim0_size += tensor.dim_size(0);
  }
  TensorShape shape = tensors[0].shape();
  shape.set_dim(0, total_dim0_size);

  const DataType dtype = tensors[0].dtype();
  for (int i = 1; i < tensors.size(); ++i) {
    if (tensors[i].dtype() != dtype) {
      return errors::InvalidArgument(
          "Cannot concatenate tensors that have different data types.", " Got ",
          DataTypeString(dtype), " and ", DataTypeString(tensors[i].dtype()),
          ".");
    }
  }
  *result = Tensor(dtype, shape);

  // We use StringPiece as a convenient map over the tensor buffer,
  // but we cast the type to get to the underlying buffer to do the
  // copy.
  StringPiece to_data = result->tensor_data();

  if (DataTypeCanUseMemcpy(dtype)) {
    int64_t offset = 0;
    for (const Tensor& tensor : tensors) {
      StringPiece from_data = tensor.tensor_data();
      CHECK_LE(offset + from_data.size(), to_data.size());
      memcpy(const_cast<char*>(to_data.data()) + offset, from_data.data(),
             from_data.size());

      offset += from_data.size();
    }
  } else {
    if (dtype != DT_STRING) {
      return errors::Internal("Unexpected data type");
    }
    tstring* to_strings =
        reinterpret_cast<tstring*>(const_cast<char*>(to_data.data()));

    int64_t offset = 0;
    for (const Tensor& tensor : tensors) {
      auto from_strings = tensor.flat<tstring>();
      CHECK_LE(offset + tensor.NumElements(), result->NumElements());
      for (int i = 0; i < tensor.NumElements(); ++i) {
        to_strings[offset + i] = from_strings(i);
      }

      offset += tensor.NumElements();
    }
  }

  return Status::OK();
}

Status Split(const Tensor& tensor, const gtl::ArraySlice<int64_t>& sizes,
             std::vector<Tensor>* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_3(mht_3_v, 301, "", "./tensorflow/core/framework/tensor_util.cc", "Split");

  if (tensor.dims() == 0) {
    return errors::InvalidArgument("Cannot split a zero-dimensional tensor");
  }
  int64_t total_size = 0;
  for (int64_t size : sizes) {
    total_size += size;
  }
  if (total_size != tensor.dim_size(0)) {
    return errors::InvalidArgument(
        "The values in 'sizes' do not sum to the zeroth-dimension size of "
        "'tensor'");
  }

  StringPiece from_data = tensor.tensor_data();

  if (DataTypeCanUseMemcpy(tensor.dtype())) {
    int64_t offset = 0;
    for (int64_t size : sizes) {
      TensorShape shape = tensor.shape();
      shape.set_dim(0, size);
      result->emplace_back(tensor.dtype(), shape);
      Tensor* split = &(*result)[result->size() - 1];

      // We use StringPiece as a convenient map over the tensor buffer,
      // but we cast the type to get to the underlying buffer to do the
      // copy.
      StringPiece to_data = split->tensor_data();
      CHECK_LE(offset + to_data.size(), from_data.size());
      memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,
             to_data.size());

      offset += to_data.size();
    }
  } else {
    if (tensor.dtype() != DT_STRING) {
      return errors::Internal("Unexpected data type");
    }
    auto from_strings = tensor.flat<tstring>();

    int64_t offset = 0;
    for (int64_t size : sizes) {
      TensorShape shape = tensor.shape();
      shape.set_dim(0, size);
      result->emplace_back(tensor.dtype(), shape);
      Tensor& split = (*result)[result->size() - 1];
      tstring* to_strings = reinterpret_cast<tstring*>(
          const_cast<char*>(split.tensor_data().data()));

      CHECK_LE(offset + split.NumElements(), tensor.NumElements());
      for (int i = 0; i < split.NumElements(); ++i) {
        to_strings[i] = from_strings(offset + i);
      }

      offset += split.NumElements();
    }
  }

  return Status::OK();
}

namespace internal {
void SetTensorProtoShape(std::vector<size_t> shape,
                         TensorShapeProto* shape_proto) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_4(mht_4_v, 367, "", "./tensorflow/core/framework/tensor_util.cc", "SetTensorProtoShape");

  for (auto dim : shape) {
    shape_proto->mutable_dim()->Add()->set_size(dim);
  }
}

template <typename T>
bool CompressTensorContent(float min_compression_ratio,
                           const TensorShape& shape, TensorProto* tensor) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_5(mht_5_v, 378, "", "./tensorflow/core/framework/tensor_util.cc", "CompressTensorContent");

  using TypeHelper = internal::TensorProtoHelper<T>;
  using FieldType = typename internal::TensorProtoHelper<T>::FieldType;
  const int64_t num_tensor_values = shape.num_elements();
  const int64_t num_bytes = tensor->tensor_content().size();
  const int64_t num_raw_values = num_bytes / sizeof(T);
  if (num_raw_values != num_tensor_values) {
    // Invalid or too small.
    return false;
  }
  int64_t last_offset = num_bytes - 1;
  int64_t prev_offset = last_offset - sizeof(T);
  // Inspect individual raw bytes sizeof(T) bytes apart in adjacent elements,
  // starting from the end, to find the last pair of elements that are not
  // identical.
  while (prev_offset >= 0) {
    if (tensor->tensor_content()[prev_offset] !=
        tensor->tensor_content()[last_offset]) {
      break;
    }
    --last_offset;
    --prev_offset;
  }
  if (prev_offset == -1) {
    // It this is a splat of value 0, it does not need an explicit value, just
    // erase the content.
    T splat_value;
    port::CopySubrangeToArray(tensor->tensor_content(), 0, sizeof(T),
                              reinterpret_cast<char*>(&splat_value));
    if (splat_value == T(0)) {
      tensor->clear_tensor_content();
      return true;
    }
  }
  // Round up to the next whole number of element of type T.
  const int64_t new_num_values = last_offset / sizeof(T) + 1;
  if (new_num_values * (is_complex<T>::value ? 2 : 1) * sizeof(FieldType) >
      static_cast<int64_t>(num_bytes / min_compression_ratio)) {
    return false;
  }
  // Copy values to truncated repeated field.
  if (sizeof(FieldType) == sizeof(T)) {
    FieldType* dst_ptr =
        TypeHelper::AppendUninitialized(new_num_values, tensor);
    port::CopySubrangeToArray(tensor->tensor_content(), 0,
                              new_num_values * sizeof(T),
                              reinterpret_cast<char*>(dst_ptr));
    tensor->clear_tensor_content();
  } else if (sizeof(T) > 1) {
    // Copy raw bytes to temp array first, then cast.
    gtl::InlinedVector<T, 64> tmp(new_num_values);
    port::CopySubrangeToArray(tensor->tensor_content(), 0,
                              new_num_values * sizeof(T),
                              reinterpret_cast<char*>(tmp.data()));
    tensor->clear_tensor_content();
    const T* begin = tmp.begin();
    const T* end = tmp.end();
    TypeHelper::AddValues(begin, end, tensor);
  } else {
    // Copy and cast, one byte at a time.
    for (int64_t i = 0; i < new_num_values; ++i) {
      char c = tensor->tensor_content()[i];
      TypeHelper::AddValue(static_cast<T>(c), tensor);
    }
    tensor->clear_tensor_content();
  }
  return true;
}

template <typename T>
inline bool PackedValuesNotEqual(T a, T b) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_6(mht_6_v, 451, "", "./tensorflow/core/framework/tensor_util.cc", "PackedValuesNotEqual");

  return a != b;
}
template <>
inline bool PackedValuesNotEqual(float a, float b) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_7(mht_7_v, 458, "", "./tensorflow/core/framework/tensor_util.cc", "PackedValuesNotEqual");

  return reinterpret_cast<int32_t&>(a) != reinterpret_cast<int32_t&>(b);
}
template <>
inline bool PackedValuesNotEqual(double a, double b) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_8(mht_8_v, 465, "", "./tensorflow/core/framework/tensor_util.cc", "PackedValuesNotEqual");

  return reinterpret_cast<int64_t&>(a) != reinterpret_cast<int64_t&>(b);
}
template <typename RealType>
inline bool PackedValuesNotEqual(const std::complex<RealType>& a,
                                 const std::complex<RealType>& b) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_9(mht_9_v, 473, "", "./tensorflow/core/framework/tensor_util.cc", "PackedValuesNotEqual");

  return PackedValuesNotEqual(a.real(), b.real()) ||
         PackedValuesNotEqual(a.imag(), b.imag());
}

// Integer can't be negative zero.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
static bool IsNegativeZero(T value) {
  return false;
}

template <typename T,
          typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
static bool IsNegativeZero(T value) {
  return value == T(0) && std::signbit(value);
}

template <typename T>
static bool IsNegativeZero(std::complex<T> value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_10(mht_10_v, 495, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");

  return IsNegativeZero(value.real()) || IsNegativeZero(value.imag());
}

static bool IsNegativeZero(Eigen::QUInt8 value) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_11(mht_11_v, 502, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");
 return false; }
static bool IsNegativeZero(Eigen::QInt8 value) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_12(mht_12_v, 506, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");
 return false; }
static bool IsNegativeZero(Eigen::QUInt16 value) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_13(mht_13_v, 510, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");
 return false; }
static bool IsNegativeZero(Eigen::QInt16 value) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_14(mht_14_v, 514, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");
 return false; }
static bool IsNegativeZero(Eigen::QInt32 value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_15(mht_15_v, 518, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");
 return false; }
static bool IsNegativeZero(Eigen::half value) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_16(mht_16_v, 522, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");

  return IsNegativeZero<float>(value);
}
static bool IsNegativeZero(Eigen::bfloat16 value) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_17(mht_17_v, 528, "", "./tensorflow/core/framework/tensor_util.cc", "IsNegativeZero");

  return IsNegativeZero<float>(value);
}

template <typename T>
bool CompressRepeatedField(float min_compression_ratio,
                           const TensorShape& shape, TensorProto* tensor) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_18(mht_18_v, 537, "", "./tensorflow/core/framework/tensor_util.cc", "CompressRepeatedField");

  using TypeHelper = internal::TensorProtoHelper<T>;
  using FieldType = typename internal::TensorProtoHelper<T>::FieldType;
  const int64_t num_tensor_values = shape.num_elements();
  const int64_t num_proto_values = TypeHelper::NumValues(*tensor);

  // Notice that for complex types the tensor is stored as an array of up to
  // 2 * num_tensor_values real values (real and imaginary parts), possibly
  // truncated. A 0-splat does not need any value present and is maximally
  // compressed.
  if (num_proto_values == 0) return false;

  const T last_value = TypeHelper::GetValue(num_proto_values - 1, *tensor);
  int64_t last_index = 0;
  for (int64_t i = num_proto_values - 2; i >= 0 && last_index == 0; --i) {
    const T cur_value = TypeHelper::GetValue(i, *tensor);
    if (PackedValuesNotEqual(cur_value, last_value)) {
      last_index = i + 1;
    }
  }

  // Detect all zeroes tensors: this is default value and the content can be
  // erased entirely.
  if (last_index == 0 && last_value == T(0) && !IsNegativeZero(last_value)) {
    TypeHelper::Truncate(0, tensor);
    return true;
  }

  const int64_t num_truncated_proto_values = last_index + 1;
  const int64_t num_bytes_as_field =
      num_truncated_proto_values * sizeof(FieldType);
  const int64_t num_bytes_as_tensor_content = num_tensor_values * sizeof(T);
  const int64_t num_bytes_before = num_proto_values * sizeof(FieldType);
  if (std::min(num_bytes_as_field, num_bytes_as_tensor_content) >
      static_cast<int64_t>(num_bytes_before / min_compression_ratio)) {
    return false;
  }
  if (num_bytes_as_field <= num_bytes_as_tensor_content) {
    TypeHelper::Truncate(num_truncated_proto_values, tensor);
  } else {
    gtl::InlinedVector<T, 64> tmp;
    if (num_proto_values == 1) {
      // Splat case.
      tmp.resize(num_tensor_values, last_value);
    } else {
      tmp.resize(num_tensor_values, T(0));
      TypeHelper::CopyValues(tmp.begin(), *tensor);
    }
    TypeHelper::Truncate(0, tensor);
    port::CopyFromArray(tensor->mutable_tensor_content(),
                        reinterpret_cast<const char*>(tmp.data()),
                        num_bytes_as_tensor_content);
  }
  return true;
}

template <typename T>
bool CompressTensorProtoInPlaceImpl(int64_t min_num_elements,
                                    float min_compression_ratio,
                                    TensorProto* tensor) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPStensor_utilDTcc mht_19(mht_19_v, 599, "", "./tensorflow/core/framework/tensor_util.cc", "CompressTensorProtoInPlaceImpl");

  const TensorShape shape(tensor->tensor_shape());
  const int64_t num_tensor_values = shape.num_elements();
  if (num_tensor_values < min_num_elements) {
    return false;
  }
  if (tensor->tensor_content().empty()) {
    return CompressRepeatedField<T>(min_compression_ratio, shape, tensor);
  } else {
    return CompressTensorContent<T>(min_compression_ratio, shape, tensor);
  }
  return true;
}

}  // namespace internal

#define HANDLE_COMPRESS_CASE(TF_TYPE)                                  \
  case TF_TYPE:                                                        \
    return internal::CompressTensorProtoInPlaceImpl<                   \
        EnumToDataType<TF_TYPE>::Type>(min_num_elements,               \
                                       min_compression_ratio, tensor); \
    break

bool CompressTensorProtoInPlace(int64_t min_num_elements,
                                float min_compression_ratio,
                                TensorProto* tensor) {
  switch (tensor->dtype()) {
    HANDLE_COMPRESS_CASE(DT_FLOAT);
    HANDLE_COMPRESS_CASE(DT_DOUBLE);
    HANDLE_COMPRESS_CASE(DT_COMPLEX64);
    HANDLE_COMPRESS_CASE(DT_COMPLEX128);
    HANDLE_COMPRESS_CASE(DT_UINT8);
    HANDLE_COMPRESS_CASE(DT_INT8);
    HANDLE_COMPRESS_CASE(DT_UINT16);
    HANDLE_COMPRESS_CASE(DT_INT16);
    HANDLE_COMPRESS_CASE(DT_UINT32);
    HANDLE_COMPRESS_CASE(DT_INT32);
    HANDLE_COMPRESS_CASE(DT_UINT64);
    HANDLE_COMPRESS_CASE(DT_INT64);
    HANDLE_COMPRESS_CASE(DT_BOOL);
    HANDLE_COMPRESS_CASE(DT_QUINT8);
    HANDLE_COMPRESS_CASE(DT_QINT8);
    HANDLE_COMPRESS_CASE(DT_QUINT16);
    HANDLE_COMPRESS_CASE(DT_QINT16);
    HANDLE_COMPRESS_CASE(DT_QINT32);
    HANDLE_COMPRESS_CASE(DT_HALF);
    HANDLE_COMPRESS_CASE(DT_BFLOAT16);
    default:
      return false;
  }
}

#undef HANDLE_COMPRESS_CASE

Status MakeShape(const Tensor& shape, TensorShape* out) {
  if (!TensorShapeUtils::IsVector(shape.shape())) {
    return errors::InvalidArgument(
        "shape must be a vector of {int32,int64}, got shape ",
        shape.shape().DebugString());
  }
  if (shape.dtype() == DataType::DT_INT32) {
    auto vec = shape.flat<int32>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else if (shape.dtype() == DataType::DT_INT64) {
    auto vec = shape.flat<int64_t>();
    return TensorShapeUtils::MakeShape(vec.data(), vec.size(), out);
  } else {
    return errors::InvalidArgument("shape must be a vector of {int32,int64}.");
  }
}

}  // namespace tensor
}  // namespace tensorflow
