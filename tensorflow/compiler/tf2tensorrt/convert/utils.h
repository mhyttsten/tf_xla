/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
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
class MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh {
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
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh() {
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
#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_tensor_proxy.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

#define TFTRT_ERROR(func, ...)                                              \
  do {                                                                      \
    return func("TFTRT::", __FUNCTION__, ":", __LINE__, ": ", __VA_ARGS__); \
  } while (0)

#define TFTRT_CHECK_SHAPE_TENSOR(tensor)                                 \
  if (!IsTrtShapeTensorCompatible(tensor)) {                             \
    TFTRT_ERROR(errors::InvalidArgument, "Tensor of type ",              \
                DebugString(tensor.dtype()), " having shape ",           \
                tensor.shape().DebugString(), " is not TRT compatible"); \
  }

namespace tensorflow {
namespace tensorrt {

static constexpr char kCastOutputTypeAttrName[] = "DstT";

#if !IS_TRT_VERSION_GE(8, 2, 0, 0)
template <typename T>
struct TrtDestroyer {
  void operator()(T* t) {
    if (t) t->destroy();
  }
};
template <typename T>
using TrtUniquePtrType = std::unique_ptr<T, TrtDestroyer<T>>;
#else
template <typename T>
using TrtUniquePtrType = std::unique_ptr<T>;
#endif

// Define a hash function for vector<TensorShape> because it is used as the key
// for the engine cache.
struct VectorTensorShapeHasher {
  std::size_t operator()(const std::vector<TensorShape>& key) const {
    return std::hash<std::string>()(TensorShapeUtils::ShapeListString(key));
  }
};

using absl::StrAppend;
using absl::StrCat;

// This utility template converts an arithmetic type to a string. This function
// is necessary to allow the following function to behave recursively:
// `string DebugString(const std::vector<CType>&)`.
template <typename CType, typename = typename std::enable_if<
                              std::is_arithmetic<CType>::value, CType>::type>
string DebugString(const CType& el) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_0(mht_0_v, 255, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DebugString");

  string el_str = std::to_string(el);
  // Prettify std::to_string which can sometimes returns 1.50000 instead of 1.5.
  // In short it removes trailing 0s in a string-formatted number.
  el_str.erase(el_str.find_last_not_of('0') + 1, std::string::npos);
  return el_str;
}
// This utility template converts nested vectors to a string for debug purposes.
template <typename CType>
string DebugString(const std::vector<CType>& vector) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_1(mht_1_v, 267, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DebugString");

  string tmp_s = "";
  for (const auto el : vector) {
    StrAppend(&tmp_s, StrCat(DebugString(el), ", "));
  }
  return StrCat("{", tmp_s.substr(0, tmp_s.length() - 2), "}");
}
string DebugString(const nvinfer1::Dims& dims);
string DebugString(const nvinfer1::DataType trt_dtype);
string DebugString(const DataType tf_type);
string DebugString(const nvinfer1::Permutation& permutation, int len);
string DebugString(const ITensorProxyPtr& tensor);
string DebugString(const nvinfer1::ITensor& tensor);
string DebugString(const std::vector<nvinfer1::Dims>& dimvec);
string DebugString(const std::vector<TensorShape>& shapes);
string DebugString(const std::vector<PartialTensorShape>& shapes);

template <size_t N>
string DebugString(const absl::InlinedVector<int64, N>& data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_2(mht_2_v, 288, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DebugString");

  return absl::StrCat("[", absl::StrJoin(data, ","), "]");
}

inline bool HasStaticShape(const nvinfer1::Dims& dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_3(mht_3_v, 295, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "HasStaticShape");

  if (dims.nbDims < 0) return false;
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] < 0) return false;
  }
  return true;
}

template <typename T>
bool HasStaticShape(const T& dims) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_4(mht_4_v, 307, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "HasStaticShape");

  return !absl::c_any_of(dims, [](int i) { return i < 0; });
}

// Returns whether a shape is compatible with a TRT shape tensor.
template <typename TensorShapeType>
inline bool IsTrtShapeTensorCompatible(const TensorShapeType& shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_5(mht_5_v, 316, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "IsTrtShapeTensorCompatible");

  return (
      shape.dims() == 0 ||
      (shape.dims() == 1 && shape.num_elements() <= nvinfer1::Dims::MAX_DIMS));
}

// Returns whether a TF tensor could be interpreted as a TRT shape tensor.
inline bool IsTrtShapeTensorCompatible(const Tensor& tensor) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_6(mht_6_v, 326, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "IsTrtShapeTensorCompatible");

  return tensor.dtype() == DT_INT32 &&
         IsTrtShapeTensorCompatible(tensor.shape());
}

// Adapts various representations of shape (TF Shape, TRT Dims, plain
// containers) and provides methods for properties (length, volume) and
// conversion between types. Note that unlike TF's TensorShape, the underlying
// storage will only contain active dimensions. In the case of scalar shapes,
// `NumDims` is allowed to return 0 or 1, but the `storage_` vector will contain
// 1 element in both cases. In the non-scalar case, `NumDims() ==
// storage_.size()`.
class DimsAdapter {
 public:
  using StorageType = absl::InlinedVector<int64_t, 4>;

 private:
  template <typename T>
  using EnableIfNotTensorShapeType =
      std::enable_if_t<!std::is_base_of<TensorShapeBase<T>, T>::value>;

  template <typename T>
  using EnableIfInt = std::enable_if_t<std::is_arithmetic<T>::value &&
                                       std::is_integral<T>::value>;

 public:
  //----- Constructors ------

  // Constructs from an absl::Span.
  template <typename T>
  explicit DimsAdapter(absl::Span<T> shape)
      : num_dims_(static_cast<int32_t>(shape.size())) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_7(mht_7_v, 360, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DimsAdapter");

    absl::c_copy(shape, std::back_inserter(storage_));
  }

  // Constructs from an absl::Span.
  template <typename T>
  explicit DimsAdapter(const std::vector<T>& shape)
      : num_dims_(static_cast<int32_t>(shape.size())) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_8(mht_8_v, 370, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DimsAdapter");

    absl::c_copy(shape, std::back_inserter(storage_));
  }

  // Constructs from a TRT dims object.
  DimsAdapter(const nvinfer1::Dims& dims) : num_dims_(dims.nbDims) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_9(mht_9_v, 378, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DimsAdapter");

    absl::c_copy(absl::MakeSpan(dims.d, dims.d + std::max(dims.nbDims, 0)),
                 std::back_inserter(storage_));
  }

  // Constructs explicitly specifing num_dims and storage data.
  DimsAdapter(int32_t num_dims, StorageType data)
      : num_dims_(num_dims), storage_(std::forward<StorageType>(data)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_10(mht_10_v, 388, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DimsAdapter");
}

  // Constructs from a TensorShape or PartialTensorShape.
  template <typename T>
  static StatusOr<DimsAdapter> Create(const TensorShapeBase<T>& shape,
                                      bool ignore_first_dim = false) {
    if (shape.dims() > nvinfer1::Dims::MAX_DIMS)
      return errors::InvalidArgument("dims of TensorShape exceed MAX_DIMS");
    if (ignore_first_dim && shape.dims() <= 0)
      return errors::InvalidArgument(
          "removing first dim requires explicit batch dimension");
    if (shape.dims() == -1) {
      return DimsAdapter(-1, StorageType{});
    }
    if (shape.dims() == 0) {
      return DimsAdapter(0, StorageType{1});
    }
    auto offt = (ignore_first_dim ? 1 : 0);
    return DimsAdapter(
        absl::MakeSpan(shape.dim_sizes().begin() + offt, shape.dims() - offt));
  }

  // Constructs from a container.
  template <typename InputSequence,
            typename = EnableIfNotTensorShapeType<InputSequence>>
  static StatusOr<DimsAdapter> Create(const InputSequence& shape,
                                      bool ignore_first_dim = false) {
    if (ignore_first_dim && shape.size() <= 0) {
      return errors::InvalidArgument(
          "removing first dim requires explicit batch dimension");
    }
    return DimsAdapter(
        absl::MakeSpan(shape).subspan(ignore_first_dim ? 1 : 0, shape.size()));
  }

  //----- Conversion Utilities ------

  //  Converts to an nvinfers::Dims and assign the result to the object passed
  //  in via the result pointer.
  void TrtDims(nvinfer1::Dims* result) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_11(mht_11_v, 430, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "TrtDims");

    result->nbDims = num_dims_;
    absl::c_copy(storage_, static_cast<int32_t*>(result->d));
  }

  // Converts to an nvinfer1::Dims and return by value.
  nvinfer1::Dims AsTrtDims() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_12(mht_12_v, 439, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "AsTrtDims");

    nvinfer1::Dims result;
    TrtDims(&result);
    return result;
  }

  // Converts to a TensorShape and assigns the result to the object passed in
  // via the shape pointer.
  Status TensorShape(TensorShape* shape,
                     absl::optional<int> batch_size = absl::nullopt) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_13(mht_13_v, 451, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "TensorShape");

    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
        static_cast<const int64_t*>(storage_.data()), storage_.size(), shape));
    if (batch_size) shape->InsertDim(0, *batch_size);
    return Status::OK();
  }

  // Converts to a PartialTensorShape and assigns the result to the object
  // passed in via the shape pointer.
  Status PartialTensorShape(
      PartialTensorShape* shape,
      absl::optional<int> batch_size = absl::nullopt) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_14(mht_14_v, 465, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "PartialTensorShape");

    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
        static_cast<const int64_t*>(storage_.data()), storage_.size(), shape));
    if (batch_size) shape->InsertDim(0, *batch_size);
    return Status::OK();
  }

  // Copies the dimension values to the vector passed in via the shape pointer.
  template <typename T, typename = EnableIfInt<T>>
  Status Vector(std::vector<T>* shape) const {
    shape->clear();
    absl::c_copy(storage_, std::back_inserter(*shape));
    return Status::OK();
  }

  //----- Property Accessors ------

  // Returns true if the shape has no dynamic dimensions.
  bool IsStatic() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_15(mht_15_v, 486, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "IsStatic");

    return !absl::c_any_of(storage_, [](auto i) { return i < 0; });
  }

  // Returns product of all dimensions.
  int64_t Volume() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_16(mht_16_v, 494, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "Volume");

    return absl::c_accumulate(storage_, static_cast<int64_t>(1),
                              std::multiplies<>());
  }

  int32_t NumDims() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_17(mht_17_v, 502, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "NumDims");
 return num_dims_; }

  // Returns true if the shape should be interpreted as a scalar. This follows
  // TensorRT conversions: a scalar shape can have NumDims()==1 or NumDims()==0,
  // but the underlying storage_ container has a single dimension of size 1.
  bool IsScalar() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_18(mht_18_v, 510, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "IsScalar");

    return (num_dims_ == 0 || num_dims_ == 1) && storage_.size() == 1 &&
           storage_[0] == 1;
  }

  // Returns true if the dimension storage is empty. This indicates an empty
  // shape in both the scalar and non-scalar case.
  bool IsEmpty() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_19(mht_19_v, 520, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "IsEmpty");
 return storage_.empty(); }

  string DebugString() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_20(mht_20_v, 525, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "DebugString");

    auto vol = absl::c_accumulate(storage_, static_cast<int64_t>(1),
                                  std::multiplies<>());
    return absl::StrCat("DimsAdapter(num_dims=", num_dims_, ",shape=[",
                        absl::StrJoin(storage_, ","), "],", "vol=", vol, ")");
  }

  // Returns beginning iterator for the underlying storage.
  StorageType::const_iterator begin() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_21(mht_21_v, 536, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "begin");
 return storage_.begin(); }

  // Returns ending iterator for the underlying storage.
  StorageType::const_iterator end() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_22(mht_22_v, 542, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "end");
 return storage_.end(); }

  // Returns the size of the dimension at `idx`.
  StorageType::value_type dim(size_t idx) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_23(mht_23_v, 548, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "dim");
 return storage_[idx]; }

  // Returns a references to the dimension at `idx`.
  StorageType::value_type& dim(size_t idx) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_24(mht_24_v, 554, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "dim");
 return storage_[idx]; }

  //----- Non-Const Operators ------

  DimsAdapter& Append(int32_t dim) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_25(mht_25_v, 561, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "Append");

    StatusOr<bool> is_scalar = IsScalar();
    if (!is_scalar.ok()) return *this;
    num_dims_ = *is_scalar ? 2 : num_dims_ + 1;
    storage_.push_back(dim);
    return *this;
  }

  DimsAdapter& Prepend(absl::optional<int32_t> dim) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_26(mht_26_v, 572, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "Prepend");

    if (dim) {
      num_dims_ = IsScalar() ? 2 : num_dims_ + 1;
      storage_.insert(storage_.begin(), *dim);
    }
    return *this;
  }

  Status RemoveBatchDimension() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPStf2tensorrtPSconvertPSutilsDTh mht_27(mht_27_v, 583, "", "./tensorflow/compiler/tf2tensorrt/convert/utils.h", "RemoveBatchDimension");

    if (storage_.empty())
      return errors::InvalidArgument(
          "attempted to remove batch dim from scalar");
    num_dims_ -= 1;
    storage_.erase(storage_.begin());
    return Status::OK();
  }

  //----- Comparison Operators ------

  bool operator==(const DimsAdapter& rhs) const {
    if (rhs.num_dims_ != num_dims_) return false;
    for (int i = 0; i < num_dims_; i++) {
      if (rhs.storage_[i] != storage_[i]) return false;
    }
    return true;
  }

  bool operator!=(const DimsAdapter& rhs) const { return !(*this == rhs); }

 private:
  int32_t num_dims_{0};
  StorageType storage_{};
};

Status GetNetworkInputShapes(const nvinfer1::INetworkDefinition* network,
                             std::vector<PartialTensorShape>* input_shapes);

Status TfTypeToTrtType(DataType tf_type, nvinfer1::DataType* trt_type);
Status TrtTypeToTfType(nvinfer1::DataType trt_type, DataType* tf_type);

// Returns true if an engine built for cached_shapes can also run actual_shapes.
bool AreShapesCompatible(const std::vector<TensorShape>& actual_shapes,
                         const std::vector<TensorShape>& cached_shapes);

// Returns the number of inputs for the engine, which also correspends to the
// number of input tensors for the network. This can differ from the number of
// input bindings, because the number of total input bindings equals the number
// of profiles times the number of engine inputs.
int GetNumberOfEngineInputs(const nvinfer1::ICudaEngine* engine);

// Returns the string representation for the assigned device or the requested
// device of the given node.
absl::string_view GetDeviceName(const Node* node);

// Returns the ParsedName representation for the assigned device or the
// requested device string of the given node. If the device string is invalid,
// returns absl::nullopt.
absl::optional<DeviceNameUtils::ParsedName> GetDeviceParsedName(
    const Node* node);

// If the given two device assignments as compatible, returns the merge of the
// two assignments. Otherwise, returns absl::nullopt.
absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, const DeviceNameUtils::ParsedName& b);
// Similar to the above, except that the second device assignment is represented
// by a string_view.
absl::optional<DeviceNameUtils::ParsedName> MergeIfCompatible(
    const DeviceNameUtils::ParsedName& a, absl::string_view b);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_UTILS_H_
