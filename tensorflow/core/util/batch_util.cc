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
class MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc() {
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

#include "tensorflow/core/util/batch_util.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

#define TF_CALL_DATASET_TYPES(m) TF_CALL_ALL_TYPES(m) TF_CALL_QUANTIZED_TYPES(m)

namespace tensorflow {
namespace batch_util {

namespace {

Status ValidateInput(const Tensor& parent, const Tensor& element,
                     int64_t index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/util/batch_util.cc", "ValidateInput");

  DCHECK_NE(parent.dim_size(0), 0);
  DCHECK_GE(index, 0);
  if (element.NumElements() != (parent.NumElements() / parent.dim_size(0))) {
    TensorShape chip_shape = parent.shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "ValidateInput Cannot perform copy: number of elements does not match. "
        " Shapes are: [element]: ",
        element.shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  return Status::OK();
}

template <typename T>
Status HandleElementToSlice(const Tensor& /* element */, T* src, T* dest,
                            int64_t num_values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/util/batch_util.cc", "HandleElementToSlice");

  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
  return Status::OK();
}

template <>
Status HandleElementToSlice<tstring>(const Tensor& element, tstring* src,
                                     tstring* dest, int64_t num_values) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_2(mht_2_v, 230, "", "./tensorflow/core/util/batch_util.cc", "HandleElementToSlice<tstring>");

  if (element.RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      *dest++ = std::move(*src++);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
  return Status::OK();
}

template <>
Status HandleElementToSlice<Variant>(const Tensor& element, Variant* src,
                                     Variant* dest, int64_t num_values) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/util/batch_util.cc", "HandleElementToSlice<Variant>");

  if (element.RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      *dest++ = std::move(*src++);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
  return Status::OK();
}

template <>
Status HandleElementToSlice<ResourceHandle>(const Tensor& /* element */,
                                            ResourceHandle* src,
                                            ResourceHandle* dest,
                                            int64_t num_values) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_4(mht_4_v, 264, "", "./tensorflow/core/util/batch_util.cc", "HandleElementToSlice<ResourceHandle>");

  std::copy_n(src, num_values, dest);
  return Status::OK();
}

template <>
Status HandleElementToSlice<Eigen::half>(const Tensor& /* element */,
                                         Eigen::half* src, Eigen::half* dest,
                                         int64_t num_values) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/util/batch_util.cc", "HandleElementToSlice<Eigen::half>");

  std::copy_n(src, num_values, dest);
  return Status::OK();
}

template <typename T>
void HandleSliceToElement(const T* src, T* dest, int64_t num_values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement");

  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
}

template <>
void HandleSliceToElement<tstring>(const tstring* src, tstring* dest,
                                   int64_t num_values) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_7(mht_7_v, 294, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<tstring>");

  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<Variant>(const Variant* src, Variant* dest,
                                   int64_t num_values) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_8(mht_8_v, 303, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<Variant>");

  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<ResourceHandle>(const ResourceHandle* src,
                                          ResourceHandle* dest,
                                          int64_t num_values) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_9(mht_9_v, 313, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<ResourceHandle>");

  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<Eigen::half>(const Eigen::half* src,
                                       Eigen::half* dest, int64_t num_values) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_10(mht_10_v, 322, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<Eigen::half>");

  std::copy_n(src, num_values, dest);
}

template <typename T>
void HandleSliceToElement(Tensor* parent, T* src, T* dest, int64_t num_values) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_11(mht_11_v, 330, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement");

  static_assert(is_simple_type<T>::value, "Memcpy requires a simple type.");
  memcpy(dest, src, num_values * sizeof(T));
}

template <>
void HandleSliceToElement<tstring>(Tensor* parent, tstring* src, tstring* dest,
                                   int64_t num_values) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_12(mht_12_v, 340, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<tstring>");

  if (parent->RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      dest[i] = std::move(src[i]);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
}

template <>
void HandleSliceToElement<Variant>(Tensor* parent, Variant* src, Variant* dest,
                                   int64_t num_values) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_13(mht_13_v, 355, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<Variant>");

  if (parent->RefCountIsOne()) {
    for (int64_t i = 0; i < num_values; ++i) {
      dest[i] = std::move(src[i]);
    }
  } else {
    std::copy_n(src, num_values, dest);
  }
}

template <>
void HandleSliceToElement<ResourceHandle>(Tensor* parent, ResourceHandle* src,
                                          ResourceHandle* dest,
                                          int64_t num_values) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_14(mht_14_v, 371, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<ResourceHandle>");

  std::copy_n(src, num_values, dest);
}

template <>
void HandleSliceToElement<Eigen::half>(Tensor* parent, Eigen::half* src,
                                       Eigen::half* dest, int64_t num_values) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_15(mht_15_v, 380, "", "./tensorflow/core/util/batch_util.cc", "HandleSliceToElement<Eigen::half>");

  std::copy_n(src, num_values, dest);
}

}  // namespace

// Copies element into the index^th slice of parent (in the 0th dimension).
Status CopyElementToSlice(Tensor element, Tensor* parent, int64_t index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_16(mht_16_v, 390, "", "./tensorflow/core/util/batch_util.cc", "CopyElementToSlice");

  TF_RETURN_IF_ERROR(ValidateInput(*parent, element, index));
  const int64_t num_values = element.NumElements();
#define HANDLE_TYPE(T)                                              \
  case DataTypeToEnum<T>::value: {                                  \
    T* src = element.base<T>();                                     \
    T* dest = parent->base<T>() + (num_values * index);             \
    return HandleElementToSlice<T>(element, src, dest, num_values); \
  }

  switch (element.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopyElementToSlice Unhandled data type: ",
                                   element.dtype());
  }
}

// Copies the index^th slice of parent (in the 0th dimension) into element.
Status CopySliceToElement(const Tensor& parent, Tensor* element,
                          int64_t index) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_17(mht_17_v, 415, "", "./tensorflow/core/util/batch_util.cc", "CopySliceToElement");

  TF_RETURN_IF_ERROR(ValidateInput(parent, *element, index));
  const int64_t num_values = element->NumElements();

#define HANDLE_TYPE(T)                                      \
  case DataTypeToEnum<T>::value: {                          \
    const T* src = parent.base<T>() + (num_values * index); \
    T* dest = element->base<T>();                           \
    HandleSliceToElement<T>(src, dest, num_values);         \
    return Status::OK();                                    \
  }

  switch (parent.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopySliceToElement Unhandled data type: ",
                                   element->dtype());
  }
}

Status CopyContiguousSlices(const Tensor& src, int64_t src_offset,
                            int64_t dst_offset, int64_t num_slices,
                            Tensor* dst) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_18(mht_18_v, 442, "", "./tensorflow/core/util/batch_util.cc", "CopyContiguousSlices");

  if (src.dtype() != dst->dtype()) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: src and dst have different "
        "dtypes. Source dtype: ",
        src.dtype(), " dstination dtype: ", dst->dtype(), ".");
  }
  if (src.dims() < 1) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: src has to be a tensor with "
        "rank >= 1. Source shape: ",
        src.shape().DebugString());
  }

  if (dst->dims() < 1) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: dst has to be a tensor "
        "with rank >= 1. Dest shape: ",
        dst->shape().DebugString());
  }

  const int64_t src_dim0 = src.dim_size(0);
  const int64_t dst_dim0 = dst->dim_size(0);
  int64_t src_chip_size = 1;
  int64_t dst_chip_size = 1;
  for (int i = 1; i < src.dims(); ++i) {
    src_chip_size *= src.dim_size(i);
  }
  for (int i = 1; i < dst->dims(); ++i) {
    dst_chip_size *= dst->dim_size(i);
  }

  if (src_chip_size != dst_chip_size) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: source and dst shapes are"
        "not compatible. Source shape: ",
        src.shape().DebugString(), ", dst shape: ", dst->shape().DebugString());
  }

  if (src_chip_size == 0 && dst_chip_size == 0) {
    return Status::OK();
  }

  if (src_offset < 0 || src_offset + num_slices > src_dim0 || dst_offset < 0 ||
      dst_offset + num_slices > dst_dim0) {
    return errors::FailedPrecondition(
        "CopyContiguousSlices cannot perform copy: index out of range. "
        "src_offset: ",
        src_offset, ", num_slices: ", num_slices, ", src_dim0: ", src_dim0,
        ", dst_offset: ", dst_offset, ", dst_dim0: ", dst_dim0, ".");
  }

#define HANDLE_TYPE(T)                                                 \
  case DataTypeToEnum<T>::value: {                                     \
    const T* src_p = src.base<T>() + (src_chip_size * src_offset);     \
    T* dst_p = dst->base<T>() + (dst_chip_size * dst_offset);          \
    HandleSliceToElement<T>(src_p, dst_p, src_chip_size * num_slices); \
    return Status::OK();                                               \
  }

  switch (src.dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented("CopyContiguousSlices unhandled data type: ",
                                   src.dtype());
  }
}

// Copies the index^th slice of parent (in the 0th dimension) into element.
//
// NOTE(mrry): The implementation may be able to optimize the copy to a move.
// This is particularly important for DT_STRING tensors.
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64_t index) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_19(mht_19_v, 519, "", "./tensorflow/core/util/batch_util.cc", "MaybeMoveSliceToElement");

  TF_RETURN_IF_ERROR(ValidateInput(*parent, *element, index));
  const int64_t num_values = element->NumElements();

#define HANDLE_TYPE(T)                                      \
  case DataTypeToEnum<T>::value: {                          \
    T* src = parent->base<T>() + (num_values * index);      \
    T* dest = element->base<T>();                           \
    HandleSliceToElement<T>(parent, src, dest, num_values); \
    return Status::OK();                                    \
  }

  switch (parent->dtype()) {
    TF_CALL_ALL_TYPES(HANDLE_TYPE);
    TF_CALL_QUANTIZED_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "MaybeMoveSliceToElement Unhandled data type: ", element->dtype());
  }
}

// The following five functions are copied from padding_fifo_queue.cc.
// TODO(mrry): Reconcile these functions with the similar methods in the
// queue implementation.
Status ValidateElementToLargerSlice(const Tensor& element, Tensor* parent) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_20(mht_20_v, 547, "", "./tensorflow/core/util/batch_util.cc", "ValidateElementToLargerSlice");

  DCHECK_NE(parent->dim_size(0), 0);
  if (element.NumElements() > (parent->NumElements() / parent->dim_size(0))) {
    TensorShape chip_shape = parent->shape();
    chip_shape.RemoveDim(0);
    return errors::Internal(
        "HandleElementToLargerSlice Cannot copy slice: number of entries in "
        "element is greater than number of elements in parent slice.  ",
        "Shapes are: [element]: ", element.shape().DebugString(),
        ", [parent slice]: ", chip_shape.DebugString());
  }
  return Status::OK();
}

template <typename T, int NDIMS>
Status HandleElementToLargerSlice(const Tensor& element, Tensor* parent,
                                  int index) {
  TF_RETURN_IF_ERROR(ValidateElementToLargerSlice(element, parent));
  if (element.NumElements() == 0) {
    return Status::OK();
  }
  auto element_t = element.tensor<T, NDIMS>();
  auto parent_t = parent->tensor<T, NDIMS + 1>();
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_indices;
  slice_indices[0] = index;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_size;
  slice_size[0] = 1;
  for (size_t i = 1; i < slice_size.size(); ++i) {
    slice_size[i] = element_t.dimension(i - 1);
  }
  parent_t.slice(slice_indices, slice_size) = element_t.reshape(slice_size);
  return Status::OK();
}

template <int NDIMS>
Status HandleElementToLargerSliceWithRank(const Tensor& element, Tensor* parent,
                                          int index) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_21(mht_21_v, 586, "", "./tensorflow/core/util/batch_util.cc", "HandleElementToLargerSliceWithRank");

#define HANDLE_TYPE(T)                                                   \
  case DataTypeToEnum<T>::value: {                                       \
    return HandleElementToLargerSlice<T, NDIMS>(element, parent, index); \
  }

  switch (element.dtype()) {
    TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "HandleElementToLargerSliceWithRank Unhandled data type: ",
          element.dtype());
  }
}

Status CopyElementToLargerSlice(const Tensor& element, Tensor* parent,
                                int index) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_22(mht_22_v, 606, "", "./tensorflow/core/util/batch_util.cc", "CopyElementToLargerSlice");

  if (parent->dims() != element.dims() + 1) {
    return errors::Internal(
        "Mismatched ranks.  Element's rank is: ", element.dims(),
        " but element is meant to be a slice in output Tensor having rank: ",
        parent->dims(), " (should be: ", element.dims() + 1, ")");
  }

#define HANDLE_DIMS(NDIMS)                                                  \
  case NDIMS: {                                                             \
    TF_RETURN_IF_ERROR(                                                     \
        HandleElementToLargerSliceWithRank<NDIMS>(element, parent, index)); \
    return Status::OK();                                                    \
  }

  switch (element.dims()) {
    HANDLE_DIMS(0);
    HANDLE_DIMS(1);
    HANDLE_DIMS(2);
    HANDLE_DIMS(3);
    HANDLE_DIMS(4);
    HANDLE_DIMS(5);
#undef HANDLE_DIMS
    default:
      return errors::Unimplemented("CopyElementToLargerSlice Unhandled rank: ",
                                   element.dims());
  }
}

Status SetElementZero(Tensor* element, const Tensor& padding) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSutilPSbatch_utilDTcc mht_23(mht_23_v, 638, "", "./tensorflow/core/util/batch_util.cc", "SetElementZero");

#define HANDLE_TYPE(T)                                     \
  if (element->dtype() == DataTypeToEnum<T>::value) {      \
    element->flat<T>().setConstant(padding.scalar<T>()()); \
    return Status::OK();                                   \
  }
  TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
  return errors::Unimplemented("SetElementZero Unhandled data type: ",
                               element->dtype());
}

}  // namespace batch_util
}  // namespace tensorflow
