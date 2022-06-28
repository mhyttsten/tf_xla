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
class MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc() {
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

#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace sparse {

namespace {

int UnsafeGetDimsFromIx(const Tensor& ix) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "UnsafeGetDimsFromIx");

  DCHECK(TensorShapeUtils::IsMatrix(ix.shape()));
  return ix.dim_size(1);
}

Status GetDimsFromIx(const Tensor& ix, int* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "GetDimsFromIx");

  if (!TensorShapeUtils::IsMatrix(ix.shape())) {
    return errors::InvalidArgument("indices must be a matrix, but got: ",
                                   ix.shape().DebugString());
  }
  *result = UnsafeGetDimsFromIx(ix);
  return Status();
}

}  // namespace

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const VarDimArray shape,
                                         const VarDimArray order,
                                         SparseTensor* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::Create");

  if (ix.dtype() != DT_INT64) {
    return errors::InvalidArgument("indices must be type int64 but got: ",
                                   ix.dtype());
  }
  if (!TensorShapeUtils::IsVector(vals.shape())) {
    return errors::InvalidArgument("vals must be a vec, but got: ",
                                   vals.shape().DebugString());
  }
  if (ix.shape().dim_size(0) != vals.shape().dim_size(0)) {
    return errors::InvalidArgument(
        "indices and values rows (indexing "
        "dimension) must match. (indices = ",
        ix.shape().dim_size(0), ", values = ", vals.shape().dim_size(0), ")");
  }
  int dims = 0;
  TF_RETURN_IF_ERROR(GetDimsFromIx(ix, &dims));
  if (order.size() != dims) {
    return errors::InvalidArgument("Order length must be SparseTensor rank.");
  }
  if (shape.size() != dims) {
    return errors::InvalidArgument("Shape rank must be SparseTensor rank.");
  }

  result->ix_ = std::move(ix);
  result->vals_ = std::move(vals);
  result->shape_.assign(shape.begin(), shape.end());
  result->order_.assign(order.begin(), order.end());
  result->dims_ = dims;
  return Status::OK();
}

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const TensorShape& shape,
                                         SparseTensor* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::Create");

  return Create(std::move(ix), std::move(vals), TensorShapeToVector(shape),
                UndefinedOrder(TensorShapeToVector(shape)), result);
}

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const VarDimArray shape,
                                         SparseTensor* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_4(mht_4_v, 266, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::Create");

  return Create(std::move(ix), std::move(vals), shape, UndefinedOrder(shape),
                result);
}

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const TensorShape& shape,
                                         const VarDimArray order,
                                         SparseTensor* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::Create");

  return Create(std::move(ix), std::move(vals), TensorShapeToVector(shape),
                order, result);
}

SparseTensor::SparseTensor(Tensor ix, Tensor vals, const VarDimArray shape,
                           const VarDimArray order)
    : ix_(std::move(ix)),
      vals_(std::move(vals)),
      shape_(shape.begin(), shape.end()),
      order_(order.begin(), order.end()),
      dims_(UnsafeGetDimsFromIx(ix_)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_6(mht_6_v, 291, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::SparseTensor");

  DCHECK_EQ(ix_.dtype(), DT_INT64)
      << "indices must be type int64 but got: " << ix_.dtype();
  DCHECK(TensorShapeUtils::IsVector(vals_.shape()))
      << "vals must be a vec, but got: " << vals_.shape().DebugString();
  DCHECK_EQ(ix_.shape().dim_size(0), vals_.shape().dim_size(0))
      << "indices and values rows (indexing dimension) must match.";
  DCHECK_EQ(order.size(), dims_) << "Order length must be SparseTensor rank.";
  DCHECK_EQ(shape.size(), dims_) << "Shape rank must be SparseTensor rank.";
}

// Optimized version of `IndicesValid()` with the following requirements:
// * The sparse tensor is one-dimensional.
//
// Returns true if the indices are valid, otherwise false.
// NOTE(mrry): If this method returns false, call IndicesValidHelper<true>()
// to obtain a meaningful error message.
bool SparseTensor::IndicesValidVectorFastPath() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_7(mht_7_v, 311, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::IndicesValidVectorFastPath");

  DCHECK_EQ(shape_.size(), 1);
  DCHECK_EQ(order_[0], 0);

  const int64_t max_index = shape_[0];

  // We maintain separate bools for each validation predicate to enable
  // vectorization across loop iterations.
  bool index_in_range_valid = true;
  bool order_valid = true;

  int64_t prev_index = -1;
  const auto ix_t = ix_.matrix<int64_t>();
  const int64_t* const index_base_ptr = ix_t.data();

  for (std::size_t n = 0; n < ix_t.dimension(0); ++n) {
    const int64_t index = index_base_ptr[n];
    index_in_range_valid = index_in_range_valid & (index < max_index);
    order_valid = order_valid & (index > prev_index);
    prev_index = index;
  }

  return index_in_range_valid & order_valid;
}

// Optimized version of `IndicesValid()` with the following requirements:
// * The sparse tensor is two-dimensional.
// * The tensor's indices are in the "standard" (lexicographic) order.
// * All of the tensor's indices fit within the range of a signed int32.
//
// Returns true if the indices are valid, otherwise false.
// NOTE(mrry): If this method returns false, call IndicesValidHelper<true>()
// to obtain a meaningful error message.
bool SparseTensor::IndicesValidMatrix32BitFastPath() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_8(mht_8_v, 347, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::IndicesValidMatrix32BitFastPath");

  const auto ix_t = ix_.matrix<int64_t>();
  const int64_t* const shape_ptr = shape_.data();

  DCHECK_EQ(shape_.size(), 2);
  DCHECK_EQ(order_[0], 0);
  DCHECK_EQ(order_[1], 1);
  DCHECK_LE(shape_ptr[0], std::numeric_limits<int32>::max());
  DCHECK_LE(shape_ptr[1], std::numeric_limits<int32>::max());

  const int32_t max_rows = static_cast<int32>(shape_ptr[0]);
  const int32_t max_cols = static_cast<int32>(shape_ptr[1]);

  // We maintain separate bools for each validation predicate to enable
  // vectorization across loop iterations.
  bool row_zeros_valid = true;
  bool row_in_range_valid = true;
  bool col_zeros_valid = true;
  bool col_in_range_valid = true;
  bool order_valid = true;

  int64_t prev_index = -1;

  // Points to the beginning of the current row of the indices matrix.
  // Each row has two int64 elements, but we use an int32 pointer to access
  // the low and high 32 bits of each element separately. This means that our
  // stride per row is 4 elements.
  const int32* const index_base_ptr =
      reinterpret_cast<const int32*>(ix_t.data());
  const size_t kInt32ElementsPerRow = 4;

  for (std::size_t n = 0; n < ix_t.dimension(0); ++n) {
    const int32* const index_ptr = index_base_ptr + n * kInt32ElementsPerRow;

    // Unpack the values on the current row of the indices matrix.
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    const int32 row_zeros = index_ptr[0];
    const int32 row_32 = index_ptr[1];
    const int32 col_zeros = index_ptr[2];
    const int32 col_32 = index_ptr[3];
#else
    const int32_t row_32 = index_ptr[0];
    const int32_t row_zeros = index_ptr[1];
    const int32_t col_32 = index_ptr[2];
    const int32_t col_zeros = index_ptr[3];
#endif

    // Validate that the high 32 bits of the row and column indices are zero.
    row_zeros_valid = row_zeros_valid & (row_zeros == 0);
    col_zeros_valid = col_zeros_valid & (col_zeros == 0);

    // Validate that the low 32 bits of the row and column indices are within
    // range of the shape.
    row_in_range_valid =
        row_in_range_valid & (row_32 >= 0) & (row_32 < max_rows);
    col_in_range_valid =
        col_in_range_valid & (col_32 >= 0) & (col_32 < max_cols);

    // Interpret the row and column as a concatenated 64-bit integer, and
    // validate that the concatenated indices are in strictly increasing order.
    const int64_t concatenated_index =
        (static_cast<int64_t>(row_32) << 32) + col_32;
    order_valid = order_valid & (concatenated_index > prev_index);
    prev_index = concatenated_index;
  }

  return row_zeros_valid & row_in_range_valid & col_zeros_valid &
         col_in_range_valid & order_valid;
}

template <bool standard_order>
Status SparseTensor::IndicesValidHelper() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_9(mht_9_v, 421, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::IndicesValidHelper");

  const auto ix_t = ix_.matrix<int64_t>();
  const int64_t* const shape_ptr = shape_.data();

  for (std::size_t n = 0; n < num_entries(); ++n) {
    bool valid = true;
    bool different = false;
    bool increasing = true;
    if (n == 0) {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_ptr[di]) valid = false;
      }
      different = true;
    } else {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_ptr[di]) valid = false;
        int ordered_dim;
        if (standard_order) {
          ordered_dim = di;
        } else {
          ordered_dim = order_[di];
        }
        int64_t diff = ix_t(n, ordered_dim) - ix_t(n - 1, ordered_dim);
        if (diff > 0) different = true;
        if (!different && diff < 0) increasing = false;
      }
    }
    if (TF_PREDICT_FALSE(!valid || !increasing || !different)) {
      string index = strings::StrCat("indices[", n, "] = [");
      for (int di = 0; di < dims_; ++di) {
        strings::StrAppend(&index, ix_t(n, di), di < dims_ - 1 ? "," : "]");
      }
      if (!valid) {
        return errors::InvalidArgument(index,
                                       " is out of bounds: need 0 <= index < [",
                                       str_util::Join(shape_, ","), "]");
      }
      if (!increasing) {
        return errors::InvalidArgument(
            index,
            " is out of order. Many sparse ops require sorted indices.\n"
            "    Use `tf.sparse.reorder` to create a correctly ordered copy."
            "\n\n");
      }
      if (!different) {
        return errors::InvalidArgument(index, " is repeated");
      }
    }
  }

  return Status::OK();
}

Status SparseTensor::IndicesValid() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSsparsePSsparse_tensorDTcc mht_10(mht_10_v, 477, "", "./tensorflow/core/util/sparse/sparse_tensor.cc", "SparseTensor::IndicesValid");

  if (shape_.size() == 1 && IndicesValidVectorFastPath()) {
    return Status::OK();
  }

  bool standard_order = true;
  for (size_t i = 0; i < order_.size(); ++i) {
    if (order_[i] < 0) {
      return errors::FailedPrecondition(
          "Order was not provided.  Provide an order at "
          "construction time or run ReorderInPlace");
    }
    standard_order = standard_order && order_[i] == i;
  }

  if (standard_order) {
    if (shape_.size() == 1) {
      if (IndicesValidVectorFastPath()) {
        return Status::OK();
      }
    } else if (shape_.size() == 2 &&
               shape_[0] <= std::numeric_limits<int32>::max() &&
               shape_[1] <= std::numeric_limits<int32>::max()) {
      if (IndicesValidMatrix32BitFastPath()) {
        return Status::OK();
      }
    }
    return IndicesValidHelper<true>();
  } else {
    return IndicesValidHelper<false>();
  }
}

}  // namespace sparse
}  // namespace tensorflow
