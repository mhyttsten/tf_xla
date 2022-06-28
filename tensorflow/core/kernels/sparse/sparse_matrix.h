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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_SPARSE_MATRIX_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_SPARSE_MATRIX_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh() {
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


#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {

class CSRSparseMatrix {
  // CreateCSRSparseMatrix is the main method used to construct a
  // CSRSparseMatrix.  The representations for both 2D and 3D
  // (batched) CSR Sparse Matrices are the same:
  //
  // dtype: The datatype of the values.
  // dense_shape: The dense shape of the matrix.
  //   * Host int64 vector, size 2 or 3.
  //   * Takes on values: (rows, cols) or (batch_size, rows, cols).
  // batch_pointers: Batch offset pointers into col_indices and values.
  //   * Host int32 vector, size (batch_size + 1).
  //   * Takes on values: (0, nnz[0], nnz[0] + nnz[1], ..., total_nnz).
  // row_pointers: Row offset pointers into col_indices and values.
  //   * Device int32 vector, size ((rows + 1) * batch_size).
  //   * Each block of size (rows + 1) takes on values:
  //     (0, num_rows{b}[0], num_rows{b}[0] + num_rows{b}[1], ..., nnz[b]).
  //     for b = 0 .. batch_size - 1.
  // col_indices: Column values for the given row and column index.
  //   * Device int32 vector, size total_nnz.
  // values: Actual values for the given row and column index.
  //   * Device dtype vector, size total_nnz.
  //
  // The storage agreement is such that for a given (batch, row, ix):
  //   offset = batch_pointers(batch) + row_pointers(batch * (rows + 1) + row)
  //   col = col_indices(offset + ix)
  //   val = values(offset + ix)
  // where ix < #nnz columns in (batch, row).
  // Then:
  //   matrix(batch, row, col) = val.
  //
  // All other elements in the dense representation are treated as 0 / empty.
  //
  // For example, for a 2D sparse matrix m shaped (3, 4) such that:
  //
  //   m[0, 0] = 1.0
  //   m[0, 1] = 2.0
  //   m[0, 2] = 3.0
  //   m[2, 2] = 4.0
  //   m[2, 3] = 5.0
  //
  // The corresponding representation is:
  //
  //   dtype: DT_FLOAT
  //   dense_shape: (3, 4)
  //   batch_pointers: (0, 5)
  //   row_pointers: (0, 3, 3, 5)
  //   col_indices: concat((0, 1, 2), (), (2, 3))
  //   values: concat((1.0, 2.0, 3.0), (), (4.0, 5.0))
  //
  // For a 3D sparse matrix m shaped (2, 3, 4) such that:
  //
  //   m[0, 0, 0] = 1.0
  //   m[0, 0, 2] = 2.0
  //   m[0, 2, 3] = 3.0
  //   m[1, 0, 3] = 4.0
  //   m[1, 1, 0] = 5.0
  //
  // The corresponding representation is:
  //   dtype: DT_FLOAT
  //   dense_shape: (2, 3, 4)
  //   batch_pointers: (0, 3, 5)
  //   row_pointers: concat((0, 2, 2, 3), (0, 1, 2, 2))
  //   col_indices: concat(concat((0, 2), (), (3,)),
  //                       concat((3,),   (), (0,)))
  //   values: concat(concat((1.0, 2.0), (3.0,), ()),
  ///                 concat((4.0,),     (5.0,), ()))
  //
 public:
  static constexpr const char kTypeName[] = "tensorflow::CSRSparseMatrix";

  CSRSparseMatrix() : metadata_{false, DT_INVALID} {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_0(mht_0_v, 274, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "CSRSparseMatrix");
}

  CSRSparseMatrix(const CSRSparseMatrix& rhs)
      : metadata_(rhs.metadata_),
        dense_shape_(rhs.dense_shape_),
        batch_pointers_(rhs.batch_pointers_),
        row_pointers_(rhs.row_pointers_),
        col_indices_(rhs.col_indices_),
        values_(rhs.values_) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_1(mht_1_v, 285, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "CSRSparseMatrix");

    SetupVecs();
  }

  CSRSparseMatrix(CSRSparseMatrix&& rhs)
      : metadata_(rhs.metadata_),
        dense_shape_(std::move(rhs.dense_shape_)),
        batch_pointers_(std::move(rhs.batch_pointers_)),
        row_pointers_(std::move(rhs.row_pointers_)),
        col_indices_(std::move(rhs.col_indices_)),
        values_(std::move(rhs.values_)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_2(mht_2_v, 298, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "CSRSparseMatrix");

    SetupVecs();
    rhs.metadata_.validated = false;
    rhs.metadata_.dtype = DT_INVALID;
    rhs.ClearVecs();
  }

  CSRSparseMatrix& operator=(CSRSparseMatrix&& rhs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_3(mht_3_v, 308, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "=");

    if (this == &rhs) return *this;
    metadata_ = rhs.metadata_;
    metadata_.validated = rhs.metadata_.validated;
    dense_shape_ = std::move(rhs.dense_shape_);
    batch_pointers_ = std::move(rhs.batch_pointers_);
    row_pointers_ = std::move(rhs.row_pointers_);
    col_indices_ = std::move(rhs.col_indices_);
    values_ = std::move(rhs.values_);
    SetupVecs();
    rhs.metadata_ = {false, DT_INVALID};
    rhs.ClearVecs();
    return *this;
  }

  static Status CreateCSRSparseMatrix(DataType dtype,
                                      const Tensor& dense_shape,     // on host
                                      const Tensor& batch_pointers,  // on host
                                      const Tensor& row_pointers,
                                      const Tensor& col_indices,
                                      const Tensor& values,
                                      CSRSparseMatrix* matrix) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_4(mht_4_v, 332, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "CreateCSRSparseMatrix");

    *matrix = CSRSparseMatrix(dtype, dense_shape, batch_pointers, row_pointers,
                              col_indices, values);
    Status s = matrix->Validate();
    matrix->metadata_.validated = s.ok();
    matrix->SetupVecs();
    return s;
  }

  Status Validate() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_5(mht_5_v, 344, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "Validate");

    return ValidateTypesAndShapes(metadata_.dtype, dense_shape_,
                                  batch_pointers_, row_pointers_, col_indices_,
                                  values_);
  }

  void Clear() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_6(mht_6_v, 353, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "Clear");

    metadata_ = {false, DT_INVALID};
    dense_shape_ = Tensor();
    batch_pointers_ = Tensor();
    row_pointers_ = Tensor();
    col_indices_ = Tensor();
    values_ = Tensor();
    ClearVecs();
  }

  bool valid() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_7(mht_7_v, 366, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "valid");

    return metadata_.validated && dense_shape_.IsInitialized() &&
           batch_pointers_.IsInitialized() && row_pointers_.IsInitialized() &&
           col_indices_.IsInitialized() && values_.IsInitialized() &&
           dense_shape_.NumElements() > 1 &&
           batch_pointers_.NumElements() > 0 && row_pointers_.NumElements() > 0;
  }

  DataType dtype() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_8(mht_8_v, 377, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "dtype");

    DCHECK(valid());
    return metadata_.dtype;
  }

  inline int dims() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_9(mht_9_v, 385, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "dims");

    DCHECK(valid());
    return dense_shape_.NumElements();
  }

  inline int nnz(int batch) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_10(mht_10_v, 393, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "nnz");

    DCHECK_LT(batch, batch_size());
    return (*batch_pointers_vec_)(batch + 1) - (*batch_pointers_vec_)(batch);
  }

  inline int batch_offset(int batch) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_11(mht_11_v, 401, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "batch_offset");

    DCHECK_LT(batch, batch_size());
    return (*batch_pointers_vec_)(batch);
  }

  inline int total_nnz() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_12(mht_12_v, 409, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "total_nnz");

    DCHECK(valid());
    return (*batch_pointers_vec_)(batch_size());
  }

  inline Tensor& dense_shape() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_13(mht_13_v, 417, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "dense_shape");

    DCHECK(valid());
    return dense_shape_;
  }

  inline const Tensor& dense_shape() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_14(mht_14_v, 425, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "dense_shape");

    DCHECK(valid());
    return dense_shape_;
  }

  inline TTypes<int32>::UnalignedVec row_pointers_vec(int batch) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_15(mht_15_v, 433, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "row_pointers_vec");

    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int64_t rows = dense_shape().vec<int64_t>()((dims() == 2) ? 0 : 1);
    const int offset = batch * (rows + 1);
    return TTypes<int32>::UnalignedVec(row_pointers_vec_->data() + offset,
                                       rows + 1);
  }

  inline TTypes<int32>::UnalignedConstVec row_pointers_vec(int batch) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_16(mht_16_v, 445, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "row_pointers_vec");

    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int64_t rows = dense_shape().vec<int64_t>()((dims() == 2) ? 0 : 1);
    const int offset = batch * (rows + 1);
    return TTypes<int32>::UnalignedConstVec(row_pointers_vec_->data() + offset,
                                            rows + 1);
  }

  inline TTypes<int32>::UnalignedVec col_indices_vec(int batch) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_17(mht_17_v, 457, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "col_indices_vec");

    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return TTypes<int32>::UnalignedVec(col_indices_vec_->data() + offset,
                                       nnz_in_batch);
  }

  inline TTypes<int32>::UnalignedConstVec col_indices_vec(int batch) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_18(mht_18_v, 469, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "col_indices_vec");

    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return TTypes<int32>::UnalignedConstVec(col_indices_vec_->data() + offset,
                                            nnz_in_batch);
  }

  template <typename T>
  inline typename TTypes<T>::UnalignedVec values_vec(int batch) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_19(mht_19_v, 482, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "values_vec");

    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return typename TTypes<T>::UnalignedVec(values().vec<T>().data() + offset,
                                            nnz_in_batch);
  }

  template <typename T>
  inline typename TTypes<T>::UnalignedConstVec values_vec(int batch) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_20(mht_20_v, 495, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "values_vec");

    DCHECK(valid());
    DCHECK_LT(batch, batch_size());
    const int offset = (*batch_pointers_vec_)(batch);
    const int nnz_in_batch = nnz(batch);
    return typename TTypes<T>::UnalignedConstVec(
        values().vec<T>().data() + offset, nnz_in_batch);
  }

  inline Tensor& row_pointers() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_21(mht_21_v, 507, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "row_pointers");

    DCHECK(valid());
    return row_pointers_;
  }

  inline const Tensor& row_pointers() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_22(mht_22_v, 515, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "row_pointers");

    DCHECK(valid());
    return row_pointers_;
  }

  inline Tensor& col_indices() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_23(mht_23_v, 523, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "col_indices");

    DCHECK(valid());
    return col_indices_;
  }

  inline const Tensor& col_indices() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_24(mht_24_v, 531, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "col_indices");

    DCHECK(valid());
    return col_indices_;
  }

  inline Tensor& values() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_25(mht_25_v, 539, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "values");

    DCHECK(valid());
    return values_;
  }

  inline const Tensor& values() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_26(mht_26_v, 547, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "values");

    DCHECK(valid());
    return values_;
  }

  inline Tensor& batch_pointers() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_27(mht_27_v, 555, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "batch_pointers");

    DCHECK(valid());
    return batch_pointers_;
  }

  inline const Tensor& batch_pointers() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_28(mht_28_v, 563, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "batch_pointers");

    DCHECK(valid());
    return batch_pointers_;
  }

  std::string TypeName() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_29(mht_29_v, 571, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "TypeName");
 return kTypeName; }

  // TODO(ebrevdo): A better debug string.
  std::string DebugString() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_30(mht_30_v, 577, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "DebugString");
 return dense_shape_.DebugString(); }

  // Returns the number of elements.  This is equal to 1 if the
  // CSRSparseMatrix is a singleton matrix (dense_shape is length 2).
  int batch_size() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_31(mht_31_v, 584, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "batch_size");

    DCHECK(valid());
    return batch_pointers_.NumElements() - 1;
  }

  bool Decode(const VariantTensorData& p) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_32(mht_32_v, 592, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "Decode");

    if (p.tensors_.empty()) return false;
    Metadata metadata;
    if (!p.get_metadata(&metadata)) return false;
    const bool validated = metadata.validated;
    const DataType dtype = metadata.dtype;

    // p.tensors_ should contain tensors {dense_shape, batch_pointers,
    // row_pointers, col_indices, values}.
    if (p.tensors_.size() != 5) return false;

    Tensor dense_shape = p.tensors_[0];
    if (dense_shape.dtype() != DT_INT64) return false;
    if (dense_shape.dims() != 1) return false;
    int rank = dense_shape.dim_size(0);
    if (rank < 2 || rank > 3) return false;

    Tensor batch_pointers(p.tensors_[1]);
    Tensor row_pointers(p.tensors_[2]);
    Tensor col_indices(p.tensors_[3]);
    Tensor values(p.tensors_[4]);

    // Check that the validated bool is consistent with the data.
    Status s = ValidateTypesAndShapes(dtype, dense_shape, batch_pointers,
                                      row_pointers, col_indices, values);
    if (s.ok() != validated) return false;

    // Save to this object.
    metadata_ = metadata;
    dense_shape_ = std::move(dense_shape);
    batch_pointers_ = std::move(batch_pointers);
    row_pointers_ = std::move(row_pointers);
    col_indices_ = std::move(col_indices);
    values_ = std::move(values);
    SetupVecs();
    return true;
  }

  void Encode(VariantTensorData* p) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_33(mht_33_v, 633, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "Encode");

    DCHECK(valid());

    // Store metadata_ to p's metadata
    p->set_metadata(metadata_);

    // Store dense_shape, row_pointers, col_indices, and values to p->tensors_.
    p->tensors_.reserve(5);
    p->tensors_.push_back(dense_shape_);
    p->tensors_.push_back(batch_pointers_);
    p->tensors_.push_back(row_pointers_);
    p->tensors_.push_back(col_indices_);
    p->tensors_.push_back(values_);
  }

  // This static method copies CSRSparseMatrices in all directions:
  //   Host->Device, Device->Host, and Device->Device.
  static Status DeviceCopy(
      const CSRSparseMatrix& from, CSRSparseMatrix* to,
      const UnaryVariantOpRegistry::AsyncTensorDeviceCopyFn& copy) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_34(mht_34_v, 655, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "DeviceCopy");

    VLOG(2) << "DeviceCopy from type: " << DataTypeString(from.dtype())
            << " and shape: " << from.dense_shape().DebugString();
    Tensor to_row_ptr(DT_INT32);
    Tensor to_col_ind(DT_INT32);
    Tensor to_values(from.dtype());
    TF_RETURN_IF_ERROR(copy(from.row_pointers(), &to_row_ptr));
    TF_RETURN_IF_ERROR(copy(from.col_indices(), &to_col_ind));
    TF_RETURN_IF_ERROR(copy(from.values(), &to_values));
    return CreateCSRSparseMatrix(from.dtype(),
                                 from.dense_shape(),     // Always on host.
                                 from.batch_pointers(),  // Always on host.
                                 to_row_ptr, to_col_ind, to_values, to);
  }

 private:
  CSRSparseMatrix(DataType dtype, const Tensor& dense_shape,
                  const Tensor& batch_pointers, const Tensor& row_pointers,
                  const Tensor& col_indices, const Tensor& values)
      : metadata_{false, dtype},
        dense_shape_(dense_shape),
        batch_pointers_(batch_pointers),
        row_pointers_(row_pointers),
        col_indices_(col_indices),
        values_(values) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_35(mht_35_v, 682, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "CSRSparseMatrix");
}

  void SetupVecs() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_36(mht_36_v, 687, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "SetupVecs");

    if (!metadata_.validated) return;
    batch_pointers_vec_.reset(
        new TTypes<int32>::Vec(batch_pointers_.vec<int32>()));
    row_pointers_vec_.reset(new TTypes<int32>::Vec(row_pointers_.vec<int32>()));
    col_indices_vec_.reset(new TTypes<int32>::Vec(col_indices_.vec<int32>()));
  }

  void ClearVecs() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_37(mht_37_v, 698, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "ClearVecs");

    batch_pointers_vec_.reset();
    row_pointers_vec_.reset();
    col_indices_vec_.reset();
  }

  static Status ValidateTypesAndShapes(DataType dtype,
                                       const Tensor& dense_shape,
                                       const Tensor& batch_pointers,
                                       const Tensor& row_pointers,
                                       const Tensor& col_indices,
                                       const Tensor& values) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_38(mht_38_v, 712, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "ValidateTypesAndShapes");

    // TODO(ebrevdo): Consider adding support for other floating point types
    // (namely, float16).
    if (dtype != DT_FLOAT && dtype != DT_DOUBLE && dtype != DT_COMPLEX64 &&
        dtype != DT_COMPLEX128) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dtype = ", DataTypeString(dtype),
          " not in {float32, float64, complex64, complex128}");
    }
    // dense_shape checks
    if (dense_shape.dtype() != DT_INT64) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape.dtype() = ",
          DataTypeString(dense_shape.dtype()), " != int64");
    }
    if (dense_shape.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape should be a vector, but saw "
          "tensor: ",
          dense_shape.DebugString());
    }
    int rank = dense_shape.dim_size(0);
    if (rank < 2 || rank > 3) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape should be a 2- or 3- vector, "
          "but saw: ",
          dense_shape.SummarizeValue(5));
    }
    auto dense_shape_t = dense_shape.vec<int64_t>();
    const int64_t batch_size = (rank == 2) ? 1 : dense_shape_t(0);
    const int64_t num_rows = (rank == 2) ? dense_shape_t(0) : dense_shape_t(1);

    if (batch_pointers.dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: batch_pointers.dtype() = ",
          DataTypeString(batch_pointers.dtype()), " != int32");
    }
    if (batch_pointers.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: batch_indices is not a vector, saw "
          "shape: ",
          batch_pointers.shape().DebugString());
    }

    // batch size checks
    if (batch_size != batch_pointers.NumElements() - 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: dense_shape is ",
          dense_shape.SummarizeValue(5),
          " but batch pointers implies batch size is ",
          batch_pointers.NumElements() - 1);
    }

    if (row_pointers.dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: row_pointers.dtype() = ",
          DataTypeString(row_pointers.dtype()), " != int32");
    }
    if (row_pointers.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: row_pointers is not a vector, saw "
          "shape: ",
          row_pointers.shape().DebugString());
    }
    if (row_pointers.dim_size(0) != batch_size * (num_rows + 1)) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: row_pointers should have size batch_size "
          "* (num_rows + 1), saw shapes: ",
          dense_shape.DebugString(), " vs. ",
          row_pointers.shape().DebugString());
    }
    if (col_indices.dtype() != DT_INT32) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: col_indices.dtype() = ",
          DataTypeString(col_indices.dtype()), " != int32");
    }
    if (col_indices.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: col_indices is not a vector, saw shape: ",
          col_indices.shape().DebugString());
    }
    if (values.dtype() != dtype) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: values.dtype() = ",
          DataTypeString(values.dtype()),
          " != dtype = ", DataTypeString(dtype));
    }
    if (values.dims() != 1) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: values is not a vector, saw shape: ",
          values.shape().DebugString());
    }
    if (col_indices.dim_size(0) != values.dim_size(0)) {
      return errors::InvalidArgument(
          "CSRSparseMatrix::Validate: size(col_indices) = ",
          col_indices.dim_size(0), " != size(values) = ", values.dim_size(0));
    }
    return Status::OK();
  }

  struct Metadata {
    bool validated;
    DataType dtype;
  };
  Metadata metadata_;
  Tensor dense_shape_;
  Tensor batch_pointers_;
  Tensor row_pointers_;
  Tensor col_indices_;
  Tensor values_;
  std::unique_ptr<TTypes<int32>::Vec> batch_pointers_vec_;
  std::unique_ptr<TTypes<int32>::Vec> row_pointers_vec_;
  std::unique_ptr<TTypes<int32>::Vec> col_indices_vec_;
};

// Call BinaryFunctor<Device, T>()(ctx, a, b, c)
// where T depends on a.dtype().  T will be one of: float, double,
// complex64, complex128.
template <typename Device, template <typename, typename> class BinaryFunctor>
Status CSRSparseMatrixBinaryHelper(OpKernelContext* ctx,
                                   const CSRSparseMatrix& a,
                                   const CSRSparseMatrix& b,
                                   CSRSparseMatrix* c) {
  DataType dt = a.dtype();
  if (dt != b.dtype()) {
    return errors::InvalidArgument(
        "CSRSparseMatrixBinaryHelper: Inconsistent dtypes for input matrices, "
        "a "
        "dtype: ",
        DataTypeString(dt), ", b dtype: ", DataTypeString(b.dtype()));
  }
  switch (dt) {
    case DT_FLOAT: {
      BinaryFunctor<Device, float> functor(ctx);
      return functor(a, b, c);
    }
    case DT_DOUBLE: {
      BinaryFunctor<Device, double> functor(ctx);
      return functor(a, b, c);
    }
    case DT_COMPLEX64: {
      BinaryFunctor<Device, complex64> functor(ctx);
      return functor(a, b, c);
    }
    case DT_COMPLEX128: {
      BinaryFunctor<Device, complex128> functor(ctx);
      return functor(a, b, c);
    }
    default:
      return errors::InvalidArgument(
          "CSRSparseMatrixBinaryHelper: a.dtype (", DataTypeString(dt),
          ") is not one of: float, double, complex64, complex128");
  }
}

// Call UnaryFunctor<Device, T>()(ctx, a, b)
// where T depends on a.dtype().  T will be one of: float, double,
// complex64, complex128.
template <typename Device, template <typename, typename> class UnaryFunctor>
Status CSRSparseMatrixUnaryHelper(OpKernelContext* ctx,
                                  const CSRSparseMatrix& a,
                                  CSRSparseMatrix* b) {
  DataType dt = a.dtype();
  switch (dt) {
    case DT_FLOAT: {
      UnaryFunctor<Device, float> functor(ctx);
      return functor(a, b);
    }
    case DT_DOUBLE: {
      UnaryFunctor<Device, double> functor(ctx);
      return functor(a, b);
    }
    case DT_COMPLEX64: {
      UnaryFunctor<Device, complex64> functor(ctx);
      return functor(a, b);
    }
    case DT_COMPLEX128: {
      UnaryFunctor<Device, complex128> functor(ctx);
      return functor(a, b);
    }
    default:
      return errors::InvalidArgument(
          "CSRSparseMatrixUnaryHelper: a.dtype (", DataTypeString(dt),
          ") is not one of: float, double, complex64, complex128");
  }
}

template <typename T>
struct ConstCSRComponent {
  TTypes<int32>::UnalignedConstVec row_ptr;
  TTypes<int32>::UnalignedConstVec col_ind;
  typename TTypes<T>::UnalignedConstVec values;
  TTypes<int64_t>::ConstVec dense_shape_host;
};

template <typename T>
struct CSRComponent {
  TTypes<int32>::UnalignedVec row_ptr;
  TTypes<int32>::UnalignedVec col_ind;
  typename TTypes<T>::UnalignedVec values;
  TTypes<int64_t>::Vec dense_shape_host;
};

template <typename T>
Status ExtractVariantFromInput(OpKernelContext* ctx, int index,
                               const T** value) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparsePSsparse_matrixDTh mht_39(mht_39_v, 920, "", "./tensorflow/core/kernels/sparse/sparse_matrix.h", "ExtractVariantFromInput");

  const Tensor& input_t = ctx->input(index);
  const Variant& input_variant = input_t.scalar<Variant>()();
  *value = input_variant.get<T>();
  if (*value == nullptr) {
    return errors::InvalidArgument("Could not retrieve Variant input ", index);
  }
  if (!(*value)->valid()) {
    return errors::InvalidArgument("Variant input ", index, " is not valid.");
  }
  return Status::OK();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_SPARSE_MATRIX_H_
