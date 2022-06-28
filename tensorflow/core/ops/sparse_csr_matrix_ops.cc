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
class MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

Status GetVariantInput(InferenceContext* c, int index,
                       ShapeAndType* shape_and_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/ops/sparse_csr_matrix_ops.cc", "GetVariantInput");

  ShapeHandle variant;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(index), 0, &variant));
  auto* shapes_and_types = c->input_handle_shapes_and_types(index);
  if (shapes_and_types == nullptr || shapes_and_types->size() != 1) {
    return errors::InvalidArgument(
        "Unable to access shape and type info from variant input ", index);
  }
  *shape_and_type = shapes_and_types->at(0);
  return Status::OK();
}

// Validates that a shape represents a (rank-2) square matrix or a (rank-3)
// batch of square matrices.
Status ValidateSquareMatrixShape(InferenceContext* c,
                                 const ShapeHandle& matrix_shape,
                                 DimensionHandle* matrix_dimension) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_opsDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/ops/sparse_csr_matrix_ops.cc", "ValidateSquareMatrixShape");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(matrix_shape, 2, &out));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(matrix_shape, 3, &out));
  if (!c->RankKnown(matrix_shape)) {
    return errors::Internal("Sparse matrix has an unknown rank.");
  }

  TF_RETURN_IF_ERROR(c->Merge(c->Dim(matrix_shape, -2),
                              c->Dim(matrix_shape, -1), matrix_dimension));
  return Status::OK();
}

REGISTER_OP("SparseTensorToCSRSparseMatrix")
    .Input("indices: int64")
    .Input("values: T")
    .Input("dense_shape: int64")
    .Attr("T: {float, double, complex64, complex128}")
    .Output("sparse_matrix: variant")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::ValidateSparseTensor(
          c, c->input(0), c->input(1), c->input(2)));
      auto rank = c->Value(c->Dim(c->input(0), 1));
      ShapeHandle dense_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(2, &dense_shape));
      TF_RETURN_IF_ERROR(c->WithRank(dense_shape, rank, &dense_shape));
      if (!c->RankKnown(dense_shape) || c->Rank(dense_shape) < 2 ||
          c->Rank(dense_shape) > 3) {
        return errors::InvalidArgument(
            "Invalid rank: ", c->Rank(dense_shape),
            ".  Expected a known rank of either 2 or 3.");
      }

      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      c->set_output(0, c->Scalar());
      c->set_output_handle_shapes_and_types(0,
                                            {ShapeAndType{dense_shape, dtype}});
      return Status::OK();
    });

REGISTER_OP("CSRSparseMatrixToSparseTensor")
    .Input("sparse_matrix: variant")
    .Output("indices: int64")
    .Output("values: type")
    .Output("dense_shape: int64")
    .Attr("type: {float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle sparse_matrix = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(sparse_matrix, 3, &sparse_matrix));
      if (!c->RankKnown(sparse_matrix)) {
        return errors::InvalidArgument("sparse_matrix has an unknown rank.");
      }
      int rank = c->Rank(sparse_matrix);
      ShapeHandle indices = c->Matrix(c->UnknownDim(), rank);
      ShapeHandle values = c->Vector(c->UnknownDim());
      ShapeHandle dense_shape = c->Vector(rank);
      c->set_output(0, indices);
      c->set_output(1, values);
      c->set_output(2, dense_shape);
      return Status::OK();
    });

REGISTER_OP("DenseToCSRSparseMatrix")
    .Input("dense_input: T")
    .Input("indices: int64")
    .Attr("T: {float, double, complex64, complex128}")
    .Output("sparse_output: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle dense_shape = c->input(0);
      if (!c->RankKnown(dense_shape) || c->Rank(dense_shape) < 2 ||
          c->Rank(dense_shape) > 3) {
        return errors::InvalidArgument(
            "Invalid rank of dense: ", c->Rank(dense_shape),
            ".  Expected a known rank of either 2 or 3.");
      }
      auto rank = c->Rank(dense_shape);

      ShapeHandle indices = c->input(1);
      if (!c->RankKnown(indices) || c->Rank(indices) != 2) {
        return errors::InvalidArgument(
            "indices must be a matrix; but its rank is not 2: ",
            c->Rank(indices));
      }
      auto indices_col = c->Dim(indices, 1);
      if (!c->ValueKnown(indices_col) || c->Value(indices_col) != rank) {
        return errors::InvalidArgument(
            "indices.shape[1] must match rank of dense; saw: ",
            c->Value(indices_col), " vs. ", rank);
      }
      ShapeHandle fake_values_vec = c->Vector(c->Dim(indices, 0));
      ShapeHandle fake_shape_shape = c->Vector(rank);
      TF_RETURN_IF_ERROR(shape_inference::ValidateSparseTensor(
          c, indices /*indices_shape*/, fake_values_vec /*values_shape*/,
          fake_shape_shape /*shape_shape*/));
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      c->set_output_handle_shapes_and_types(0,
                                            {ShapeAndType{dense_shape, dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("CSRSparseMatrixToDense")
    .Input("sparse_input: variant")
    .Output("dense_output: type")
    .Attr("type: {float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle sparse_matrix = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(sparse_matrix, 3, &sparse_matrix));
      if (!c->RankKnown(sparse_matrix)) {
        return errors::InvalidArgument("sparse_matrix has an unknown rank.");
      }
      c->set_output(0, sparse_matrix);
      return Status::OK();
    });

REGISTER_OP("CSRSparseMatrixComponents")
    .Input("csr_sparse_matrix: variant")
    .Input("index: int32")
    .Output("row_ptrs: int32")
    .Output("col_inds: int32")
    .Output("values: type")
    .Attr("type: {float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle csr_sparse_matrix = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(csr_sparse_matrix, 2, &csr_sparse_matrix));
      TF_RETURN_IF_ERROR(
          c->WithRankAtMost(csr_sparse_matrix, 3, &csr_sparse_matrix));
      ShapeHandle index;
      if (c->Rank(c->input(1)) != 0) {
        return errors::InvalidArgument("index must be a scalar.");
      }
      if (!c->RankKnown(csr_sparse_matrix)) {
        return errors::InvalidArgument(
            "csr_sparse_matrix has an unknown rank.");
      }
      auto row_ptrs_dh = c->Dim(csr_sparse_matrix, -2);
      TF_RETURN_IF_ERROR(c->Add(row_ptrs_dh, 1, &row_ptrs_dh));
      ShapeHandle row_ptrs = c->Vector(row_ptrs_dh);
      c->set_output(0, row_ptrs);
      c->set_output(1, c->Vector(c->UnknownDim()));
      c->set_output(2, c->Vector(c->UnknownDim()));
      return Status::OK();
    });

REGISTER_OP("SparseMatrixNNZ")
    .Input("sparse_matrix: variant")
    .Output("nnz: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle sparse_matrix = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(sparse_matrix, 2, &sparse_matrix));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(sparse_matrix, 3, &sparse_matrix));
      if (!c->RankKnown(sparse_matrix)) {
        return errors::InvalidArgument("sparse_matrix has an unknown rank.");
      }
      ShapeHandle out;
      if (c->Rank(sparse_matrix) == 3) {
        out = c->Vector(c->Dim(sparse_matrix, 0));
      } else {
        out = c->Scalar();
      }
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("SparseMatrixMatMul")
    .Input("a: variant")
    .Input("b: T")
    .Attr("T: type")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .Attr("transpose_output: bool = false")
    .Attr("conjugate_output: bool = false")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle a_shape = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(a_shape, 3, &a_shape));
      if (!c->RankKnown(a_shape)) {
        return errors::Internal("a has an unknown rank.");
      }
      ShapeHandle b_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &b_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(b_shape, 3, &b_shape));

      bool transpose_a = false;
      bool transpose_b = false;
      bool transpose_output = false;

      // TODO(ebrevdo): Add transpose support.
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_output", &transpose_output));

      bool adjoint_a = false;
      bool adjoint_b = false;
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_a", &adjoint_a));
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_b", &adjoint_b));
      if (adjoint_a && transpose_a) {
        return errors::InvalidArgument(
            "Only one of adjoint_a and transpose_a may be true.");
      }
      if (adjoint_b && transpose_b) {
        return errors::InvalidArgument(
            "Only one of adjoint_b and transpose_b may be true.");
      }
      transpose_a = transpose_a || adjoint_a;
      transpose_b = transpose_b || adjoint_b;

      auto output_rows = c->Dim(a_shape, transpose_a ? -1 : -2);
      auto output_cols = c->Dim(b_shape, transpose_b ? -2 : -1);
      if (transpose_output) {
        std::tie(output_rows, output_cols) =
            std::make_tuple(output_cols, output_rows);
      }

      // Batch dims match between inputs.
      ShapeHandle a_batch_dims;
      ShapeHandle b_batch_dims;
      ShapeHandle batch_dims;
      TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, b_batch_dims, &batch_dims));

      // Assert inner dims match.
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(a_shape, transpose_a ? -2 : -1),
                                  c->Dim(b_shape, transpose_b ? -1 : -2),
                                  &unused));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          batch_dims, c->Matrix(output_rows, output_cols), &out));

      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("SparseMatrixMul")
    .Input("a: variant")
    .Input("b: T")
    .Attr("T: type")
    .Output("output: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle a_shape = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(a_shape, 3, &a_shape));
      if (!c->RankKnown(a_shape)) {
        return errors::Internal("a has an unknown rank.");
      }
      ShapeHandle b_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 3, &b_shape));
      if (!c->RankKnown(b_shape)) {
        return errors::Internal("b has an unknown rank.");
      }
      ShapeHandle out;
      if (c->Rank(b_shape) == 0) {
        out = a_shape;
      } else if (c->Rank(b_shape) == 3) {
        if (c->Rank(a_shape) != 3) {
          return errors::Unimplemented("rank of b is 3 but rank of a is not.");
        }
        if (!(c->Value(c->Dim(b_shape, 1)) == 1 &&
              c->Value(c->Dim(b_shape, 2)) == 1)) {
          return errors::Unimplemented(
              "b must be a scalar or shaped [batch_size, 1, 1]");
        }
        DimensionHandle batch_size = c->Dim(a_shape, 0);
        TF_RETURN_IF_ERROR(
            c->Merge(batch_size, c->Dim(b_shape, 0), &batch_size));
        TF_RETURN_IF_ERROR(c->ReplaceDim(b_shape, 0, batch_size, &b_shape));
        TF_RETURN_IF_ERROR(c->ReplaceDim(a_shape, 0, batch_size, &a_shape));
        out = a_shape;
      } else {
        return errors::Unimplemented(
            "b must be a scalar or shaped [batch_size, 1, 1]");
      }
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{out, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("SparseMatrixAdd")
    .Input("a: variant")
    .Input("b: variant")
    .Input("alpha: T")
    .Input("beta: T")
    .Attr("T: {float, double, complex64, complex128}")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      // alpha and beta are scalars.
      ShapeHandle unused_scalar_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused_scalar_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_scalar_shape));

      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle a_shape = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(a_shape, 3, &a_shape));
      if (!c->RankKnown(a_shape)) {
        return errors::InvalidArgument("a has an unknown rank.");
      }

      TF_RETURN_IF_ERROR(GetVariantInput(c, 1, &sparse_matrix_shape_and_type));
      ShapeHandle b_shape = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(b_shape, 2, &b_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(b_shape, 3, &b_shape));
      if (!c->RankKnown(b_shape)) {
        return errors::InvalidArgument("b has an unknown rank.");
      }
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Merge(a_shape, b_shape, &out));
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{out, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("SparseMatrixSparseMatMul")
    .Input("a: variant")
    .Input("b: variant")
    .Attr("type: {float, double, complex64, complex128}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("adjoint_a: bool = false")
    .Attr("adjoint_b: bool = false")
    .Output("c: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle a_shape = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(a_shape, 2, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(a_shape, 3, &a_shape));
      if (!c->RankKnown(a_shape)) {
        return errors::Internal("a has an unknown rank.");
      }

      TF_RETURN_IF_ERROR(GetVariantInput(c, 1, &sparse_matrix_shape_and_type));
      ShapeHandle b_shape = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(b_shape, 2, &b_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(b_shape, 3, &b_shape));
      if (!c->RankKnown(b_shape)) {
        return errors::Internal("b has an unknown rank.");
      }

      bool transpose_a = false;
      bool transpose_b = false;
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
      TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
      bool adjoint_a = false;
      bool adjoint_b = false;
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_a", &adjoint_a));
      TF_RETURN_IF_ERROR(c->GetAttr("adjoint_b", &adjoint_b));
      if (adjoint_a && transpose_a) {
        return errors::InvalidArgument(
            "Only one of adjoint_a and transpose_a may be true.");
      } else if (adjoint_b && transpose_b) {
        return errors::InvalidArgument(
            "Only one of adjoint_b and transpose_b may be true.");
      }
      transpose_a = transpose_a || adjoint_a;
      transpose_b = transpose_b || adjoint_b;

      auto output_rows = c->Dim(a_shape, transpose_a ? -1 : -2);
      auto output_cols = c->Dim(b_shape, transpose_b ? -2 : -1);

      // Batch dims match between inputs.
      ShapeHandle a_batch_dims;
      ShapeHandle b_batch_dims;
      ShapeHandle batch_dims;
      TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, b_batch_dims, &batch_dims));

      // Assert inner dims match.
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(a_shape, transpose_a ? -2 : -1),
                                  c->Dim(b_shape, transpose_b ? -1 : -2),
                                  &unused));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          batch_dims, c->Matrix(output_rows, output_cols), &out));

      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{out, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("SparseMatrixZeros")
    .Input("dense_shape: int64")
    .Attr("type: {float, double, complex64, complex128}")
    .Output("sparse_matrix: variant")
    .SetShapeFn([](InferenceContext* c) {
      auto rank = c->NumElements(c->input(0));
      ShapeHandle dense_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &dense_shape));
      TF_RETURN_IF_ERROR(
          c->WithRank(dense_shape, c->Value(rank), &dense_shape));
      if (!c->RankKnown(dense_shape) || c->Rank(dense_shape) < 2 ||
          c->Rank(dense_shape) > 3) {
        return errors::InvalidArgument(
            "Invalid rank: ", c->Rank(dense_shape),
            ".  Expected a known rank of either 2 or 3.");
      }
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("type", &dtype));
      c->set_output_handle_shapes_and_types(0,
                                            {ShapeAndType{dense_shape, dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("SparseMatrixTranspose")
    .Input("input: variant")
    .Attr("conjugate: bool = false")
    .Attr("type: {float, double, complex64, complex128}")
    .Output("output: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle input = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 2, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(input, 3, &input));
      if (!c->RankKnown(input)) {
        return errors::InvalidArgument("input has an unknown rank.");
      }
      ShapeHandle output;
      if (c->Rank(input) == 2) {
        output = c->Matrix(c->Dim(input, 1), c->Dim(input, 0));
      } else {
        output = c->MakeShape(
            {c->Dim(input, 0), c->Dim(input, 2), c->Dim(input, 1)});
      }
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{output, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());

      return Status::OK();
    });

REGISTER_OP("SparseMatrixSoftmax")
    .Input("logits: variant")
    .Attr("type: {float, double}")
    .Output("softmax: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle logits = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(logits, 2, &logits));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(logits, 3, &logits));
      if (!c->RankKnown(logits)) {
        return errors::InvalidArgument("logits has an unknown rank.");
      }
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{logits, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("SparseMatrixSoftmaxGrad")
    .Input("softmax: variant")
    .Input("grad_softmax: variant")
    .Attr("type: {float, double}")
    .Output("gradient: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle softmax = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(softmax, 2, &softmax));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(softmax, 3, &softmax));
      if (!c->RankKnown(softmax)) {
        return errors::InvalidArgument("softmax has an unknown rank.");
      }
      TF_RETURN_IF_ERROR(GetVariantInput(c, 1, &sparse_matrix_shape_and_type));
      ShapeHandle grad_softmax = sparse_matrix_shape_and_type.shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(grad_softmax, 2, &grad_softmax));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(grad_softmax, 3, &grad_softmax));
      if (!c->RankKnown(grad_softmax)) {
        return errors::InvalidArgument("grad_softmax has an unknown rank.");
      }
      TF_RETURN_IF_ERROR(c->Merge(softmax, grad_softmax, &softmax));
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{softmax, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("SparseMatrixOrderingAMD")
    .Input("input: variant")
    .Output("output: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle matrix_shape = sparse_matrix_shape_and_type.shape;
      DimensionHandle n;
      TF_RETURN_IF_ERROR(ValidateSquareMatrixShape(c, matrix_shape, &n));

      ShapeHandle output;
      if (c->Rank(matrix_shape) == 2) {
        output = c->Vector(c->Dim(matrix_shape, 0));
      } else {
        output = c->Matrix(c->Dim(matrix_shape, 0), c->Dim(matrix_shape, 1));
      }
      c->set_output(0, output);
      return Status::OK();
    });

REGISTER_OP("SparseMatrixSparseCholesky")
    .Input("input: variant")
    .Input("permutation: int32")
    .Attr("type: {float, double, complex64, complex128}")
    .Output("output: variant")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle matrix_shape = sparse_matrix_shape_and_type.shape;
      DimensionHandle n;
      TF_RETURN_IF_ERROR(ValidateSquareMatrixShape(c, matrix_shape, &n));

      ShapeHandle perm_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &perm_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 2, &perm_shape));
      if (!c->RankKnown(perm_shape)) {
        return errors::Internal("permutation has an unknown rank.");
      }

      // Each batch component of permutation must have the same number of
      // elements as number of rows of sparse_matrix.
      TF_RETURN_IF_ERROR(c->Merge(n, c->Dim(perm_shape, -1), &n));
      ShapeHandle matrix_batch_shape;
      ShapeHandle perm_batch_shape;

      // Make the common batch subshape.
      TF_RETURN_IF_ERROR(c->Subshape(matrix_shape, 0, -2, &matrix_batch_shape));
      TF_RETURN_IF_ERROR(c->Subshape(perm_shape, 0, -1, &perm_shape));
      // Make sure the batch dimensions match between sparse_matrix and
      // permutation.
      TF_RETURN_IF_ERROR(
          c->Merge(matrix_batch_shape, perm_batch_shape, &matrix_batch_shape));

      ShapeHandle out = matrix_shape;
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{out, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, c->Scalar());

      return Status::OK();
    });

}  // namespace tensorflow
