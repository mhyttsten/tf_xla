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
class MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

// Return in <out> the result of making the end of <s> a square matrix.
Status MakeBatchSquareMatrix(InferenceContext* c, ShapeHandle input,
                             ShapeHandle* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/ops/linalg_ops.cc", "MakeBatchSquareMatrix");

  ShapeHandle s;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(input, 2, &s));

  DimensionHandle d;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(s, -2), c->Dim(s, -1), &d));

  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(s, 0, -2, &batch_shape));
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(d, d), out));
  return Status::OK();
}

Status BatchUnchangedSquareShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/ops/linalg_ops.cc", "BatchUnchangedSquareShapeFn");

  ShapeHandle out;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &out));
  c->set_output(0, out);
  return Status::OK();
}

// The first input is [...,K,M] and second input is [...,M,N].
Status BandedTriangularSolveShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/ops/linalg_ops.cc", "BandedTriangularSolveShapeFn");

  ShapeHandle lhs;
  ShapeHandle rhs;

  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &lhs));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  // Check K > 0.
  DimensionHandle num_bands = c->Dim(lhs, -2);
  DimensionHandle m = c->Dim(lhs, -1);
  if (c->ValueKnown(num_bands) && c->Value(num_bands) <= 0) {
    return errors::InvalidArgument("Number of bands must be positive, but is ",
                                   c->Value(num_bands));
  }
  if (c->ValueKnown(num_bands) && c->ValueKnown(m) &&
      c->Value(num_bands) > c->Value(m)) {
    return errors::InvalidArgument("Number of bands ", c->Value(num_bands),
                                   " cannot exceed the size of the matrix ",
                                   c->Value(m));
  }

  ShapeHandle lhs_batch_shape;
  ShapeHandle rhs_batch_shape;
  ShapeHandle output_batch_shape;
  // Make the common batch subshape.
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, lhs_batch_shape, rhs_batch_shape, true, &output_batch_shape));

  // lhs and rhs have the same value for M to be compatible.
  TF_RETURN_IF_ERROR(c->Merge(m, c->Dim(rhs, -2), &m));

  // Build final shape (batch_shape + m + n) in <out>.
  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(output_batch_shape, c->Matrix(m, c->Dim(rhs, -1)), &out));

  c->set_output(0, out);
  return Status::OK();
}

// The first input is [...,M,N] and second input is either [...,M,K] or [...,M].
// Output is [...,N,K] or [...,N]. If <square>, then input is [...,M,M].
Status MatrixSolveShapeFn(InferenceContext* c, bool square) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_3(mht_3_v, 273, "", "./tensorflow/core/ops/linalg_ops.cc", "MatrixSolveShapeFn");

  ShapeHandle lhs;
  ShapeHandle rhs;
  if (square) {
    TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &lhs));
  } else {
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &lhs));
  }
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  ShapeHandle lhs_batch_shape;
  ShapeHandle rhs_batch_shape;
  // Make the common batch subshape.
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  // Make sure the batch dimensions match between lhs and rhs.
  TF_RETURN_IF_ERROR(
      c->Merge(lhs_batch_shape, rhs_batch_shape, &lhs_batch_shape));

  DimensionHandle m;
  // lhs and rhs have the same value for m to be compatible.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, -2), c->Dim(rhs, -2), &m));
  DimensionHandle n = c->Dim(lhs, -1);
  if (square) {
    TF_RETURN_IF_ERROR(c->Merge(m, n, &n));
  }

  ShapeHandle out;
  // Build final shape (batch_shape + n + k) in <out>.
  TF_RETURN_IF_ERROR(c->Concatenate(lhs_batch_shape, c->Vector(n), &out));
  TF_RETURN_IF_ERROR(c->Concatenate(out, c->Vector(c->Dim(rhs, -1)), &out));
  c->set_output(0, out);
  return Status::OK();
}

// The first input is [...,M,M] and second input is [...,M,N].
// Output is [...,M,N].
Status MatrixTriangularSolveShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_4(mht_4_v, 313, "", "./tensorflow/core/ops/linalg_ops.cc", "MatrixTriangularSolveShapeFn");

  ShapeHandle lhs;
  ShapeHandle rhs;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &lhs));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  ShapeHandle lhs_batch_shape;
  ShapeHandle rhs_batch_shape;
  ShapeHandle output_batch_shape;
  // Make the common batch subshape.
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(
      c, lhs_batch_shape, rhs_batch_shape, true, &output_batch_shape));
  DimensionHandle m;
  // lhs and rhs have the same value for m to be compatible.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(lhs, -1), c->Dim(rhs, -2), &m));

  ShapeHandle out;
  // Build final shape (batch_shape + m + n) in <out>.
  TF_RETURN_IF_ERROR(
      c->Concatenate(output_batch_shape, c->Matrix(m, c->Dim(rhs, -1)), &out));
  c->set_output(0, out);
  return Status::OK();
}

// Input is [...,N,N]. Outputs are:
//   [...,N];[0], if compute_v is false,
//   [...,N];[...,N,N], if compute_v is true.
Status SelfAdjointEigV2ShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_5(mht_5_v, 345, "", "./tensorflow/core/ops/linalg_ops.cc", "SelfAdjointEigV2ShapeFn");

  ShapeHandle input;
  TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &input));
  DimensionHandle n;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, -2), c->Dim(input, -1), &n));
  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  ShapeHandle e_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(n), &e_shape));
  c->set_output(0, e_shape);
  bool compute_v;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_v", &compute_v));
  if (compute_v) {
    ShapeHandle v_shape;
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(n, n), &v_shape));
    c->set_output(1, v_shape);
  } else {
    c->set_output(1, c->Vector(0ll));
  }
  return Status::OK();
}

// Input is [...,N,N].
// First and second outputs are:
//   [...,N,N]; [...,N].
Status LuShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_6(mht_6_v, 373, "", "./tensorflow/core/ops/linalg_ops.cc", "LuShapeFn");

  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));

  DimensionHandle n;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(input, -2), c->Dim(input, -1), &n));

  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));

  ShapeHandle lu_shape;
  ShapeHandle p_shape;

  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(n, n), &lu_shape));
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(n), &p_shape));

  c->set_output(0, lu_shape);
  c->set_output(1, p_shape);
  return Status::OK();
}

// Input is [...,M,N].
// First and second outputs are:
//   [...,M,M]; [...,M,N], if full_matrices is true,
//   [...,M,P]; [...,P,N], if full_matrices is false,
// where P = min(M,N).
Status QrShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_7(mht_7_v, 402, "", "./tensorflow/core/ops/linalg_ops.cc", "QrShapeFn");

  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
  DimensionHandle m = c->Dim(input, -2);
  DimensionHandle n = c->Dim(input, -1);
  DimensionHandle p;
  TF_RETURN_IF_ERROR(c->Min(m, n, &p));
  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  ShapeHandle q_shape;
  ShapeHandle r_shape;
  bool full_matrices;
  TF_RETURN_IF_ERROR(c->GetAttr("full_matrices", &full_matrices));
  if (full_matrices) {
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(m, m), &q_shape));
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(m, n), &r_shape));
  } else {
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(m, p), &q_shape));
    TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Matrix(p, n), &r_shape));
  }
  c->set_output(0, q_shape);
  c->set_output(1, r_shape);
  return Status::OK();
}

// Input is [...,M,N].  First output is [...,min(M,N)].
// Second and third outputs are:
//   [0]; [0], if compute_uv is false.
//   [...,M,M]; [...,N,N], if compute_uv is true and full_matrices is true,
//   [...,M,P]; [...,N,P], if compute_uv is true and full_matrices is false,
// where P = min(M,N).
Status SvdShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_8(mht_8_v, 436, "", "./tensorflow/core/ops/linalg_ops.cc", "SvdShapeFn");

  ShapeHandle input;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));
  DimensionHandle m = c->Dim(input, -2);
  DimensionHandle n = c->Dim(input, -1);
  DimensionHandle p;
  TF_RETURN_IF_ERROR(c->Min(m, n, &p));
  ShapeHandle batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &batch_shape));
  ShapeHandle e_shape;
  TF_RETURN_IF_ERROR(c->Concatenate(batch_shape, c->Vector(p), &e_shape));
  c->set_output(0, e_shape);
  bool compute_uv;
  TF_RETURN_IF_ERROR(c->GetAttr("compute_uv", &compute_uv));
  if (compute_uv) {
    ShapeHandle u_shape;
    ShapeHandle v_shape;
    bool full_matrices;
    TF_RETURN_IF_ERROR(c->GetAttr("full_matrices", &full_matrices));
    if (full_matrices) {
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(m, m), &u_shape));
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(n, n), &v_shape));
    } else {
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(m, p), &u_shape));
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, c->Matrix(n, p), &v_shape));
    }
    c->set_output(1, u_shape);
    c->set_output(2, v_shape);
  } else {
    c->set_output(1, c->Vector(0ll));
    c->set_output(2, c->Vector(0ll));
  }
  return Status::OK();
}

// Inputs: [...,1,M], [...,1,M], [...,1,M],[...,M,N].
// Output is [...,M,N].
Status TridiagonalMatMulShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_9(mht_9_v, 480, "", "./tensorflow/core/ops/linalg_ops.cc", "TridiagonalMatMulShapeFn");

  ShapeHandle superdiag;
  ShapeHandle maindiag;
  ShapeHandle subdiag;
  ShapeHandle rhs;

  // Check that rank is at least 2.
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &superdiag));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &maindiag));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &subdiag));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 2, &rhs));

  // Extract batch dimensions and check they are the same.
  ShapeHandle superdiag_batch_shape;
  ShapeHandle maindiag_batch_shape;
  ShapeHandle subdiag_batch_shape;
  ShapeHandle rhs_batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(superdiag, 0, -2, &superdiag_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(maindiag, 0, -2, &maindiag_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(subdiag, 0, -2, &subdiag_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Merge(superdiag, maindiag, &superdiag));
  TF_RETURN_IF_ERROR(
      c->Merge(maindiag_batch_shape, rhs_batch_shape, &rhs_batch_shape));
  TF_RETURN_IF_ERROR(
      c->Merge(subdiag_batch_shape, rhs_batch_shape, &rhs_batch_shape));

  // Check that diagonals have the same shape.
  TF_RETURN_IF_ERROR(c->Merge(superdiag, maindiag, &maindiag));
  TF_RETURN_IF_ERROR(c->Merge(subdiag, maindiag, &maindiag));

  // Check that size of tri-diagonal matrix is the same as height of matrix on
  // the right.
  DimensionHandle m_lhs = c->Dim(maindiag, -1);
  DimensionHandle m_rhs = c->Dim(rhs, -2);
  TF_RETURN_IF_ERROR(c->Merge(m_lhs, m_rhs, &m_lhs));

  // Check that next-to-last dimension of diagonals is 1.
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(maindiag, -2), 1, &unused));

  // The output shape is the same as rhs shape.
  c->set_output(0, rhs);
  return Status::OK();
}

// The first input is [...,3,M] and second input is [...,M,K].
// Output is [...,M,K].
Status TridiagonalSolveShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_opsDTcc mht_10(mht_10_v, 531, "", "./tensorflow/core/ops/linalg_ops.cc", "TridiagonalSolveShapeFn");

  ShapeHandle lhs;
  ShapeHandle rhs;
  // Check that rank is at least 2.
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &lhs));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &rhs));

  // Extract batch dimensions and check they are the same.
  ShapeHandle lhs_batch_shape;
  ShapeHandle rhs_batch_shape;
  TF_RETURN_IF_ERROR(c->Subshape(lhs, 0, -2, &lhs_batch_shape));
  TF_RETURN_IF_ERROR(c->Subshape(rhs, 0, -2, &rhs_batch_shape));
  TF_RETURN_IF_ERROR(
      c->Merge(lhs_batch_shape, rhs_batch_shape, &lhs_batch_shape));

  // Check that "M" is the same in both inputs.
  DimensionHandle m_lhs = c->Dim(lhs, -1);
  DimensionHandle m_rhs = c->Dim(rhs, -2);
  TF_RETURN_IF_ERROR(c->Merge(m_lhs, m_rhs, &m_lhs));

  // Check that next-to-last dimension of the first input is 3.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(lhs, -2), 3, &m_lhs));

  // The output shape is the same as rhs shape.
  c->set_output(0, rhs);
  return Status::OK();
}

}  // namespace

REGISTER_OP("MatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -1), c->Dim(input, -2), &unused));

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("LogMatrixDeterminant")
    .Input("input: T")
    .Output("sign: T")
    .Output("log_abs_determinant: T")
    .Attr("T: {half, float, double, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input));

      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input, -1), c->Dim(input, -2), &unused));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &s));
      c->set_output(0, s);

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &out));
      c->set_output(1, out);
      return Status::OK();
    });

REGISTER_OP("MatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("MatrixExponential")
    .Deprecated(
        27, "Use Python implementation tf.linalg.matrix_exponential instead.")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("MatrixLogarithm")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("Cholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("CholeskyGrad")
    .Input("l: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("SelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float, half}")
    .Deprecated(11, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(MakeBatchSquareMatrix(c, c->input(0), &input));

      DimensionHandle d = c->Dim(input, -1);
      DimensionHandle d_plus_1;
      TF_RETURN_IF_ERROR(c->Add(d, 1, &d_plus_1));

      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->Subshape(input, 0, -2, &s));
      TF_RETURN_IF_ERROR(c->Concatenate(s, c->Matrix(d_plus_1, d), &s));
      c->set_output(0, s);
      return Status::OK();
    });

REGISTER_OP("Eig")
    .Input("input: T")
    .Output("e: Tout")
    .Output("v: Tout")
    .Attr("compute_v: bool = True")
    .Attr("T: {float, double, complex64, complex128}")
    .Attr("Tout: {complex64, complex128}")
    .SetShapeFn(SelfAdjointEigV2ShapeFn);

REGISTER_OP("SelfAdjointEigV2")
    .Input("input: T")
    .Output("e: T")
    .Output("v: T")
    .Attr("compute_v: bool = True")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(SelfAdjointEigV2ShapeFn);

REGISTER_OP("Lu")
    .Input("input: T")
    .Output("lu: T")
    .Output("p: output_idx_type")
    .Attr("T: {double, float, half, complex64, complex128}")
    .Attr("output_idx_type: {int32, int64} = DT_INT32")
    .SetShapeFn(LuShapeFn);

REGISTER_OP("MatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return MatrixSolveShapeFn(c, true /* square (*/);
    });

REGISTER_OP("BandedTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return BandedTriangularSolveShapeFn(c);
    });

REGISTER_OP("MatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {bfloat16, double, float, half, complex64, complex128}")
    .SetShapeFn([](InferenceContext* c) {
      return MatrixTriangularSolveShapeFn(c);
    });

REGISTER_OP("MatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {double, float, half, complex64, complex128}")
    .Attr("fast: bool = True")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle l2_regularizer;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &l2_regularizer));
      return MatrixSolveShapeFn(c, false /* square */);
    });

REGISTER_OP("MatrixSquareRoot")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(BatchUnchangedSquareShapeFn);

REGISTER_OP("Qr")
    .Input("input: T")
    .Output("q: T")
    .Output("r: T")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(QrShapeFn);

REGISTER_OP("Svd")
    .Input("input: T")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .Attr("compute_uv: bool = True")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float, half, complex64, complex128}")
    .SetShapeFn(SvdShapeFn);

REGISTER_OP("TridiagonalMatMul")
    .Input("superdiag: T")
    .Input("maindiag: T")
    .Input("subdiag: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(TridiagonalMatMulShapeFn);

REGISTER_OP("TridiagonalSolve")
    .Input("diagonals: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("partial_pivoting: bool = True")
    .Attr("perturb_singular: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .SetShapeFn(TridiagonalSolveShapeFn);

REGISTER_OP("Einsum")
    .Input("inputs: N * T")
    .Output("output: T")
    .Attr("equation: string")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .SetShapeFn(shape_inference::EinsumShape);

// Deprecated op registrations:

// Can be deleted after 3feb2017.
REGISTER_OP("BatchSelfAdjointEig")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Deprecated(11, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn(shape_inference::UnknownShape);

// Can all be deleted after 9mar2017.
REGISTER_OP("BatchMatrixDeterminant")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {float, double, complex64, complex128}")
    .Deprecated(13, "Use MatrixDeterminant instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixInverse")
    .Input("input: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use MatrixInverse instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchCholesky")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use Cholesky instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchCholeskyGrad")
    .Input("l: T")
    .Input("grad: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Deprecated(13, "Use CholeskyGrad instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchSelfAdjointEigV2")
    .Input("input: T")
    .Output("e: T")
    .Output("v: T")
    .Attr("compute_v: bool = True")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use SelfAdjointEigV2 instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use MatrixSolve instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixTriangularSolve")
    .Input("matrix: T")
    .Input("rhs: T")
    .Output("output: T")
    .Attr("lower: bool = True")
    .Attr("adjoint: bool = False")
    .Attr("T: {double, float}")
    .Deprecated(13, "Use MatrixTriangularSolve instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchMatrixSolveLs")
    .Input("matrix: T")
    .Input("rhs: T")
    .Input("l2_regularizer: double")
    .Output("output: T")
    .Attr("T: {double, float}")
    .Attr("fast: bool = True")
    .Deprecated(13, "Use MatrixSolveLs instead.")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("BatchSvd")
    .Input("input: T")
    .Output("s: T")
    .Output("u: T")
    .Output("v: T")
    .Attr("compute_uv: bool = True")
    .Attr("full_matrices: bool = False")
    .Attr("T: {double, float, complex64, complex128}")
    .Deprecated(13, "Use Svd instead.")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow
