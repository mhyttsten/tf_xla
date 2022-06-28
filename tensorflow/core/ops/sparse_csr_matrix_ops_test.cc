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
class MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc() {
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

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

TEST(SparseMatrixOpsTest, SparseTensorToCSRSparseMatrix_ShapeFn) {
  ShapeInferenceTestOp op("SparseTensorToCSRSparseMatrix");
  (*op.node_def.mutable_attr())["T"].set_type(DT_FLOAT);
  op.input_tensors.resize(3);
  // inputs: indices, values, dense_shape
  INFER_ERROR("Expected a known rank", op, "?;?;?");
  INFER_ERROR("either 2 or 3", op, "[?,4];?;?");
  INFER_OK(op, "[?,2];?;?", "[]");
  INFER_OK(op, "[?,3];?;?", "[]");
  Tensor dense_shape_t = test::AsTensor<int64_t>({5, 6});
  op.input_tensors[2] = &dense_shape_t;
  INFER_ERROR("Shape must be rank 3 but is rank 2 for", op, "[?,3];?;?");
  INFER_OK(op, "[?,2];?;?", "[]");
}

TEST(SparseMatrixOpsTest, CSRSparseMatrixToSparseTensor_ShapeFn) {
  ShapeInferenceTestOp op("CSRSparseMatrixToSparseTensor");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  // outputs: indices, values, dense_shape
  shapes_and_types[0].first = "[4,5]";
  INFER_OK(op, "[]", "[?,2];[?];[2]");
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[]", "[?,2];[?];[2]");
  shapes_and_types[0].first = "[4,5,6]";
  INFER_OK(op, "[]", "[?,3];[?];[3]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[]", "[?,3];[?];[3]");
}

TEST(SparseMatrixOpsTest, DenseToCSRSparseMatrix_ShapeFn) {
  ShapeInferenceTestOp op("DenseToCSRSparseMatrix");
  (*op.node_def.mutable_attr())["T"].set_type(DT_FLOAT);
  INFER_ERROR("Expected a known rank", op, "?;?");
  INFER_ERROR("either 2 or 3", op, "[?];?");
  INFER_OK(op, "[?,?];[?,2]", "[]");
  INFER_OK(op, "[?,?,?];[?,3]", "[]");
  INFER_ERROR("indices.shape[1] must match rank of dense; saw: 2 vs. 3", op,
              "[?,?,?];[?,2]");
}

TEST(SparseMatrixOpsTest, CSRSparseMatrixToDense_ShapeFn) {
  ShapeInferenceTestOp op("CSRSparseMatrixToDense");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  // outputs: dense
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[]", "[?,?]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[]", "[?,?,?]");
}

TEST(SparseMatrixOpsTest, CSRSparseMatrixComponents_ShapeFn) {
  ShapeInferenceTestOp op("CSRSparseMatrixComponents");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  // inputs: csr_sparse_matrix, index
  // outputs: row_ptrs, col_inds, values
  shapes_and_types[0].first = "[4,5]";
  INFER_OK(op, "[];[]", "[5];[?];[?]");
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[];[]", "[?];[?];[?]");
  shapes_and_types[0].first = "[19,34,55]";
  INFER_OK(op, "[];[]", "[35];[?];[?]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[]", "[?];[?];[?]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("index must be a scalar", op, "[];?");
}

TEST(SparseMatrixOpsTest, SparseMatrixMatMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixMatMul");
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  auto set_options = [&op](bool transpose_a, bool transpose_b, bool adjoint_a,
                           bool adjoint_b, bool transpose_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc mht_0(mht_0_v, 278, "", "./tensorflow/core/ops/sparse_csr_matrix_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "SparseMatrixMatMul")
                     .Input("a", 0, DT_VARIANT)
                     .Input("b", 1, DT_FLOAT)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Attr("transpose_output", transpose_output)
                     .Finalize(&op.node_def));
  };
  // inputs: a <CSR>, b <T>
  // output: matmul(a, b)
  set_options(false, false, false, false, false /*transpose_output*/);
  a_shapes_and_types[0].first = "?";
  INFER_ERROR("a has an unknown rank", op, "[];?");
  a_shapes_and_types[0].first = "[?]";
  INFER_ERROR("must be at least rank 2 but is rank 1", op, "[];?");
  a_shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[];?", "[?,?]");
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];?", "[?,?,?]");
  a_shapes_and_types[0].first = "[?,3,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,3,d1_2]");
  a_shapes_and_types[0].first = "[?,3,?]";
  INFER_OK(op, "[];[?,?,4]", "[?,3,d1_2]");  // [B,3,?] . [B,?,4]
  a_shapes_and_types[0].first = "[?,?,6]";
  INFER_OK(op, "[];[?,6,?]", "[?,?,d1_2]");  // [B,?,6] . [B,6,?]
  a_shapes_and_types[0].first = "[?,?,5]";
  INFER_ERROR("must be equal, but are 5 and 6 for", op, "[];[?,6,?]");

  set_options(false, false, false, false, true /*transpose_output*/);
  a_shapes_and_types[0].first = "[?,3,?]";
  INFER_OK(op, "[];[?,?,4]", "[?,d1_2,3]");
  a_shapes_and_types[0].first = "[3,?]";
  INFER_OK(op, "[];[?,4]", "[d1_1,3]");

  set_options(/*transpose_a=*/true, /*transpose_b=*/true,
              /*adjoint_a=*/false, /*adjoint_b=*/false,
              false /*transpose_output*/);
  // t([B,W,X]) . t([B,Y,Z]) => [B,X,Y]
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,?,d1_1]");

  set_options(/*transpose_a=*/false, /*transpose_b=*/false,
              /*adjoint_a=*/true, /*adjoint_b=*/true,
              false /*transpose_output*/);
  // adj([B,W,X]) . adj([B,Y,Z]) => [B,X,Y]
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,?,d1_1]");

  set_options(true /*transpose_a*/, true /*transpose_b*/,
              /*adjoint_a=*/false, /*adjoint_b=*/false,
              true /*transpose_output*/);
  // t(t([B,W,X]) . t([B,Y,Z])) => [B,Y,X]
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,d1_1,?]");

  set_options(/*transpose_a=*/true, /*transpose_b=*/false,
              /*adjoint_a=*/true, /*adjoint_b=*/true,
              false /*transpose_output*/);
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("Only one of adjoint_a and transpose_a", op, "[];[?,?,?]");
  set_options(/*transpose_a=*/false, /*transpose_b=*/true,
              /*adjoint_a=*/true, /*adjoint_b=*/true,
              false /*transpose_output*/);
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("Only one of adjoint_b and transpose_b", op, "[];[?,?,?]");
}

TEST(SparseMatrixOpsTest, SparseMatrixAdd_ShapeFn) {
  // inputs: a <CSR>, b <CSR>, alpha <scalar>, beta <scalar>
  // output: alpha * a + beta * b
  ShapeInferenceTestOp op("SparseMatrixAdd");
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  std::vector<ShapeInferenceTestOp::ShapeAndType> b_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  b_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(&b_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  auto set_shapes = [&a_shapes_and_types, &b_shapes_and_types](
                        const string& a_shape, const string& b_shape) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("a_shape: \"" + a_shape + "\"");
   mht_1_v.push_back("b_shape: \"" + b_shape + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc mht_1(mht_1_v, 366, "", "./tensorflow/core/ops/sparse_csr_matrix_ops_test.cc", "lambda");

    a_shapes_and_types[0].first = a_shape;
    b_shapes_and_types[0].first = b_shape;
  };
  // TODO(ebrevdo): Update shape_inference_testutil to be able to properly test
  // output handle shapes and types.
  set_shapes("[?,?]", "[?,?]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [?,?]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [?,?,?]
  set_shapes("[3,4]", "[3,4]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [3,4]
  set_shapes("[3,4,5]", "[3,4,5]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [3,4,5]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[];[];[]", "[]");  // output handle: [?,?,?]
  // non-scalar beta.
  set_shapes("[?,?]", "[?,?]");
  INFER_ERROR("must be rank 0 but is rank 1", op, "[];[];?;[?]");
  // unknown rank b.
  set_shapes("[?,?,?]", "?");
  INFER_ERROR("b has an unknown rank", op, "[];[];?;?");
  // different ranks of a and b.
  set_shapes("[?,?,?]", "[?,?]");
  INFER_ERROR("must be equal", op, "[];[];?;?");
}

TEST(SparseMatrixOpsTest, SparseMatrixSparseMatMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixSparseMatMul");
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  std::vector<ShapeInferenceTestOp::ShapeAndType> b_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  b_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(&b_shapes_and_types);
  auto set_shapes = [&a_shapes_and_types, &b_shapes_and_types](
                        const string& a_shape, const string& b_shape) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("a_shape: \"" + a_shape + "\"");
   mht_2_v.push_back("b_shape: \"" + b_shape + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc mht_2(mht_2_v, 407, "", "./tensorflow/core/ops/sparse_csr_matrix_ops_test.cc", "lambda");

    a_shapes_and_types[0].first = a_shape;
    b_shapes_and_types[0].first = b_shape;
  };
  auto set_options = [&op](bool transpose_a, bool transpose_b, bool adjoint_a,
                           bool adjoint_b) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc mht_3(mht_3_v, 415, "", "./tensorflow/core/ops/sparse_csr_matrix_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "SparseMatrixMatMul")
                     .Input("a", 0, DT_VARIANT)
                     .Input("b", 1, DT_FLOAT)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Finalize(&op.node_def));
  };
  // inputs: a <CSR>, b <CSR>
  // output: matmul(a, b) <CSR>
  set_options(false, false, false, false);
  set_shapes("?", "?");
  INFER_ERROR("has an unknown rank", op, "[];[]");
  set_shapes("[?]", "[?,?]");
  INFER_ERROR("must be at least rank 2 but is rank 1", op, "[];[]");
  set_shapes("[?,?]", "[?,?]");
  INFER_OK(op, "[];[]", "[]");  // [d0_0,d1_1]"
  set_shapes("[?,?,?]", "[?,?]");
  INFER_ERROR("must be equal rank, but are", op, "[];[]");
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_1,d1_2]"
  set_shapes("[?,3,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_1,d1_2]"
  set_shapes("[?,3,?]", "[?,?,4]");
  INFER_OK(op, "[];[]", "[]");  // [d0_0,d0_1,d1_2]"
  set_shapes("[?,?,6]", "[?,6,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_1,d1_2]"
  set_shapes("[?,?,5]", "[?,6,?]");
  INFER_ERROR("must be equal, but are 5 and 6 for", op, "[];[]");

  set_options(/*transpose_a=*/true, /*transpose_b=*/true, /*adjoint_a=*/false,
              /*adjoint_b=*/false);
  // t([B,W,X]) . t([B,Y,Z]) => [B,X,Y]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // [d0_0,d0_2,d1_1]"

  set_options(/*transpose_a=*/false, /*transpose_b=*/false, /*adjoint_a=*/true,
              /*adjoint_b=*/true);
  // adj([B,W,X]) . adj([B,Y,Z]) => [B,X,Y]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_2,d1_1]"

  set_options(/*transpose_a=*/true, /*transpose_b=*/false,
              /*adjoint_a=*/true, /*adjoint_b=*/true);
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_ERROR("Only one of adjoint_a and transpose_a", op, "[];[]");
  set_options(/*transpose_a=*/false, /*transpose_b=*/true,
              /*adjoint_a=*/true, /*adjoint_b=*/true);
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_ERROR("Only one of adjoint_b and transpose_b", op, "[];[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixTranspose_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixTranspose");
  // inputs: input
  // outputs: output
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  shapes_and_types[0].first = "[3,4,5]";
  INFER_OK(op, "[]", "[]");  // [3,5,4]"
  shapes_and_types[0].first = "[3,4]";
  INFER_OK(op, "[]", "[]");  // "[4, 3]";
  shapes_and_types[0].first = "?";
  INFER_ERROR("input has an unknown rank", op, "[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixSoftmax_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixSoftmax");
  // inputs: logits
  // outputs: softmax
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[]", "[]");  // "in0"
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[]", "[]");  // "in0"
  shapes_and_types[0].first = "?";
  INFER_ERROR("logits has an unknown rank", op, "[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixSoftmaxGrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixSoftmaxGrad");
  // inputs: softmax, grad_softmax
  // outputs: gradient
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  std::vector<ShapeInferenceTestOp::ShapeAndType> b_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  b_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(&b_shapes_and_types);
  auto set_shapes = [&a_shapes_and_types, &b_shapes_and_types](
                        const string& a_shape, const string& b_shape) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("a_shape: \"" + a_shape + "\"");
   mht_4_v.push_back("b_shape: \"" + b_shape + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSsparse_csr_matrix_ops_testDTcc mht_4(mht_4_v, 515, "", "./tensorflow/core/ops/sparse_csr_matrix_ops_test.cc", "lambda");

    a_shapes_and_types[0].first = a_shape;
    b_shapes_and_types[0].first = b_shape;
  };
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "in0"
  set_shapes("[?,?]", "[?,?]");
  INFER_OK(op, "[];[]", "[]");  // "in0"
  set_shapes("[3,4]", "[5,6]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 3 and 5", op,
              "[];[]");
  set_shapes("?", "[?,?]");
  INFER_ERROR("softmax has an unknown rank", op, "[];[]");
  set_shapes("[?,?,?]", "?");
  INFER_ERROR("grad_softmax has an unknown rank", op, "[];[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixMul");
  // inputs: a <CSR>, b <dense>
  // output: a * b
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  shapes_and_types[0].first = "[3,4]";
  INFER_OK(op, "[];[]", "[]");  // "[3,4]"
  shapes_and_types[0].first = "[5,3,4]";
  INFER_OK(op, "[];[?,1,1]", "[]");  // "[5,3,4]"
  // b not scalar, doesn't match a.
  shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("b must be a scalar or shaped [batch_size, 1, 1]", op,
              "[];[3,4]");
  shapes_and_types[0].first = "[3,4]";
  INFER_ERROR("b must be a scalar or shaped", op, "[];[3,4]");
  shapes_and_types[0].first = "[3,4,5]";
  INFER_ERROR("b must be a scalar or shaped", op, "[];[3,4,5]");
  shapes_and_types[0].first = "[3,4,5]";
  INFER_ERROR("must be equal, but are 3 and 4", op, "[];[4,1,1]");
}

}  // namespace tensorflow
