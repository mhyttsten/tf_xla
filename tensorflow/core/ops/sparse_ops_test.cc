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
class MHTracer_DTPStensorflowPScorePSopsPSsparse_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSsparse_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSsparse_ops_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(SparseOpsTest, SparseTensorDenseAdd_ShapeFn) {
  ShapeInferenceTestOp op("SparseTensorDenseAdd");

  // Copies input 3 to output 0.
  INFER_OK(op, "?;?;?;?", "in3");
}

TEST(SparseOpsTest, SparseAdd_ShapeFn) {
  ShapeInferenceTestOp op("SparseAdd");

  INFER_OK(op, "?;?;?;?;?;?;?", "[?,?];[?];[?]");

  // input(2) determines the output[0].
  INFER_OK(op, "?;?;[?];?;?;?;?", "[?,d2_0];[?];in2");
  INFER_OK(op, "?;?;[1];?;?;?;?", "[?,d2_0];[?];in2");
}

TEST(SparseOpsTest, SparseAddGrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseAddGrad");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "?;?;[1];?");
  INFER_ERROR("must be rank 2", op, "?;[1];?;?");

  INFER_OK(op, "?;?;?;?", "[?];[?]");

  // input[1].dim(0) and input[2].dim(0) determine output.
  INFER_OK(op, "?;[?,?];[?,?];?", "[d1_0];[d2_0]");
}

TEST(SparseOpsTest, SparseSliceGrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseSliceGrad");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "?;[1];?;?");

  INFER_OK(op, "?;?;?;?", "[?]");

  // input[1].dim(0) determine output.
  INFER_OK(op, "?;[?,?];?;?", "[d1_0]");
}

TEST(SparseOpsTest, SparseReorder_ShapeFn) {
  ShapeInferenceTestOp op("SparseReorder");

  // Inputs are input_indices, input_values, and input_shape.

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix and vector.
  INFER_OK(op, "?;?;?", "[?,?];[?]");

  // input_indices and input_values and transferred to outputs 0 and 1.
  INFER_OK(op, "[?,?];[?];?", "in0;in1");
}

TEST(SparseOpsTest, SparseReshape_ShapeFn) {
  ShapeInferenceTestOp op("SparseReshape");

  // Inputs are input_indices, input_shape, and new_shape.

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix and vector.
  INFER_OK(op, "?;?;?", "[?,?];[?]");

  // first output is matrix [input_indices.dim(0), new_shape.dim(0)].
  // new_shape is transferred to second output.
  INFER_OK(op, "[?,?];?;[?]", "[d0_0,d2_0];in2");
}

TEST(SparseOpsTest, SparseSplit_ShapeFn) {
  ShapeInferenceTestOp op("SparseSplit");
  TF_ASSERT_OK(NodeDefBuilder("test", "SparseSplit")
                   .Input({"split_dim", 0, DT_INT64})
                   .Input({"indices", 1, DT_INT64})
                   .Input({"values", 2, DT_INT64})
                   .Input({"shape", 3, DT_INT64})
                   .Attr("num_split", 2)  // each output is copied twice.
                   .Finalize(&op.node_def));

  // output has three shape types, derived from input_shape (which is input(3)).
  // each type is copied #splits times.
  // First output is [?, NumElements(input_shape)].
  // Second output is [?]
  // Third output is input_shape.
  INFER_OK(op, "?;?;?;?", "[?,?];[?,?];[?];[?];in3;in3");
  INFER_OK(op, "?;?;?;[5,4,3,2,1]", "[?,120];[?,120];[?];[?];in3;in3");
}

TEST(SparseOpsTest, SparseToDense_ShapeFn) {
  ShapeInferenceTestOp op("SparseToDense");
  op.input_tensors.resize(4);

  // input[1] is the shape tensor.
  INFER_OK(op, "?;?;?;?", "?");
  INFER_OK(op, "?;[?];?;?", "?");
  INFER_OK(op, "?;[4];?;?", "[?,?,?,?]");
  Tensor in_t = test::AsTensor<int32>({1, 2, 3, 4});
  op.input_tensors[1] = &in_t;
  INFER_OK(op, "?;[4];?;?", "[1,2,3,4]");
}

TEST(SparseOpsTest, SparseReduceSum_ShapeFn) {
  ShapeInferenceTestOp op("SparseReduceSum");
  TF_ASSERT_OK(NodeDefBuilder("test", "SparseReduceSum")
                   .Input({"input_indices", 0, DT_INT64})
                   .Input({"input_values", 1, DT_INT64})
                   .Input({"input_shape", 2, DT_INT64})
                   .Input({"reduction_axes", 3, DT_INT32})
                   .Attr("keep_dims", false)
                   .Finalize(&op.node_def));

  // Shape fn always yields unknown.
  INFER_OK(op, "?;?;?;?", "?");
}

TEST(SparseOpsTest, SerializeSparse_ShapeFn) {
  ShapeInferenceTestOp op("SerializeSparse");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always vector of size 3.
  INFER_OK(op, "?;?;?", "[3]");
}

TEST(SparseOpsTest, SerializeManySparse_ShapeFn) {
  ShapeInferenceTestOp op("SerializeManySparse");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix of [?,3].
  INFER_OK(op, "?;?;?", "[?,3]");
}

TEST(SparseOpsTest, DeserializeManySparse_ShapeFn) {
  ShapeInferenceTestOp op("DeserializeManySparse");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1]");
  INFER_ERROR("must be 3", op, "[?,4]");

  // output is always [?,?];[?];[?].
  INFER_OK(op, "?", "[?,?];[?];[?]");
  INFER_OK(op, "[?,3]", "[?,?];[?];[?]");
}

TEST(SparseOpsTest, SparseTensorDenseMatMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseTensorDenseMatMul");
  auto set_adjoints = [&op](bool adjoint_a, bool adjoint_b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSsparse_ops_testDTcc mht_0(mht_0_v, 355, "", "./tensorflow/core/ops/sparse_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "SparseTensorDenseMatMul")
                     .Input({"a_indices", 1, DT_INT64})
                     .Input({"a_values", 2, DT_INT64})
                     .Input({"a_shape", 3, DT_INT64})
                     .Input({"b", 3, DT_INT64})
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Finalize(&op.node_def));
  };

  // Inputs are a_indices, a_values, a_shape, b.
  set_adjoints(false, false);

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?;?");
  INFER_ERROR("must be rank 1", op, "?;?;[];?");
  INFER_ERROR("must be rank 2", op, "?;?;[3];?");
  INFER_ERROR("must be rank 2", op, "?;?;?;[]");

  // second output dim comes from b, depending on adjoint_b value.
  INFER_OK(op, "?;?;?;?", "[?,?]");
  INFER_OK(op, "?;?;?;[?,?]", "[?,d3_1]");    // use d3_1, !adjoint_b.
  INFER_OK(op, "?;?;?;[1,2]", "[?,d3_1]");    // use d3_1, !adjoint_b.
  INFER_OK(op, "?;?;[2];[1,2]", "[?,d3_1]");  // use d3_1, !adjoint_b.

  set_adjoints(false, true);
  INFER_OK(op, "?;?;?;[?,?]", "[?,d3_0]");  // use d3_0, adjoint_b.
  INFER_OK(op, "?;?;?;[1,2]", "[?,d3_0]");  // use d3_0, adjoint_b.

  // first output comes from a, depending on adjoint_a value.
  // When input tensor is known, its values determine output shape.
  Tensor a_shape_t = test::AsTensor<int64_t>(std::vector<int64_t>{3, 1});
  op.input_tensors.resize(4);
  op.input_tensors[2] = &a_shape_t;

  // Multiplying matrices of shape [3, 1] x [1, 2]
  set_adjoints(false, false);
  INFER_OK(op, "?;?;[2];[1,2]", "[3,d3_1]");  // use d3_1, !adjoint_b.
  INFER_OK(op, "?;?;?;[1,2]", "[3,d3_1]");    // use d3_1, !adjoint_b.

  set_adjoints(true, false);
  // Trying to multiply matrices of [1, 3] x [1, 2]
  INFER_ERROR("must be equal", op, "?;?;[2];[1,2]");  // adjoint_a, !adjoint_b.

  // Try with shape tensor describing shape of rank 3.
  a_shape_t = test::AsTensor<int64_t>(std::vector<int64_t>{3, 1, 2});
  INFER_ERROR("must be rank 2 but is rank 3", op, "?;?;[3];[1,2]");
}

TEST(SparseOpsTest, SparseSoftmax_ShapeFn) {
  ShapeInferenceTestOp op("SparseSoftmax");

  // Inputs are sp_indices, sp_values, sp_shape.

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is values_shape.
  INFER_OK(op, "?;?;?", "[?]");
  INFER_OK(op, "?;[?];?", "in1");
  INFER_OK(op, "?;[5];?", "in1");
}

TEST(SparseOpsTest, SparseSparseMinAndMin_ShapeFn) {
  for (const char* op_name : {"SparseSparseMaximum", "SparseSparseMinimum"}) {
    ShapeInferenceTestOp op(op_name);

    // Rank checks.
    INFER_ERROR("must be rank 2", op, "[1];?;?;?;?;?");  // a_indices
    INFER_ERROR("must be rank 1", op, "?;[];?;?;?;?");   // a_values
    INFER_ERROR("must be rank 1", op, "?;?;[];?;?;?");   // a_shape
    INFER_ERROR("must be rank 2", op, "?;?;?;[];?;?");   // b_indices
    INFER_ERROR("must be rank 1", op, "?;?;?;?;[];?");   // b_values
    INFER_ERROR("must be rank 1", op, "?;?;?;?;?;[]");   // b_shape

    // output is always [?,?];[?]
    INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?]");
    INFER_OK(op, "?;[?];?;?;?;?", "[?,?];[?]");
    INFER_OK(op, "?;[5];?;?;?;?", "[?,?];[?]");
  }
}

TEST(SparseOpsTest, SparseConcat_ShapeFn) {
  ShapeInferenceTestOp op("SparseConcat");
  std::vector<NodeDefBuilder::NodeOut> src_list;
  int n = 2;
  src_list.reserve(n);
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_INT64);
  TF_ASSERT_OK(NodeDefBuilder("test", "SparseConcat")
                   .Input(src_list)
                   .Input(src_list)
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?;?;?;?");  // indices
  INFER_ERROR("must be rank 2", op, "?;[1];?;?;?;?");  // indices
  INFER_ERROR("must be rank 1", op, "?;?;[];?;?;?");   // values
  INFER_ERROR("must be rank 1", op, "?;?;?;[];?;?");   // values
  INFER_ERROR("must be rank 1", op, "?;?;?;?;[];?");   // shapes
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;[]");   // shapes

  // row count is sum of (indices[i].dim(0) merge values[i].dim(0))
  // ind_cols is merge of (indices[i].dim(1))
  //
  // output 0 is matrix [row_count, ind_cols]
  // output 1 is matrix [row_count]
  // output 2 is merge of all shapes

  // Test merge of shapes.
  INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?];[?]");
  INFER_OK(op, "?;?;?;?;[?];[?]", "[?,?];[?];in4|in5");
  INFER_OK(op, "?;?;?;?;[?];[5]", "[?,?];[?];in5");

  // Test accumulation of row_count and ind_cols from indices.
  INFER_OK(op, "[4,5];[3,?];?;?;?;?", "[7,d0_1];[7];[?]");

  // Test accumulation of row_count and ind_cols from values.
  INFER_OK(op, "?;?;[4];[3];?;?", "[7,?];[7];[?]");

  // Test merge between row_count and ind_cols.
  INFER_OK(op, "[?,2];[3,?];[4];[?];?;?", "[7,d0_1];[7];[?]");

  // Test some errors during merge.
  INFER_ERROR("but are 100 and 200", op, "[100,?];[?,?];[200];[?];?;?");
  INFER_ERROR("but are 2 and 3", op, "[?,2];[?,3];[?];[?];?;?");
  INFER_ERROR("but are 4 and 5", op, "?;?;?;?;[4];[5]");
}

TEST(SparseOpsTest, SparseDenseCwise_ShapeFn) {
  for (const char* op_name :
       {"SparseDenseCwiseMul", "SparseDenseCwiseDiv", "SparseDenseCwiseAdd"}) {
    ShapeInferenceTestOp op(op_name);

    // output is always a vector.
    INFER_OK(op, "?;?;?;?", "[?]");

    // input(0).dim(0) determines output[0].
    INFER_OK(op, "[?,?];?;?;?", "[d0_0]");

    // Rank checks.
    INFER_ERROR("must be rank 2", op, "[1];?;?;?");
  }
}

TEST(SparseOpsTest, AddSparseToTensorsMap_ShapeFn) {
  ShapeInferenceTestOp op("AddSparseToTensorsMap");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always scalar
  INFER_OK(op, "?;?;?", "[]");
}

TEST(SparseOpsTest, AddManySparseToTensorsMap_ShapeFn) {
  ShapeInferenceTestOp op("AddManySparseToTensorsMap");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix of [?].
  INFER_OK(op, "?;?;?", "[?]");
}

TEST(SparseOpsTest, TakeManySparseFromTensorsMap_ShapeFn) {
  ShapeInferenceTestOp op("TakeManySparseFromTensorsMap");

  // Rank checks.
  INFER_ERROR("must be rank 1", op, "[?,1]");

  // output is always [?,?];[?];[?].
  INFER_OK(op, "?", "[?,?];[?];[?]");
  INFER_OK(op, "[?]", "[?,?];[?];[?]");
}

}  // end namespace tensorflow
