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
class MHTracer_DTPStensorflowPScorePSopsPSlinalg_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSlinalg_ops_testDTcc() {
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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(LinalgOpsTest, MatrixDeterminant_ShapeFn) {
  ShapeInferenceTestOp op("MatrixDeterminant");
  INFER_OK(op, "?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 2 and 1", op, "[1,?,3,4,1,2]");

  INFER_OK(op, "[?,?]", "[]");
  INFER_OK(op, "[1,?]", "[]");
  INFER_OK(op, "[?,1]", "[]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[1,?,3,4,?,?]", "[d0_0,d0_1,d0_2,d0_3]");
  INFER_OK(op, "[1,?,3,4,1,?]", "[d0_0,d0_1,d0_2,d0_3]");
  INFER_OK(op, "[1,?,3,4,?,1]", "[d0_0,d0_1,d0_2,d0_3]");
}

TEST(LinalgOpsTest, UnchangedSquare_ShapeFn) {
  for (const char* op_name : {"Cholesky", "CholeskyGrad", "MatrixInverse"}) {
    ShapeInferenceTestOp op(op_name);

    const string extra_shape = (op.name == "CholeskyGrad" ? ";?" : "");

    INFER_OK(op, "?" + extra_shape, "?");
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
                "[1]" + extra_shape);
    INFER_ERROR("Dimensions must be equal, but are 1 and 2", op,
                "[1,2]" + extra_shape);

    INFER_OK(op, "[?,?]" + extra_shape, "[d0_0|d0_1,d0_0|d0_1]");
    INFER_OK(op, "[1,?]" + extra_shape, "[d0_0,d0_0]");
    INFER_OK(op, "[?,1]" + extra_shape, "[d0_1,d0_1]");

    // Repeat previous block of tests with input rank > 2.
    INFER_OK(op, "[5,?,7,?,?]" + extra_shape,
             "[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
    INFER_OK(op, "[5,?,7,1,?]" + extra_shape, "[d0_0,d0_1,d0_2,d0_3,d0_3]");
    INFER_OK(op, "[5,?,7,?,1]" + extra_shape, "[d0_0,d0_1,d0_2,d0_4,d0_4]");
  }
}

TEST(LinalgOpsTest, SelfAdjointEig_ShapeFn) {
  ShapeInferenceTestOp op("SelfAdjointEig");
  INFER_OK(op, "?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2]");

  INFER_OK(op, "[?,?]", "[?,d0_0|d0_1]");
  INFER_OK(op, "[1,?]", "[2,d0_0]");
  INFER_OK(op, "[?,1]", "[2,d0_1]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[5,?,7,?,?]", "[d0_0,d0_1,d0_2,?,d0_3|d0_4]");
  INFER_OK(op, "[5,?,7,1,?]", "[d0_0,d0_1,d0_2,2,d0_3]");
  INFER_OK(op, "[5,?,7,?,1]", "[d0_0,d0_1,d0_2,2,d0_4]");
}

TEST(LinalgOpsTest, SelfAdjointEigV2_ShapeFn) {
  ShapeInferenceTestOp op("SelfAdjointEigV2");
  auto set_compute_v = [&op](bool compute_v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_ops_testDTcc mht_0(mht_0_v, 251, "", "./tensorflow/core/ops/linalg_ops_test.cc", "lambda");

    // Test for float32
    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input({{"input", 0, DT_FLOAT}})
                     .Attr("compute_v", compute_v)
                     .Finalize(&op.node_def));

    // Test for float16
    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input({{"input", 0, DT_HALF}})
                     .Attr("compute_v", compute_v)
                     .Finalize(&op.node_def));
  };
  set_compute_v(false);
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[3,1,2]");

  INFER_OK(op, "?", "?;[0]");
  INFER_OK(op, "[?,?]", "[d0_0|d0_1];[0]");
  INFER_OK(op, "[1,?]", "[d0_0|d0_1];[0]");
  INFER_OK(op, "[?,1]", "[d0_0|d0_1];[0]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[5,?,7,?,?]", "[d0_0,d0_1,d0_2,d0_3|d0_4];[0]");
  INFER_OK(op, "[5,?,7,1,?]", "[d0_0,d0_1,d0_2,d0_3|d0_4];[0]");
  INFER_OK(op, "[5,?,7,?,1]", "[d0_0,d0_1,d0_2,d0_3|d0_4];[0]");

  set_compute_v(true);
  INFER_OK(op, "?", "?;?");
  INFER_OK(op, "[?,?]", "[d0_0|d0_1];[d0_0|d0_1,d0_0|d0_1]");
  INFER_OK(op, "[1,?]", "[d0_0|d0_1];[d0_0|d0_1,d0_0|d0_1]");
  INFER_OK(op, "[?,1]", "[d0_0|d0_1];[d0_0|d0_1,d0_0|d0_1]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[5,?,7,?,?]",
           "[d0_0,d0_1,d0_2,d0_3|d0_4];[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
  INFER_OK(op, "[5,?,7,1,?]",
           "[d0_0,d0_1,d0_2,d0_3|d0_4];[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
  INFER_OK(op, "[5,?,7,?,1]",
           "[d0_0,d0_1,d0_2,d0_3|d0_4];[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
}

TEST(LinalgOpsTest, MatrixSolve_ShapeFn) {
  ShapeInferenceTestOp op("MatrixSolve");
  INFER_OK(op, "?;?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1];?");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2];?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[5,?,?];[6]");
  INFER_ERROR("Shapes must be equal rank, but are 0 and 1", op,
              "[5,?];[6,?,?]");

  INFER_OK(op, "[?,?];?", "[d0_0|d0_1,?]");

  // Inputs are [...,M,M] and [...,M,K].  Output is [...,M,K].
  // First test where ... is empty.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[?,?];[1,?]", "[d1_0,d1_1]");
  INFER_OK(op, "[1,?];[1,?]", "[d0_0|d1_0,d1_1]");
  INFER_OK(op, "[?,1];[1,?]", "[d0_1|d1_0,d1_1]");
  INFER_OK(op, "[1,1];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[1,1];[1,?]", "[d0_0|d0_1|d1_0,d1_1]");
  // Test with ... being 2-d.
  INFER_OK(op, "[10,?,?,?];[?,20,1,?]", "[d0_0,d1_1,d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,?];[?,20,1,?]", "[d0_0,d1_1,d0_2|d1_2,d1_3]");
  INFER_OK(op, "[10,?,?,1];[?,20,1,?]", "[d0_0,d1_1,d0_3|d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,?,?]", "[d0_0,d1_1,d0_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,1,?]", "[d0_0,d1_1,d0_2|d0_3|d1_2,d1_3]");
}

TEST(LinalgOpsTest, MatrixTriangularSolve_ShapeFn) {
  ShapeInferenceTestOp op("MatrixTriangularSolve");
  INFER_OK(op, "?;?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1];?");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2];?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[5,?,?];[6]");

  // Inputs are [...,M,M] and [...,M,K].  Output is [...,M,K].
  // First test where ... is empty.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[?,?];[1,?]", "[d1_0,d1_1]");
  INFER_OK(op, "[1,?];[1,?]", "[d0_0|d1_0,d1_1]");
  INFER_OK(op, "[?,1];[1,?]", "[d0_1|d1_0,d1_1]");
  INFER_OK(op, "[1,1];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[1,1];[1,?]", "[d0_0|d0_1|d1_0,d1_1]");
  // Test with ... being 2-d.
  INFER_OK(op, "[10,?,?,?];[?,20,1,?]", "[d0_0,d1_1,d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,?];[?,20,1,?]", "[d0_0,d1_1,d0_2|d1_2,d1_3]");
  INFER_OK(op, "[10,?,?,1];[?,20,1,?]", "[d0_0,d1_1,d0_3|d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,?,?]", "[d0_0,d1_1,d0_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,1,?]", "[d0_0,d1_1,d0_2|d0_3|d1_2,d1_3]");
}

TEST(LinalgOpsTest, MatrixSolveLs_ShapeFn) {
  ShapeInferenceTestOp op("MatrixSolveLs");
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "?;?;[]", "?");

  // Inputs are [...,M,N], [...,M,K], and scalar regularizer.
  // Output is [...,N,K]
  // Test with no batch dims.
  INFER_OK(op, "[1,?];[1,?];?", "[d0_1,d1_1]");
  INFER_OK(op, "[1,2];[1,3];?", "[d0_1,d1_1]");
  INFER_ERROR("Dimensions must be equal, but are 5 and 6", op, "[5,?];[6,?];?");
  // Test with batch dims.
  INFER_OK(op, "[10,?,1,?];[?,20,1,?];?", "[d0_0,d1_1,d0_3,d1_3]");
  INFER_OK(op, "[10,20,1,2];[10,20,1,3];?", "[d0_0|d1_0,d0_1|d1_1,d0_3,d1_3]");
  INFER_ERROR("Dimensions must be equal, but are 5 and 6", op,
              "[10,?,5,?];[?,20,6,?];?");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 10 and 11", op,
              "[10,?,5,?];[11,?,5,?];?");

  // Rank checks.
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[?];?;?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "?;[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[1]");
}

TEST(LinalgOpsTest, Qr_ShapeFn) {
  ShapeInferenceTestOp op("Qr");
  auto set_attrs = [&op](bool full_matrices) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_ops_testDTcc mht_1(mht_1_v, 374, "", "./tensorflow/core/ops/linalg_ops_test.cc", "lambda");

    // Test float32
    TF_ASSERT_OK(NodeDefBuilder("test", "Qr")
                     .Input({"input", 0, DT_FLOAT})
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));

    // Test float16
    TF_ASSERT_OK(NodeDefBuilder("test", "Qr")
                     .Input({"input", 0, DT_HALF})
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));
  };

  // Defining `P` = min(`M`, `N`), if full_matrices = False, then Q should be
  // `M` x `P` and `R` should be `P` x `N`. Otherwise, Q should be
  // `M` x `M` and `R` should be `M` x `N`.
  //
  // For rank-3 tensors, `M` = d0_1 and `N` = d0_2.
  //
  set_attrs(false);
  INFER_OK(op, "?", "?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[4,?,?]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[4,2,?]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[4,?,2]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");

  set_attrs(true);
  INFER_OK(op, "?", "?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,?,?]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,?]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,?,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
}

TEST(LinalgOpsTest, Svd_ShapeFn) {
  ShapeInferenceTestOp op("Svd");
  auto set_attrs = [&op](bool compute_uv, bool full_matrices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPSlinalg_ops_testDTcc mht_2(mht_2_v, 428, "", "./tensorflow/core/ops/linalg_ops_test.cc", "lambda");

    // Test for float32
    TF_ASSERT_OK(NodeDefBuilder("test", "Svd")
                     .Input({"input", 0, DT_FLOAT})
                     .Attr("compute_uv", compute_uv)
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));

    // Test for float16
    TF_ASSERT_OK(NodeDefBuilder("test", "Svd")
                     .Input({"input", 0, DT_HALF})
                     .Attr("compute_uv", compute_uv)
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));
  };

  // Defining `P` = min(`M`, `N`), if full_matrices = False, then U should be
  // `M` x `P` and `V` should be `N` x `P`. Otherwise, U should be
  // `M` x `M` and `V` should be `N` x `N`.
  //
  // For rank-3 tensors, `M` = d0_1 and `N` = d0_2.
  //
  set_attrs(false, false);
  INFER_OK(op, "?", "?;[0];[0]");
  INFER_OK(op, "[?,?,?]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[4,?,?]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[4,2,?]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[4,?,2]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1];[0];[0]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1];[0];[0]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_2];[0];[0]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_2];[0];[0]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1];[0];[0]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1];[0];[0]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");

  set_attrs(true, false);
  INFER_OK(op, "?", "?;?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[4,?,?]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[4,2,?]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[4,?,2]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");

  set_attrs(true, true);
  INFER_OK(op, "?", "?;?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,?,?]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,2,?]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,?,2]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
}

TEST(LinalgOpsTest, Lu_ShapeFn) {
  ShapeInferenceTestOp op("Lu");
  INFER_OK(op, "?", "?;?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,?,3,4,1,2]");

  INFER_OK(op, "[?,?]", "[d0_0,d0_0];[d0_0]");
  INFER_OK(op, "[1,?]", "[d0_0,d0_0];[d0_0]");
  INFER_OK(op, "[?,1]", "[d0_1,d0_1];[d0_1]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[1,?,3,4,?,?]",
           "[d0_0,d0_1,d0_2,d0_3,d0_4,d0_4];[d0_0,d0_1,d0_2,d0_3,d0_4]");
  INFER_OK(op, "[1,?,3,4,1,?]",
           "[d0_0,d0_1,d0_2,d0_3,d0_4,d0_4];[d0_0,d0_1,d0_2,d0_3,d0_4]");
  INFER_OK(op, "[1,?,3,4,?,1]",
           "[d0_0,d0_1,d0_2,d0_3,d0_5,d0_5];[d0_0,d0_1,d0_2,d0_3,d0_5]");
}

TEST(LinalgOpsTest, TridiagonalMatMul_ShapeFn) {
  ShapeInferenceTestOp op("TridiagonalMatMul");
  INFER_OK(op, "?;?;?;?", "in3");
  INFER_OK(op, "[1,5];[1,5];[1,5];[?,1]", "in3");
  INFER_OK(op, "[1,5];[1,5];[1,5];[5,1]", "in3");

  INFER_OK(op, "[?,1,?];[?,1,?];[?,1,?];[?,?,?]", "in3");
  INFER_OK(op, "[?,1,5];[?,1,5];[?,1,5];[7,5,2]", "in3");
  INFER_OK(op, "[7,1,5];[7,1,5];[7,1,5];[?,5,2]", "in3");
  INFER_OK(op, "[7,1,5];[7,1,5];[7,1,5];[7,5,2]", "in3");

  INFER_OK(op, "[7,?,1,5];[7,?,1,5];[7,?,1,5];[7,8,5,2]", "in3");
  INFER_OK(op, "[7,8,1,5];[7,8,1,5];[7,8,1,5];[7,8,5,2]", "in3");

  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[3];[3];[3];[5,1]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[3,5];[3,5];[3,5];[5]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [6,4] and [6,8].",
      op, "[6,4,3,5];[6,4,3,5];[6,4,3,5];[6,8,5,2]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [?,4] and [6,8].",
      op, "[?,4,3,5];[?,4,3,5];[?,4,3,5];[6,8,5,2]");

  // Diagonals must have the same length.
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 5 and 6. "
      "Shapes are [1,5] and [1,6]",
      op, "[1,5];[1,6];[1,5];[6,2]");

  // Diagonals must be 1-row matrices.
  INFER_ERROR("Dimension must be 1 but is 3", op, "[3,5];[3,5];[3,5];[5,2]");
}

TEST(LinalgOpsTest, TridiagonalSolve_ShapeFn) {
  ShapeInferenceTestOp op("TridiagonalSolve");
  INFER_OK(op, "?;?", "in1");
  INFER_OK(op, "[3,5];[?,1]", "in1");
  INFER_OK(op, "[?,5];[5,1]", "in1");
  INFER_OK(op, "[?,5];[?,?]", "in1");
  INFER_OK(op, "[?,?];[?,?]", "in1");
  INFER_OK(op, "[3,5];[5,1]", "in1");
  INFER_OK(op, "[3,5];[5,2]", "in1");

  INFER_OK(op, "[?,?,?];[?,?,?]", "in1");
  INFER_OK(op, "[?,3,5];[7,5,2]", "in1");
  INFER_OK(op, "[7,3,5];[?,5,2]", "in1");
  INFER_OK(op, "[7,?,5];[?,5,?]", "in1");
  INFER_OK(op, "[7,3,5];[7,5,2]", "in1");

  INFER_OK(op, "[7,?,3,5];[7,8,5,2]", "in1");
  INFER_OK(op, "[7,8,3,5];[7,8,5,2]", "in1");

  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[3];[5,1]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[3,5];[5]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [6,4] and [6,8].",
      op, "[6,4,3,5];[6,8,5,2]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [?,4] and [6,8].",
      op, "[?,4,3,5];[6,8,5,2]");
  INFER_ERROR("Dimension must be 3 but is 4", op, "[4,5];[5,2]");
  INFER_ERROR("Dimension must be 3 but is 4", op, "[6,4,5];[6,5,2]");
  INFER_ERROR("Dimensions must be equal, but are 9 and 5", op, "[3,9];[5,2]");
  INFER_ERROR("Dimensions must be equal, but are 9 and 5", op,
              "[6,3,9];[6,5,2]");
}

}  // end namespace tensorflow
