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
class MHTracer_DTPStensorflowPScorePSopsPStraining_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPStraining_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPStraining_ops_testDTcc() {
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Used for testing the grad+indices handling for SparseApplyXYZ tests.
static void TestGradAndIndicesErrorHandling(const ShapeInferenceTestOp& op,
                                            string shape_spec_middle,
                                            const string& shape_spec_end = "") {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("shape_spec_middle: \"" + shape_spec_middle + "\"");
   MHTracer_DTPStensorflowPScorePSopsPStraining_ops_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/ops/training_ops_test.cc", "TestGradAndIndicesErrorHandling");

  auto shape_spec = [&shape_spec_middle, shape_spec_end](
                        const char* var_spec, const char* grad_indices_spec) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("var_spec: \"" + (var_spec == nullptr ? std::string("nullptr") : std::string((char*)var_spec)) + "\"");
   mht_1_v.push_back("grad_indices_spec: \"" + (grad_indices_spec == nullptr ? std::string("nullptr") : std::string((char*)grad_indices_spec)) + "\"");
   MHTracer_DTPStensorflowPScorePSopsPStraining_ops_testDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/ops/training_ops_test.cc", "lambda");

    return strings::StrCat(var_spec, ";", shape_spec_middle, ";",
                           grad_indices_spec, shape_spec_end);
  };

  // mismatch between grad[1] and var[1].
  INFER_ERROR("Dimension 1 in both shapes must be equal", op,
              shape_spec("[?,1]", "[?,2];[?]"));
  // grad[0] and indices[0] must match.
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op,
              shape_spec("?", "[2,?];[1]"));
  // grad is wrong rank.
  INFER_ERROR("must be equal rank", op, shape_spec("[1]", "[?,2];[?]"));
  // indices is wrong rank.
  INFER_ERROR("Shape must be rank 1 but is rank 2", op,
              shape_spec("[?]", "[?];[1,2]"));
}

TEST(TrainingOpsTest, ApplyGradientDescent_ShapeFn) {
  ShapeInferenceTestOp op("ApplyGradientDescent");

  // Output is a merge of inputs 0 and 2 (var and delta).
  INFER_OK(op, "[1,?];[];[?,2]", "[d0_0,d2_1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[];[2]");

  // alpha must be a scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?];?");
}

TEST(TrainingOpsTest, ApplyProximalGradientDescent_ShapeFn) {
  ShapeInferenceTestOp op("ApplyProximalGradientDescent");

  // Output is a merge of inputs 0 and 4 (var and delta).
  INFER_OK(op, "[1,?];[];[];[];[?,2]", "[d0_0,d4_1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[];[];[];[2]");

  // alpha, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyProximalGradientDescent_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyProximalGradientDescent");

  // Output is a merge of inputs 0 (var) and the non-indices part of 4 (delta).
  INFER_OK(op, "[1,?];[];[];[];[?,2];[3]", "[d0_0,d4_1]");

  TestGradAndIndicesErrorHandling(op, "[];[];[]");

  // alpha, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyAdadelta_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAdadelta");

  // Output is a merge of inputs 0, 1, 2, and 6 (var, accum, accum_update,
  // grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[?,?,?,4]",
           "[d0_0,d1_1,d2_2,d6_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[];[];[];[2]");

  // lr, rho, and epsilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyAdadelta_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyAdadelta");

  // Output is a merge of inputs 0, 1, 2, and non-indices part of 6 (var, accum,
  // accum_update, grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[?,?,?,4];?",
           "[d0_0,d1_1,d2_2,d6_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[1];?");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[1];?");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[?,1];[];[];[];[?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?;?;?;?");

  // lr, rho, and epsilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAdagrad");

  // Output is a merge of inputs 0, 1, and 3 (var, accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3]", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[2]");

  // lr must be a scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyAdagrad");

  // Output is a merge of inputs 0, 1, and non-indices part of 3 (var, accum,
  // grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3];?", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,2];[];[?,1];?");
  INFER_ERROR("Shapes must be equal rank, but are 2 and 3", op,
              "[?,1];[?,1];[];[?,?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?");

  // lr must be a scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyProximalAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("ApplyProximalAdagrad");

  // Output is a merge of inputs 0, 1, and 5 (var, accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[];[];[?,?,3]", "[d0_0,d1_1,d5_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[];[];[2]");

  // lr, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyProximalAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyProximalAdagrad");

  // Output is a merge of inputs 0, 1, and the non-indices part of 5 (var,
  // accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[];[];[?,?,3];?", "[d0_0,d1_1,d5_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[];[];[?,1];?");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[];[];[];[?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?;?;?");

  // lr, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyFtrl_ShapeFn) {
  ShapeInferenceTestOp op("ApplyFtrl");

  // Output is a merge of inputs 0, 1, 2, and 3 (var, accum, linear, grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[?,?,?,4];[];[];[];[]",
           "[d0_0,d1_1,d2_2,d3_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[1];[];[];[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[1];[];[];[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[2];[];[];[];[]");

  // lr, l1, l2, and lr_power must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;?;[?]");
}

TEST(TrainingOpsTest, SparseApplyFtrl_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyFtrl");

  // Output is a merge of inputs 0, 1, 2, and non-indices part of 3 (var, accum,
  // linear, grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[?,?,?,4];?;[];[];[];[]",
           "[d0_0,d1_1,d2_2,d3_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[?,1];?;[];[];[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[?,1];?;[];[];[];[]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[?,1];[?,2];?;[];[];[];[]");

  TestGradAndIndicesErrorHandling(op, "?;?", ";?;?;?;?");

  // lr, l1, l2, and lr_power must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;?;[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;?;?;[?]");
}

TEST(TrainingOpsTest, ApplyMomentum_ShapeFn) {
  ShapeInferenceTestOp op("ApplyMomentum");

  // Output is a merge of inputs 0, 1, and 3 (var, accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3];[]", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[1];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[2];[]");

  // lr and momentum must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?]");
}

TEST(TrainingOpsTest, SparseApplyMomentum_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyMomentum");

  // Output is a merge of inputs 0, 1, and non-indices part of 3 (var, accum,
  // grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3];?;[]", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,2];[];[?,1];?;[]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[];[?,2];?;[]");

  TestGradAndIndicesErrorHandling(op, "?;?", ";?");

  // lr and momentum must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?]");
}

TEST(TrainingOpsTest, ApplyAdam_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAdam");

  // Output is a merge of inputs 0, 1, 2, and 9 (var, m, v, and grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[];[];[];[?,?,?,4]",
           "[d0_0,d1_1,d2_2,d9_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[];[];[];[];[];[];[2]");

  // beta1_power, beta2_power, lr, beta1, beta2, and epsilon must be scalars.
  const char err[] = "Shape must be rank 0 but is rank 1";
  INFER_ERROR(err, op, "?;?;?;[?];?;?;?;?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;[?];?;?;?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;[?];?;?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;?;[?];?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;?;?;[?];?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, ApplyRMSProp_ShapeFn) {
  ShapeInferenceTestOp op("ApplyRMSProp");

  // Output is a merge of inputs 0, 1, 2, and 7 (var, ms, mom, and grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[];[?,?,?,4]",
           "[d0_0,d1_1,d2_2,d7_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[];[];[];[];[2]");

  // lr, rho, momentum, and epsilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyRMSProp_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyRMSProp");

  // Output is a merge of inputs 0, 1, 2, and the non-indices part of 7 (var,
  // ms, mom, and grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[];[?,?,?,4];?",
           "[d0_0,d1_1,d2_2,d7_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[];[1];?");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[];[1];?");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[?,1];[];[];[];[];[?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?;?;?;?;?");

  // lr, rho, momentum, and epsilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyAddSign_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAddSign");

  // Output is a merge of inputs 0, 1, and 6 (var, ms, and grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[];[];[];[?,?,2]", "[d0_0,d1_1,d6_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[];[];[];[2]");

  // lr, alpha, sign_decay, and beta must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, ApplyPowerSign_ShapeFn) {
  ShapeInferenceTestOp op("ApplyPowerSign");

  // Output is a merge of inputs 0, 1, and 6 (var, ms, and grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[];[];[];[?,?,2]", "[d0_0,d1_1,d6_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[];[];[];[2]");

  // lr, logbase, sign_decay, and beta must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?");
}

}  // end namespace tensorflow
