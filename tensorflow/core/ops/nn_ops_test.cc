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
class MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc() {
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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(NNOpsTest, TopK_ShapeFn) {
  ShapeInferenceTestOp op("TopK");
  auto set_k = [&op](int k) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/ops/nn_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input({{"a", 0, DT_FLOAT}})
                     .Attr("k", k)
                     .Finalize(&op.node_def));
  };

  set_k(20);
  // With known input, each output is an unknown shape.
  INFER_OK(op, "?", "?;?");
  // With vector input, each output is [k].
  INFER_OK(op, "[20]", "[20];[20]");
  INFER_OK(op, "[21]", "[20];[20]");

  // With input rank 3, each output is the two first 2 dims of input, plus k.
  INFER_OK(op, "[1,?,21]", "[d0_0,d0_1,20];[d0_0,d0_1,20]");
  // With input rank 4, each output is the two first 3 dims of input, plus k.
  INFER_OK(op, "[1,?,21,?]", "[d0_0,d0_1,d0_2,20];[d0_0,d0_1,d0_2,20]");

  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 1", op, "[1]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 4", op,
              "[1,2,3,4]");
  set_k(-1);
  INFER_ERROR("Need k >= 0, got -1", op, "[1,2,3,4]");
}

TEST(NNOpsTest, TopKV2_ShapeFn) {
  ShapeInferenceTestOp op("TopKV2");
  op.input_tensors.resize(2);

  Tensor k_t;
  op.input_tensors[1] = &k_t;

  k_t = test::AsScalar<int32>(20);
  // With known input, each output is an unknown shape.
  INFER_OK(op, "?;[]", "?;?");
  // With vector input, each output is [k].
  INFER_OK(op, "[20];[]", "[20];[20]");

  // With input rank 3, each output is the two first 2 dims of input, plus k.
  INFER_OK(op, "[1,?,21];[]", "[d0_0,d0_1,20];[d0_0,d0_1,20]");
  // With input rank 4, each output is the two first 3 dims of input, plus k.
  INFER_OK(op, "[1,?,21,?];[]", "[d0_0,d0_1,d0_2,20];[d0_0,d0_1,d0_2,20]");

  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 1", op,
              "[1];[]");
  INFER_ERROR("input must have last dimension >= k = 20 but is 4", op,
              "[1,2,3,4];[]");
  k_t = test::AsScalar<int32>(-1);
  INFER_ERROR(
      "Dimension size, given by scalar input 1, must be non-negative but is -1",
      op, "[1,2,3,4];[]");
}

TEST(NNOpsTest, NthElement_ShapeFn) {
  ShapeInferenceTestOp op("NthElement");
  op.input_tensors.resize(2);

  Tensor n_t;
  op.input_tensors[1] = &n_t;
  n_t = test::AsScalar<int32>(20);

  INFER_OK(op, "?;[]", "?");
  INFER_OK(op, "[21];[]", "[]");
  INFER_OK(op, "[2,?,?];[]", "[d0_0,d0_1]");
  INFER_OK(op, "[?,3,?,21];[]", "[d0_0,d0_1,d0_2]");

  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];[]");
  INFER_ERROR("Input must have last dimension > n = 20 but is 1", op, "[1];[]");
  INFER_ERROR("Input must have last dimension > n = 20 but is 20", op,
              "[1,2,3,20];[]");
  n_t = test::AsScalar<int32>(-1);
  INFER_ERROR(
      "Dimension size, given by scalar input 1, must be non-negative but is -1",
      op, "[1,2,3,4];[]");
}

TEST(NNOpsTest, BatchNormWithGlobalNormalization_ShapeFn) {
  ShapeInferenceTestOp op("BatchNormWithGlobalNormalization");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;?;[1,2,3]");

  // last dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,?,?,d3_0]");
  INFER_OK(op, "?;?;?;?;[1]", "[?,?,?,d4_0]");
  INFER_OK(op, "[1,2,3,4];[4];[4];[4];[4]",
           "[d0_0,d0_1,d0_2,d0_3|d1_0|d2_0|d3_0|d4_0]");
}

TEST(NNOpsTest, QuantizedBatchNormWithGlobalNormalization_ShapeFn) {
  // These are the same tests as BatchNormWithGlobalNormalization tests, but
  // with extra scalar inputs and outputs for the mins and maxes.

  ShapeInferenceTestOp op("QuantizedBatchNormWithGlobalNormalization");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op,
              "[1,2,3];?;?;?;?;?;?;?;?;?;?;?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op,
              "?;?;?;[1,2,3];?;?;?;?;?;?;?;?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op,
              "?;?;?;?;?;?;[1,2,3];?;?;?;?;?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op,
              "?;?;?;?;?;?;?;?;?;[1,2,3];?;?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op,
              "?;?;?;?;?;?;?;?;?;?;?;?;[1,2,3];?;?");

  // last dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;[];[];?;[];[];?;[];[];?;[];[];?;[];[]", "[?,?,?,?];[];[]");
  INFER_OK(op, "?;[];[];[1];[];[];?;[];[];?;[];[];?;[];[]",
           "[?,?,?,d3_0];[];[]");
  INFER_OK(op, "?;[];[];?;[];[];[1];[];[];?;[];[];?;[];[]",
           "[?,?,?,d6_0];[];[]");
  INFER_OK(op, "?;[];[];?;[];[];?;[];[];[1];[];[];?;[];[]",
           "[?,?,?,d9_0];[];[]");
  INFER_OK(op, "?;[];[];?;[];[];?;[];[];?;[];[];[1];[];[]",
           "[?,?,?,d12_0];[];[]");
  INFER_OK(op, "[1,2,3,4];[];[];[4];[];[];[4];[];[];[4];[];[];[4];[];[]",
           "[d0_0,d0_1,d0_2,d0_3|d3_0|d6_0|d9_0|d12_0];[];[]");
}

TEST(NNOpsTest, BatchNormWithGlobalNormalizationGrad_ShapeFn) {
  ShapeInferenceTestOp op("BatchNormWithGlobalNormalizationGrad");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shapes must be equal rank, but are 4 and 3", op,
              "?;?;?;?;[1,2,3]");

  // The first output comes from the first and last inputs merged together.
  // Other inputs are merged with the last dim of that merge result, and that
  // merged vector dim is the last 4 outputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,?,?,d3_0];[d3_0];[d3_0];[d3_0];[d3_0]");
  INFER_OK(op, "[1,?,3,?];[?];[?];[?];[?,2,?,4]",
           "[d0_0,d4_1,d0_2,d4_3];[d4_3];[d4_3];[d4_3];[d4_3]");
}

TEST(NNOpsTest, FusedBatchNorm_ShapeFn) {
  ShapeInferenceTestOp op("FusedBatchNorm");

  auto set_op = [&op](bool is_training, float exponential_avg_factor,
                      string data_format) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("data_format: \"" + data_format + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc mht_1(mht_1_v, 358, "", "./tensorflow/core/ops/nn_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "FusedBatchNorm")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("data_format", data_format)
                     .Attr("is_training", is_training)
                     .Attr("exponential_avg_factor", exponential_avg_factor)
                     .Finalize(&op.node_def));
  };

  set_op(true, 1.0, "NHWC");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "[1,2,3,4];[4];[4];?;?",
           "[d0_0,d0_1,d0_2,d0_3|d1_0|d2_0];"
           "[d0_3|d1_0|d2_0];[d0_3|d1_0|d2_0];"
           "[d0_3|d1_0|d2_0];[d0_3|d1_0|d2_0]");

  set_op(true, 0.5, "NHWC");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "[1,2,3,4];[4];[4];?;?",
           "[d0_0,d0_1,d0_2,d0_3|d1_0|d2_0];"
           "[d0_3|d1_0|d2_0];[d0_3|d1_0|d2_0];"
           "[d0_3|d1_0|d2_0];[d0_3|d1_0|d2_0]");

  set_op(true, 1.0, "NCHW");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,d1_0,?,?];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,d2_0,?,?];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "[1,4,2,3];[4];[4];?;?",
           "[d0_0,d0_1|d1_0|d2_0,d0_2,d0_3];"
           "[d0_1|d1_0|d2_0];[d0_1|d1_0|d2_0];"
           "[d0_1|d1_0|d2_0];[d0_1|d1_0|d2_0]");

  set_op(false, 1.0, "NHWC");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;?;[1,2,3]");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,?,?,d1_0];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,?,?,d3_0];[d3_0];[d3_0];[d3_0];[d3_0]");
  INFER_OK(op, "?;?;?;?;[1]", "[?,?,?,d4_0];[d4_0];[d4_0];[d4_0];[d4_0]");
  INFER_OK(op, "[1,2,3,4];[4];[4];[4];[4]",
           "[d0_0,d0_1,d0_2,d0_3|d1_0|d2_0|d3_0|d4_0];"
           "[d0_3|d1_0|d2_0|d3_0|d4_0];[d0_3|d1_0|d2_0|d3_0|d4_0];"
           "[d0_3|d1_0|d2_0|d3_0|d4_0];[d0_3|d1_0|d2_0|d3_0|d4_0]");

  set_op(false, 1.0, "NCHW");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;?;[1,2,3]");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[?];[?]");
  INFER_OK(op, "?;[1];?;?;?", "[?,d1_0,?,?];[d1_0];[d1_0];[d1_0];[d1_0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,d2_0,?,?];[d2_0];[d2_0];[d2_0];[d2_0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,d3_0,?,?];[d3_0];[d3_0];[d3_0];[d3_0]");
  INFER_OK(op, "?;?;?;?;[1]", "[?,d4_0,?,?];[d4_0];[d4_0];[d4_0];[d4_0]");
  INFER_OK(op, "[1,4,2,3];[4];[4];[4];[4]",
           "[d0_0,d0_1|d1_0|d2_0|d3_0|d4_0,d0_2,d0_3];"
           "[d0_1|d1_0|d2_0|d3_0|d4_0];[d0_1|d1_0|d2_0|d3_0|d4_0];"
           "[d0_1|d1_0|d2_0|d3_0|d4_0];[d0_1|d1_0|d2_0|d3_0|d4_0]");
}

TEST(NNOpsTest, FusedBatchNormGrad_ShapeFn) {
  ShapeInferenceTestOp op("FusedBatchNormGrad");
  auto set_op = [&op](string data_format) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("data_format: \"" + data_format + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc mht_2(mht_2_v, 456, "", "./tensorflow/core/ops/nn_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "FusedBatchNormGrad")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("data_format", data_format)
                     .Finalize(&op.node_def));
  };

  set_op("NCHW");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;?;[1,2,3]");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[0];[0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,d2_0,?,?];[d2_0];[d2_0];[0];[0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,d3_0,?,?];[d3_0];[d3_0];[0];[0]");
  INFER_OK(op, "?;?;?;?;[1]", "[?,d4_0,?,?];[d4_0];[d4_0];[0];[0]");
  INFER_OK(op, "[1,4,2,3];[1,4,2,3];[4];[4];[4]",
           "[d0_0,d0_1|d2_0|d3_0|d4_0,d0_2,d0_3];"
           "[d0_1|d2_0|d3_0|d4_0];[d0_1|d2_0|d3_0|d4_0];[0];[0]");

  set_op("NHWC");
  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?;?;?");
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "?;[1,2,3];?;?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;[1,2,3];?;?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 3", op, "?;?;?;?;[1,2,3]");
  // Channel dim of first input is merged with the single dim in other 4 inputs.
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?];[?];[?];[0];[0]");
  INFER_OK(op, "?;?;[1];?;?", "[?,?,?,d2_0];[d2_0];[d2_0];[0];[0]");
  INFER_OK(op, "?;?;?;[1];?", "[?,?,?,d3_0];[d3_0];[d3_0];[0];[0]");
  INFER_OK(op, "?;?;?;?;[1]", "[?,?,?,d4_0];[d4_0];[d4_0];[0];[0]");
  INFER_OK(op, "[1,2,3,4];[1,2,3,4];[4];[4];[4]",
           "[d0_0,d0_1,d0_2,d0_3|d2_0|d3_0|d4_0];"
           "[d0_3|d2_0|d3_0|d4_0];[d0_3|d2_0|d3_0|d4_0];[0];[0]");
}

TEST(NNOpsTest, Conv2DBackpropInput_ShapeFn) {
  ShapeInferenceTestOp op("Conv2DBackpropInput");

  // Test rank error.
  INFER_ERROR("input_sizes to contain 4 values or 2 values", op,
              "[3];[?,?,?,?];[?,?,?,?]");
  INFER_ERROR("Shape must be rank 4 but is rank 3", op,
              "[4];[?,?,?,?];[?,?,?]");

  // When input_sizes is a 4D shape and the convolution is grouped, the channel
  // size of the input grad doesn't always equal the input channel size of the
  // filter. So, when input_sizes is a 4D shape, the channel size of the input
  // grad is determined by the content of input_sizes.
  INFER_OK(op, "[4];[?,?,2,?];[1,?,?,?]", "[d2_0,?,?,?]");
  // When input_sizes is a 2D shape, the channel size of the input grad always
  // matches the filter shape.
  INFER_OK(op, "[2];[?,?,2,?];[1,?,?,?]", "[d2_0,?,?,d1_2]");
}

TEST(NNOpsTest, Conv3DBackpropInput_ShapeFn) {
  ShapeInferenceTestOp op("Conv3DBackpropInput");

  // Test rank error.
  INFER_ERROR("Shape must be rank 5 but is rank 3", op, "[1,2,3];?;?");

  // input[1] is transferred to output after asserting its rank.
  INFER_OK(op, "?;?;?", "[?,?,?,?,?]");
  INFER_OK(op, "[?,?,?,?,?];?;?", "in0");
  INFER_OK(op, "[?,2,?,4,?];?;?", "in0");
}

TEST(NNOpsTest, Conv3DBackpropFilter_ShapeFn) {
  ShapeInferenceTestOp op("Conv3DBackpropFilter");

  // Test rank error.
  INFER_ERROR("Shape must be rank 5 but is rank 3", op, "?;[1,2,3];?");

  // input[1] is transferred to output after asserting its rank.
  INFER_OK(op, "?;?;?", "[?,?,?,?,?]");
  INFER_OK(op, "?;[?,?,?,?,?];?", "in1");
  INFER_OK(op, "?;[?,2,?,4,?];?", "in1");
}

TEST(NNOpsTest, MaxPool3DGrad_ShapeFn) {
  ShapeInferenceTestOp op("MaxPool3DGrad");

  // Test rank error.
  INFER_ERROR("Shape must be rank 5 but is rank 3", op, "[1,2,3];?;?");

  // input[0] is transferred to output after asserting its rank.
  INFER_OK(op, "?;?;?", "[?,?,?,?,?]");
  INFER_OK(op, "[?,?,?,?,?];?;?", "in0");
  INFER_OK(op, "[?,2,?,4,?];?;?", "in0");
}

TEST(NNOpsTest, LRNGrad_ShapeFn) {
  ShapeInferenceTestOp op("LRNGrad");

  // LRN Grad is a merge of all three inputs, of rank 4.
  INFER_OK(op, "[1,?,?,4];[?,2,?,?];[?,?,3,?]", "[d0_0,d1_1,d2_2,d0_3]");

  // Test rank errors.
  INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?");
  INFER_ERROR("Shapes must be equal rank, but are 4 and 3", op, "?;[1,2,3];?");
  INFER_ERROR("Shapes must be equal rank, but are 4 and 3", op, "?;?;[1,2,3]");
}

TEST(NNOpsTest, MaxPoolGrad_ShapeFn) {
  for (const char* op_name : {"MaxPoolGrad", "MaxPoolGradWithArgmax"}) {
    ShapeInferenceTestOp op(op_name);

    // Test rank error.
    INFER_ERROR("Shape must be rank 4 but is rank 3", op, "[1,2,3];?;?");

    // input[0] is transferred to output after asserting its rank.
    INFER_OK(op, "?;?;?", "[?,?,?,?]");
    INFER_OK(op, "[?,?,?,?];?;?", "in0");
    INFER_OK(op, "[?,2,?,4];?;?", "in0");
  }
}

TEST(NNOpsTest, Dilation2DBackpropInput_ShapeFn) {
  ShapeInferenceTestOp op("Dilation2DBackpropInput");

  // input[0] is transferred to output.
  INFER_OK(op, "?;?;?", "in0");
  INFER_OK(op, "?;[?,?,?,?,?];?", "in0");
  INFER_OK(op, "?;[?,2,?,4,?];?", "in0");
}

TEST(NNOpsTest, Dilation2DBackpropFilter_ShapeFn) {
  ShapeInferenceTestOp op("Dilation2DBackpropFilter");

  // input[1] is transferred to output.
  INFER_OK(op, "?;?;?", "in1");
  INFER_OK(op, "?;[?,?,?,?,?];?", "in1");
  INFER_OK(op, "?;[?,2,?,4,?];?", "in1");
}

TEST(NNOpsTest, MergeBothInputs_ShapeFn) {
  for (const char* op_name : {"ReluGrad", "Relu6Grad", "EluGrad", "SeluGrad",
                              "SoftplusGrad", "SoftsignGrad"}) {
    ShapeInferenceTestOp op(op_name);

    INFER_OK(op, "?;?", "in0|in1");
    INFER_OK(op, "?;[1,?,3]", "in1");
    INFER_OK(op, "[1,?,3];?", "in0");
    INFER_OK(op, "[1,?];[?,2]", "[d0_0,d1_1]");
    INFER_ERROR("Dimension 1 in both shapes must be equal, but are 3 and 2", op,
                "[1,3];[?,2]");
  }
}

TEST(NNOpsTest, SoftmaxCrossEntropyWithLogits_ShapeFn) {
  ShapeInferenceTestOp op("SoftmaxCrossEntropyWithLogits");

  // Inputs are [batch_size,N] and [batch_size,N], and outputs are [batch_size]
  // and
  // [batch_size,N].
  INFER_OK(op, "?;?", "[?];[?,?]");
  INFER_OK(op, "[?,?];[?,?]", "[d0_0|d1_0];in0|in1");
  INFER_OK(op, "[1,2];[?,2]", "[d0_0];in0");
  INFER_OK(op, "[1,?];[?,2]", "[d0_0];[d0_0,d0_1|d1_1]");
  INFER_OK(op, "[?,2];[1,2]", "[d1_0];in1");

  INFER_ERROR("Shape must be broadcasted with rank 2", op, "[1,2,3];?");
  INFER_ERROR("Shape must be broadcasted with rank 2", op, "?;[1,2,3]");

  // Broadcast example
  // [1,4] and [2,4] are broadcasted to [2,4]
  INFER_OK(op, "[1,4];[2,4]", "[d1_0];[d1_0,d0_1|d1_1]");
  // [2,4] and [2,1] are broadcasted to [2,4]
  INFER_OK(op, "[2,4];[2,1]", "[d0_0];[d0_0|d1_0,d0_1]");
  // [1,?] and [2,4] are broadcasted to [2,4]
  INFER_OK(op, "[1,?];[2,4]", "[d1_0];[d1_0,d0_1|d1_1]");
  // [2,4] and [?,1] are broadcasted to [2,4]
  INFER_OK(op, "[2,4];[?,1]", "[d0_0];[d0_0|d1_0,d0_1]");
}

TEST(NNOpsTest, SparseSoftmaxCrossEntropyWithLogits_ShapeFn) {
  ShapeInferenceTestOp op("SparseSoftmaxCrossEntropyWithLogits");

  // Inputs are [batch_size,N] and [batch_size], and outputs are [batch_size]
  // and [batch_size,N].
  INFER_OK(op, "?;?", "[?];[?,?]");
  INFER_OK(op, "[?,?];[?]", "[d0_0|d1_0];[d0_0|d1_0,d0_1]");
  INFER_OK(op, "[1,2];[1]", "[d0_0|d1_0];[d0_0|d1_0,d0_1]");
  INFER_OK(op, "[?,2];[1]", "[d1_0];[d1_0,d0_1]");

  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,?];[2]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[1,2]");
}

TEST(NNOpsTest, InTopK_ShapeFn) {
  ShapeInferenceTestOp op("InTopK");

  // Inputs are [batch_size,N] and [batch_size], and output is [batch_size].
  INFER_OK(op, "?;?", "[?]");
  INFER_OK(op, "[?,?];[?]", "[d0_0|d1_0]");
  INFER_OK(op, "[1,2];[1]", "[d0_0|d1_0]");
  INFER_OK(op, "[?,2];[1]", "[d1_0]");

  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,?];[2]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[1,2,3];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[1,2]");
}

TEST(NNOpsTest, Dilation2DShapeTest) {
  ShapeInferenceTestOp op("Dilation2D");
  auto set_op = [&op](const std::vector<int32>& strides,
                      const std::vector<int32>& rates, const string& padding) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("padding: \"" + padding + "\"");
   MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc mht_3(mht_3_v, 675, "", "./tensorflow/core/ops/nn_ops_test.cc", "lambda");

    TF_ASSERT_OK(NodeDefBuilder("test", "Dilation2D")
                     .Input("input", 0, DT_FLOAT)
                     .Input("filter", 0, DT_FLOAT)
                     .Attr("strides", strides)
                     .Attr("rates", rates)
                     .Attr("padding", padding)
                     .Finalize(&op.node_def));
  };

  // rate rows and cols is 1, so filter_rows and cols are unchanged.
  // We have a 1x1 filter so the output is still 2x2.
  set_op({1, 1, 1, 1}, {1, 1, 1, 1}, "VALID");
  INFER_OK(op, "[1,2,2,2];[1,1,2]", "[d0_0,2,2,d1_2]");

  // rate rows and cols is 2, so filter_rows and cols are changed to
  // be 2 + (2 - 1) = 3.  7x7 input with 3x3 filter and 1x1 stride
  // gives a 5x5 output.
  set_op({1, 1, 1, 1}, {1, 2, 2, 1}, "VALID");
  INFER_OK(op, "[1,7,7,2];[2,2,2]", "[d0_0,5,5,d1_2]");
}

TEST(NNOpsTest, FractionalPool_ShapeFn) {
  for (const char* op_name : {"FractionalAvgPool", "FractionalMaxPool"}) {
    ShapeInferenceTestOp op(op_name);
    auto set_op = [&op, op_name](const std::vector<float>& pooling_ratio) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPSnn_ops_testDTcc mht_4(mht_4_v, 703, "", "./tensorflow/core/ops/nn_ops_test.cc", "lambda");

      TF_ASSERT_OK(NodeDefBuilder("test", op_name)
                       .Input("input", 0, DT_FLOAT)
                       .Attr("pooling_ratio", pooling_ratio)
                       .Finalize(&op.node_def));
    };

    set_op(std::vector<float>{2.0f, 1, 1 / 1.5f, 1 / 2.0f});

    // Rank check.
    INFER_ERROR("must be rank 4", op, "[?,?,?]");

    // Unknown inputs.
    INFER_OK(op, "?", "[?,?,?,?];[?];[?]");
    INFER_OK(op, "[?,?,?,?]", "[?,?,?,?];[?];[?]");

    INFER_OK(op, "[10,20,30,40]", "[5,20,45,80];[20];[45]");
    INFER_OK(op, "[?,20,30,40]", "[?,20,45,80];[20];[45]");
    INFER_OK(op, "[10,?,30,40]", "[5,?,45,80];[?];[45]");
    INFER_OK(op, "[10,20,?,40]", "[5,20,?,80];[20];[?]");
    INFER_OK(op, "[10,20,30,?]", "[5,20,45,?];[20];[45]");

    // Wrong number of values for pooling_ratio.
    set_op(std::vector<float>{.5, 1.0, 1.5});
    INFER_ERROR("pooling_ratio field", op, "?");
    set_op(std::vector<float>{1, 2, 3, 4, 5});
    INFER_ERROR("pooling_ratio field", op, "?");

    // Check dim size >= 0.
    set_op(std::vector<float>{-1, 2, 3, 4});
    INFER_ERROR("is negative", op, "[1,2,3,4]");
  }
}

TEST(NNOpsTest, FractionalMaxPoolGrad) {
  ShapeInferenceTestOp op("FractionalMaxPoolGrad");

  // Note that the shape fn only uses input[0] for computation.
  INFER_ERROR("must be rank 4", op, "[?,?,?];?;?;?;?");
  INFER_OK(op, "?;?;?;?;?", "[?,?,?,?]");
  INFER_OK(op, "[?,?,3,4];?;?;?;?", "in0");
}

TEST(NNOpsTest, FractionalAvgPoolGrad) {
  ShapeInferenceTestOp op("FractionalAvgPoolGrad");
  op.input_tensors.resize(1);

  // With no input shape tensor, returns unknown of rank 4.
  INFER_OK(op, "?;?;?;?", "[?,?,?,?]");

  // When input tensor is known, its values determine output shape.
  std::vector<int32> shape{1, 2, 3, 4};
  Tensor shape_t = test::AsTensor<int32>(shape);
  op.input_tensors[0] = &shape_t;
  INFER_OK(op, "[5];?;?;?", "[1,2,3,4]");
}

}  // end namespace tensorflow
