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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_op_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedRangeOpTest : public ::tensorflow::OpsTestBase {
 protected:
  // Indices of output tensors.
  static constexpr int kSplitsOutput = 0;
  static constexpr int kValuesOutput = 1;

  // Builds the tensorflow test graph for the RaggedRange op.
  template <typename T>
  void BuildRaggedRangeGraph() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_range_op_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/ragged_range_op_test.cc", "BuildRaggedRangeGraph");

    const auto& dtype = DataTypeToEnum<T>::v();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedRange")
                     .Input(FakeInput(dtype))  // starts
                     .Input(FakeInput(dtype))  // limits
                     .Input(FakeInput(dtype))  // deltas
                     .Attr("T", dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RaggedRangeOpTest, IntValues) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<int>(TensorShape({4}), {8, 7, 8, 1});   // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 2, 4, 6], [5, 6], [], [5, 4, 3, 2]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 4, 6, 6, 10}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({0, 2, 4, 6, 5, 6, 5, 4, 3, 2}));
}

TEST_F(RaggedRangeOpTest, FloatValues) {
  BuildRaggedRangeGraph<float>();
  AddInputFromArray<float>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<float>(TensorShape({4}), {8, 7, 8, 1});   // limits
  AddInputFromArray<float>(TensorShape({4}), {2, 1, 1, -1});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 2, 4, 6], [5, 6], [], [5, 4, 3, 2]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 4, 6, 6, 10}));
  test::ExpectTensorNear<float>(
      *GetOutput(kValuesOutput),
      test::AsTensor<float>({0, 2, 4, 6, 5, 6, 5, 4, 3, 2}), 0.1);
}

TEST_F(RaggedRangeOpTest, BroadcastDeltas) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({3}), {0, 5, 8});  // starts
  AddInputFromArray<int>(TensorShape({3}), {8, 7, 8});  // limits
  AddInputFromArray<int>(TensorShape({}), {1});         // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 1, 2, 3, 4, 5, 6, 7], [5, 6], []]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 8, 10, 10}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({0, 1, 2, 3, 4, 5, 6, 7, 5, 6}));
}

TEST_F(RaggedRangeOpTest, BroadcastLimitsAndDeltas) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({}), {0});         // starts
  AddInputFromArray<int>(TensorShape({3}), {3, 0, 2});  // limits
  AddInputFromArray<int>(TensorShape({}), {1});         // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 1, 2], [], [0, 1]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 3, 3, 5}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({0, 1, 2, 0, 1}));
}

TEST_F(RaggedRangeOpTest, BroadcastStartsAndLimits) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({}), {0});         // starts
  AddInputFromArray<int>(TensorShape({}), {12});        // limits
  AddInputFromArray<int>(TensorShape({3}), {3, 4, 5});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 3, 6, 9], [0, 4, 8], [0, 5, 10]]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 4, 7, 10}));
  test::ExpectTensorEqual<int>(
      *GetOutput(kValuesOutput),
      test::AsTensor<int>({0, 3, 6, 9, 0, 4, 8, 0, 5, 10}));
}

TEST_F(RaggedRangeOpTest, AllScalarInputs) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({}), {0});  // starts
  AddInputFromArray<int>(TensorShape({}), {5});  // limits
  AddInputFromArray<int>(TensorShape({}), {1});  // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 1, 2, 3, 4]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 5}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({0, 1, 2, 3, 4}));
}

TEST_F(RaggedRangeOpTest, InvalidArgsStarts) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4, 1}), {0, 5, 8, 5});  // starts
  AddInputFromArray<int>(TensorShape({4}), {8, 7, 8, 1});     // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});    // deltas
  EXPECT_EQ("starts must be a scalar or vector", RunOpKernel().error_message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsLimits) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});     // starts
  AddInputFromArray<int>(TensorShape({4, 1}), {8, 7, 8, 1});  // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});    // deltas
  EXPECT_EQ("limits must be a scalar or vector", RunOpKernel().error_message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsDeltas) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});      // starts
  AddInputFromArray<int>(TensorShape({4}), {8, 7, 8, 1});      // limits
  AddInputFromArray<int>(TensorShape({4, 1}), {2, 1, 1, -1});  // deltas
  EXPECT_EQ("deltas must be a scalar or vector", RunOpKernel().error_message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsShapeMismatch) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<int>(TensorShape({3}), {7, 8, 1});      // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 1, -1});  // deltas
  EXPECT_EQ("starts, limits, and deltas must have the same shape",
            RunOpKernel().error_message());
}

TEST_F(RaggedRangeOpTest, InvalidArgsZeroDelta) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({4}), {0, 5, 8, 5});   // starts
  AddInputFromArray<int>(TensorShape({4}), {7, 8, 8, 1});   // limits
  AddInputFromArray<int>(TensorShape({4}), {2, 1, 0, -1});  // deltas
  EXPECT_EQ("Requires delta != 0", RunOpKernel().error_message());
}

TEST_F(RaggedRangeOpTest, EmptyRangePositiveDelta) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({2}), {0, 5});  // starts
  AddInputFromArray<int>(TensorShape({2}), {5, 0});  // limits
  AddInputFromArray<int>(TensorShape({}), {2});      // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[0, 2, 4], []]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 3, 3}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({0, 2, 4}));
}

TEST_F(RaggedRangeOpTest, EmptyRangeNegativeDelta) {
  BuildRaggedRangeGraph<int>();
  AddInputFromArray<int>(TensorShape({2}), {0, 5});  // starts
  AddInputFromArray<int>(TensorShape({2}), {5, 0});  // limits
  AddInputFromArray<int>(TensorShape({}), {-2});     // deltas
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[], [5, 3, 1]]
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSplitsOutput),
                                   test::AsTensor<int64_t>({0, 0, 3}));
  test::ExpectTensorEqual<int>(*GetOutput(kValuesOutput),
                               test::AsTensor<int>({5, 3, 1}));
}

TEST_F(RaggedRangeOpTest, ShapeFn) {
  // RaggedRange(starts, limits, deltas) -> [splits, values]
  ShapeInferenceTestOp op("RaggedRange");
  INFER_OK(op, "?;?;?", "[?];[?]");
  INFER_OK(op, "[3];[3];[3]", "[4];[?]");
  INFER_OK(op, "[3];[3];[]", "[4];[?]");  // broadcast deltas
  INFER_OK(op, "[3];[];[3]", "[4];[?]");  // broadcast limits
  INFER_OK(op, "[];[3];[3]", "[4];[?]");  // broadcast starts
  INFER_OK(op, "[];[];[]", "[2];[?]");    // degenerate case: all scalar inputs
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[5,5];[5];[5]");
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[5];[5,5];[5]");
  INFER_ERROR("Shape must be at most rank 1 but is rank 2", op,
              "[5];[5];[5,5]");
  INFER_ERROR("Dimensions must be equal, but are 4 and 3", op, "[3];[4];[3]");
}

}  // namespace
}  // namespace tensorflow
