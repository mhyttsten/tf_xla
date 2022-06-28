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
class MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_sparse_kernel_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_sparse_kernel_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_sparse_kernel_testDTcc() {
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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedTensorToSparseTest : public ::tensorflow::OpsTestBase {
 protected:
  static constexpr int kSparseIndicesOutput = 0;
  static constexpr int kSparseValuesOutput = 1;
  static constexpr int kSparseDenseShapeOutput = 2;
  // Builds the tensorflow test graph for the RaggedTensorToSparse op, and
  // populates the `splits` input with the given values.
  template <typename T>
  void BuildRaggedTensorToSparseGraph(
      const std::vector<std::vector<int64_t>>& rt_nested_splits,
      const TensorShape& rt_dense_values_shape,
      const std::vector<T>& rt_dense_values) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSragged_tensor_to_sparse_kernel_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/ragged_tensor_to_sparse_kernel_test.cc", "BuildRaggedTensorToSparseGraph");

    const auto& dtype = DataTypeToEnum<T>::v();
    int64_t num_splits = rt_nested_splits.size();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedTensorToSparse")
                     .Input(FakeInput(num_splits))  // rt_nested_splits
                     .Input(FakeInput(dtype))       // rt_dense_values
                     .Attr("RAGGED_RANK", num_splits)
                     .Attr("T", dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    for (const auto& splits : rt_nested_splits) {
      int64_t splits_size = splits.size();
      AddInputFromArray<int64_t>(TensorShape({splits_size}), splits);
    }
    AddInputFromArray<T>(rt_dense_values_shape, rt_dense_values);
  }
};

TEST_F(RaggedTensorToSparseTest, OneSplits_Values1D) {
  // ragged_tensor=[[1, 2, 3], [], [4, 5], [6]]
  BuildRaggedTensorToSparseGraph<int>({{0, 3, 3, 5, 6}},    // splits
                                      TensorShape({6}),     // values.shape
                                      {1, 2, 3, 4, 5, 6});  // values
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64_t>({0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 3, 0}, {6, 2}));
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>({1, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSparseDenseShapeOutput),
                                   test::AsTensor<int64_t>({4, 3}));
}

TEST_F(RaggedTensorToSparseTest, EmptyRows) {
  // Empty rows at the beginning, middle, and end of the RaggedTensor.
  // ragged_tensor=[[], [1, 2, 3, 4], [], [5, 6], []]
  BuildRaggedTensorToSparseGraph<int>({{0, 0, 4, 4, 6, 6}},  // splits
                                      TensorShape({6}),      // values.shape
                                      {1, 2, 3, 4, 5, 6});   // values
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64_t>({1, 0, 1, 1, 1, 2, 1, 3, 3, 0, 3, 1}, {6, 2}));
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>({1, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSparseDenseShapeOutput),
                                   test::AsTensor<int64_t>({5, 4}));
}

TEST_F(RaggedTensorToSparseTest, OneSplits_Values2D) {
  // ragged_tensor=[[[1, 2], [3, 4], [5, 6]], [], [[7, 8], [9, 10]], [[11, 12]]]
  BuildRaggedTensorToSparseGraph<int>(
      {{0, 3, 3, 5, 6}},                         // splits
      TensorShape({6, 2}),                       // values.shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});  // values
  TF_ASSERT_OK(RunOpKernel());
  std::vector<int64_t> expected_splits_12_3 = {
      0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1,
      2, 0, 0, 2, 0, 1, 2, 1, 0, 2, 1, 1, 3, 0, 0, 3, 0, 1};
  std::vector<int> expected_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64_t>(expected_splits_12_3, {12, 3}));
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>(expected_values));
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSparseDenseShapeOutput),
                                   test::AsTensor<int64_t>({4, 3, 2}));
}

TEST_F(RaggedTensorToSparseTest, TwoSplits_Values1D) {
  // ragged_tensor =
  //        0             1           2
  // -+--------------------------------------
  // 0| [[ [x],         [x x],       [] ],
  // 1|  [                              ],
  // 2|  [ [x x x x x], [x x x]         ],
  // 3|  [ [],          [x x x x]       ]]
  BuildRaggedTensorToSparseGraph<int>(
      {{0, 3, 3, 5, 7}, {0, 1, 3, 3, 8, 11, 11, 15}},        // splits
      TensorShape({15}),                                     // values.shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});  // values
  TF_ASSERT_OK(RunOpKernel());
  std::vector<int64_t> expected_splits_15_3 = {
      0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 0,
      4, 2, 1, 0, 2, 1, 1, 2, 1, 2, 3, 1, 0, 3, 1, 1, 3, 1, 2, 3, 1, 3};
  std::vector<int> expected_values = {1, 2,  3,  4,  5,  6,  7, 8,
                                      9, 10, 11, 12, 13, 14, 15};
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>(expected_values));
  test::ExpectTensorEqual<int64_t>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64_t>(expected_splits_15_3, {15, 3}));
  test::ExpectTensorEqual<int64_t>(*GetOutput(kSparseDenseShapeOutput),
                                   test::AsTensor<int64_t>({4, 3, 5}));
}

TEST_F(RaggedTensorToSparseTest, ShapeFn) {
  // RaggedSplitsToIndices(rt_nested_splits+, rt_dense_values)
  //     -> [sparse_indices, sparse_values, sparse_dense_shape]
  // The output shape will always have the following form:
  //     [nvals, dense_dims];[nvals];[dense_dims]
  ShapeInferenceTestOp op("RaggedTensorToSparse");

  // Tests with len(rt_nested_splits)==0.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(0);
  INFER_ERROR("Requires RAGGED_RANK>0", op, "?");

  // Tests with len(rt_nested_splits)==1.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(1);
  INFER_OK(op, "?;?", "[?,?];[?];[?]");          // nvals=?, dense_dims=?
  INFER_OK(op, "?;[?]", "[?,2];[?];[2]");        // nvals=?, dense_dims=2
  INFER_OK(op, "?;[?,?]", "[?,3];[?];[3]");      // nvals=?, dense_dims=3
  INFER_OK(op, "[?];[5]", "[5,2];[5];[2]");      // nvals=5, dense_dims=2
  INFER_OK(op, "[?];[5,2]", "[10,3];[10];[3]");  // nvals=10, dense_dims=3
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[5,5];?");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "?;[]");

  // Tests with len(rt_nested_splits)==2
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(2);
  INFER_OK(op, "?;?;?", "[?,?];[?];[?]");            // nvals=?, dense_dims=?
  INFER_OK(op, "?;?;[?]", "[?,3];[?];[3]");          // nvals=?, dense_dims=3
  INFER_OK(op, "?;?;[?,?]", "[?,4];[?];[4]");        // nvals=?, dense_dims=4
  INFER_OK(op, "[?];[?];[5]", "[5,3];[5];[3]");      // nvals=5, dense_dims=3
  INFER_OK(op, "[?];[?];[5,2]", "[10,4];[10];[4]");  // nvals=10, dense_dims=4
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[5,5];?");

  // Tests with len(rt_nested_splits)==3
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(3);
  INFER_OK(op, "?;?;?;?", "[?,?];[?];[?]");    // nvals=?, dense_dims=?
  INFER_OK(op, "?;?;?;[?]", "[?,4];[?];[4]");  // nvals=?, dense_dims=4
  INFER_OK(op, "?;?;?;[5]", "[5,4];[5];[4]");  // nvals=5, dense_dims=4
}

TEST_F(RaggedTensorToSparseTest, NoSplits) {
  const auto& dtype = DataTypeToEnum<int>::v();
  TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedTensorToSparse")
                   .Input(FakeInput(0))
                   .Input(FakeInput(dtype))
                   .Attr("RAGGED_RANK", 0)
                   .Attr("T", dtype)
                   .Finalize(node_def()));
  EXPECT_TRUE(absl::StartsWith(
      InitOp().error_message(),
      "Value for attr 'RAGGED_RANK' of 0 must be at least minimum 1"));
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_BadSplitStart) {
  BuildRaggedTensorToSparseGraph<int>({{5, 7, 10}},      // splits
                                      TensorShape({0}),  // values.shape
                                      {});               // values
  EXPECT_EQ("First value of ragged splits must be 0.",
            RunOpKernel().error_message());
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_BadSplitLengths1) {
  BuildRaggedTensorToSparseGraph<int>({{0, 5}, {0, 2, 4, 6}},  // splits
                                      TensorShape({0}),        // values.shape
                                      {});                     // values
  EXPECT_EQ(
      "Final value of ragged splits must match the length "
      "the corresponding ragged values.",
      RunOpKernel().error_message());
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_BadSplitLengths2) {
  BuildRaggedTensorToSparseGraph<int>({{0, 5}},          // splits
                                      TensorShape({0}),  // values.shape
                                      {});               // values
  EXPECT_EQ(
      "Final value of ragged splits must match the length "
      "the corresponding ragged values.",
      RunOpKernel().error_message());
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_EmptySplits) {
  BuildRaggedTensorToSparseGraph<int>({{}},              // splits
                                      TensorShape({0}),  // values.shape
                                      {});               // values
  EXPECT_EQ("ragged splits may not be empty.", RunOpKernel().error_message());
}

}  // namespace
}  // namespace tensorflow
