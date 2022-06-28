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
class MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc() {
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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class ScatterNdUpdateOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterNdUpdate")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// TODO(simister): Re-enable this once binary size is under control.
// TEST_F(ScatterNdUpdateOpTest, Simple_StringType) {
//   MakeOp(DT_STRING_REF, DT_INT32);
//   AddInputFromArray<tstring>(TensorShape({1}), {"Brain"});
//   AddInputFromArray<int32>(TensorShape({1}), {0});
//   AddInputFromArray<tstring>(TensorShape({1}), {"TensorFlow"});
//   TF_ASSERT_OK(RunOpKernel());
//   // Check the new state of the input
//   Tensor params_tensor = *mutable_input(0).tensor;
//   Tensor expected(allocator(), DT_STRING, TensorShape({1}));
//   test::FillValues<tstring>(&expected, {"TensorFlow"});
//   test::ExpectTensorEqual<tstring>(expected, params_tensor);
// }

// TEST_F(ScatterNdUpdateOpTest, Simple_BoolType) {
//   MakeOp(DT_BOOL_REF, DT_INT32);
//   AddInputFromArray<bool>(TensorShape({1}), {false});
//   AddInputFromArray<int32>(TensorShape({1}), {0});
//   AddInputFromArray<bool>(TensorShape({1}), {true});
//   TF_ASSERT_OK(RunOpKernel());
//   // Check the new state of the input
//   Tensor params_tensor = *mutable_input(0).tensor;
//   Tensor expected(allocator(), DT_BOOL, TensorShape({1}));
//   test::FillValues<bool>(&expected, {true});
//   test::ExpectTensorEqual<bool>(expected, params_tensor);
// }

TEST_F(ScatterNdUpdateOpTest, Simple_TwoD32) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, Simple_Two64) {
  MakeOp(DT_FLOAT_REF, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int64_t>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5, 3}));
  test::FillValues<float>(&expected, {100, 101, 102, 0, 0, 0, 10000, 10001,
                                      10002, 0, 0, 0, 777, 778, 779});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

/*TEST_F(ScatterNdUpdateOpTest, Simple_ZeroElements) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<int32>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString())
                  .contains("Output must not have 0 elements, got shape: "))
      << s;
}*/

TEST_F(ScatterNdUpdateOpTest, Simple_ZeroD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1}), {3});
  AddInputFromArray<float>(TensorShape({1}), {101});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0, 0, 0, 101, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, Simple_OneD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, HigherRank) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({8}), {0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2, 3, 1}), {0, 4, 2, 1, 3, 6});
  AddInputFromArray<float>(TensorShape({2, 3}), {10, 20, 30, 40, 50, 60});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {10, 40, 30, 50, 20, 0, 60, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterNdUpdateOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(), "indices[2] = [99] does not index into shape [5,3]"))
      << s;
}

TEST_F(ScatterNdUpdateOpTest, Error_WrongDimsIndices) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1, 3, 1}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "Dimensions [0,1) of indices[shape=[1,3,1]] = 1 must match dimensions "
      "[0,1) of updates[shape=[3,3]] = 3"))
      << s;
}

TEST_F(ScatterNdUpdateOpTest, Error_MismatchedParamsAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(
      TensorShape({3, 4}),
      {100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "Dimensions [1,2) of input[shape=[5,3]] must match dimensions [1,2) of "
      "updates[shape=[3,4]]"))
      << s;
}

TEST_F(ScatterNdUpdateOpTest, Error_MismatchedIndicesAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3, 1}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {100, 101, 102, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(
      s.ToString(),
      "Dimensions [0,1) of indices[shape=[3,1]] = 3 must match dimensions [0,1)"
      " of updates[shape=[2,3]] = 2"))
      << s;
}

class ScatterNdUpdateBM : public ScatterNdUpdateOpTest {
 public:
  void TestBody() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_1(mht_1_v, 417, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "TestBody");
}
  void MakeBenchmarkOp(const char* op, DataType index_type) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_2(mht_2_v, 422, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "MakeBenchmarkOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", op)
                     .Input(FakeInput(DT_FLOAT_REF))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
  }
};

template <typename Index>
void BM_ScatterNdHelper(::testing::benchmark::State& state, int embedding_size,
                        const char* op) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_3(mht_3_v, 438, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "BM_ScatterNdHelper");

  const int kRows = 10000000 / embedding_size;
  std::vector<float> values;
  values.reserve(kRows);
  for (int i = 0; i < kRows * embedding_size; i++) {
    values.push_back(i);
  }
  const int kNumUpdates = 1000;
  random::PhiloxRandom philox(301, 17);
  random::SimplePhilox rnd(&philox);
  std::vector<Index> indices;
  std::vector<float> updates;
  for (int i = 0; i < kNumUpdates; i++) {
    indices.push_back(rnd.Uniform(kRows));
    for (int j = 0; j < embedding_size; j++) {
      updates.push_back(i * 10 + j);
    }
  }

  ScatterNdUpdateBM bm;
  bm.MakeBenchmarkOp(op, DataTypeToEnum<Index>::v());
  bm.AddInputFromArray<float>(TensorShape({kRows, embedding_size}), values);
  bm.AddInputFromArray<Index>(TensorShape({kNumUpdates}), indices);
  bm.AddInputFromArray<float>(TensorShape({kNumUpdates, embedding_size}),
                              updates);
  for (auto i : state) {
    Status s = bm.RunOpKernel();
  }
  state.SetItemsProcessed((static_cast<int64_t>(kNumUpdates) * embedding_size) *
                          state.iterations());
}

void BM_ScatterNdUpdateInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_4(mht_4_v, 473, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "BM_ScatterNdUpdateInt32");

  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int32>(state, embedding_size, "ScatterNdUpdate");
}
void BM_ScatterNdUpdateInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_5(mht_5_v, 481, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "BM_ScatterNdUpdateInt64");

  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int64_t>(state, embedding_size, "ScatterNdUpdate");
}

void BM_ScatterNdAddInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_6(mht_6_v, 490, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "BM_ScatterNdAddInt32");

  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int32>(state, embedding_size, "ScatterNdAdd");
}
void BM_ScatterNdAddInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_nd_op_testDTcc mht_7(mht_7_v, 498, "", "./tensorflow/core/kernels/scatter_nd_op_test.cc", "BM_ScatterNdAddInt64");

  const int embedding_size = state.range(0);

  BM_ScatterNdHelper<int64_t>(state, embedding_size, "ScatterNdAdd");
}

BENCHMARK(BM_ScatterNdUpdateInt32)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);
BENCHMARK(BM_ScatterNdUpdateInt64)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

BENCHMARK(BM_ScatterNdAddInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterNdAddInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

}  // namespace
}  // namespace tensorflow
