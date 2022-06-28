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
class MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc() {
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

class ScatterUpdateOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/scatter_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterUpdate")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};
class ScatterSubOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType variable_ref_type, DataType index_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/scatter_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", "ScatterSub")
                     .Input(FakeInput(variable_ref_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(RemoveRefType(variable_ref_type)))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ScatterUpdateOpTest, Simple_StringType) {
  MakeOp(DT_STRING_REF, DT_INT32);
  AddInputFromArray<tstring>(TensorShape({1}), {"Brain"});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<tstring>(TensorShape({1}), {"TensorFlow"});
  TF_ASSERT_OK(RunOpKernel());
  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_STRING, TensorShape({1}));
  test::FillValues<tstring>(&expected, {"TensorFlow"});
  test::ExpectTensorEqual<tstring>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_BoolType) {
  MakeOp(DT_BOOL_REF, DT_INT32);
  AddInputFromArray<bool>(TensorShape({1}), {false});
  AddInputFromArray<int32>(TensorShape({1}), {0});
  AddInputFromArray<bool>(TensorShape({1}), {true});
  TF_ASSERT_OK(RunOpKernel());
  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_BOOL, TensorShape({1}));
  test::FillValues<bool>(&expected, {true});
  test::ExpectTensorEqual<bool>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_TwoD32) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
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

TEST_F(ScatterUpdateOpTest, Simple_Two64) {
  MakeOp(DT_FLOAT_REF, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int64_t>(TensorShape({3}), {0, 4, 2});
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

TEST_F(ScatterUpdateOpTest, Simple_ZeroD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({}), {3});
  AddInputFromArray<float>(TensorShape({}), {101});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0, 0, 0, 101, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Simple_OneD) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5}), {0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {100, 0, 102, 0, 101});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, HigherRank) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({8}), {0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({2, 3}), {0, 4, 2, 1, 3, 6});
  AddInputFromArray<float>(TensorShape({2, 3}), {10, 20, 30, 40, 50, 60});
  TF_ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_FLOAT, TensorShape({8}));
  test::FillValues<float>(&expected, {10, 40, 30, 50, 20, 0, 60, 0});
  test::ExpectTensorEqual<float>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 5)"))
      << s;
}

TEST_F(ScatterSubOpTest, Error_IndexOutOfRange) {
  MakeOp(DT_FLOAT_REF, DT_INT32);
  // Feed and run
  AddInputFromArray<float>(TensorShape({14}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 1, 99});
  AddInputFromArray<float>(TensorShape({3}), {100, 101, 102});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      absl::StrContains(s.ToString(), "indices[2] = 99 is not in [0, 14)"))
      << s;
}

TEST_F(ScatterSubOpTest, StressIndexTest) {
  MakeOp(DT_INT32_REF, DT_INT32);
  // Feed and run
  const int kRows = 1;
  std::vector<int32> values(kRows, 0);
  const int kNumUpdates = 1000000;
  std::vector<int32> indices(kNumUpdates, 0);
  std::vector<int32> updates(kNumUpdates, 1);
  AddInputFromArray<int32>(TensorShape({kRows}), values);
  AddInputFromArray<int32>(TensorShape({kNumUpdates}), indices);
  AddInputFromArray<int32>(TensorShape({kNumUpdates}), updates);
  Status s = RunOpKernel();
  Tensor params_tensor = *mutable_input(0).tensor;
  Tensor expected(allocator(), DT_INT32, TensorShape({1}));
  test::FillValues<int32>(&expected, {-1000000});
  test::ExpectTensorEqual<int32>(expected, params_tensor);
}

TEST_F(ScatterUpdateOpTest, Error_WrongDimsIndices) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({2, 3}), {0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({1, 3}), {0, 4, 99});
  AddInputFromArray<float>(TensorShape({3, 3}),
                           {100, 101, 102, 777, 778, 779, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Must have updates.shape = indices.shape + "
                                "params.shape[1:] or updates.shape = [], got "))
      << s;
}

TEST_F(ScatterUpdateOpTest, Error_MismatchedParamsAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(
      TensorShape({3, 4}),
      {100, 101, 102, 103, 777, 778, 779, 780, 10000, 10001, 10002, 10004});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Must have updates.shape = indices.shape + "
                                "params.shape[1:] or updates.shape = [], got "))

      << s;
}

TEST_F(ScatterUpdateOpTest, Error_MismatchedIndicesAndUpdateDimensions) {
  MakeOp(DT_FLOAT_REF, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({5, 3}),
                           {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  AddInputFromArray<int32>(TensorShape({3}), {0, 4, 2});
  AddInputFromArray<float>(TensorShape({2, 3}),
                           {100, 101, 102, 10000, 10001, 10002});
  Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(),
                                "Must have updates.shape = indices.shape + "
                                "params.shape[1:] or updates.shape = [], got "))
      << s;
}

class ScatterUpdateBM : public ScatterUpdateOpTest {
 public:
  void TestBody() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_2(mht_2_v, 446, "", "./tensorflow/core/kernels/scatter_op_test.cc", "TestBody");
}
  void MakeBenchmarkOp(const char* op, DataType index_type) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_3(mht_3_v, 451, "", "./tensorflow/core/kernels/scatter_op_test.cc", "MakeBenchmarkOp");

    TF_ASSERT_OK(NodeDefBuilder("myop", op)
                     .Input(FakeInput(DT_FLOAT_REF))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_CHECK_OK(InitOp());
  }
};

template <typename Index>
void BM_ScatterHelper(::testing::benchmark::State& state, int embedding_size,
                      const char* op, bool big_num_updates = false) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_4(mht_4_v, 467, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterHelper");

  const int kRows = 10000000 / embedding_size;
  std::vector<float> values;
  values.reserve(kRows);
  for (int i = 0; i < kRows * embedding_size; i++) {
    values.push_back(i);
  }
  const int kNumUpdates = big_num_updates ? 1000000 : 1000;
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

  ScatterUpdateBM bm;
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

void BM_ScatterUpdateInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_5(mht_5_v, 502, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterUpdateInt32");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterUpdate");
}
void BM_ScatterUpdateInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_6(mht_6_v, 510, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterUpdateInt64");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64_t>(state, embedding_size, "ScatterUpdate");
}

void BM_ScatterAddInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_7(mht_7_v, 519, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterAddInt32");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterAdd");
}

void BM_ScatterAddInt32Large(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_8(mht_8_v, 528, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterAddInt32Large");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterAdd", true);
}
void BM_ScatterAddInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_9(mht_9_v, 536, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterAddInt64");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64_t>(state, embedding_size, "ScatterAdd");
}

void BM_ScatterMulInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_10(mht_10_v, 545, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterMulInt32");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterMul");
}
void BM_ScatterMulInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_11(mht_11_v, 553, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterMulInt64");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64_t>(state, embedding_size, "ScatterMul");
}

void BM_ScatterDivInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_12(mht_12_v, 562, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterDivInt32");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterDiv");
}
void BM_ScatterDivInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_13(mht_13_v, 570, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterDivInt64");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64_t>(state, embedding_size, "ScatterDiv");
}

void BM_ScatterMinInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_14(mht_14_v, 579, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterMinInt32");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterMin");
}
void BM_ScatterMinInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_15(mht_15_v, 587, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterMinInt64");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64_t>(state, embedding_size, "ScatterMin");
}

void BM_ScatterMaxInt32(::testing::benchmark::State& state) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_16(mht_16_v, 596, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterMaxInt32");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int32>(state, embedding_size, "ScatterMax");
}
void BM_ScatterMaxInt64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_op_testDTcc mht_17(mht_17_v, 604, "", "./tensorflow/core/kernels/scatter_op_test.cc", "BM_ScatterMaxInt64");

  const int embedding_size = state.range(0);

  BM_ScatterHelper<int64_t>(state, embedding_size, "ScatterMax");
}

BENCHMARK(BM_ScatterUpdateInt32)
    ->Arg(1)
    ->Arg(10)
    ->Arg(32)
    ->Arg(50)
    ->Arg(64)
    ->Arg(80)
    ->Arg(96)
    ->Arg(112)
    ->Arg(192)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(10000)
    ->Arg(100000)
    ->Arg(1000000);
BENCHMARK(BM_ScatterUpdateInt64)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->Arg(100000);

BENCHMARK(BM_ScatterAddInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterAddInt32Large)
    ->Arg(1)
    ->Arg(10)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

BENCHMARK(BM_ScatterAddInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterMulInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterMulInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterDivInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterDivInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterMinInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterMinInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

BENCHMARK(BM_ScatterMaxInt32)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK(BM_ScatterMaxInt64)->Arg(1)->Arg(10)->Arg(64)->Arg(256)->Arg(1024);

}  // namespace
}  // namespace tensorflow
