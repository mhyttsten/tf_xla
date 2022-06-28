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
class MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc() {
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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

using Eigen::half;

namespace tensorflow {

template <typename Src, typename Dst>
static Graph* Cast(int num) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor data(DataTypeToEnum<Src>::value,
              TensorShape({64, 64, num / (64 * 64)}));
  data.flat<Src>().setRandom();
  test::graph::Cast(g, test::graph::Constant(g, data),
                    DataTypeToEnum<Dst>::value);
  return g;
}

class CastOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType src, DataType dst, bool trunc) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/cast_op_test.cc", "MakeOp");

    if (trunc) {
      TF_EXPECT_OK(NodeDefBuilder("cast_op", "Cast")
                       .Input(FakeInput(src))
                       .Attr("SrcT", src)
                       .Attr("DstT", dst)
                       .Attr("Truncate", true)
                       .Finalize(node_def()));
    } else {
      TF_EXPECT_OK(NodeDefBuilder("cast_op", "Cast")
                       .Input(FakeInput(src))
                       .Attr("SrcT", src)
                       .Attr("DstT", dst)
                       .Finalize(node_def()));
    }

    TF_EXPECT_OK(InitOp());
  }

  template <typename INPUT, typename OUTPUT>
  void CheckCast(bool trunc) {
    DataType in_type = DataTypeToEnum<INPUT>::v();
    DataType out_type = DataTypeToEnum<OUTPUT>::v();
    MakeOp(in_type, out_type, trunc);
    AddInputFromArray<INPUT>(TensorShape({1, 2, 2, 1}),
                             {INPUT(1), INPUT(2), INPUT(3), INPUT(4)});
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(allocator(), out_type, TensorShape({1, 2, 2, 1}));
    test::FillValues<OUTPUT>(&expected,
                             {OUTPUT(1), OUTPUT(2), OUTPUT(3), OUTPUT(4)});
    test::ExpectTensorEqual<OUTPUT>(expected, *GetOutput(0));
  }
};

#define TEST_CAST(in, out)                                                   \
  TEST_F(CastOpTest, TestCast##_##in##_##out) { CheckCast<in, out>(false); } \
  TEST_F(CastOpTest, TestCastTruncate_##_##in##_##out) {                     \
    CheckCast<in, out>(true);                                                \
  }

#define TEST_ALL_CASTS_FROM(in) \
  TEST_CAST(in, uint8);         \
  TEST_CAST(in, uint16);        \
  TEST_CAST(in, uint32);        \
  TEST_CAST(in, uint64);        \
  TEST_CAST(in, int16);         \
  TEST_CAST(in, int32);         \
  TEST_CAST(in, int64_t);       \
  TEST_CAST(in, half);          \
  TEST_CAST(in, float);         \
  TEST_CAST(in, double);        \
  TEST_CAST(in, bfloat16);      \
  TEST_CAST(in, quint8);        \
  TEST_CAST(in, qint8);         \
  TEST_CAST(in, qint32);        \
  TEST_CAST(in, qint16);        \
  TEST_CAST(in, quint16);

TEST_ALL_CASTS_FROM(uint8)
TEST_ALL_CASTS_FROM(uint16)
TEST_ALL_CASTS_FROM(uint32)
TEST_ALL_CASTS_FROM(uint64)
TEST_ALL_CASTS_FROM(int16)
TEST_ALL_CASTS_FROM(int32)
TEST_ALL_CASTS_FROM(int64_t)
TEST_ALL_CASTS_FROM(half)
TEST_ALL_CASTS_FROM(float)
TEST_ALL_CASTS_FROM(double)
TEST_ALL_CASTS_FROM(bfloat16)
TEST_ALL_CASTS_FROM(quint8)
TEST_ALL_CASTS_FROM(qint8)
TEST_ALL_CASTS_FROM(qint32)
TEST_ALL_CASTS_FROM(qint16)
TEST_ALL_CASTS_FROM(quint16)

#undef TEST_ALL_CASTS_FROM
#undef TEST_CAST

// TODO(wicke): check conversions from/to bool, and bfloat16

static void BM_cpu_float_int64(::testing::benchmark::State& state) {
  const int num = state.range(0);
  test::Benchmark("cpu", Cast<float, int64_t>(num), /*old_benchmark_api=*/false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(int64_t)));
}
BENCHMARK(BM_cpu_float_int64)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_float_int64(::testing::benchmark::State& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_1(mht_1_v, 306, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_gpu_float_int64");

  const int num = state.range(0);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  test::Benchmark("gpu", Cast<float, int64_t>(num), /*old_benchmark_api=*/false)
      .Run(state);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(int64_t)));
}
BENCHMARK(BM_gpu_float_int64)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_bool_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_2(mht_2_v, 322, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_cpu_bool_float");

  const int num = state.range(0);

  test::Benchmark("cpu", Cast<bool, float>(num), /*old_benchmark_api=*/false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(bool) + sizeof(float)));
}
BENCHMARK(BM_cpu_bool_float)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_bool_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_3(mht_3_v, 336, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_gpu_bool_float");

  const int num = state.range(0);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  test::Benchmark("gpu", Cast<bool, float>(num), /*old_benchmark_api=*/false)
      .Run(state);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(bool) + sizeof(float)));
}
BENCHMARK(BM_gpu_bool_float)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_float_bfloat16(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_4(mht_4_v, 352, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_cpu_float_bfloat16");

  const int num = state.range(0);
  test::Benchmark("cpu", Cast<float, bfloat16>(num),
                  /*old_benchmark_api=*/false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(bfloat16)));
}
BENCHMARK(BM_cpu_float_bfloat16)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_bfloat16_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_5(mht_5_v, 366, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_cpu_bfloat16_float");

  const int num = state.range(0);
  test::Benchmark("cpu", Cast<bfloat16, float>(num),
                  /*old_benchmark_api=*/false)
      .Run(state);

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(bfloat16)));
}
BENCHMARK(BM_cpu_bfloat16_float)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_float_half(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_6(mht_6_v, 381, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_cpu_float_half");

  const int num = state.range(0);

  test::Benchmark("cpu", Cast<float, Eigen::half>(num),
                  /*old_benchmark_api=*/false)
      .Run(state);

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
}
BENCHMARK(BM_cpu_float_half)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_cpu_half_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_7(mht_7_v, 397, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_cpu_half_float");

  const int num = state.range(0);

  test::Benchmark("cpu", Cast<Eigen::half, float>(num),
                  /*old_benchmark_api=*/false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
}
BENCHMARK(BM_cpu_half_float)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_float_half(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_8(mht_8_v, 412, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_gpu_float_half");

  const int num = state.range(0);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  test::Benchmark("gpu", Cast<float, Eigen::half>(num),
                  /*old_benchmark_api=*/false)
      .Run(state);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
}
BENCHMARK(BM_gpu_float_half)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

static void BM_gpu_half_float(::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScast_op_testDTcc mht_9(mht_9_v, 429, "", "./tensorflow/core/kernels/cast_op_test.cc", "BM_gpu_half_float");

  const int num = state.range(0);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  test::Benchmark("gpu", Cast<Eigen::half, float>(num),
                  /*old_benchmark_api=*/false)
      .Run(state);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * num *
                          (sizeof(float) + sizeof(Eigen::half)));
}
BENCHMARK(BM_gpu_half_float)->UseRealTime()->Arg(64 << 10)->Arg(32 << 20);

}  // end namespace tensorflow
