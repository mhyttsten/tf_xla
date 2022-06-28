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
class MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc() {
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
#include <random>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

class DequantizeOpTest : public OpsTestBase {
 protected:
  template <typename T>
  void ComputeDequantizeMinCombinedUsingEigen(const Tensor& input,
                                              float min_range, float max_range,
                                              Tensor* output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "ComputeDequantizeMinCombinedUsingEigen");

    float half_range =
        !std::is_signed<T>::value
            ? 0.0f
            : (static_cast<float>(std::numeric_limits<T>::max()) -
               std::numeric_limits<T>::min() + 1) /
                  2.0f;
    const float scale_factor =
        (max_range - min_range) /
        (static_cast<float>(std::numeric_limits<T>::max()) -
         std::numeric_limits<T>::min());
    output->flat<float>() =
        ((input.flat<T>().template cast<int>().template cast<float>() +
          half_range) *
         scale_factor) +
        min_range;
  }

  // Compares dequantize min vs the same using eigen. This tests that a change
  // to not use eigen gives equivalent results to using eigen.
  template <typename T>
  void RunDequantizeMinCombinedTest(float min_range, float max_range,
                                    const string& op_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "RunDequantizeMinCombinedTest");

    TF_ASSERT_OK(NodeDefBuilder("dequantize_op", op_name)
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("mode", "MIN_COMBINED")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    std::vector<T> input;
    for (int64_t i = std::numeric_limits<T>::min();
         i < std::numeric_limits<T>::max(); ++i) {
      input.push_back(static_cast<T>(i));
    }
    TensorShape shape({static_cast<int64_t>(input.size())});
    AddInputFromArray<T>(shape, input);
    AddInputFromArray<float>(TensorShape({}), {min_range});
    AddInputFromArray<float>(TensorShape({}), {max_range});
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected(allocator(), DT_FLOAT, shape);
    ComputeDequantizeMinCombinedUsingEigen<T>(GetInput(0), min_range, max_range,
                                              &expected);
    test::ExpectTensorEqual<float>(expected, *GetOutput(0));
  }

  // Compares dequantize min vs the same using eigen. This tests that a change
  // to not use eigen gives equivalent results to using eigen.
  template <typename T>
  void RunDequantizeBfloat16MinCombinedTest(float min_range, float max_range) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_2(mht_2_v, 271, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "RunDequantizeBfloat16MinCombinedTest");

    TF_ASSERT_OK(NodeDefBuilder("dequantize_op_bfloat16", "Dequantize")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("mode", "MIN_COMBINED")
                     .Attr("dtype", DT_BFLOAT16)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    std::vector<T> input;
    for (int64_t i = std::numeric_limits<T>::min();
         i < std::numeric_limits<T>::max(); ++i) {
      input.push_back(static_cast<T>(i));
    }
    TensorShape shape({static_cast<int64_t>(input.size())});
    AddInputFromArray<T>(shape, input);
    AddInputFromArray<float>(TensorShape({}), {min_range});
    AddInputFromArray<float>(TensorShape({}), {max_range});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_float32(allocator(), DT_FLOAT, shape);
    ComputeDequantizeMinCombinedUsingEigen<T>(GetInput(0), min_range, max_range,
                                              &expected_float32);
    Tensor expected(allocator(), DT_BFLOAT16, shape);
    expected.flat<bfloat16>() = expected_float32.flat<float>().cast<bfloat16>();

    test::ExpectTensorEqual<bfloat16>(expected, *GetOutput(0));
  }

  // Creates a tensor with the specified dims, using values chosen from data,
  // multiplied by (1 + index) along the axis dimension.
  template <typename T>
  std::vector<T> ScalePerSliceAlongAxis(std::vector<int64_t> dims, int axis,
                                        const std::vector<T>& data) {
    uint32 seed = 123;
    std::minstd_rand rng(seed);
    int64_t out_size = 1;
    for (int dim : dims) {
      out_size *= dim;
    }
    int minor_size = 1;
    for (int i = axis + 1; i < dims.size(); ++i) {
      minor_size *= dims[i];
    }
    std::vector<T> out(out_size);
    int num_slices = (axis == -1) ? 1 : dims[axis];
    for (int out_idx = 0; out_idx < out_size; ++out_idx) {
      int in_idx = rng() % data.size();
      T multiplier = ((out_idx / minor_size) % num_slices) + 1;
      out[out_idx] = data[in_idx] * multiplier;
    }
    return out;
  }

  template <typename T>
  void RunDequantizeScaledTest(float min_range, float max_range, int axis,
                               const std::vector<T>& values,
                               const std::vector<float>& expected) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_3(mht_3_v, 333, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "RunDequantizeScaledTest");

    const std::vector<int64_t> dims = {2, 3, 4, 5};
    int num_slices = (axis == -1) ? 1 : dims[axis];
    TF_ASSERT_OK(NodeDefBuilder("dequantize_op", "Dequantize")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("mode", "SCALED")
                     .Attr("axis", axis)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    AddInputFromArray<T>(TensorShape(dims),
                         ScalePerSliceAlongAxis(dims, -1, values));
    std::vector<float> min_ranges(num_slices), max_ranges(num_slices);
    for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
      min_ranges[slice_idx] = (slice_idx + 1) * min_range;
      max_ranges[slice_idx] = (slice_idx + 1) * max_range;
    }
    AddInputFromArray<float>(TensorShape({num_slices}), min_ranges);
    AddInputFromArray<float>(TensorShape({num_slices}), max_ranges);
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected_tensor(allocator(), DT_FLOAT, TensorShape(dims));
    test::FillValues<float>(&expected_tensor,
                            ScalePerSliceAlongAxis(dims, axis, expected));
    test::ExpectClose(expected_tensor, *GetOutput(0));
  }
};

struct ParameterizedDequantizeOpTest
    : public OpsTestBase,
      public ::testing::WithParamInterface<int> {};

TEST_F(DequantizeOpTest, DequantizeMinCombinedQuint8) {
  RunDequantizeMinCombinedTest<quint8>(0, 255.0f, "Dequantize");
}
TEST_F(DequantizeOpTest, DequantizeMinCombinedQint8) {
  RunDequantizeMinCombinedTest<qint8>(0, 255.0f, "Dequantize");
}
TEST_F(DequantizeOpTest, DequantizeMinCombinedQint16) {
  RunDequantizeMinCombinedTest<qint16>(0, 255.0f, "Dequantize");
}
TEST_F(DequantizeOpTest, DequantizeMinCombinedQuint16) {
  RunDequantizeMinCombinedTest<quint16>(0, 255.0f, "Dequantize");
}

TEST_F(DequantizeOpTest, DequantizeBfloat16MinCombinedQuint8) {
  RunDequantizeBfloat16MinCombinedTest<quint8>(0, 255.0f);
}
TEST_F(DequantizeOpTest, DequantizeBfloat16MinCombinedQint8) {
  RunDequantizeBfloat16MinCombinedTest<qint8>(0, 255.0f);
}
TEST_F(DequantizeOpTest, DequantizeBfloat16MinCombinedQint16) {
  RunDequantizeBfloat16MinCombinedTest<qint16>(0, 255.0f);
}
TEST_F(DequantizeOpTest, DequantizeBfloat16MinCombinedQuint16) {
  RunDequantizeBfloat16MinCombinedTest<quint16>(0, 255.0f);
}

TEST_F(DequantizeOpTest, DequantizeScaledQuint8Zero) {
  RunDequantizeScaledTest<quint8>(-255.0f, 127.0f, -1, {0}, {0.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQuint8CheckIgnoresNegative) {
  RunDequantizeScaledTest<quint8>(-512.0f, 255.0f, -1, {255}, {255.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQuint8ScaleDown) {
  RunDequantizeScaledTest<quint8>(-1.0f, 2.0f, -1, {255}, {2.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQuint8ScaleUp) {
  RunDequantizeScaledTest<quint8>(200.0f, 400.0f, -1, {255}, {400.0});
}

TEST_F(DequantizeOpTest, DequantizeScaledQint8Zero) {
  RunDequantizeScaledTest<qint8>(-255.0f, 127.0f, -1, {0}, {0.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8ScaleIdentity) {
  RunDequantizeScaledTest<qint8>(-10.0f, 127.0f, -1, {-127}, {-127.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8ScaleDown) {
  RunDequantizeScaledTest<qint8>(-2.0f, 1.0f, -1, {-128}, {-2.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8ScaleUp) {
  RunDequantizeScaledTest<qint8>(-1.0f, 300.0f, -1, {42}, {99.212601});
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8Axis1) {
  RunDequantizeScaledTest<qint8>(-12.8f, 12.7f, 1, {-20, -10, 0, 1, 10, 20},
                                 {-2.0, -1.0, 0.0, 0.1, 1.0, 2.0});
}
TEST_F(DequantizeOpTest, DequantizeScaledQint8Axis3) {
  RunDequantizeScaledTest<qint8>(-12.8f, 12.7f, 3, {-20, -10, 0, 1, 10, 20},
                                 {-2.0, -1.0, 0.0, 0.1, 1.0, 2.0});
}

template <typename T>
static void BM_DequantizeMinCombinedCpu(::testing::benchmark::State& state) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_4(mht_4_v, 432, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeMinCombinedCpu");

  auto root = Scope::NewRootScope().ExitOnError();
  const int64_t num_values = 1500 * 250;
  std::vector<T> inputs;

  inputs.reserve(num_values);
  for (int i = 0; i < num_values; ++i) inputs.push_back(i);

  ops::Dequantize(root, test::AsTensor<T>(inputs), test::AsScalar<float>(-1.5f),
                  test::AsScalar<float>(20.5f),
                  ops::Dequantize::Attrs().Mode("MIN_COMBINED"));
  TF_CHECK_OK(root.status());
  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(g));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(state.iterations() * num_values *
                          (sizeof(float) + sizeof(T)));
  state.SetItemsProcessed(state.iterations());
}

void BM_DequantizeMinCombinedCpuQuint16(::testing::benchmark::State& state) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_5(mht_5_v, 456, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeMinCombinedCpuQuint16");

  BM_DequantizeMinCombinedCpu<quint16>(state);
}

void BM_DequantizeMinCombinedCpuQint16(::testing::benchmark::State& state) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_6(mht_6_v, 463, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeMinCombinedCpuQint16");

  BM_DequantizeMinCombinedCpu<qint16>(state);
}

void BM_DequantizeMinCombinedCpuQuint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_7(mht_7_v, 470, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeMinCombinedCpuQuint8");

  BM_DequantizeMinCombinedCpu<quint8>(state);
}

void BM_DequantizeMinCombinedCpuQint8(::testing::benchmark::State& state) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_8(mht_8_v, 477, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeMinCombinedCpuQint8");

  BM_DequantizeMinCombinedCpu<qint8>(state);
}

BENCHMARK(BM_DequantizeMinCombinedCpuQuint16);
BENCHMARK(BM_DequantizeMinCombinedCpuQint16);
BENCHMARK(BM_DequantizeMinCombinedCpuQuint8);
BENCHMARK(BM_DequantizeMinCombinedCpuQint8);

template <typename T>
static void BM_DequantizeBfloat16MinCombinedCpu(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_9(mht_9_v, 491, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeBfloat16MinCombinedCpu");

  auto root = Scope::NewRootScope().ExitOnError();
  const int64_t num_values = 1500 * 250;
  std::vector<T> inputs;

  inputs.reserve(num_values);
  for (int i = 0; i < num_values; ++i) inputs.push_back(i);

  ops::Dequantize(root, test::AsTensor<T>(inputs), test::AsScalar<float>(-1.5f),
                  test::AsScalar<float>(20.5f),
                  ops::Dequantize::Attrs().Dtype(DT_BFLOAT16));
  TF_CHECK_OK(root.status());
  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(g));

  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetBytesProcessed(state.iterations() * num_values *
                          (sizeof(bfloat16) + sizeof(T)));
  state.SetItemsProcessed(state.iterations());
}

void BM_DequantizeBfloat16MinCombinedCpuQuint16(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_10(mht_10_v, 516, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeBfloat16MinCombinedCpuQuint16");

  BM_DequantizeBfloat16MinCombinedCpu<quint16>(state);
}

void BM_DequantizeBfloat16MinCombinedCpuQint16(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_11(mht_11_v, 524, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeBfloat16MinCombinedCpuQint16");

  BM_DequantizeBfloat16MinCombinedCpu<qint16>(state);
}

void BM_DequantizeBfloat16MinCombinedCpuQuint8(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_12(mht_12_v, 532, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeBfloat16MinCombinedCpuQuint8");

  BM_DequantizeBfloat16MinCombinedCpu<quint8>(state);
}

void BM_DequantizeBfloat16MinCombinedCpuQint8(
    ::testing::benchmark::State& state) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdequantize_op_testDTcc mht_13(mht_13_v, 540, "", "./tensorflow/core/kernels/dequantize_op_test.cc", "BM_DequantizeBfloat16MinCombinedCpuQint8");

  BM_DequantizeBfloat16MinCombinedCpu<qint8>(state);
}

BENCHMARK(BM_DequantizeBfloat16MinCombinedCpuQuint16);
BENCHMARK(BM_DequantizeBfloat16MinCombinedCpuQint16);
BENCHMARK(BM_DequantizeBfloat16MinCombinedCpuQuint8);
BENCHMARK(BM_DequantizeBfloat16MinCombinedCpuQint8);

}  // namespace
}  // namespace tensorflow
