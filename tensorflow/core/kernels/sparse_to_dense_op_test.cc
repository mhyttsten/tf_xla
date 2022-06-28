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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_op_testDTcc() {
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
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {

class SparseToDenseTest : public OpsTestBase {
 protected:
  void MakeOp(int dim, DataType index_type, DataType value_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_op_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/kernels/sparse_to_dense_op_test.cc", "MakeOp");

    TF_ASSERT_OK(NodeDefBuilder("sparsetodense", "SparseToDense")
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(value_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseToDenseTest, OneD_OneValue) {
  MakeOp(1, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32>(TensorShape({3}), {1, 3, 4});
  // output_shape
  AddInputFromArray<int32>(TensorShape({1}), {5});
  // sparse_values
  AddInputFromArray<float>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {5});
  test::FillValues<float>(&expected, {-2, 2, -2, 2, 2});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, OneD_OneValue_int64_double) {
  MakeOp(1, DT_INT64, DT_DOUBLE);

  // sparse_indices
  AddInputFromArray<int64_t>(TensorShape({3}), {1, 3, 4});
  // output_shape
  AddInputFromArray<int64_t>(TensorShape({1}), {5});
  // sparse_values
  AddInputFromArray<double>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<double>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_DOUBLE, {5});
  test::FillValues<double>(&expected, {-2, 2, -2, 2, 2});
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, OneD_MultValues) {
  MakeOp(1, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32>({3}, {1, 3, 4});
  // output_shape
  AddInputFromArray<int32>({1}, {5});
  // sparse_values
  AddInputFromArray<float>({3}, {3, 4, 5});
  // default_value
  AddInputFromArray<float>({}, {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {5});
  test::FillValues<float>(&expected, {-2, 3, -2, 4, 5});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, TwoD_OneValue) {
  MakeOp(2, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32>(TensorShape({3, 2}), {0, 1, 0, 2, 2, 3});
  // output_shape
  AddInputFromArray<int32>(TensorShape({2}), {3, 4});
  // sparse_values
  AddInputFromArray<float>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 2>()(0, 1) = 2;
  expected.tensor<float, 2>()(0, 2) = 2;
  expected.tensor<float, 2>()(2, 3) = 2;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, TwoD_MultValues) {
  MakeOp(2, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32>(TensorShape({3, 2}), {0, 1, 0, 2, 2, 3});
  // output_shape
  AddInputFromArray<int32>(TensorShape({2}), {3, 4});
  // sparse_values
  AddInputFromArray<float>(TensorShape({3}), {3, 4, 5});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 2>()(0, 1) = 3;
  expected.tensor<float, 2>()(0, 2) = 4;
  expected.tensor<float, 2>()(2, 3) = 5;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, ThreeD_OneValue) {
  MakeOp(3, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32>(TensorShape({3, 3}), {0, 1, 1, 0, 2, 0, 2, 3, 1});
  // output_shape
  AddInputFromArray<int32>(TensorShape({3}), {3, 4, 2});
  // sparse_values
  AddInputFromArray<float>(TensorShape({}), {2});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4, 2});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 3>()(0, 1, 1) = 2;
  expected.tensor<float, 3>()(0, 2, 0) = 2;
  expected.tensor<float, 3>()(2, 3, 1) = 2;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(SparseToDenseTest, ThreeD_MultValues) {
  MakeOp(3, DT_INT32, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int32>(TensorShape({3, 3}), {0, 1, 1, 0, 2, 0, 2, 3, 1});
  // output_shape
  AddInputFromArray<int32>(TensorShape({3}), {3, 4, 2});
  // sparse_values
  AddInputFromArray<float>(TensorShape({3}), {3, 4, 5});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {-2});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_FLOAT, {3, 4, 2});
  expected.flat<float>().setConstant(-2);
  expected.tensor<float, 3>()(0, 1, 1) = 3;
  expected.tensor<float, 3>()(0, 2, 0) = 4;
  expected.tensor<float, 3>()(2, 3, 1) = 5;
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

}  // namespace

static void BM_SparseToDense(::testing::benchmark::State& state) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_op_testDTcc mht_1(mht_1_v, 373, "", "./tensorflow/core/kernels/sparse_to_dense_op_test.cc", "BM_SparseToDense");

  const int NDIM = state.range(0);
  const int N = state.range(1);

  // TODO(zhifengc): Switch to use kernel_benchmark_testlib.h

  const int IndexDim = (NDIM == 1) ? 0 : 1;

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  gtl::InlinedVector<TensorValue, 4> inputs;

  // Create a dense tensor with dims [1, ..., 1, N]
  Tensor output_shape(DT_INT32, TensorShape({NDIM}));
  Tensor sparse_indices(DT_INT64, TensorShape({N, NDIM}));
  Tensor sparse_values(DT_FLOAT, TensorShape({N}));
  Tensor default_value(DT_FLOAT, TensorShape({}));
  auto output_shape_t = output_shape.vec<int32>();
  for (int d = 0; d < NDIM; ++d) {
    output_shape_t(d) = (d == IndexDim) ? N : 3;
  }

  auto sparse_indices_t = sparse_indices.matrix<int64_t>();
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < NDIM; ++d)
      sparse_indices_t(n, d) = (d == IndexDim) ? n : 0;
  }

  for (auto* ptr :
       {&sparse_indices, &output_shape, &sparse_values, &default_value}) {
    inputs.push_back({nullptr, ptr});
  }

  NodeDef sparse_node_def;
  TF_CHECK_OK(NodeDefBuilder("sparsetodense", "SparseToDense")
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&sparse_node_def));

  Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), sparse_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> sparse_context(new OpKernelContext(&params));
  op->Compute(sparse_context.get());
  for (auto s : state) {
    delete sparse_context->release_output(0).tensor;
    op->Compute(sparse_context.get());
    TF_ASSERT_OK(sparse_context->status());
  }

  // processing input, mainly
  int64_t bytes_per_iter = static_cast<int64_t>((N + N * NDIM) * sizeof(float));
  state.SetBytesProcessed(bytes_per_iter * state.iterations());
}

BENCHMARK(BM_SparseToDense)
    ->ArgPair(1, 10)
    ->ArgPair(1, 100)
    ->ArgPair(1, 1000)
    ->ArgPair(1, 10000)
    ->ArgPair(2, 10)
    ->ArgPair(2, 100)
    ->ArgPair(2, 1000)
    ->ArgPair(2, 10000)
    ->ArgPair(3, 10)
    ->ArgPair(3, 100)
    ->ArgPair(3, 1000)
    ->ArgPair(3, 10000)
    ->ArgPair(5, 10)
    ->ArgPair(5, 100)
    ->ArgPair(5, 1000)
    ->ArgPair(5, 10000);

}  // namespace tensorflow
