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
class MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc() {
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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class ConstantOpTest : public OpsTestBase {
 protected:
  void PersistentMemoryTrackingTest(bool on_gpu);
};

void ConstantOpTest::PersistentMemoryTrackingTest(bool on_gpu) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/constant_op_test.cc", "ConstantOpTest::PersistentMemoryTrackingTest");

  DataType data_type = DT_INT32;
  std::initializer_list<int64_t> dims = {2, 3, 4, 5};
  Tensor tensor(data_type, TensorShape(dims));
  for (int i = 0; i < 2 * 3 * 4 * 5; ++i) {
    tensor.flat<int32>()(i) = i;
  }

  NodeDef const_node;
  TF_ASSERT_OK(NodeDefBuilder("some_node", "Const")
                   .Attr("dtype", data_type)
                   .Attr("value", tensor)
                   .Finalize(&const_node));

  string device_string = "CPU";
  DeviceType device_type = DEVICE_CPU;
  if (on_gpu) {
    device_string = "GPU";
    DeviceType device_type = DEVICE_GPU;
  }
  std::unique_ptr<Device> device(DeviceFactory::NewDevice(
      device_string, {}, "/job:worker/replica:0/task:0"));

  Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(device_type, device.get(),
                                              cpu_allocator(), const_node,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_ASSERT_OK(status);

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.op_kernel = op.get();
  params.track_allocations = true;

  OpKernelContext ctx(&params);
  op->Compute(&ctx);
  TF_EXPECT_OK(ctx.status());

  if (on_gpu) {
    EXPECT_EQ(ctx.persistent_memory_allocated(), 512);
  } else {
    EXPECT_EQ(ctx.persistent_memory_allocated(), 480);
  }

  // Remove memory leak errors.
  for (auto allocator_pair : ctx.ConsumeWrappedAllocators()) {
    allocator_pair.second->GetRecordsAndUnRef();
  }
}

TEST_F(ConstantOpTest, PersistentMemoryTracking) {
  PersistentMemoryTrackingTest(false);
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
  PersistentMemoryTrackingTest(true);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

// Returns graph containing "num" const nodes.  If 'sequential' is
// true, make sure all constants are executed sequentially in the
// graph by adding control dependencies.
static Graph* ManyConsts(int num, bool sequential) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc mht_1(mht_1_v, 274, "", "./tensorflow/core/kernels/constant_op_test.cc", "ManyConsts");

  Graph* g = new Graph(OpRegistry::Global());
  Node* prev = nullptr;
  for (int i = 0; i < num; ++i) {
    Tensor c(DT_FLOAT, TensorShape({}));
    c.scalar<float>()() = i;
    Node* curr = test::graph::Constant(g, c);
    if (sequential && prev != nullptr) {
      g->AddControlEdge(prev, curr);
    }
    prev = curr;
  }
  return g;
}

static void BM_ManyConsts_Parallel(::testing::benchmark::State& state) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc mht_2(mht_2_v, 292, "", "./tensorflow/core/kernels/constant_op_test.cc", "BM_ManyConsts_Parallel");

  const int num = state.range(0);

  test::Benchmark("cpu", ManyConsts(num, false /* !sequential */),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
}
BENCHMARK(BM_ManyConsts_Parallel)->Range(1, 1 << 10);

static void BM_ManyConsts_Sequential(::testing::benchmark::State& state) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSconstant_op_testDTcc mht_3(mht_3_v, 305, "", "./tensorflow/core/kernels/constant_op_test.cc", "BM_ManyConsts_Sequential");

  const int num = state.range(0);

  test::Benchmark("cpu", ManyConsts(num, true /* sequential */),
                  /*old_benchmark_api*/ false)
      .Run(state);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num);
}
BENCHMARK(BM_ManyConsts_Sequential)->Range(1, 1 << 10);

}  // end namespace tensorflow
