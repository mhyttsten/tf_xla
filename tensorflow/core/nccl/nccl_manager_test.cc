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
class MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc() {
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
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>
#include <random>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {

static std::vector<std::unique_ptr<BaseGPUDevice>> GetGPUDevices() {
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory(DEVICE_GPU)
                  ->AddDevices(SessionOptions(), "", &devices));
  std::vector<std::unique_ptr<BaseGPUDevice>> gpus;
  for (std::unique_ptr<Device>& device : devices) {
    if (device->device_type() == "GPU") {
      // If `device_type()` is GPU, this `Device` is guaranteed to be a
      // `BaseGPUDevice`, which is a subclass of `Device`.
      gpus.emplace_back(static_cast<BaseGPUDevice*>(device.release()));
    }
  }
  return gpus;
}

template <typename Scalar>
class NcclManagerTest : public ::testing::Test {
 public:
  // A single all-reduce to apply.
  struct TestCase {
    TestCase(int num_nodes, int num_ranks_per_node)
        : num_nodes(num_nodes), num_ranks_per_node(num_ranks_per_node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_0(mht_0_v, 224, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "TestCase");
}
    std::vector<Tensor> ins;
    std::vector<Tensor> outs;
    Tensor expected;
    const int num_nodes;
    const int num_ranks_per_node;

    mutex mu;
    Status final_status;
    int num_completed TF_GUARDED_BY(mu) = 0;
    condition_variable done_cv;
  };

  static void SetUpTestSuite() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "SetUpTestSuite");

    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    devices_ = new std::vector<std::unique_ptr<BaseGPUDevice>>(GetGPUDevices());
    VLOG(1) << "Running test with " << devices_->size() << " gpus";
    if (devices_->size() <= 1) {
      LOG(FATAL) << "Cannot run NCCL test without multiple GPUs";
    }
    work_queue_ = new UnboundedWorkQueue(Env::Default(), "nccl_manager_test");
  }

  void SetUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_2(mht_2_v, 254, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "SetUp");

    ASSERT_GT(devices_->size(), 0) << "No GPUs found";
    ASSERT_NE(work_queue_, nullptr);
  }

  static int32 NumGPUs() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_3(mht_3_v, 262, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "NumGPUs");
 return static_cast<int32>(devices_->size()); }

  // Let N = #GPUs.  When N is even, num_nodes=2 and num_ranks_per_node=N/2.
  // When N is odd, num_nodes=2 and num_ranks_per_node=(N-1)/2.
  static void PopulateMultiNodeParams(int* num_nodes, int* num_ranks_per_node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_4(mht_4_v, 269, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "PopulateMultiNodeParams");

    const auto num_gpus = NumGPUs();
    CHECK_GT(num_gpus, 1);
    *num_nodes = 2;
    if (num_gpus % 2 == 0) {
      *num_ranks_per_node = num_gpus / 2;
    } else {
      *num_ranks_per_node = (num_gpus - 1) / 2;
    }
  }

  static void TearDownTestSuite() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "TearDownTestSuite");

    delete devices_;
    delete work_queue_;
  }

  TestCase* MakeReductionTestCase(int num_nodes, int num_ranks_per_node,
                                  ncclRedOp_t reduction_op, TensorShape shape,
                                  float value_offset) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "MakeReductionTestCase");

    TestCase* test_case = new TestCase(num_nodes, num_ranks_per_node);
    test_case->expected = Tensor(data_type_, shape);
    if (reduction_op == ncclProd) {
      test::FillFn<Scalar>(&test_case->expected,
                           [](int) { return static_cast<Scalar>(1); });
    } else if (reduction_op == ncclSum) {
      test::FillFn<Scalar>(&test_case->expected,
                           [](int) { return static_cast<Scalar>(0); });
    } else if (reduction_op == ncclMax) {
      test::FillFn<Scalar>(&test_case->expected, [](int) { return -max_; });
    } else if (reduction_op == ncclMin) {
      test::FillFn<Scalar>(&test_case->expected, [](int) { return max_; });
    } else {
      LOG(FATAL) << "Invalid reduction_op " << reduction_op;
    }

    float value_scale = 0.01;  // Small scale to avoid fp16 overflow.
    for (int node = 0; node < num_nodes; ++node) {
      for (int local_rank = 0; local_rank < num_ranks_per_node; ++local_rank) {
        auto* device = GetDevice(num_ranks_per_node, node, local_rank);
        auto* stream = device->tensorflow_accelerator_device_info()->stream;

        Tensor in_cpu(data_type_, shape);
        test::FillFn<Scalar>(&in_cpu, [&](int index) {
          return static_cast<Scalar>((index + 1) * value_scale + value_offset);
        });
        for (int j = 0; j < shape.num_elements(); ++j) {
          auto in_val = in_cpu.flat<Scalar>()(j);
          auto out_expr = test_case->expected.template flat<Scalar>();
          if (reduction_op == ncclProd) {
            out_expr(j) = out_expr(j) * in_val;
          } else if (reduction_op == ncclSum) {
            out_expr(j) = out_expr(j) + in_val;
          } else if (reduction_op == ncclMax) {
            if (in_val > out_expr(j)) {
              out_expr(j) = in_val;
            }
          } else if (reduction_op == ncclMin) {
            if (in_val < out_expr(j)) {
              out_expr(j) = in_val;
            }
          }
        }

        value_scale *= 10;
        test_case->ins.emplace_back(GpuAllocator(device), data_type_, shape);
        test_case->outs.emplace_back(GpuAllocator(device), data_type_, shape);

        const Tensor& in_gpu = test_case->ins.back();
        auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<Scalar>().data());
        stream->ThenMemcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                           in_cpu.TotalBytes());
      }
    }

    return test_case;
  }

  TestCase* MakeGatherTestCase(int num_nodes, int num_ranks_per_node,
                               TensorShape in_shape, TensorShape out_shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_7(mht_7_v, 356, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "MakeGatherTestCase");

    TestCase* test_case = new TestCase(num_nodes, num_ranks_per_node);
    test_case->expected = Tensor(data_type_, out_shape);
    test::FillFn<Scalar>(&test_case->expected,
                         [](int) { return static_cast<Scalar>(0); });

    float value_scale = 0.01;  // Small scale to avoid fp16 overflow.
    for (int node = 0; node < num_nodes; ++node) {
      for (int i = 0; i < num_ranks_per_node; ++i) {
        auto* device = GetDevice(num_ranks_per_node, node, i);
        auto* stream = device->tensorflow_accelerator_device_info()->stream;

        Tensor in_cpu(data_type_, in_shape);
        test::FillFn<Scalar>(&in_cpu, [&](int index) {
          return static_cast<Scalar>((index + 1) * value_scale);
        });
        // Starting index for this rank's tensor in the all-gathered output.
        int32_t gather_idx =
            (node * num_ranks_per_node + i) * in_shape.num_elements();
        for (int j = 0; j < in_shape.num_elements(); ++j) {
          auto in_val = in_cpu.flat<Scalar>()(j);
          auto out_expr = test_case->expected.template flat<Scalar>();
          out_expr(gather_idx + j) = in_val;
        }

        value_scale *= 10;
        test_case->ins.emplace_back(GpuAllocator(device), data_type_, in_shape);
        test_case->outs.emplace_back(GpuAllocator(device), data_type_,
                                     out_shape);

        const Tensor& in_gpu = test_case->ins.back();
        auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<Scalar>().data());
        stream->ThenMemcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                           in_cpu.TotalBytes());
      }
    }

    return test_case;
  }

  // Make a broadcast test which broadcasts a tensor with shape `shape` from
  // `src_node`, `src_rank` to all other ranks.
  // If `in_place` is true, input and output are the same for the source,
  // otherwise they are tensors backed by different buffers.
  TestCase* MakeBroadcastTestCase(int num_nodes, int num_ranks_per_node,
                                  TensorShape shape, int src_node, int src_rank,
                                  bool in_place) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_8(mht_8_v, 405, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "MakeBroadcastTestCase");

    TestCase* test_case = new TestCase(num_nodes, num_ranks_per_node);
    test_case->expected = Tensor(data_type_, shape);
    test::FillFn<Scalar>(&test_case->expected,
                         [](int) { return static_cast<Scalar>(1); });

    for (int node = 0; node < num_nodes; ++node) {
      for (int local_rank = 0; local_rank < num_ranks_per_node; ++local_rank) {
        auto* device = GetDevice(num_ranks_per_node, node, local_rank);
        if (node == src_node && local_rank == src_rank) {
          test_case->ins.emplace_back(GpuAllocator(device), data_type_, shape);
          if (in_place) {
            test_case->outs.emplace_back(test_case->ins.back());
          } else {
            test_case->outs.emplace_back(GpuAllocator(device), data_type_,
                                         shape);
          }
          Tensor in_cpu(data_type_, shape);
          test::FillFn<Scalar>(&in_cpu,
                               [](int) { return static_cast<Scalar>(1); });
          const Tensor& in_gpu = test_case->ins.back();
          auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<Scalar>().data());
          auto* stream = device->tensorflow_accelerator_device_info()->stream;
          stream->ThenMemcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                             in_cpu.TotalBytes());
        } else {
          test_case->ins.emplace_back(Tensor());
          test_case->outs.emplace_back(GpuAllocator(device), data_type_, shape);
        }
      }
    }

    return test_case;
  }

  // Waits for the done callback to be called for each participant.
  void WaitForTestCompletion(TestCase* test_case) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_9(mht_9_v, 444, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "WaitForTestCompletion");

    mutex_lock l(test_case->mu);
    while (test_case->num_completed != test_case->outs.size()) {
      test_case->done_cv.wait(l);
    }
  }

  void VerifyResults(TestCase* test_case) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_10(mht_10_v, 454, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "VerifyResults");

    WaitForTestCompletion(test_case);
    TF_ASSERT_OK(test_case->final_status);
    // Copy memory to host and verify.
    for (int node = 0; node < test_case->num_nodes; ++node) {
      for (int local_rank = 0; local_rank < test_case->num_ranks_per_node;
           ++local_rank) {
        auto* device =
            GetDevice(test_case->num_ranks_per_node, node, local_rank);
        auto* stream = device->tensorflow_accelerator_device_info()->stream;
        const int global_rank =
            GlobalRank(test_case->num_ranks_per_node, node, local_rank);
        const Tensor& out_gpu = test_case->outs[global_rank];
        Tensor out_cpu(data_type_, out_gpu.shape());
        auto out_gpu_mem = AsDeviceMemory(out_gpu.flat<Scalar>().data());
        stream->ThenMemcpy(out_cpu.flat<Scalar>().data(), out_gpu_mem,
                           out_cpu.TotalBytes());
        SE_ASSERT_OK(stream->BlockHostUntilDone());
        VLOG(1) << "Verifying rank " << global_rank << " expected shape "
                << test_case->expected.shape() << " out shape "
                << out_cpu.shape();
        test::ExpectClose(test_case->expected, out_cpu);
      }
    }
  }

  void VerifyError(TestCase* test_case) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_11(mht_11_v, 483, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "VerifyError");

    WaitForTestCompletion(test_case);
    LOG(INFO) << test_case->final_status;
    EXPECT_EQ(test_case->final_status.code(), error::INTERNAL);
  }

  NcclManager::DoneCallback CreateDoneCallback(TestCase* test_case) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_12(mht_12_v, 492, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "CreateDoneCallback");

    return [this, test_case](Status s) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_13(mht_13_v, 496, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "lambda");

      mutex_lock l(test_case->mu);
      test_case->final_status.Update(s);
      if (++test_case->num_completed == test_case->outs.size()) {
        test_case->done_cv.notify_one();
      }
    };
  }

  struct NodeState {
    NcclManager nccl_manager;
    std::atomic<int> launched{0};
  };

  void RunMultiNodeAllReduceTest(const int num_nodes,
                                 const int num_ranks_per_node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_14(mht_14_v, 514, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "RunMultiNodeAllReduceTest");

    std::vector<NodeState> node_states(num_nodes);
    RunMultiNodeAllReduceTest(node_states, num_ranks_per_node);
  }

  void RunMultiNodeAllReduceTest(std::vector<NodeState>& node_states,
                                 const int num_ranks_per_node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_15(mht_15_v, 523, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "RunMultiNodeAllReduceTest");

    const int num_nodes = node_states.size();
    const int num_global_ranks = num_nodes * num_ranks_per_node;
    const string collective_key = "allreduce";
    // The NcclManagers in this test synchronize in real-time, so we need to run
    // each node's code in a separate thread.
    // Specifically, the call to ncclGroupEnd() after calling ncclCommInitRank
    // waits for all communicators before returning.

    // First, initialize the communicator_key used for this collective.
    const string communicator_key =
        node_states[0].nccl_manager.GenerateCommunicatorKey();

    for (int op = 0; op < 4; ++op) {
      ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
      std::unique_ptr<TestCase> test_case(
          this->MakeReductionTestCase(num_nodes, num_ranks_per_node,
                                      reduction_op, TensorShape({2, 3}), 0.0f));
      for (int node = 0; node < num_nodes; ++node) {
        auto node_fn = [this, node, num_ranks_per_node, num_global_ranks,
                        &node_states, &communicator_key, &collective_key,
                        reduction_op, &test_case] {
          for (int local_rank = 0; local_rank < num_ranks_per_node;
               ++local_rank) {
            auto* device = GetDevice(num_ranks_per_node, node, local_rank);
            auto* info = device->tensorflow_accelerator_device_info();
            auto* stream = device->tensorflow_accelerator_device_info()->stream;
            const int global_rank =
                GlobalRank(num_ranks_per_node, node, local_rank);
            auto participant = absl::make_unique<NcclManager::Participant>(
                device->executor(), stream, info, &test_case->ins[global_rank],
                &test_case->outs[global_rank], global_rank,
                this->CreateDoneCallback(test_case.get()));
            node_states[node].nccl_manager.AddToAllReduce(
                std::move(participant),
                {collective_key, num_ranks_per_node, num_global_ranks,
                 communicator_key, /*source_rank=*/-1},
                reduction_op);
            VLOG(1) << "AddToAllReduce node " << node << " global_rank "
                    << global_rank;
          }

          // Signal collective ready to launch at this node.
          node_states[node].nccl_manager.SignalMultiNodeReady(collective_key);
        };
        this->work_queue_->Schedule(node_fn);
      }

      VLOG(2) << "Verifying results";
      this->VerifyResults(test_case.get());
    }
  }

  void RunMultiNodeBroadcastTest(const int num_nodes,
                                 const int num_ranks_per_node,
                                 const int src_node, const int src_local_rank,
                                 const bool in_place) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_16(mht_16_v, 582, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "RunMultiNodeBroadcastTest");

    const int num_global_ranks = num_nodes * num_ranks_per_node;
    const int src_global_rank = src_node * num_ranks_per_node + src_local_rank;
    const string collective_key = "broadcast";
    std::vector<NodeState> node_states(num_nodes);
    const string communicator_key =
        node_states[0].nccl_manager.GenerateCommunicatorKey();
    std::unique_ptr<TestCase> test_case(this->MakeBroadcastTestCase(
        num_nodes, num_ranks_per_node, TensorShape({5, 6}), src_node,
        src_local_rank, in_place));
    for (int node = 0; node < num_nodes; ++node) {
      for (int local_rank = 0; local_rank < num_ranks_per_node; ++local_rank) {
        // Launch each rank in a separate thread to test concurrent,
        // randomly-ordered calls into NcclManager.
        auto rank_fn = [this, node, num_ranks_per_node, num_global_ranks,
                        src_global_rank, local_rank, &node_states,
                        &collective_key, &communicator_key, &test_case]() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_17(mht_17_v, 601, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "lambda");

          auto* device = GetDevice(num_ranks_per_node, node, local_rank);
          auto* info = device->tensorflow_accelerator_device_info();
          auto* stream = device->tensorflow_accelerator_device_info()->stream;
          const int global_rank =
              GlobalRank(num_ranks_per_node, node, local_rank);
          auto* input = global_rank == src_global_rank
                            ? &test_case->ins[global_rank]
                            : nullptr;
          auto* output = test_case->outs[global_rank].NumElements() == 0
                             ? nullptr
                             : &test_case->outs[global_rank];
          auto participant = absl::make_unique<NcclManager::Participant>(
              device->executor(), stream, info, input, output, global_rank,
              this->CreateDoneCallback(test_case.get()));
          if (global_rank == src_global_rank) {
            node_states[node].nccl_manager.AddBroadcastSend(
                std::move(participant),
                {collective_key, num_ranks_per_node, num_global_ranks,
                 communicator_key, src_global_rank});
          } else {
            node_states[node].nccl_manager.AddBroadcastRecv(
                std::move(participant),
                {collective_key, num_ranks_per_node, num_global_ranks,
                 communicator_key, src_global_rank});
          }

          if (++node_states[node].launched == num_ranks_per_node) {
            // Signal collective ready to launch at this node.
            node_states[node].nccl_manager.SignalMultiNodeReady(collective_key);
          }
        };
        this->work_queue_->Schedule(std::move(rank_fn));
      }
    }

    VLOG(2) << "Verifying results";
    this->VerifyResults(test_case.get());
  }

  static int GlobalRank(int num_ranks_per_node, int node, int local_rank) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_18(mht_18_v, 644, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "GlobalRank");

    return node * num_ranks_per_node + local_rank;
  }

  static BaseGPUDevice* GetDevice(int num_ranks_per_node, int node,
                                  int local_rank) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_19(mht_19_v, 652, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "GetDevice");

    const int device_idx = GlobalRank(num_ranks_per_node, node, local_rank);
    CHECK_LT(device_idx, devices_->size());
    return (*devices_)[device_idx].get();
  }

  static UnboundedWorkQueue* work_queue_;

 private:
  static Allocator* GpuAllocator(BaseGPUDevice* device) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_20(mht_20_v, 664, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "GpuAllocator");

    return device->GetAllocator(AllocatorAttributes());
  }

  static se::DeviceMemory<Scalar> AsDeviceMemory(const Scalar* cuda_memory) {
    se::DeviceMemoryBase wrapped(const_cast<Scalar*>(cuda_memory));
    se::DeviceMemory<Scalar> typed(wrapped);
    return typed;
  }

  static std::vector<std::unique_ptr<BaseGPUDevice>>* devices_;
  static const DataType data_type_;
  static const Scalar max_;
};

template <typename Scalar>
std::vector<std::unique_ptr<BaseGPUDevice>>* NcclManagerTest<Scalar>::devices_ =
    nullptr;
template <typename Scalar>
const DataType NcclManagerTest<Scalar>::data_type_ =
    DataTypeToEnum<Scalar>::value;
template <typename Scalar>
const Scalar NcclManagerTest<Scalar>::max_ =
    Eigen::NumTraits<Scalar>::highest();
template <typename Scalar>
UnboundedWorkQueue* NcclManagerTest<Scalar>::work_queue_ = nullptr;

// Instantiate tests for float and double.
using TypeList = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NcclManagerTest, TypeList);

// Test basic sum reduction.
TYPED_TEST(NcclManagerTest, BasicSumReduction) {
  const int num_ranks = this->NumGPUs();

  for (int op = 0; op < 4; ++op) {
    ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, reduction_op,
                                    TensorShape({2, 3}), 0.0f));
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
      VLOG(2) << "rank " << rank << " device " << device->name();
      auto* info = device->tensorflow_accelerator_device_info();
      auto* stream = device->tensorflow_accelerator_device_info()->stream;
      auto participant = absl::make_unique<NcclManager::Participant>(
          device->executor(), stream, info, &test_case->ins[rank],
          &test_case->outs[rank], /*global_rank=*/-1,
          this->CreateDoneCallback(test_case.get()));
      NcclManager::instance()->AddToAllReduce(
          std::move(participant),
          {"allreduce", /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks, /*communicator_key=*/"",
           /*source_rank=*/-1},
          reduction_op);
    }

    LOG(INFO) << "Verifying results";
    this->VerifyResults(test_case.get());
  }
}

// Same as the Basic test, but with multiple threads launching parts of many
// reductions.
//
// To run test longer, increase num_ranks, num_collectives_per_iteration and
// time_limit_micros.
TYPED_TEST(NcclManagerTest, MultipleCallers) {
  const int num_ranks = this->NumGPUs();
  const int num_collectives_per_iteration = 10;
  const int time_limit_micros = 1 * 1000 * 1000;  // 1 second

  int64_t start = Env::Default()->NowMicros();
  srand(Env::Default()->NowMicros());

  for (;;) {
    std::vector<std::pair<int, int>> case_and_rank;
    std::vector<std::unique_ptr<typename TestFixture::TestCase>> test_cases;
    for (int i = 0; i < num_collectives_per_iteration; ++i) {
      test_cases.emplace_back(this->MakeReductionTestCase(
          /*num_nodes=*/1, num_ranks, ncclSum,
          TensorShape({100, i % 5 + 1, i % 3 + 1}), 1.1f * i));
      for (int j = 0; j < num_ranks; ++j) {
        case_and_rank.emplace_back(i, j);
      }
    }

    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
      auto* stream = device->tensorflow_accelerator_device_info()->stream;
      SE_ASSERT_OK(stream->BlockHostUntilDone());
    }

    std::shuffle(case_and_rank.begin(), case_and_rank.end(),
                 std::mt19937(std::random_device()()));

    mutex mu;  // guards case_and_rank.
    const int to_schedule = case_and_rank.size();
    for (int i = 0; i < to_schedule; ++i) {
      auto fn = [&]() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_21(mht_21_v, 766, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "lambda");

        int rank;
        int test_num;
        {
          mutex_lock l(mu);
          test_num = case_and_rank.back().first;
          rank = case_and_rank.back().second;
          case_and_rank.pop_back();
        }
        auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
        auto* info = device->tensorflow_accelerator_device_info();
        auto* stream = device->tensorflow_accelerator_device_info()->stream;
        typename TestFixture::TestCase* test_case = test_cases[test_num].get();
        auto participant = absl::make_unique<NcclManager::Participant>(
            device->executor(), stream, info, &test_case->ins[rank],
            &test_case->outs[rank], /*global_rank=*/-1,
            this->CreateDoneCallback(test_case));
        NcclManager::instance()->AddToAllReduce(
            std::move(participant),
            {strings::StrCat("allreduce", test_num),
             /*num_local_devices=*/num_ranks,
             /*num_global_devices=*/num_ranks,
             /*communicator_key=*/"", /*source_rank=*/-1},
            ncclSum);
      };
      this->work_queue_->Schedule(fn);
    }

    VLOG(2) << "Verifying results for " << num_collectives_per_iteration
            << " collectives";
    for (int i = 0; i < test_cases.size(); ++i) {
      this->VerifyResults(test_cases[i].get());
    }

    int64_t delta = Env::Default()->NowMicros() - start;
    if (delta > time_limit_micros) {
      LOG(INFO) << "Ran for " << delta << " microsecs, now quitting";
      break;
    }
  }
}

// Test basic all-gather.
TYPED_TEST(NcclManagerTest, BasicAllGather) {
  const int num_ranks = this->NumGPUs();
  for (int i = 0; i < num_ranks; ++i) {
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeGatherTestCase(/*num_nodes=*/1, num_ranks,
                                 TensorShape({2, 3}),
                                 TensorShape({2 * num_ranks, 3})));
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
      VLOG(2) << "rank " << rank << " device " << device->name();
      auto* info = device->tensorflow_accelerator_device_info();
      auto* stream = device->tensorflow_accelerator_device_info()->stream;
      auto participant = absl::make_unique<NcclManager::Participant>(
          device->executor(), stream, info, &test_case->ins[rank],
          &test_case->outs[rank], rank,
          this->CreateDoneCallback(test_case.get()));
      NcclManager::instance()->AddToAllGather(
          std::move(participant),
          {"allgather", /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks, /*communicator_key=*/"",
           /*source_rank=*/-1});
    }

    LOG(INFO) << "Verifying results";
    this->VerifyResults(test_case.get());
  }
}

// Test basic broadcast.
TYPED_TEST(NcclManagerTest, BasicBroadcast) {
  this->RunMultiNodeBroadcastTest(/*num_nodes=*/1,
                                  /*num_ranks_per_node=*/this->NumGPUs(),
                                  /*src_node=*/0, /*src_local_rank=*/0,
                                  /*in_place=*/false);
}

// Test in-place broadcast.
TYPED_TEST(NcclManagerTest, InPlaceBroadcast) {
  this->RunMultiNodeBroadcastTest(/*num_nodes=*/1,
                                  /*num_ranks_per_node=*/this->NumGPUs(),
                                  /*src_node=*/0, /*src_local_rank=*/0,
                                  /*in_place=*/true);
}

// Test broadcast with increasing ranks.
TYPED_TEST(NcclManagerTest, BroadcastWithDifferentRanks) {
  for (int num_ranks = 1; num_ranks <= this->NumGPUs(); ++num_ranks) {
    const int src_rank = static_cast<int>(random::New64() % num_ranks);
    for (int in_place_idx = 0; in_place_idx <= 1; ++in_place_idx) {
      const bool in_place = in_place_idx == 0;
      this->RunMultiNodeBroadcastTest(/*num_nodes=*/1, num_ranks,
                                      /*src_node=*/0, src_rank, in_place);
    }
  }
}

// Multi-node NCCL tests.

TEST(NcclManagerTest, CommunicatorKey) {
  const string communicator_key =
      NcclManager::instance()->GenerateCommunicatorKey();
  EXPECT_EQ(communicator_key.size(), NCCL_UNIQUE_ID_BYTES);
}

#if !TENSORFLOW_USE_ROCM
// ROCm platform currently does not support simulating a multi-node
// environment, on a single node with multiple GPUS. So tests that rely
// upon such simulation need to be skipped on the ROCm platform

// This test creates `num_nodes` NcclManagers to simulate a multi-node
// environment.  It works on a single node with multiple GPUs.  It enqueues NCCL
// kernels on separate stream per rank.
TYPED_TEST(NcclManagerTest, MultiNode) {
  int num_nodes;
  int num_ranks_per_node;
  this->PopulateMultiNodeParams(&num_nodes, &num_ranks_per_node);
  VLOG(1) << "Calling RunMultiNodeAllReduceTest with num_nodes=" << num_nodes
          << " and num_ranks_per_node=" << num_ranks_per_node;
  this->RunMultiNodeAllReduceTest(num_nodes, num_ranks_per_node);
}
#endif

// Tests that specifying `communicator_key` with a single node NCCL collective
// works well.
TYPED_TEST(NcclManagerTest, MultiNodeSingle) {
  this->RunMultiNodeAllReduceTest(/*num_nodes=*/1,
                                  /*num_ranks_per_node=*/this->NumGPUs());
}

#if !TENSORFLOW_USE_ROCM
// ROCm platform currently does not support simulating a multi-node
// environment, on a single node with multiple GPUS. So tests that rely
// upon such simulation need to be skipped on the ROCm platform

// Multi-node broadcast.
TYPED_TEST(NcclManagerTest, MultiNodeBroadcast) {
  int num_nodes;
  int num_ranks_per_node;
  this->PopulateMultiNodeParams(&num_nodes, &num_ranks_per_node);
  VLOG(1) << "Calling RunMultiNodeBroadcastTest with num_nodes=" << num_nodes
          << " and num_ranks_per_node=" << num_ranks_per_node;
  this->RunMultiNodeBroadcastTest(num_nodes, num_ranks_per_node,
                                  /*src_node=*/0, /*src_local_rank=*/0,
                                  /*in_place=*/true);
}
#endif

// Checks that we return error status if a collective_key is used for different
// types of collectives, e.g.a reduction and a broadcast.
TYPED_TEST(NcclManagerTest, ConsistentCollectiveType) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, ncclSum,
                                  TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[rank],
        &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    if (rank == 0) {
      NcclManager::instance()->AddToAllReduce(std::move(participant),
                                              {"bad_coll_type",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/-1},
                                              ncclSum);
    } else {
      NcclManager::instance()->AddBroadcastSend(
          std::move(participant),
          {"bad_coll_type",
           /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks,
           /*communicator_key=*/"", /*source_rank=*/-1});
    }
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if different communicator_key is passed to
// same collective.
TYPED_TEST(NcclManagerTest, ConsistentCommunicatorKey) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, ncclSum,
                                  TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[rank],
        &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddToAllReduce(
        std::move(participant),
        {"bad_coll_type",
         /*num_local_devices=*/num_ranks,
         /*num_global_devices=*/num_ranks,
         rank == 0 ? "" : NcclManager::instance()->GenerateCommunicatorKey(),
         /*source_rank=*/-1},
        ncclSum);
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if the number of devices is inconsistent
// across multiple participants of a collective.
TYPED_TEST(NcclManagerTest, ConsistentNumberOfDevices) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, ncclSum,
                                  TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    int num_devices = rank == 0 ? num_ranks : num_ranks + 1;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[rank],
        &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddToAllReduce(std::move(participant),
                                            {"bad_coll_type",
                                             /*num_local_devices=*/num_devices,
                                             /*num_global_devices=*/num_devices,
                                             /*communicator_key=*/"",
                                             /*source_rank=*/-1},
                                            ncclSum);
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if a broadcast does not have source.
TYPED_TEST(NcclManagerTest, BroadcastNoSource) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeBroadcastTestCase(/*num_nodes=*/1, num_ranks,
                                  TensorShape({2, 3}), /*src_node=*/-1,
                                  /*src_rank=*/-1, false));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, nullptr, &test_case->outs[rank], rank,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddBroadcastRecv(std::move(participant),
                                              {"bcast_no_send",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/-1});
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if a broadcast has multiple sends.
TYPED_TEST(NcclManagerTest, BroadcastMultipleSends) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeBroadcastTestCase(/*num_nodes=*/1, num_ranks,
                                  TensorShape({2, 3}), /*src_node=*/-1,
                                  /*src_rank=*/-1, false));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->outs[rank],
        &test_case->outs[rank], rank,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddBroadcastSend(std::move(participant),
                                              {"bcast_multiple_send",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/-1});
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if a broadcast has inconsistent source
// ranks.
TYPED_TEST(NcclManagerTest, BroadcastInconsistentSource) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeBroadcastTestCase(/*num_nodes=*/1, num_ranks,
                                  TensorShape({2, 3}), /*src_node=*/-1,
                                  /*src_rank=*/-1, false));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->outs[rank],
        &test_case->outs[rank], rank,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddBroadcastRecv(std::move(participant),
                                              {"bcast_inconsistent_source",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/rank});
  }

  this->VerifyError(test_case.get());
}

#if !TENSORFLOW_USE_ROCM
// ROCm platform currently does not support simulating a multi-node
// environment, on a single node with multiple GPUS. So tests that rely
// upon such simulation need to be skipped on the ROCm platform

TYPED_TEST(NcclManagerTest, AbortThenReset) {
  using NodeState = typename TestFixture::NodeState;
  using TestCase = typename TestFixture::TestCase;
  const int num_nodes = 2;
  std::vector<NodeState> nodes(num_nodes);
  // First do a normal all-reduce to simulate the case when there're
  // multiple communicators.
  this->RunMultiNodeAllReduceTest(nodes, /* num_ranks_per_node */ 1);

  const string collective_key = "allreduce";
  ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(0);
  auto node_fn = [&](TestCase* test_case, int node,
                     const string& communicator_key) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("communicator_key: \"" + communicator_key + "\"");
   MHTracer_DTPStensorflowPScorePSncclPSnccl_manager_testDTcc mht_22(mht_22_v, 1113, "", "./tensorflow/core/nccl/nccl_manager_test.cc", "lambda");

    auto* device = this->GetDevice(/* num_ranks_per_node */ 1, node,
                                   /* local_rank */ 0);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[node],
        &test_case->outs[node], /* global_rank */ node,
        this->CreateDoneCallback(test_case));
    nodes[node].nccl_manager.AddToAllReduce(
        std::move(participant),
        {collective_key, /* num_local_devices */ 1,
         /* num_global_devices */ num_nodes, communicator_key,
         /*source_rank=*/-1},
        reduction_op);
    nodes[node].nccl_manager.SignalMultiNodeReady(collective_key);
  };

  // Use a new communicator_key, which uses a new set of ncclComm underneath.
  string communicator_key = nodes[0].nccl_manager.GenerateCommunicatorKey();
  // Do a normal all-reduce with this communicator key to initialize ncclComm.
  // This is because ncclCommInitRank waits for all ranks and is blocking.
  {
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeReductionTestCase(
            /* num_nodes */ num_nodes, /* num_ranks_per_node */ 1, reduction_op,
            TensorShape({2, 3}), 0.0f));
    for (int i = 0; i < num_nodes; ++i) {
      this->work_queue_->Schedule(
          [&node_fn, &test_case, i, communicator_key]() {
            node_fn(test_case.get(), i, communicator_key);
          });
    }
    this->VerifyResults(test_case.get());
  }

  // A hanging all-reduce.
  ASSERT_GT(num_nodes, 1);
  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(
          /* num_nodes */ num_nodes, /* num_ranks_per_node */ 1, reduction_op,
          TensorShape({2, 3}), 0.0f));
  node_fn(test_case.get(), 0, communicator_key);
  Env::Default()->SleepForMicroseconds(1000000);
  for (auto& node : nodes) {
    node.nccl_manager.StartAbort(errors::Unavailable("peer down"));
  }
  {
    mutex_lock l(test_case->mu);
    while (test_case->num_completed != 1) {
      test_case->done_cv.wait(l);
    }
  }

  // Reset the aborted NcclManager and then run another all-reduce with the
  // resetted NcclManagers.
  for (auto& node : nodes) {
    node.nccl_manager.Reset();
  }
  // Regenerate the communicator_key, because this is needed to create new
  // communicators.
  communicator_key = nodes[0].nccl_manager.GenerateCommunicatorKey();
  {
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeReductionTestCase(
            /* num_nodes */ num_nodes, /* num_ranks_per_node */ 1, reduction_op,
            TensorShape({2, 3}), 0.0f));
    for (int i = 0; i < num_nodes; ++i) {
      this->work_queue_->Schedule(
          [&node_fn, &test_case, i, communicator_key]() {
            node_fn(test_case.get(), i, communicator_key);
          });
    }
    this->VerifyResults(test_case.get());
  }
}

#endif

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
