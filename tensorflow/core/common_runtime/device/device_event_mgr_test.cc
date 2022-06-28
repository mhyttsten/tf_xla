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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <atomic>

#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// Subclass EventMgr to access its private constructor.
class TEST_EventMgr : public EventMgr {
 public:
  TEST_EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
      : EventMgr(se, gpu_options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "TEST_EventMgr");
}
};

class TEST_EventMgrHelper {
 public:
  explicit TEST_EventMgrHelper(EventMgr* em) : em_(em) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "TEST_EventMgrHelper");

    // The polling loop can interfere with the measurements made here, and
    // isn't needed since the member PollEvents() always clears the queue.
    // The tested behavior is slightly different from what may occur in
    // ordinary execution.
    StopPollingLoop();
  }

  size_t queue_size() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "queue_size");

    mutex_lock l(em_->mu_);
    return em_->used_events_.size();
  }

  size_t free_size() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "free_size");

    mutex_lock l(em_->mu_);
    return em_->free_events_.size();
  }

  void PollEvents() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "PollEvents");

    while (queue_size() > 0) {
      // For ordinary tensor frees, this function
      // should synchronously harvest all complete
      // events and execute the corresponding memory frees.
      EventMgr::ToFreeVector to_free;
      {
        mutex_lock l(em_->mu_);
        em_->PollEvents(true, &to_free);
      }
      em_->FreeMemory(to_free);
    }
  }

  void StopPollingLoop() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_5(mht_5_v, 264, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "StopPollingLoop");
 return em_->StopPollingLoop(); }

  void StartPollingLoop() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_6(mht_6_v, 269, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "StartPollingLoop");
 return em_->StartPollingLoop(); }

 private:
  EventMgr* em_;
};

static std::atomic_int_fast64_t live_tensor_bytes(0);

// A TensorBuffer that counts live memory usage for testing
class TestTensorBuffer : public TensorBuffer {
 public:
  explicit TestTensorBuffer(size_t bytes)
      : TensorBuffer(nullptr), bytes_(bytes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_7(mht_7_v, 284, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "TestTensorBuffer");

    live_tensor_bytes += bytes_;
  }
  ~TestTensorBuffer() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_8(mht_8_v, 290, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "~TestTensorBuffer");
 live_tensor_bytes -= bytes_; }

  size_t size() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_9(mht_9_v, 295, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "size");
 return bytes_; }

  // Not used in this test
  TensorBuffer* root_buffer() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_10(mht_10_v, 301, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "root_buffer");
 return nullptr; }
  void FillAllocationDescription(AllocationDescription* arg) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_11(mht_11_v, 305, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "FillAllocationDescription");
}

 private:
  size_t bytes_;
};

namespace {

TEST(EventMgr, Empty) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  TEST_EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  EXPECT_EQ(0, th.queue_size());
  EXPECT_EQ(0, th.free_size());
}

// Tests that WarnIfInCallback() triggers correctly.
TEST(EventMgr, WarnIfInCallback) {
  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  TEST_EventMgr em(stream_exec, GPUOptions());
  TEST_EventMgrHelper th(&em);
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  bool hit = false;
  th.StartPollingLoop();
  device_event_mgr::WarnIfInCallback([&hit] { hit = true; });
  EXPECT_FALSE(hit);
  Notification note;
  em.ThenExecute(stream.get(), [&hit, &note]() {
    device_event_mgr::WarnIfInCallback([&hit, &note] {
      hit = true;
      note.Notify();
    });
  });
  note.WaitForNotification();
  EXPECT_TRUE(hit);
}
}  // namespace

// Provides access to private resources of BaseGPUDevice.
class GPUDeviceTestHelper {
 public:
  GPUDeviceTestHelper(size_t memory_limit, int pending_cap) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_12(mht_12_v, 351, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "GPUDeviceTestHelper");

    SessionOptions sops;
    device_ =
        DeviceFactory::NewDevice(DEVICE_GPU, sops, "/job:a/replica:0/task:0");
    gpu_.reset(reinterpret_cast<BaseGPUDevice*>(device_.release()));
    gpu_allocator_ = GPUProcessState::singleton()->GetGPUAllocator(
        GPUOptions(), TfDeviceId(0), memory_limit, /*peer_gpu_ids=*/{});
    host_allocator_ = GPUProcessState::singleton()->GetGpuHostAllocator(0);
  }

  BaseGPUDevice* gpu() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_13(mht_13_v, 364, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "gpu");
 return gpu_.get(); }
  Allocator* gpu_allocator() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_14(mht_14_v, 368, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "gpu_allocator");
 return gpu_allocator_; }
  Allocator* host_allocator() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_15(mht_15_v, 372, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "host_allocator");
 return host_allocator_; }
  se::Stream* compute_stream() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_16(mht_16_v, 376, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "compute_stream");
 return gpu_->stream_->compute; }
  se::Stream* h2d_stream() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_17(mht_17_v, 380, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "h2d_stream");
 return gpu_->stream_->host_to_device; }
  se::Stream* d2h_stream() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_18(mht_18_v, 384, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "d2h_stream");
 return gpu_->stream_->device_to_host; }
  se::Stream* d2d_stream() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_19(mht_19_v, 388, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "d2d_stream");
 return gpu_->stream_->device_to_device[0]; }
  EventMgr* event_mgr() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_20(mht_20_v, 392, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "event_mgr");
 return gpu_->em_; }
  int pending_cap() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_21(mht_21_v, 396, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "pending_cap");
 return gpu_->pending_cap_; }

 private:
  std::unique_ptr<Device> device_;
  std::unique_ptr<BaseGPUDevice> gpu_;
  Allocator* gpu_allocator_;
  Allocator* host_allocator_;
};

namespace {

// Class that can queue some GPU data transfers and simple kernels.
class EMBenchmarkHelper {
  GPUDeviceTestHelper* gpu_helper_;
  // We need one of these for each Add op in the chain.
  std::vector<std::unique_ptr<OpKernel>> add_kernels_;
  std::vector<OpKernelContext::Params*> add_params_;
  std::vector<std::unique_ptr<OpKernelContext>> add_contexts_;
  // The rest of these are one per chain.
  NodeDef add_node_def_;
  NodeDef id_node_def_;
  gtl::InlinedVector<TensorValue, 4> add_inputs_;
  std::vector<AllocatorAttributes> allocator_attrs_;
  gtl::InlinedVector<Tensor, 4> gpu_inputs_;
  gtl::InlinedVector<Tensor, 4> gpu_outputs_;
  gtl::InlinedVector<Tensor, 4> host_inputs_;
  gtl::InlinedVector<Tensor, 4> host_outputs_;

 public:
  // Length of tensors.  TODO(tucker): make this a variable parameter.
  static constexpr int kTDim = 1024;

  int num_ops() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_22(mht_22_v, 431, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "num_ops");
 return add_kernels_.size(); }
  size_t tensor_size() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_23(mht_23_v, 435, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "tensor_size");

    return add_inputs_.empty() ? 0 : add_inputs_[0]->NumElements();
  }

  Tensor& host_outputs(int i) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_24(mht_24_v, 442, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "host_outputs");
 return host_outputs_[i]; }
  Tensor& host_inputs(int i) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_25(mht_25_v, 446, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "host_inputs");
 return host_inputs_[i]; }

  EMBenchmarkHelper(GPUDeviceTestHelper* h) : gpu_helper_(h) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_26(mht_26_v, 451, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "EMBenchmarkHelper");
}

  void ReInit(int num_ops, int tensor_size) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_27(mht_27_v, 456, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "ReInit");

    gpu_inputs_.clear();
    while (gpu_inputs_.size() < 2) {
      gpu_inputs_.push_back(Tensor(gpu_helper_->gpu_allocator(), DT_FLOAT,
                                   {tensor_size}, AllocationAttributes()));
    }
    gpu_outputs_.clear();
    while (gpu_outputs_.size() < 1) {
      gpu_outputs_.push_back(Tensor(gpu_helper_->gpu_allocator(), DT_FLOAT,
                                    {tensor_size}, AllocationAttributes()));
    }
    host_inputs_.clear();
    while (host_inputs_.size() < 2) {
      int instance_index = host_inputs_.size();
      host_inputs_.push_back(Tensor(gpu_helper_->host_allocator(), DT_FLOAT,
                                    {tensor_size}, AllocationAttributes()));
      for (int i = 0; i < tensor_size; ++i) {
        host_inputs_.back().flat<float>()(i) =
            i * (1.0 + (0.5 * instance_index));
      }
    }
    host_outputs_.clear();
    while (host_outputs_.size() < 1) {
      host_outputs_.push_back(Tensor(gpu_helper_->host_allocator(), DT_FLOAT,
                                     {tensor_size}, AllocationAttributes()));
      for (int i = 0; i < tensor_size; ++i) {
        host_outputs_.back().flat<float>()(i) = -1;
      }
    }
    add_kernels_.clear();
    add_params_.clear();
    while (add_kernels_.size() < num_ops) {
      MakeAddOp();
    }
  }

  std::unique_ptr<OpKernel> GetOpKernel(const NodeDef& node_def,
                                        Status* status) {
    return CreateOpKernel("GPU", gpu_helper_->gpu(),
                          gpu_helper_->gpu_allocator(), node_def,
                          TF_GRAPH_DEF_VERSION, status);
  }

  void MakeAddOp() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_28(mht_28_v, 502, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "MakeAddOp");

    if (add_kernels_.empty()) {
      TF_ASSERT_OK(NodeDefBuilder("add_op", "Add")
                       .Input(FakeInput(DT_FLOAT))
                       .Input(FakeInput(DT_FLOAT))
                       .Device("/job:a/replica:0/task:0/GPU:0")
                       .Finalize(&add_node_def_));
    }
    Status status;
    add_kernels_.emplace_back(GetOpKernel(add_node_def_, &status));
    TF_ASSERT_OK(status);
    add_params_.push_back(new OpKernelContext::Params);
    PrepOpKernel(add_params_.back(), add_kernels_.back().get());
  }

  void SetOutputAttrs(OpKernelContext::Params* params,
                      std::vector<AllocatorAttributes>* attrs) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_29(mht_29_v, 521, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "SetOutputAttrs");

    attrs->clear();
    for (int index = 0; index < params->op_kernel->num_outputs(); index++) {
      AllocatorAttributes attr;
      const bool on_host =
          (params->op_kernel->output_memory_types()[index] == HOST_MEMORY);
      attr.set_on_host(on_host);
      attrs->push_back(attr);
    }
    params->output_attr_array = attrs->data();
    params->forward_from_array = {};
  }

  void PrepOpKernel(OpKernelContext::Params* params, OpKernel* kernel) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_30(mht_30_v, 537, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "PrepOpKernel");

    // This mimics what happens in ExecutorState::Process to run
    // a single graph node.
    params->step_id = 1;
    params->device = gpu_helper_->gpu();
    params->log_memory = false;
    params->rendezvous = nullptr;
    params->collective_executor = nullptr;
    params->session_state = nullptr;  // ???
    params->session_handle = "session_handle";
    params->tensor_store = nullptr;
    params->cancellation_manager = nullptr;

    params->call_frame = nullptr;
    params->function_library = nullptr;
    params->runner = nullptr;
    params->graph_collector = nullptr;

    params->step_container = nullptr;
    params->slice_reader_cache = nullptr;
    params->resource_manager = gpu_helper_->gpu()->resource_manager();

    params->stats_collector = nullptr;
    params->inc_num_deferred_ops_function = nullptr;
    params->dec_num_deferred_ops_function = nullptr;

    params->op_device_context = nullptr;
    params->track_allocations = false;
    params->op_kernel = kernel;
    params->frame_iter = FrameAndIter(0, 0);
    params->is_input_dead = false;

    if (add_inputs_.empty()) {
      add_inputs_.resize(2);
      add_inputs_[0] = TensorValue(&gpu_inputs_[0]);
      add_inputs_[1] = TensorValue(&gpu_inputs_[1]);
    }
    params->inputs = &add_inputs_;
    params->input_alloc_attrs = nullptr;
    SetOutputAttrs(params, &allocator_attrs_);
  }

  struct TimeSet {
    int iter = 0;
    int64_t start = 0;
    int64_t copy_done = 0;
    int64_t compute_done = 0;
    int64_t final_copy = 0;
    int64_t all_done = 0;
  };

  // Display sampled iteration times giving the approximate breakdown
  // within iterations and overall curve.
  void DisplayTimes(std::vector<TimeSet>* times) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_31(mht_31_v, 593, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "DisplayTimes");

    LOG(INFO) << "Summarize set of " << times->size() << " iters";
    for (auto& ts : *times) {
      ts.final_copy = ts.all_done - ts.compute_done;
      ts.compute_done = ts.compute_done - ts.copy_done;
      ts.copy_done = ts.copy_done - ts.start;
      ts.all_done = ts.all_done - ts.start;
    }
    struct TSSort {
      bool operator()(const TimeSet& a, const TimeSet& b) {
        return a.all_done < b.all_done;
      }
    };
    std::sort(times->begin(), times->end(), TSSort());
    int64_t last_time = 0;
    // Display first, last and every > 5% change.
    for (int i = 0; i < times->size(); ++i) {
      if (i == (times->size() - 1) ||
          (times->at(i).all_done >= (1.05 * last_time))) {
        LOG(INFO) << "rank " << i << " iter: " << times->at(i).iter
                  << " copy: " << times->at(i).copy_done
                  << " compute: " << times->at(i).compute_done
                  << " copy back: " << times->at(i).final_copy
                  << " sum: " << times->at(i).all_done;
        last_time = times->at(i).all_done;
      }
    }
  }

  // Queue one work unit on the GPU as follows:
  // 1. Copy 2 input tensors from CPU to GPU using h2d stream.
  // 2. Instruct compute stream to wait on h2d stream.
  // 3. Queue a sequence of Add ops on the compute stream, all using
  //    the same input tensors, allocating their own output tensors.
  // 4. Instruct d2h stream to wait on the compute stream.
  // 5. Copy final output tensor back to the CPU.
  // 6. Instruct the EventMgr to execute callback when the final tensor
  //    copy completes.
  // If event_after_add == true then additionally instruct the EventMgr
  //    to execute the callback after each Add completes.
  // The optional times parameter is used for gathering detailed timing
  // data.
  void DoAddChain(int adds_per_copy, int rounds, bool event_after_add,
                  std::function<void()> callback, std::vector<TimeSet>* times) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_32(mht_32_v, 639, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "DoAddChain");

    // Take an extra ref on the inputs so that the add doesn't compute in place.
    Tensor alias0(gpu_inputs_[0]);
    Tensor alias1(gpu_inputs_[1]);
    for (int r = 0; r < rounds; ++r) {
      if (times) {
        times->at(r).iter = r;
        times->at(r).start = Env::Default()->NowMicros();
      }
      gpu_helper_->h2d_stream()->ThenWaitFor(gpu_helper_->compute_stream());
      // Begin by copying the input values from CPU to GPU.
      const int64_t src_bytes = host_inputs_[0].TotalBytes();
      se::DeviceMemoryBase gpu_dst_ptr0(DMAHelper::base(&gpu_inputs_[0]),
                                        src_bytes);
      gpu_helper_->h2d_stream()->ThenMemcpy(
          &gpu_dst_ptr0, DMAHelper::base(&host_inputs_[0]), src_bytes);
      se::DeviceMemoryBase gpu_dst_ptr1(DMAHelper::base(&gpu_inputs_[1]),
                                        src_bytes);
      gpu_helper_->h2d_stream()->ThenMemcpy(
          &gpu_dst_ptr1, DMAHelper::base(&host_inputs_[1]), src_bytes);
      gpu_helper_->compute_stream()->ThenWaitFor(gpu_helper_->h2d_stream());
      if (times) {
        gpu_helper_->event_mgr()->ThenExecute(
            gpu_helper_->compute_stream(), [times, r]() {
              times->at(r).copy_done = Env::Default()->NowMicros();
            });
      }
      std::unique_ptr<OpKernelContext> ctx;
      for (int apc = 0; apc < adds_per_copy; ++apc) {
        ctx.reset(new OpKernelContext(add_params_[apc], 1));
        gpu_helper_->gpu()->Compute(add_kernels_[apc].get(), ctx.get());
        TF_ASSERT_OK(ctx->status());
        if (event_after_add) {
          gpu_helper_->event_mgr()->ThenExecute(gpu_helper_->compute_stream(),
                                                callback);
        }
      }
      // Finish by copying output back to CPU.
      if (times) {
        gpu_helper_->event_mgr()->ThenExecute(
            gpu_helper_->compute_stream(), [times, r]() {
              times->at(r).compute_done = Env::Default()->NowMicros();
            });
      }
      gpu_helper_->d2h_stream()->ThenWaitFor(gpu_helper_->compute_stream());
      const int64_t return_bytes = ctx->mutable_output(0)->TotalBytes();
      se::DeviceMemoryBase gpu_src_ptr(DMAHelper::base(ctx->mutable_output(0)),
                                       return_bytes);
      gpu_helper_->d2h_stream()->ThenMemcpy(DMAHelper::base(&host_outputs_[0]),
                                            gpu_src_ptr, return_bytes);
      gpu_helper_->event_mgr()->ThenExecute(gpu_helper_->d2h_stream(),
                                            callback);
      if (times) {
        gpu_helper_->event_mgr()->ThenExecute(
            gpu_helper_->d2h_stream(), [times, r]() {
              times->at(r).all_done = Env::Default()->NowMicros();
            });
      }
    }
  }
};

static void BM_no_ops(::testing::benchmark::State& state) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_33(mht_33_v, 704, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_no_ops");

  const int threads = state.range(0);
  const int iters = state.max_iterations;

  auto stream_exec = GPUMachineManager()->ExecutorForDevice(0).ValueOrDie();
  std::unique_ptr<se::Stream> stream(new se::Stream(stream_exec));
  CHECK(stream);
  stream->Init();
  TEST_EventMgr em(stream_exec, GPUOptions());

  auto benchmark_exec = [&]() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_34(mht_34_v, 717, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "lambda");

    std::atomic<int> counter;
    counter.store(0, std::memory_order_seq_cst);
    se::Stream* stream_ptr = stream.get();
    auto runner = [&em, &counter, stream_ptr, iters]() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_35(mht_35_v, 724, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "lambda");

      auto callback = [&counter]() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_36(mht_36_v, 728, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "lambda");
 counter.fetch_add(1); };
      for (int i = 0; i < iters; ++i) {
        em.ThenExecute(stream_ptr, callback);
      }
    };
    for (int t = 0; t < threads; ++t) {
      Env::Default()->SchedClosure(runner);
    }
    int expected = iters * threads;
    while (counter < expected) {
      Env::Default()->SleepForMicroseconds(1);
    }
  };

#ifdef PLATFORM_GOOGLE

  // The timer starts automatically
  while (state.KeepRunningBatch(state.max_iterations)) {
    benchmark_exec();
  }
#else
  // The tensorflow's own implementation of the benchmark does not support
  // running-batch (yet), therefore we had to use the Stop/StartTimer.
  // FIXME: Remove this if-def once we switched all tensorflow's benchmarks to
  // using the OSS benchmark library.

  state.ResumeTiming();
  benchmark_exec();
  state.PauseTiming();
#endif
}
BENCHMARK(BM_no_ops)->UseRealTime()->Arg(4)->Arg(8)->Arg(32);

// Benchmark functions are defined at top level.  In order to provide a real,
// persistent GPUDevice to the following function it also needs to be at top
// level.  But then we can't clean it up without a cuda runtime error, so we
// just leak it.
GPUDeviceTestHelper* gpu_helper = nullptr;
EMBenchmarkHelper* bm_helper = nullptr;
mutex helper_mu;

#ifdef PLATFORM_GOOGLE
static void BM_chain_ops(::testing::benchmark::State& state, int tensor_size,
                         int adds_per_round, bool event_after_add,
                         int pending_cap) {
#else
static void BM_chain_ops(::testing::benchmark::State& state, int tensor_size,
                         int adds_per_round, bool event_after_add,
                         int pending_cap, int threads) {
#endif
  const int iters = state.max_iterations;
  {
    mutex_lock l(helper_mu);
    if (gpu_helper && gpu_helper->pending_cap() != pending_cap) {
      delete bm_helper;
      bm_helper = nullptr;
      delete gpu_helper;
      gpu_helper = nullptr;
    }
    if (!gpu_helper) {
      gpu_helper = new GPUDeviceTestHelper(1 << 24, pending_cap);
      bm_helper = new EMBenchmarkHelper(gpu_helper);
    }
    if (bm_helper->num_ops() != adds_per_round ||
        bm_helper->tensor_size() != tensor_size) {
      bm_helper->ReInit(adds_per_round, tensor_size);
    }
  }
  std::vector<EMBenchmarkHelper::TimeSet> times;
  std::vector<EMBenchmarkHelper::TimeSet>* time_ptr = nullptr;
  if (VLOG_IS_ON(1)) {
    times.resize(iters);
    time_ptr = &times;
  }
  std::atomic<int> counter;
  counter.store(0, std::memory_order_seq_cst);
  auto callback = [&counter]() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_37(mht_37_v, 807, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "lambda");
 counter.fetch_add(1); };
  // First iter is always slow, so do one prior to the timed loop.
  int expected = 1 + (event_after_add ? adds_per_round : 0);
  bm_helper->DoAddChain(adds_per_round, 1, event_after_add, callback, nullptr);
  while (counter < expected) {
    Env::Default()->SleepForMicroseconds(1);
  }
  counter = 0;

#ifdef PLATFORM_GOOGLE
  while (state.KeepRunningBatch(state.max_iterations)) {
    expected = iters * (1 + (event_after_add ? adds_per_round : 0));
    bm_helper->DoAddChain(adds_per_round, iters, event_after_add, callback,
                          time_ptr);
    while (counter < expected) {
      Env::Default()->SleepForMicroseconds(1);
    }
  }
#else
  state.ResumeTiming();
  expected = threads * iters * (1 + (event_after_add ? adds_per_round : 0));
  for (int i = 0; i < threads; ++i) {
    Env::Default()->SchedClosure(
        [callback, iters, adds_per_round, event_after_add, time_ptr]() {
          bm_helper->DoAddChain(adds_per_round, iters, event_after_add,
                                callback, time_ptr);
        });
  }
  while (counter < expected) {
    Env::Default()->SleepForMicroseconds(1);
  }
  state.PauseTiming();
#endif
  VLOG(1) << "counter = " << counter << " post_execute Output: "
          << bm_helper->host_outputs(0).SummarizeValue(64);
  if (time_ptr) bm_helper->DisplayTimes(time_ptr);
}

#ifdef PLATFORM_GOOGLE
static void BM_chain_1024_1_false(::testing::benchmark::State& state) {
  BM_chain_ops(state, 1024, 1, false, 0);
}

static void BM_chain_1024_1_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_38(mht_38_v, 853, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_1_true");

  BM_chain_ops(state, 1024, 1, true, 0);
}

static void BM_chain_1024_10_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_39(mht_39_v, 860, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_10_false");

  BM_chain_ops(state, 1024, 10, false, 0);
}

static void BM_chain_1024_10_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_40(mht_40_v, 867, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_10_true");

  BM_chain_ops(state, 1024, 10, true, 0);
}

static void BM_chain_1024_100_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_41(mht_41_v, 874, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_100_false");

  BM_chain_ops(state, 1024, 100, false, 0);
}

static void BM_chain_1024_100_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_42(mht_42_v, 881, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_100_true");

  BM_chain_ops(state, 1024, 100, true, 0);
}

static void BM_chain_1M_1_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_43(mht_43_v, 888, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_1_false");

  BM_chain_ops(state, 1 << 20, 1, false, 0);
}

static void BM_chain_1M_1_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_44(mht_44_v, 895, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_1_true");

  BM_chain_ops(state, 1 << 20, 1, true, 0);
}

static void BM_chain_1M_10_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_45(mht_45_v, 902, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_10_false");

  BM_chain_ops(state, 1 << 20, 10, false, 0);
}

static void BM_chain_1M_10_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_46(mht_46_v, 909, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_10_true");

  BM_chain_ops(state, 1 << 20, 10, true, 0);
}

static void BM_chain_1M_100_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_47(mht_47_v, 916, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_100_false");

  BM_chain_ops(state, 1 << 20, 100, false, 0);
}

static void BM_chain_1M_100_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_48(mht_48_v, 923, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_100_true");

  BM_chain_ops(state, 1 << 20, 100, true, 0);
}

BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Threads(8);

BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Threads(1);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Threads(2);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Threads(8);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Threads(8);
#else
static void BM_chain_1024_1_false(::testing::benchmark::State& state) {
  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 1, false, 0, threads);
}

static void BM_chain_1024_1_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_49(mht_49_v, 969, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_1_true");

  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 1, true, 0, threads);
}

static void BM_chain_1024_10_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_50(mht_50_v, 977, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_10_false");

  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 10, false, 0, threads);
}

static void BM_chain_1024_10_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_51(mht_51_v, 985, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_10_true");

  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 10, true, 0, threads);
}

static void BM_chain_1024_100_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_52(mht_52_v, 993, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_100_false");

  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 100, false, 0, threads);
}

static void BM_chain_1024_100_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_53(mht_53_v, 1001, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1024_100_true");

  const int threads = state.range(0);
  BM_chain_ops(state, 1024, 100, true, 0, threads);
}

static void BM_chain_1M_1_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_54(mht_54_v, 1009, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_1_false");

  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 1, false, 0, threads);
}

static void BM_chain_1M_1_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_55(mht_55_v, 1017, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_1_true");

  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 1, true, 0, threads);
}

static void BM_chain_1M_10_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_56(mht_56_v, 1025, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_10_false");

  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 10, false, 0, threads);
}

static void BM_chain_1M_10_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_57(mht_57_v, 1033, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_10_true");

  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 10, true, 0, threads);
}

static void BM_chain_1M_100_false(::testing::benchmark::State& state) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_58(mht_58_v, 1041, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_100_false");

  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 100, false, 0, threads);
}

static void BM_chain_1M_100_true(::testing::benchmark::State& state) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSdevicePSdevice_event_mgr_testDTcc mht_59(mht_59_v, 1049, "", "./tensorflow/core/common_runtime/device/device_event_mgr_test.cc", "BM_chain_1M_100_true");

  const int threads = state.range(0);
  BM_chain_ops(state, 1 << 20, 100, true, 0, threads);
}

BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_1_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_1_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_10_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_10_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1024_100_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1024_100_true)->UseRealTime()->Arg(8);

BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_1_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_1_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_10_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_10_true)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Arg(1);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Arg(2);
BENCHMARK(BM_chain_1M_100_false)->UseRealTime()->Arg(8);
BENCHMARK(BM_chain_1M_100_true)->UseRealTime()->Arg(8);
#endif
}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
