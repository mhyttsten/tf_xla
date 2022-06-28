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
class MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/device_attributes.pb.h"
#ifdef GOOGLE_CUDA

#include "tensorflow/core/kernels/collective_nccl.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/collective_nccl_broadcaster.h"
#include "tensorflow/core/kernels/collective_nccl_gatherer.h"
#include "tensorflow/core/kernels/collective_nccl_reducer.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
static constexpr int kStepId = 10;

std::unique_ptr<OpKernel> GetKernel(const NodeDef& node, DeviceBase* device) {
  Status status;
  std::unique_ptr<OpKernel> k = CreateOpKernel(
      DEVICE_GPU, device, device->GetAllocator(AllocatorAttributes()), node,
      TF_GRAPH_DEF_VERSION, &status);
  if (!status.ok()) LOG(FATAL) << status;
  return k;
}

std::unique_ptr<OpKernel> GetAdd(DeviceBase* device) {
  NodeDef node_def;
  NodeDefBuilder builder("add_node", "Add");
  TF_CHECK_OK(builder.Attr("T", DT_FLOAT)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&node_def));
  return GetKernel(node_def, device);
}

std::unique_ptr<OpKernel> GetDiv(DeviceBase* device) {
  NodeDef node_def;
  NodeDefBuilder builder("add_node", "Div");
  TF_CHECK_OK(builder.Attr("T", DT_FLOAT)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&node_def));
  return GetKernel(node_def, device);
}

class NcclTestBase : public ::testing::Test {
 protected:
  class DeviceInstance;

  NcclTestBase(CollectiveType collective_type, const string& collective_name)
      : collective_type_(collective_type),
        collective_name_(collective_name),
        nccl_communicator_(MaybeCreateNcclCommunicator(config_proto_)),
        work_queue_(std::make_shared<UnboundedWorkQueue>(
            Env::Default(), "collective_executor")),
        col_exec_(nullptr),
        col_params_(nullptr) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("collective_name: \"" + collective_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_0(mht_0_v, 263, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "NcclTestBase");
}

  ~NcclTestBase() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_1(mht_1_v, 268, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "~NcclTestBase");

    if (col_exec_) col_exec_->Unref();
    if (col_params_) col_params_->Unref();
  }

  void SetUp() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_2(mht_2_v, 276, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "SetUp");

    std::vector<std::unique_ptr<Device>> all_devices;
    TF_CHECK_OK(DeviceFactory::GetFactory(DEVICE_GPU)
                    ->AddDevices(SessionOptions(), "", &all_devices));
    for (std::unique_ptr<Device>& d : all_devices) {
      if (d->device_type() == "GPU") {
        gpus_.emplace_back(std::move(d));
      }
    }
  }

  void Init(const int num_ranks, const int instance_key) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_3(mht_3_v, 290, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "Init");

    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    std::vector<std::unique_ptr<Device>> local_devices;
    std::vector<string> device_names;
    CHECK_LE(num_ranks, gpus_.size());
    for (int rank = 0; rank < num_ranks; ++rank) {
      local_devices.emplace_back(std::move(gpus_[rank]));
    }
    int num_gpus = local_devices.size();
    for (const auto& device : local_devices) {
      device_names.push_back(device->name());
      VLOG(2) << device->name();
    }
    if (!dev_mgr_)
      dev_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(local_devices));
    col_exec_ =
        new BaseCollectiveExecutor(&col_exec_mgr_, /*remote_access=*/nullptr,
                                   kStepId, dev_mgr_.get(), work_queue_);

    // Initialize collective params.
    col_params_ = new CollectiveParams();
    col_params_->name = "test_nccl_collective_op";
    const int group_key = num_ranks;
    col_params_->group.group_key = group_key;
    col_params_->group.device_type = DEVICE_GPU;
    col_params_->group.group_size = num_ranks;
    col_params_->instance.instance_key = instance_key;
    col_params_->instance.type = collective_type_;
    col_params_->instance.data_type = DT_FLOAT;
    col_params_->instance.impl_details.collective_name = collective_name_;
    const string task_name = "/job:worker/replica:0/task:0";
    col_params_->group.num_devices_per_task[task_name] = num_ranks;
    for (int rank = 0; rank < num_ranks; ++rank) {
      CollGroupMember member;
      member.device.set_name(device_names[rank % num_gpus]);
      col_params_->group.members.push_back(member);
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
      instances_.push_back(absl::make_unique<DeviceInstance>(
          rank, col_params_->group.members[rank].device.name(), this));
    }
  }

  // Initialize `input` tensor at rank `rank`.
  virtual void InitInput(Tensor* input, const int rank) = 0;

  // Initialize `expected` output at all `num_ranks` ranks.
  virtual void InitExpected(std::vector<float>* expected,
                            const int tensor_length, const int num_ranks) = 0;

  // Initialize device `di` specific to the collective op.
  virtual void InitDevice(DeviceInstance* di) = 0;

  // Run collective op on device `di`.
  virtual void RunCollectiveOnDevice(DeviceInstance* di) = 0;

  void RunCollective() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_4(mht_4_v, 350, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunCollective");

    int done = 0;
    mutex done_mu;
    condition_variable done_cv;
    for (const auto& instance : instances_) {
      DeviceInstance* di = instance.get();
      InitDevice(di);
      SchedClosure([this, di, &done, &done_mu, &done_cv] {
        RunCollectiveOnDevice(di);
        mutex_lock l(done_mu);
        ++done;
        done_cv.notify_all();
      });
    }

    mutex_lock l(done_mu);
    while (done < instances_.size()) done_cv.wait(l);
  }

  void RunTest(int num_ranks, int input_length, int instance_key) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_5(mht_5_v, 372, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunTest");

    if (num_ranks > gpus_.size()) {
      LOG(WARNING) << "Skipping test because required " << num_ranks
                   << " GPUs but found " << gpus_.size();
      return;
    }
    Init(num_ranks, instance_key);
    std::vector<float> expected;
    InitExpected(&expected, input_length, num_ranks);
    if (VLOG_IS_ON(3)) {
      string str_buf;
      for (const auto& x : expected) {
        strings::StrAppend(&str_buf, " ", x);
      }
      VLOG(3) << "Expected output " << str_buf;
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
      DeviceInstance* instance = instances_[rank].get();
      instance->InitTensor(DT_FLOAT, TensorShape({input_length}),
                           [this, rank](Tensor* t) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_6(mht_6_v, 394, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "lambda");
 InitInput(t, rank); });
    }
    RunCollective();
    // Confirm that every rank computed the same correct value.
    for (int rank = 0; rank < instances_.size(); ++rank) {
      TF_ASSERT_OK(instances_[rank]->status_);
      Tensor* output = &instances_[rank]->output_;
      const int output_length = output->NumElements();
      VLOG(2) << "rank " << rank << " output " << output << " buf "
              << DMAHelper::base(output);
      Tensor actual(DT_FLOAT, TensorShape({output_length}));
      Device* dev = instances_[rank]->device_;
      auto* dev_info = dev->tensorflow_accelerator_device_info();
      TF_CHECK_OK(dev_info->default_context->CopyDeviceTensorToCPUSync(
          output, /*tensor_name=*/"", dev, &actual));
      VLOG(3) << "rank " << rank << " got output tensor "
              << actual.DebugString(output_length);
      for (int i = 0; i < output_length; ++i) {
        EXPECT_FLOAT_EQ(expected[i], actual.template flat<float>()(i))
            << "Mismatch at rank " << rank << " index " << i;
      }
    }
  }

  std::unique_ptr<OpKernel> GetCollectiveReduceOpKernel(
      const CollectiveParams& params, Tensor* input, DeviceBase* device) {
    mutex_lock l(mu_);
    NodeDef node_def;
    NodeDefBuilder builder(strings::StrCat("collective_reduce_", op_counter_++),
                           "CollectiveReduce");
    TF_CHECK_OK(
        builder.Attr("T", params.instance.data_type)
            .Attr("merge_op", "Add")
            .Attr("final_op", "Div")
            .Attr("group_size", params.group.group_size)
            .Attr("group_key", params.group.group_key)
            .Attr("instance_key", params.instance.instance_key)
            .Attr("subdiv_offsets", params.instance.impl_details.subdiv_offsets)
            .Input(FakeInput(params.instance.data_type))
            .Finalize(&node_def));
    return GetKernel(node_def, device);
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, const string& device_name, NcclTestBase* parent)
        : parent_(parent),
          device_name_(device_name),
          rank_(rank),
          col_params_(new CollectiveParams()) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_7(mht_7_v, 447, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "DeviceInstance");

      TF_CHECK_OK(parent_->dev_mgr_->LookupDevice(device_name_, &device_))
          << "Could not find device " << device_name_ << " existing devices "
          << parent_->dev_mgr_->DebugString();
      merge_op_ = GetAdd(device_);
      final_op_ = GetDiv(device_);
      col_params_->name = parent_->col_params_->name;
      col_params_->default_rank = rank;
      col_params_->group = parent_->col_params_->group;
      col_params_->instance = parent->col_params_->instance;
    }

    ~DeviceInstance() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_8(mht_8_v, 462, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "~DeviceInstance");
 col_params_->Unref(); }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const std::function<void(Tensor*)>& init_f) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_9(mht_9_v, 468, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitTensor");

      input_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      Tensor cpu_tensor(dtype, shape);
      init_f(&cpu_tensor);
      if (VLOG_IS_ON(3)) {
        VLOG(3) << "input tensor "
                << cpu_tensor.DebugString(shape.num_elements());
      } else {
        VLOG(2) << "input tensor " << cpu_tensor.DebugString();
      }
      auto* dev_info = device_->tensorflow_accelerator_device_info();
      TF_CHECK_OK(dev_info->default_context->CopyCPUTensorToDeviceSync(
          &cpu_tensor, device_, &input_));
    }

    void PrepareDeviceContext(OpKernelContext::Params* params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_10(mht_10_v, 487, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "PrepareDeviceContext");

      params->step_id = kStepId;
      params->device = device_;
      DeviceContext* dev_ctx = nullptr;
      auto* dev_info = device_->tensorflow_accelerator_device_info();
      if (dev_info) {
        dev_ctx = dev_info->default_context;
        dev_ctx->Ref();
      } else {
        dev_ctx = new DeviceContext;
      }
      params->op_device_context = dev_ctx;
    }

    void RunReduce() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_11(mht_11_v, 504, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunReduce");

      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      PrepareDeviceContext(&op_params);

      // Prepare inputs and outputs to OpKernel.
      gtl::InlinedVector<TensorValue, 4> inputs;
      inputs.push_back(TensorValue(&input_));
      op_params.inputs = &inputs;
      gtl::InlinedVector<AllocatorAttributes, 4> input_aa(
          {AllocatorAttributes()});
      op_params.input_alloc_attrs = &input_aa;
      int forward_from = 0;
      op_params.forward_from_array = &forward_from;
      AllocatorAttributes generic_alloc_attr;
      op_params.output_attr_array = &generic_alloc_attr;
      std::unique_ptr<OpKernel> op =
          parent_->GetCollectiveReduceOpKernel(*col_params_, &input_, device_);
      op_params.op_kernel = op.get();
      OpKernelContext ctx(&op_params, 1);
      // We never actually execute the kernel, so we need to do the output
      // allocation it would do, ourselves.
      Tensor* output_tensor_ptr = nullptr;
      TF_CHECK_OK(ctx.forward_input_or_allocate_output({0}, 0, input_.shape(),
                                                       &output_tensor_ptr));
      CHECK_EQ(output_tensor_ptr, ctx.mutable_output(0));

      // Run the all-reduce.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      auto* reducer = new NcclReducer();
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, parent_->nccl_communicator_.get(),
          parent_->dev_mgr_.get(),
          /*OpKernelContext=*/&ctx, &op_params, col_params_, exec_key, kStepId,
          /*input=*/&input_, /*output=*/&input_);
      TF_CHECK_OK(reducer->InitializeCollectiveContext(col_ctx));
      Notification note;
      reducer->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();
      if (status_.ok()) {
        CHECK(output_.CopyFrom(*ctx.mutable_output(0), input_.shape()));
      }

      reducer->Unref();
      op_params.op_device_context->Unref();
    }

    void RunBroadcast() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_12(mht_12_v, 558, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunBroadcast");

      VLOG(2) << "RunBroadcast name " << parent_->collective_name_ << " rank "
              << col_params_->default_rank;
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      PrepareDeviceContext(&op_params);
      OpKernelContext ctx(&op_params, 1);

      // Run broadcast.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      auto* broadcaster = new NcclBroadcaster();
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, parent_->nccl_communicator_.get(),
          parent_->dev_mgr_.get(),
          /*OpKernelContext=*/&ctx, &op_params, col_params_, exec_key, kStepId,
          /*input=*/col_params_->is_source ? &input_ : nullptr,
          /*output=*/&input_);
      TF_CHECK_OK(broadcaster->InitializeCollectiveContext(col_ctx));
      Notification note;
      broadcaster->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();
      if (status_.ok()) {
        CHECK(output_.CopyFrom(input_, input_.shape()));
      }

      broadcaster->Unref();
      op_params.op_device_context->Unref();
    }

    void RunGather() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_13(mht_13_v, 594, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunGather");

      VLOG(2) << "RunGather name " << parent_->collective_name_ << " rank "
              << col_params_->default_rank;
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      PrepareDeviceContext(&op_params);
      OpKernelContext ctx(&op_params, 1);

      // Allocate output.  We can't reuse the input because output has a
      // different shape.
      auto output_shape = input_.shape();
      output_shape.set_dim(
          0, output_shape.dim_size(0) * col_params_->group.group_size);
      output_ = Tensor(device_->GetAllocator(AllocatorAttributes()), DT_FLOAT,
                       output_shape);

      // Run gather.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      auto* gatherer = new NcclGatherer();
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, parent_->nccl_communicator_.get(),
          parent_->dev_mgr_.get(),
          /*OpKernelContext=*/&ctx, &op_params, col_params_, exec_key, kStepId,
          /*input=*/&input_,
          /*output=*/&output_);
      TF_CHECK_OK(gatherer->InitializeCollectiveContext(col_ctx));
      Notification note;
      gatherer->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();

      gatherer->Unref();
      op_params.op_device_context->Unref();
    }

    NcclTestBase* parent_;
    string device_name_;
    int rank_;
    Tensor input_;
    Tensor output_;
    Device* device_;
    CollectiveParams* col_params_;
    std::unique_ptr<OpKernel> merge_op_;
    std::unique_ptr<OpKernel> final_op_;
    Status status_;
  };

  CollectiveType collective_type_;
  const string collective_name_;
  std::vector<std::unique_ptr<tensorflow::Device>> gpus_;
  TestCollectiveExecutorMgr col_exec_mgr_;
  ConfigProto config_proto_;
  std::unique_ptr<NcclCommunicatorInterface> nccl_communicator_;
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  CollectiveExecutor* col_exec_;
  std::unique_ptr<DeviceMgr> dev_mgr_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
  CollectiveParams* col_params_;
  mutex mu_;
  int32 op_counter_ TF_GUARDED_BY(mu_) = 0;
};

class NcclReducerTest : public NcclTestBase {
 protected:
  NcclReducerTest()
      : NcclTestBase(/*collective_type=*/REDUCTION_COLLECTIVE,
                     /*collective_name=*/"NcclReduce") {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_14(mht_14_v, 666, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "NcclReducerTest");
}
  ~NcclReducerTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_15(mht_15_v, 672, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitInput");

    for (size_t i = 0; i < input->NumElements(); ++i) {
      float value = pow(10, rank) * i;
      input->flat<float>()(i) = value;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int num_ranks) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_16(mht_16_v, 683, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitExpected");

    expected->resize(tensor_length);
    for (int i = 0; i < tensor_length; ++i) {
      float expected_sum = 0.0;
      for (int rank = 0; rank < num_ranks; ++rank) {
        float value = pow(10, rank) * i;
        expected_sum += value;
      }
      (*expected)[i] = expected_sum / num_ranks;
    }
  }

  void InitDevice(DeviceInstance* di) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_17(mht_17_v, 698, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitDevice");

    di->col_params_->merge_op = di->merge_op_.get();
    di->col_params_->final_op = di->final_op_.get();
  }

  void RunCollectiveOnDevice(DeviceInstance* di) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_18(mht_18_v, 706, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunCollectiveOnDevice");
 di->RunReduce(); }
};

class NcclBroadcasterTest : public NcclTestBase {
 protected:
  NcclBroadcasterTest()
      : NcclTestBase(/*collective_type=*/BROADCAST_COLLECTIVE,
                     /*collective_name=*/"NcclBroadcast") {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_19(mht_19_v, 716, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "NcclBroadcasterTest");
}
  ~NcclBroadcasterTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_20(mht_20_v, 722, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitInput");

    bool source = rank == source_rank_;
    for (size_t i = 0; i < input->NumElements(); ++i) {
      input->flat<float>()(i) = source ? static_cast<float>(i) : -1.0;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int num_ranks) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_21(mht_21_v, 733, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitExpected");

    expected->resize(tensor_length);
    for (int i = 0; i < tensor_length; ++i) {
      (*expected)[i] = i;
    }
  }

  void InitDevice(DeviceInstance* di) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_22(mht_22_v, 743, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitDevice");

    di->col_params_->source_rank = source_rank_;
    di->col_params_->is_source = di->col_params_->default_rank == source_rank_;
  }

  void RunCollectiveOnDevice(DeviceInstance* di) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_23(mht_23_v, 751, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunCollectiveOnDevice");

    di->RunBroadcast();
  }

  int source_rank_ = 0;
};

class NcclGathererTest : public NcclTestBase {
 protected:
  NcclGathererTest()
      : NcclTestBase(/*collective_type=*/GATHER_COLLECTIVE,
                     /*collective_name=*/"NcclGather") {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_24(mht_24_v, 765, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "NcclGathererTest");
}
  ~NcclGathererTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_25(mht_25_v, 771, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitInput");

    for (size_t i = 0; i < input->NumElements(); ++i) {
      float value = pow(10, rank) * i;
      input->flat<float>()(i) = value;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int num_ranks) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_26(mht_26_v, 782, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitExpected");

    expected->resize(tensor_length * num_ranks, -1);
    for (int rank = 0, i = 0; rank < num_ranks; ++rank) {
      for (int j = 0; j < tensor_length; ++j, ++i) {
        (*expected)[i] = pow(10, rank) * j;
      }
    }
  }

  void InitDevice(DeviceInstance* di) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_27(mht_27_v, 794, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "InitDevice");
}

  void RunCollectiveOnDevice(DeviceInstance* di) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScollective_nccl_testDTcc mht_28(mht_28_v, 799, "", "./tensorflow/core/kernels/collective_nccl_test.cc", "RunCollectiveOnDevice");
 di->RunGather(); }

  int source_rank_ = 0;
};

TEST_F(NcclReducerTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test8Dev1045991Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576, /*instance_key=*/23);
}

TEST_F(NcclBroadcasterTest, Test2Dev16LenSrc0) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclBroadcasterTest, Test4Dev16LenSrc1) {
  source_rank_ = 1;
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclBroadcasterTest, Test8Dev16LenSrc7) {
  source_rank_ = 7;
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclBroadcasterTest, Test8Dev128LenSrc0) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128, /*instance_key=*/24);
}
TEST_F(NcclBroadcasterTest, Test8Dev1045991LenSrc0) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576, /*instance_key=*/23);
}

TEST_F(NcclGathererTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclGathererTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclGathererTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclGathererTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128, /*instance_key=*/24);
}
TEST_F(NcclGathererTest, Test8Dev1045991Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576, /*instance_key=*/23);
}

}  // namespace tensorflow

#endif
