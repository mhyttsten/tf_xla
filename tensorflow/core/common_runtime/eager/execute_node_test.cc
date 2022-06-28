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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_node_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_node_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_node_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/execute_node.h"

#include <memory>

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestKernelAndDeviceFunc final : public KernelAndDeviceFunc {
 public:
  TestKernelAndDeviceFunc(std::vector<Device*> input_devices,
                          Device* host_cpu_device)
      : KernelAndDeviceFunc(
            /*flr=*/nullptr, /*pflr=*/nullptr, /*input_devices=*/{},
            /*composite_devices=*/{}, /*input_resource_dtypes_and_shapes=*/{},
            /*runner=*/nullptr, /*collective_executor=*/nullptr,
            host_cpu_device, /*name=*/"", /*outputs_on_op_device=*/false,
            /*allow_small_function_optimizations=*/false,
            /*allow_control_flow_sync_execution=*/false,
            /*shape_inference_on_tfe_dialect_import=*/true,
            /*int_args_and_retvals_on_device=*/false,
            /*xla_compile_device_type=*/absl::nullopt,
            /*rendezvous_creator=*/nullptr, /*get_op_id=*/nullptr),
        test_input_devices_(std::move(input_devices)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_node_testDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/common_runtime/eager/execute_node_test.cc", "TestKernelAndDeviceFunc");
}

  Device* InputDevice(int i) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSexecute_node_testDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/common_runtime/eager/execute_node_test.cc", "InputDevice");
 return test_input_devices_[i]; }

 private:
  std::vector<Device*> test_input_devices_;
};

TEST(ExecuteNodeTest, ExecuteNodeArgs) {
  StaticDeviceMgr device_mgr(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
  Device* device0 = device_mgr.ListDevices().at(0);
  auto remote_device_mgr = absl::make_unique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> remote_devices;
  remote_devices.emplace_back(
      DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:1"));
  TF_ASSERT_OK(remote_device_mgr->AddDevices(std::move(remote_devices)));
  Device* device1 = remote_device_mgr->ListDevices().at(0);

  Status s;
  std::unique_ptr<CompositeDevice> composite_device =
      CompositeDevice::MakeDevice({device0->name(), device1->name()},
                                  /*unique_device_id=*/0,
                                  device_mgr.HostCPU()->parsed_name(), &s);
  TF_ASSERT_OK(s);

  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr);

  // Set a RemoteMgr to the EagerContext.
  auto remote_mgr = absl::make_unique<eager::RemoteMgr>(
      /*is_master=*/true, ctx);
  TF_ASSERT_OK(ctx->InitializeRemoteMaster(
      /*server=*/nullptr, /*worker_env=*/nullptr,
      /*worker_session=*/nullptr, /*remote_eager_workers=*/nullptr,
      std::move(remote_device_mgr), /*remote_contexts=*/{},
      EagerContext::NewContextId(),
      /*r=*/nullptr, &device_mgr, /*keep_alive_secs*/ 600,
      /*cluster_flr=*/nullptr, std::move(remote_mgr)));

  DataType dtype = DT_FLOAT;
  Tensor t0(dtype, TensorShape({}));
  // Create two local TensorHandles
  t0.scalar<float>()() = {1.0f};
  TensorHandle* h0 =
      TensorHandle::CreateLocalHandle(std::move(t0), device0, device0, ctx);
  Tensor t1(dtype, TensorShape({}));
  t1.scalar<float>()() = {2.0f};
  TensorHandle* h1 =
      TensorHandle::CreateLocalHandle(std::move(t1), device0, device0, ctx);
  // Create two remote TensorHandles
  TensorHandle* h2 = TensorHandle::CreateLazyRemoteHandle(
      /*op_id=*/1, /*output_num=*/0, dtype, device1, /*is_ready=*/true, ctx);
  TensorHandle* h3 = TensorHandle::CreateLazyRemoteHandle(
      /*op_id=*/2, /*output_num=*/1, dtype, device1, /*is_ready=*/true, ctx);
  // Create a packed TensorHandle
  TensorHandle* packed_h = nullptr;
  TF_ASSERT_OK(TensorHandle::CreatePackedHandle({h1, h2}, ctx, &packed_h));

  // LOCAL, PACKED, REMOTE
  absl::InlinedVector<TensorHandle*, 4> inputs = {h0, packed_h, h3};

  std::vector<Device*> input_devices;
  for (auto* h : inputs) {
    input_devices.push_back(h->DeviceOrHostCPU(*ctx));
  }
  const core::RefCountPtr<KernelAndDevice> kernel(
      new TestKernelAndDeviceFunc(std::move(input_devices), device0));

  ExecuteNodeArgs args(inputs.size());
  TF_EXPECT_OK(args.Init(ctx, inputs, kernel));
  EXPECT_TRUE(args.HasRemoteOrPackedInputs());
  Tensor local0;
  TF_EXPECT_OK(args.GetLocalArg(FunctionArgIndex(0), &local0));
  EXPECT_EQ(local0.flat<float>().size(), 1);
  EXPECT_EQ(local0.flat<float>()(0), 1.0);
  Tensor local1;
  TF_EXPECT_OK(args.GetLocalArg(FunctionArgIndex(1, 0), &local1));
  EXPECT_EQ(local1.flat<float>().size(), 1);
  EXPECT_EQ(local1.flat<float>()(0), 2.0);
  eager::RemoteTensorHandle remote0;
  TF_EXPECT_OK(args.GetRemoteArg(FunctionArgIndex(1, 1), &remote0));
  EXPECT_EQ(remote0.op_id(), 1);
  EXPECT_EQ(remote0.output_num(), 0);
  eager::RemoteTensorHandle remote1;
  TF_EXPECT_OK(args.GetRemoteArg(FunctionArgIndex(2), &remote1));
  EXPECT_EQ(remote1.op_id(), 2);
  EXPECT_EQ(remote1.output_num(), 1);

  h0->Unref();
  h1->Unref();
  h2->Unref();
  h3->Unref();
  packed_h->Unref();
  ctx->Unref();
}

}  // namespace
}  // namespace tensorflow
