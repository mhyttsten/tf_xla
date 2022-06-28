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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc() {
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

#include "tensorflow/core/common_runtime/eager/context.h"

#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;

typedef FunctionDefHelper FDH;

// Return a fake device.
static Device* CreateDevice(const string& type, int n) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "CreateDevice");

  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_2(mht_2_v, 214, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_3(mht_3_v, 218, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "GetAllocator");
 return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name("/job:localhost/replica:0/task:0/device:" + type + ":" +
                std::to_string(n));
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

class EagerContextTest : public ::testing::Test {
 public:
  EagerContext* context() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_4(mht_4_v, 232, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "context");
 return context_.get(); }

  void InitContext(const SessionOptions& opts,
                   ContextDevicePlacementPolicy policy, bool async = false) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_5(mht_5_v, 238, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "InitContext");

    ASSERT_EQ(context_, nullptr);
    InitDeviceManager();
    context_ = core::RefCountPtr<EagerContext>(
        new EagerContext(opts, policy, async, device_manager_.get(),
                         /*device_mgr_owned=*/false, /*rendezvous=*/nullptr,
                         /*cluster_flr=*/nullptr));
  }

 protected:
  void InitDeviceManager() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_6(mht_6_v, 251, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "InitDeviceManager");

    ASSERT_EQ(device_manager_, nullptr);
    device_manager_ = absl::make_unique<DynamicDeviceMgr>();
    std::vector<std::unique_ptr<Device>> added_devices;
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_CPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 0));
    added_devices.emplace_back(CreateDevice(DEVICE_GPU, 1));
    added_devices.emplace_back(CreateDevice(DEVICE_TPU, 0));

    TF_CHECK_OK(device_manager_->AddDevices(std::move(added_devices)));
  }

  std::unique_ptr<DynamicDeviceMgr> device_manager_;
  core::RefCountPtr<EagerContext> context_;
};

TEST_F(EagerContextTest, CompositeDevice) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  std::vector<string> underlying_devices = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_0));
  EXPECT_EQ(composite_device_0->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:0");
  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:0", &device));
  EXPECT_EQ(device, composite_device_0);
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
  underlying_devices.push_back("/job:worker/replica:0/task:0/device:CPU:2");
  CompositeDevice* composite_device_2 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(underlying_devices,
                                                      /*device_name=*/"",
                                                      &composite_device_2));
  EXPECT_EQ(composite_device_2->name(),
            "/job:localhost/replica:0/task:0/device:COMPOSITE:1");
  TF_EXPECT_OK(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:1", &device));
  EXPECT_EQ(device, composite_device_2);

  EXPECT_TRUE(errors::IsNotFound(context()->FindCompositeDeviceFromName(
      "/job:localhost/replica:0/task:0/device:COMPOSITE:2", &device)));
}

TEST_F(EagerContextTest, CompositeDeviceWithGivenName) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const std::vector<string> underlying_devices_0 = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:0/device:CPU:1"};
  const string composite_device_name =
      "/job:worker1/replica:0/task:0/device:COMPOSITE:5";
  // Create a CompositeDevice with the given name.
  CompositeDevice* composite_device_0 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_0, composite_device_name, &composite_device_0));
  EXPECT_EQ(composite_device_0->name(), composite_device_name);

  CompositeDevice* device = nullptr;
  TF_EXPECT_OK(
      context()->FindCompositeDeviceFromName(composite_device_name, &device));
  EXPECT_EQ(device, composite_device_0);

  std::vector<string> underlying_devices_1 = {
      "/job:worker/replica:0/task:0/device:CPU:1",
      "/job:worker/replica:0/task:0/device:CPU:2"};
  // Find a CompositeDevice with the given name.
  CompositeDevice* composite_device_1 = nullptr;
  TF_ASSERT_OK(context()->FindOrCreateCompositeDevice(
      underlying_devices_1, composite_device_0->name(), &composite_device_1));
  EXPECT_EQ(composite_device_1, composite_device_0);
}

TEST_F(EagerContextTest, AddFunctionDef) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatSame) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two_copy));
}

TEST_F(EagerContextTest, AddFunctionDefRepeatDifferent) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  TF_EXPECT_OK(context()->AddFunctionDef(x_times_two));
  const Tensor kThree = test::AsScalar<int64_t>(3);
  // Same function name but body is different. This should error out.
  const FunctionDef x_times_two_copy = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kThree}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
  Status s = context()->AddFunctionDef(x_times_two_copy);
  EXPECT_FALSE(s.ok());
}

TEST_F(EagerContextTest, FunctionErrorRecovery) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, /*async=*/true);
  context()->SetReuseRendezvousForFunctions(true);
  const FunctionDef assert_and_identity = FDH::Define(
      // Name
      "AssertAndIdentity",
      // Args
      {"x: float", "condition: bool"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"assert"},
           "Assert",
           {"condition", "x"},
           {{"T", std::vector<DataType>{DT_FLOAT}}}},
          {{"y"},
           "Identity",
           {"x"},
           {{"T", DT_FLOAT}},
           /*dep=*/{"assert"}},
      });
  Status s = context()->AddFunctionDef(assert_and_identity);
  auto fail_op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(fail_op->Reset("AssertAndIdentity",
                              "/job:localhost/replica:0/task:0/device:CPU:0"));
  Tensor float_tensor = test::AsScalar<float>(3.0);
  auto input_float = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          float_tensor, context()->HostCPUName().c_str()));
  Tensor bool_tensor_false = test::AsScalar<bool>(false);
  auto input_bool_false = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          bool_tensor_false, context()->HostCPUName().c_str()));
  TF_ASSERT_OK(fail_op->AddInput(input_float.get()));
  TF_ASSERT_OK(fail_op->AddInput(input_bool_false.get()));
  std::vector<AbstractTensorHandle*> retvals(1);
  int num_retvals = retvals.size();
  StatusGroup op_and_sync_status;
  op_and_sync_status.Update(
      fail_op->Execute(absl::MakeSpan(retvals), &num_retvals));
  op_and_sync_status.Update(context()->SyncExecutors());
  ASSERT_THAT(op_and_sync_status.as_summary_status().error_message(),
              HasSubstr("assertion failed"));
  if (retvals[0] != nullptr) {
    retvals[0]->Unref();
    retvals[0] = nullptr;
  }

  Tensor bool_tensor_true = test::AsScalar<bool>(true);
  auto input_bool_true = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          bool_tensor_true, context()->HostCPUName().c_str()));
  auto success_op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(success_op->Reset(
      "AssertAndIdentity", "/job:localhost/replica:0/task:0/device:CPU:0"));
  TF_ASSERT_OK(success_op->AddInput(input_float.get()));
  TF_ASSERT_OK(success_op->AddInput(input_bool_true.get()));
  // A second run of the function should work, despite the previous failure.
  TF_ASSERT_OK(success_op->Execute(absl::MakeSpan(retvals), &num_retvals));
  TF_ASSERT_OK(context()->SyncExecutors());
  retvals[0]->Unref();
  retvals[0] = nullptr;
}

TEST_F(EagerContextTest, XlaCompileDeviceType) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT, /*async=*/true);
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const FunctionDef x_times_two = FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: int64"},
      // Return values
      {"y: int64"}, {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"y"}, "Mul", {"x", "two"}, {{"T", DT_INT64}}},
      });

  Status s = context()->AddFunctionDef(x_times_two);
  context()->SetJitCompileRewrite(true);
  auto op = ImmediateOpPtr(context()->CreateOperation());
  TF_ASSERT_OK(
      op->Reset("XTimesTwo", "/job:localhost/replica:0/task:0/device:TPU:0"));
  Tensor int_tensor = test::AsScalar<int64_t>(3);
  auto input_int = core::RefCountPtr<ImmediateExecutionTensorHandle>(
      context()->CreateLocalHandleFromTFTensor(
          int_tensor, context()->HostCPUName().c_str()));
  TF_ASSERT_OK(op->AddInput(input_int.get()));
  std::vector<AbstractTensorHandle*> retvals(1);
  int num_retvals = retvals.size();
  TF_ASSERT_OK(op->Execute(absl::MakeSpan(retvals), &num_retvals));
  retvals[0]->Unref();
  retvals[0] = nullptr;
}

TEST_F(EagerContextTest, LocalRendezvousCreation) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  std::function<Rendezvous*(const int64_t)> rendezvous_creator =
      context()->RendezvousCreator();

  // Create a new rendezvous instance.
  // Initially its ref-count is 2:
  // one added upopn rendezvous creation, the other one added by EagerContext.
  Rendezvous* rendezvous_1 = rendezvous_creator(1);
  EXPECT_EQ(rendezvous_1->RefCount(), 2);

  // Create another rendezvous instance with the same step-id.
  // This would add one more ref-count to the existing rendezvous insteance
  // insted of creating a new instance.
  Rendezvous* rendezvous_2 = rendezvous_creator(1);
  EXPECT_EQ(rendezvous_2->RefCount(), 3);

  // Caller releases rendezvous-1.
  rendezvous_1->Unref();
  EXPECT_EQ(rendezvous_1->RefCount(), 2);

  // Caller releases rendezvous-2.
  rendezvous_2->Unref();
  EXPECT_EQ(rendezvous_2->RefCount(), 1);
}

void TestGlobalRendezvous(EagerContext* context, bool reuse_global_rendezvous) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPScontext_testDTcc mht_7(mht_7_v, 558, "", "./tensorflow/core/common_runtime/eager/context_test.cc", "TestGlobalRendezvous");

  context->SetReuseRendezvousForFunctions(reuse_global_rendezvous);
  EXPECT_EQ(context->GetReuseRendezvousForFunctions(), reuse_global_rendezvous);

  auto rendezvous_creator = context->RendezvousCreator();
  Rendezvous* rendezvous_1 = rendezvous_creator(-1);
  EXPECT_EQ(rendezvous_1->RefCount(), 2);
  Rendezvous* rendezvous_2 = rendezvous_creator(-1);
  EXPECT_EQ(rendezvous_2->RefCount(), 3);

  // Global rendezvous's ref-count should be back to 1 after resetting.
  context->ResetGlobalRendezvousForFunction();

  Rendezvous* rendezvous_3 = rendezvous_creator(-1);
  EXPECT_EQ(rendezvous_3->RefCount(), 2);

  // Callers release rendezvous.
  rendezvous_1->Unref();
  rendezvous_2->Unref();
  rendezvous_3->Unref();
}

TEST_F(EagerContextTest, GlobalRendezvousCreation) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);

  TestGlobalRendezvous(context(), false);
}

TEST_F(EagerContextTest, ReuseGlobalRendezvous) {
  InitContext(SessionOptions(), DEVICE_PLACEMENT_EXPLICIT);
  EXPECT_FALSE(context()->GetReuseRendezvousForFunctions());

  TestGlobalRendezvous(context(), true);
}

}  // namespace
}  // namespace tensorflow
