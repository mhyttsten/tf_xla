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
class MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_test_util.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace {

class WhileOpTest : public OpsTestBase {
 protected:
  WhileOpTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/while_op_test.cc", "WhileOpTest");
}
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/kernels/while_op_test.cc", "SetUp");

    stream_executor::test_util::PopulateDefaultPlatform(&platform_,
                                                        &platform_fns_);
    stream_executor::test_util::PopulateDefaultDeviceFns(&device_fns_);
    stream_executor::test_util::PopulateDefaultStreamExecutor(&se_);
    stream_executor::test_util::PopulateDefaultTimerFns(&timer_fns_);
  }
  void TearDown() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/kernels/while_op_test.cc", "TearDown");
}

  SP_Platform platform_;
  SP_PlatformFns platform_fns_;
  SP_DeviceFns device_fns_;
  SP_StreamExecutor se_;
  SP_TimerFns timer_fns_;
};

FunctionDef LessThanOrEqualToNWithCast(int64_t N) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_3(mht_3_v, 228, "", "./tensorflow/core/kernels/while_op_test.cc", "LessThanOrEqualToNWithCast");

  typedef FunctionDefHelper FDH;
  const Tensor kN = test::AsScalar<int64_t>(N);
  return FDH::Define(
      // Name
      "LessThanOrEqualToNWithCast",
      // Args
      {"x: T"},
      // Return values
      {"z: bool"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", kN}, {"dtype", DT_INT64}}},
          {{"y"}, "_HostCast", {"N"}, {{"SrcT", DT_INT64}, {"DstT", DT_INT32}}},
          {{"x_cst"}, "_HostCast", {"x"}, {{"SrcT", "$T"}, {"DstT", DT_INT32}}},
          {{"z"}, "LessEqual", {"x_cst", "y"}, {{"T", DT_INT32}}},
      });
}

FunctionDef XTimesTwoWithCast() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/kernels/while_op_test.cc", "XTimesTwoWithCast");

  typedef FunctionDefHelper FDH;
  const Tensor kTwo = test::AsScalar<int64_t>(2);
  return FDH::Define(
      // Name
      "XTimesTwoWithCast",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"two_cst"},
           "_HostCast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_INT32}}},
          {{"x_cst"}, "_HostCast", {"x"}, {{"SrcT", "$T"}, {"DstT", DT_INT32}}},
          {{"y_cast"}, "Mul", {"x_cst", "two_cst"}, {{"T", DT_INT32}}},
          {{"y"},
           "_HostCast",
           {"y_cast"},
           {{"SrcT", DT_INT32}, {"DstT", "$T"}}},
      });
}

TEST_F(WhileOpTest, WhileOpCPUBuildWithPluggableDevice) {
  const std::string platform_name = "MY_TEST";
  const std::string platform_type = "FAKE";
  platform_.name = platform_name.c_str();
  platform_.type = platform_type.c_str();

  // TODO(penporn): Refactor this to test_pluggable_device.cc.
  static bool memcpy_d2h_called = false;
  se_.memcpy_dtoh = [](const SP_Device* device, SP_Stream stream,
                       void* host_dst, const SP_DeviceMemoryBase* device_src,
                       uint64_t size, TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/kernels/while_op_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    memcpy_d2h_called = true;
    std::memcpy(host_dst, device_src->opaque, size);
  };
  se_.memcpy_htod = [](const SP_Device* const device, SP_Stream stream,
                       SP_DeviceMemoryBase* const device_dst,
                       const void* host_src, uint64_t size,
                       TF_Status* const status) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_6(mht_6_v, 304, "", "./tensorflow/core/kernels/while_op_test.cc", "lambda");

    TF_SetStatus(status, TF_OK, "");
    std::memcpy(device_dst->opaque, host_src, size);
  };

  se_.host_memory_allocate = [](const SP_Device* const device, uint64_t size) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_7(mht_7_v, 312, "", "./tensorflow/core/kernels/while_op_test.cc", "lambda");

#if EIGEN_MAX_ALIGN_BYTES == 0
    return malloc(size);
#else
    return tensorflow::port::AlignedMalloc(size, EIGEN_MAX_ALIGN_BYTES);
#endif
  };
  se_.host_memory_deallocate = [](const SP_Device* const device, void* mem) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_8(mht_8_v, 322, "", "./tensorflow/core/kernels/while_op_test.cc", "lambda");

    free(mem);
  };

  se_.allocate = [](const SP_Device* const device, uint64_t size,
                    int64_t memory_space, SP_DeviceMemoryBase* const mem) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_9(mht_9_v, 330, "", "./tensorflow/core/kernels/while_op_test.cc", "lambda");

    mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
#if EIGEN_MAX_ALIGN_BYTES == 0
    mem->opaque = malloc(size);
#else
    mem->opaque = tensorflow::port::AlignedMalloc(size, EIGEN_MAX_ALIGN_BYTES);
#endif
    mem->size = size;
  };
  se_.deallocate = [](const SP_Device* const device,
                      SP_DeviceMemoryBase* const mem) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhile_op_testDTcc mht_10(mht_10_v, 343, "", "./tensorflow/core/kernels/while_op_test.cc", "lambda");

    free(mem->opaque);
    mem->opaque = nullptr;
    mem->size = 0;
  };

  static SE_EventStatus event_status = SE_EVENT_COMPLETE;
  se_.create_event = [](const SP_Device* const device, SP_Event* event,
                        TF_Status* const status) -> void {
    *event = new SP_Event_st(666);
  };
  se_.destroy_event = [](const SP_Device* const device,
                         SP_Event event) -> void { delete event; };
  se_.get_event_status = [](const SP_Device* const device,
                            SP_Event event) -> SE_EventStatus {
    EXPECT_EQ(event->event_id, 666);
    return event_status;
  };

  std::unique_ptr<stream_executor::CPlatform> cplatform(
      new stream_executor::CPlatform(
          std::move(platform_), stream_executor::test_util::DestroyPlatform,
          std::move(platform_fns_),
          stream_executor::test_util::DestroyPlatformFns,
          std::move(device_fns_), std::move(se_), std::move(timer_fns_)));
  SE_CHECK_OK(stream_executor::MultiPlatformManager::RegisterPlatform(
      std::move(cplatform)));

  DeviceFactory::Register(
      platform_type, new PluggableDeviceFactory(platform_type, platform_name),
      /*priority=*/220, /*is_pluggable_device=*/true);
  std::unique_ptr<Device> plug_device(
      DeviceFactory::NewDevice(platform_type, {}, "/job:a/replica:0"));
  OpsTestBase::SetDevice(platform_type.c_str(), std::move(plug_device));
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope root = Scope::NewRootScope().ExitOnError();
  FunctionDef x_times_two = XTimesTwoWithCast();
  FunctionDef less_than_or_eq = LessThanOrEqualToNWithCast(8);

  FunctionDefLibrary f_lib_proto;
  *f_lib_proto.add_function() = x_times_two;
  *f_lib_proto.add_function() = less_than_or_eq;

  TF_ASSERT_OK(root.graph()->AddFunctionLibrary(f_lib_proto));
  auto a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);
  AttrValue cond_func;
  cond_func.mutable_func()->set_name("LessThanOrEqualToNWithCast");
  (*cond_func.mutable_func()->mutable_attr())["T"].set_type(DT_FLOAT);
  AttrValue body_func;
  body_func.mutable_func()->set_name("XTimesTwoWithCast");
  (*body_func.mutable_func()->mutable_attr())["T"].set_type(DT_FLOAT);

  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* node;
  TF_EXPECT_OK(NodeBuilder("while_test", "While", &root.graph()->flib_def())
                   .Input(inputs)
                   .Attr("T", {DT_FLOAT})
                   .Attr("cond", cond_func)
                   .Attr("body", body_func)
                   .Attr("parallel_iterations", 100)
                   .Finalize(root.graph(), &node));
  auto c = ops::Identity(
      root.WithOpName("C").WithControlDependencies(Output(node)), Output(node));

  TF_ASSERT_OK(root.DoShapeInference(node));
  TF_ASSERT_OK(root.ToGraph(graph.get()));
  ClientSession session(root);
  {
    ClientSession::FeedType feeds;
    feeds.emplace(Output(a.node()), Input::Initializer(1.f));
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, {Output(c.node())}, &out_tensors));
    ASSERT_EQ(memcpy_d2h_called, true);
    ASSERT_EQ(out_tensors.size(), 1);
    EXPECT_EQ(out_tensors[0].scalar<float>()(), 16.f);
  }
}

}  // namespace
}  // namespace tensorflow
