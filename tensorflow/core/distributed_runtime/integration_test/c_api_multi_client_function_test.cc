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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc() {
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

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/eager/kernel_and_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace {

std::string SendFunction(const std::string& send_device,
                         const std::string& recv_device,
                         const tensorflow::int64 send_device_incarnation) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("send_device: \"" + send_device + "\"");
   mht_0_v.push_back("recv_device: \"" + recv_device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_function_test.cc", "SendFunction");

  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      absl::StrCat("    signature {"
                   "      name: 'SendFunction'"
                   "      input_arg {"
                   "        name: 'in'"
                   "        type: DT_FLOAT"
                   "      }"
                   "      control_output: 'send_tensor'"
                   "    }"
                   "    node_def {"
                   "      name: 'send'"
                   "      op: '_Send'"
                   "      input: 'in'"
                   "      device: '",
                   send_device, "'",
                   "      attr {"
                   "        key: 'T'"
                   "        value {"
                   "          type: DT_FLOAT"
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'tensor_name'"
                   "        value {"
                   "          s: 'dummy'"
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'send_device'"
                   "        value {"
                   "          s: '",
                   send_device, "'",
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'recv_device'"
                   "        value {"
                   "          s: '",
                   recv_device, "'",
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'send_device_incarnation'"
                   "        value {"
                   "          i: ",
                   absl::StrCat(send_device_incarnation),
                   "        }"
                   "      }"
                   "    }"
                   "    control_ret {"
                   "      key: 'send_tensor'"
                   "      value: 'send'"
                   "    }"),
      &def));
  return def.SerializeAsString();
}

std::string RecvFunction(const std::string& send_device,
                         const std::string& recv_device,
                         const tensorflow::int64 send_device_incarnation) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("send_device: \"" + send_device + "\"");
   mht_1_v.push_back("recv_device: \"" + recv_device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc mht_1(mht_1_v, 279, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_function_test.cc", "RecvFunction");

  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      absl::StrCat("    signature {"
                   "      name: 'RecvFunction'"
                   "      output_arg {"
                   "        name: 'out'"
                   "        type: DT_FLOAT"
                   "      }"
                   "    }"
                   "    node_def {"
                   "      name: 'recv'"
                   "      op: '_Recv'"
                   "      device: '",
                   recv_device, "'",
                   "      attr {"
                   "        key: 'tensor_type'"
                   "        value {"
                   "          type: DT_FLOAT"
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'tensor_name'"
                   "        value {"
                   "          s: 'dummy'"
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'send_device'"
                   "        value {"
                   "          s: '",
                   send_device, "'",
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'recv_device'"
                   "        value {"
                   "          s: '",
                   recv_device, "'",
                   "        }"
                   "      }"
                   "      attr {"
                   "        key: 'send_device_incarnation'"
                   "        value {"
                   "          i: ",
                   absl::StrCat(send_device_incarnation),
                   "        }"
                   "      }"
                   "    }"
                   "    ret {"
                   "      key: 'out'"
                   "      value: 'recv:tensor'"
                   "    }"),
      &def));
  return def.SerializeAsString();
}

TFE_TensorHandle* DummyTensorHandleWithValue(TFE_Context* ctx, float v) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc mht_2(mht_2_v, 339, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_function_test.cc", "DummyTensorHandleWithValue");

  // Initialize matrix values.
  int64_t dims[] = {2, 2};
  float data[4];
  for (int i = 0; i < 4; i++) {
    data[i] = v * (i + 1);
  }

  return TestTensorHandleWithDimsFloat(ctx, data, &dims[0],
                                       sizeof(dims) / sizeof(int64_t));
}

struct MultiClientSendRecvTestParams {
  std::string test_name;
  bool use_tfrt = false;
  uint num_steps = 1;
  uint delay_recv_sec = 0;
  uint delay_send_sec = 0;
};

using MultiClientSendRecvTest =
    testing::TestWithParam<MultiClientSendRecvTestParams>;

TEST_P(MultiClientSendRecvTest, TestMultiClientSendRecv) {
  const MultiClientSendRecvTestParams& params = GetParam();
  // Use a mutex to enforce a serialized operation between the two
  // worker-threads since some of their operations involve updating the global
  // singleton instances (in TFRT scenario), which otherwise would cause a data
  // race.
  tensorflow::mutex mu;

  const int cluster_size = 2;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);

  // Enable coordination service for propagating remote device attributess
  auto* coord_config = server_def.mutable_default_session_config()
                           ->mutable_experimental()
                           ->mutable_coordination_config();
  coord_config->set_service_type("standalone");
  coord_config->set_service_leader("/job:worker/replica:0/task:0");

  // The blocking counter makes sure that worker/0 thread (leader that starts
  // the coordination service) does not exit early while other workers are still
  // interacting with the coordination service.
  tensorflow::BlockingCounter counter(cluster_size);

  auto worker_thread_fn = [&](int worker_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc mht_3(mht_3_v, 389, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_function_test.cc", "lambda");

    tensorflow::ServerDef server_def_copy = server_def;
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* context_opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(context_opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(context_opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    // use-tfrt flag.
    context_opts->use_tfrt = params.use_tfrt;
    tensorflow::SessionOptions session_opts;
    session_opts.config = server_def_copy.default_session_config();
    context_opts->session_options.options = session_opts;

    TFE_Context* ctx;
    {
      tensorflow::mutex_lock l(mu);
      ctx = TFE_NewContext(context_opts, status);
    }
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(context_opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    const std::string& send_device =
        "/job:worker/replica:0/task:0/device:CPU:0";
    const std::string& recv_device =
        "/job:worker/replica:0/task:1/device:CPU:0";

    std::vector<tensorflow::DeviceAttributes> device_attrs;
    tensorflow::unwrap(ctx)->ListDevices(&device_attrs);
    tensorflow::uint64 send_device_incarnation = 0;
    for (const auto& device_attr : device_attrs) {
      if (device_attr.name() == send_device) {
        send_device_incarnation = device_attr.incarnation();
        break;
      }
    }

    if (worker_id == 0) {
      // Sender worker.
      tensorflow::Env::Default()->SleepForMicroseconds(params.delay_send_sec *
                                                       1000);

      const std::string& fdef =
          SendFunction(send_device, recv_device, send_device_incarnation);
      TFE_ContextAddFunctionDef(ctx, fdef.data(), fdef.size(), status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      // Run multiple steps.
      for (int s = 1; s <= params.num_steps; s++) {
        TFE_Op* send_func = TFE_NewOp(ctx, "SendFunction", status);
        EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

        if (params.use_tfrt) {
          // TODO (@chienchunh): Add support for step id configuration in TFRT.
          EXPECT_TRUE(tensorflow::unwrap(send_func)
                          ->Reset("SendFunction", send_device.c_str())
                          .ok());
        } else {
          tensorflow::EagerOperation* op =
              tensorflow::OperationFromInterface(tensorflow::unwrap(send_func));
          EXPECT_TRUE(op->Reset("SendFunction", send_device.c_str(),
                                /*remote=*/false, /*executor=*/nullptr,
                                tensorflow::EagerFunctionParams{/*op_id=*/s,
                                                                /*step_id=*/s})
                          .ok());
        }

        TFE_TensorHandle* in = DummyTensorHandleWithValue(ctx, 1.0f * s);
        TFE_OpAddInput(send_func, in, status);
        EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
        int num_retvals = 0;
        {
          tensorflow::mutex_lock l(mu);
          TFE_Execute(send_func, nullptr, &num_retvals, status);
        }
        EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
        TFE_DeleteOp(send_func);
        TFE_DeleteTensorHandle(in);
      }
    } else {
      // Receiver worker.
      tensorflow::Env::Default()->SleepForMicroseconds(params.delay_recv_sec *
                                                       1000);

      const std::string& fdef =
          RecvFunction(send_device, recv_device, send_device_incarnation);
      TFE_ContextAddFunctionDef(ctx, fdef.data(), fdef.size(), status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      // Run multiple steps.
      for (int s = 1; s <= params.num_steps; s++) {
        TFE_Op* recv_func = TFE_NewOp(ctx, "RecvFunction", status);
        EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

        if (params.use_tfrt) {
          // TODO (@chienchunh): Add support for step id configuration in TFRT.
          EXPECT_TRUE(tensorflow::unwrap(recv_func)
                          ->Reset("RecvFunction", recv_device.c_str())
                          .ok());
        } else {
          tensorflow::EagerOperation* op =
              tensorflow::OperationFromInterface(tensorflow::unwrap(recv_func));
          EXPECT_TRUE(op->Reset("RecvFunction", recv_device.c_str(),
                                /*remote=*/false, /*executor=*/nullptr,
                                tensorflow::EagerFunctionParams{/*op_id=*/s,
                                                                /*step_id=*/s})
                          .ok());
        }

        TFE_TensorHandle* retvals[1];
        int num_retvals = 1;
        {
          tensorflow::mutex_lock l(mu);
          TFE_Execute(recv_func, &retvals[0], &num_retvals, status);
        }
        EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
        TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
        TFE_DeleteOp(recv_func);
        TFE_DeleteTensorHandle(retvals[0]);

        float result[4] = {0};
        EXPECT_EQ(sizeof(result), TF_TensorByteSize(t));
        memcpy(&result[0], TF_TensorData(t), TF_TensorByteSize(t));
        TF_DeleteTensor(t);
        for (int i = 0; i < 4; i++) {
          EXPECT_EQ(result[i], 1.0 * s * (i + 1));
        }
      }
    }

    // To make sure the sender won't delete the data it sent before the receiver
    // retrieves it, we need to do the following steps:
    // 1. Since we created async EagerContext, we need to force each worker to
    //    wait until all pening operations finish before deleting the context.
    // 2. In addition, use the blocking counter to notify the 2 workers when
    //    it is safe to clean up all the data.
    TFE_ContextAsyncWait(ctx, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter.DecrementCount();
    counter.Wait();

    {
      tensorflow::mutex_lock l(mu);
      TFE_DeleteContext(ctx);
    }
    TF_DeleteStatus(status);
  };

  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });

  thread_worker1.join();
  thread_worker2.join();
}

INSTANTIATE_TEST_SUITE_P(
    MultiClientSendRecvTests, MultiClientSendRecvTest,
    testing::ValuesIn<MultiClientSendRecvTestParams>({
        {"MultiClientSingleStepFunction", false, 1, 0, 0},
        {"MultiClientMultiStepFunction", false, 3, 0, 0},
        {"MultiClientMultiStepFunctionWithRecvDelay", false, 5, 2, 0},
        {"MultiClientMultiStepFunctionWithSendDelay", false, 5, 0, 2},
        {"MultiClientSingleStepFunctionTfrt", true, 1, 0, 0},
        {"MultiClientMultiStepFunctionTfrt", true, 3, 0, 0},
        {"MultiClientMultiStepFunctionWithRecvDelayTfrt", true, 5, 2, 0},
        {"MultiClientMultiStepFunctionWithSendDelayTfrt", true, 5, 0, 2},
    }),
    [](const testing::TestParamInfo<MultiClientSendRecvTest::ParamType>& info) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_function_testDTcc mht_4(mht_4_v, 565, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_function_test.cc", "lambda");

      return info.param.test_name;
    });

}  // namespace
