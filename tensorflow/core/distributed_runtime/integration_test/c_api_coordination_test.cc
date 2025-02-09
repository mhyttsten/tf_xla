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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc() {
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

#include <string>

#include "absl/time/time.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

constexpr char kCoordinationServiceType[] = "standalone";

void ConfigCoordinationService(
    tensorflow::ServerDef* server_def,
    bool agent_destruction_without_shutdown = false) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "ConfigCoordinationService");

  auto coord_config = server_def->mutable_default_session_config()
                          ->mutable_experimental()
                          ->mutable_coordination_config();
  coord_config->set_service_type(kCoordinationServiceType);
  coord_config->set_service_leader("/job:worker/replica:0/task:0");
  coord_config->set_agent_destruction_without_shutdown(
      agent_destruction_without_shutdown);
  coord_config->set_heartbeat_timeout_in_ms(
      absl::ToInt64Milliseconds(absl::Seconds(5)));
  coord_config->set_shutdown_barrier_timeout_in_ms(
      absl::ToInt64Milliseconds(absl::Seconds(5)));
}

string SetConfigKeyValueFn() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_1(mht_1_v, 232, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "SetConfigKeyValueFn");

  FunctionDef fdef;
  tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'SetConfigKeyValueFn'"
      "      input_arg {"
      "        name: 'config_key'"
      "        type: DT_STRING"
      "      }"
      "      input_arg {"
      "        name: 'config_value'"
      "        type: DT_STRING"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'set0'"
      "      op: 'TestSetConfigKeyValue'"
      "      input: 'config_key'"
      "      input: 'config_value'"
      "    }"
      "    ret {"
      "    }",
      &fdef);
  return fdef.SerializeAsString();
}

string GetConfigKeyValueFn() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "GetConfigKeyValueFn");

  FunctionDef fdef;
  tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'GetConfigKeyValueFn'"
      "      input_arg {"
      "        name: 'config_key'"
      "        type: DT_STRING"
      "      }"
      "      output_arg {"
      "        name: 'config_value'"
      "        type: DT_STRING"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'get0'"
      "      op: 'TestGetConfigKeyValue'"
      "      input: 'config_key'"
      "    }"
      "    ret {"
      "      key: 'config_value'"
      "      value: 'get0:value:0'"
      "    }",
      &fdef);
  return fdef.SerializeAsString();
}

TEST(CAPI, MultiClientCoordinationService) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  // Agent needs to be destroyed without shutdown to simulate network failure,
  // which would trigger stale heartbeat detection on the service-side.
  ConfigCoordinationService(&server_def,
                            /*agent_destruction_without_shutdown=*/true);
  auto worker_thread_fn = [&](int worker_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_3(mht_3_v, 299, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "lambda");

    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    // Normal execution: all cluster members are online.
    std::this_thread::sleep_for(std::chrono::seconds(5));
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    // Sleep for 10 seconds and run colletive ops on cluster except worker/1.
    // Since worker/1 thread directly exits here, its heartbeat will expire,
    // leading to UnavailableError on leader and then propagate to all other
    // members in cluster.
    if (worker_id != 1) {
      // Wait for 10 seconds, during this period of time worker/1 exits and
      // its heartbeat will expire.
      std::this_thread::sleep_for(std::chrono::seconds(10));
      TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
      TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
      TFE_TensorHandle* retvals[1];
      int num_retvals = 1;
      TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

      TFE_DeleteTensorHandle(in);
      TFE_DeleteTensorHandle(retvals[0]);
      TFE_DeleteOp(allreduce);

      // Since we created async executor, op status is eventually reported at
      // the sync barrier.
      TFE_ExecutorWaitForAllPendingNodes(executor, status);
      ASSERT_EQ(TF_UNAVAILABLE, TF_GetCode(status)) << TF_Message(status);
    }
    TFE_DeleteExecutor(executor);
    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

TEST(CAPI, MultiClientSetGetConfigInOp) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  ConfigCoordinationService(&server_def);
  BlockingCounter finish_counter(cluster_size);
  auto worker_thread_fn = [&](int worker_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_4(mht_4_v, 369, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "lambda");

    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TFE_Op* set_op = TFE_NewOp(ctx, "TestSetConfigKeyValue", status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* my_key = TestScalarTensorHandle(
        ctx, tstring(strings::StrCat("worker_", worker_id)));
    TFE_OpAddInput(set_op, my_key, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* my_val = TestScalarTensorHandle(
        ctx, tstring(strings::StrCat("value_", worker_id)));
    TFE_OpAddInput(set_op, my_val, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    int num_retvals = 0;
    TFE_Execute(set_op, nullptr, &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteTensorHandle(my_key);
    TFE_DeleteTensorHandle(my_val);
    TFE_DeleteOp(set_op);

    TFE_Op* get_op = TFE_NewOp(ctx, "TestGetConfigKeyValue", status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_TensorHandle* next_key = TestScalarTensorHandle(
        ctx,
        tstring(strings::StrCat("worker_", (worker_id + 1) % cluster_size)));
    TFE_OpAddInput(get_op, next_key, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TFE_TensorHandle* retvals[1];
    num_retvals = 1;
    TFE_Execute(get_op, retvals, &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    const tstring& next_val = *static_cast<tstring*>(TF_TensorData(t));
    const tstring& expected_val =
        tstring(strings::StrCat("value_", (worker_id + 1) % cluster_size));
    EXPECT_EQ(next_val, expected_val) << strings::StrCat(
        "Expecting value ", expected_val, ", but got ", next_val);

    TFE_DeleteTensorHandle(next_key);
    TFE_DeleteTensorHandle(retvals[0]);
    TF_DeleteTensor(t);
    TFE_DeleteOp(get_op);

    // Since we created async executor, op status is eventually reported at
    // the sync barrier.
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TF_DeleteStatus(status);
    finish_counter.DecrementCount();
    finish_counter.Wait();
    TFE_DeleteExecutor(executor);
    TFE_DeleteContext(ctx);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

TEST(CAPI, MultiClientCoordinationSetGetConfigs) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  ConfigCoordinationService(&server_def);
  tensorflow::BlockingCounter counter1(cluster_size);
  tensorflow::BlockingCounter counter2(cluster_size);
  tensorflow::BlockingCounter counter3(cluster_size);

  auto worker_thread_fn = [&](int worker_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_5(mht_5_v, 462, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "lambda");

    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    // For each worker i, set (keyi, valuei)
    const std::string& key = tensorflow::strings::StrCat("key", worker_id);
    TFE_InsertConfigKeyValue(
        ctx, key.c_str(),
        tensorflow::strings::StrCat("value", worker_id).c_str(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter1.DecrementCount();
    counter1.Wait();

    const int next_id = (worker_id + 1) % cluster_size;
    // Setting next_key errors out because it has been set by another worker
    const std::string& next_key = tensorflow::strings::StrCat("key", next_id);
    TFE_InsertConfigKeyValue(ctx, next_key.c_str(), "some_value", status);
    EXPECT_EQ(TF_ALREADY_EXISTS, TF_GetCode(status)) << TF_Message(status);
    // Getting next_key returns the value set by another worker
    TF_Buffer* value_buf = TF_NewBuffer();
    TFE_GetConfigKeyValue(ctx, next_key.c_str(), value_buf, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    std::string value_str{static_cast<const char*>(value_buf->data),
                          value_buf->length};
    EXPECT_EQ(value_str, tensorflow::strings::StrCat("value", next_id));
    TF_DeleteBuffer(value_buf);
    counter2.DecrementCount();
    counter2.Wait();

    // Delete key
    TFE_DeleteConfigKeyValue(ctx, key.c_str(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter3.DecrementCount();
    counter3.Wait();

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

TEST(CAPI, MultiClientPropagateError) {
  const int cluster_size = 3;
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
  ConfigCoordinationService(&server_def);
  // Barrier for initializing the cluster.
  tensorflow::BlockingCounter counter1(cluster_size);
  // Barrier for finishing executing operations on all workers.
  tensorflow::BlockingCounter counter2(cluster_size);

  auto worker_thread_fn = [&](int worker_id) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_coordination_testDTcc mht_6(mht_6_v, 536, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_coordination_test.cc", "lambda");

    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/false));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    counter1.DecrementCount();
    counter1.Wait();

    // Set error from worker/1
    if (worker_id == 1) {
      TFE_ReportErrorToCluster(ctx, TF_INVALID_ARGUMENT, "my_error", status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    }

    // Run collective on all workers. The collective will not finish because
    // worker/1 already in error status. Check that all workers get the same
    // error reported from running the collective ops.
    TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
    TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
    EXPECT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status)) << TF_Message(status);

    TFE_DeleteTensorHandle(in);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteOp(allreduce);
    counter2.DecrementCount();
    counter2.Wait();

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  std::thread thread_worker3([&] { worker_thread_fn(2); });
  thread_worker1.join();
  thread_worker2.join();
  thread_worker3.join();
}

class SingleClientCoordinationServiceTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<bool> {};

TEST_P(SingleClientCoordinationServiceTest, TestSetGetConfigInOp) {
  const bool use_worker0_as_client = GetParam();
  tensorflow::ServerDef server_def = GetServerDef("worker", 3);
  const char task0_name[] = "/job:worker/replica:0/task:0/device:CPU:0";
  const char task1_name[] = "/job:worker/replica:0/task:1/device:CPU:0";
  const char task2_name[] = "/job:worker/replica:0/task:2/device:CPU:0";

  ConfigCoordinationService(&server_def);
  ServerFactory* factory;
  ASSERT_TRUE(ServerFactory::GetFactory(server_def, &factory).ok());
  server_def.set_job_name("worker");
  server_def.set_task_index(0);
  std::unique_ptr<tensorflow::ServerInterface> w0;
  if (!use_worker0_as_client) {
    // Start a separate server for worker0 if it's not used as the client
    ASSERT_TRUE(
        factory->NewServer(server_def, ServerFactory::Options(), &w0).ok());
    ASSERT_TRUE(w0->Start().ok());
  }
  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::ServerInterface> w1;
  ASSERT_TRUE(
      factory->NewServer(server_def, ServerFactory::Options(), &w1).ok());
  ASSERT_TRUE(w1->Start().ok());
  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::ServerInterface> w2;
  ASSERT_TRUE(
      factory->NewServer(server_def, ServerFactory::Options(), &w2).ok());
  ASSERT_TRUE(w2->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(/*enable=*/true));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  server_def.set_task_index(0);
  if (!use_worker0_as_client) {
    // Add localhost job for the remote client task
    auto cluster = server_def.mutable_cluster();
    auto client_job = cluster->add_job();
    client_job->set_name("localhost");
    const int client_port = tensorflow::testing::PickUnusedPortOrDie();
    client_job->mutable_tasks()->insert(
        {0, strings::StrCat("localhost:", client_port)});
    server_def.set_job_name("localhost");
  }
  server_def.mutable_default_session_config()
      ->mutable_experimental()
      ->mutable_coordination_config()
      ->set_service_leader(task0_name);
  const std::string serialized = server_def.SerializeAsString();

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_Op* set_op = TFE_NewOp(ctx, "TestSetConfigKeyValue", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* set_key = TestScalarTensorHandle(ctx, tstring("test_key"));
  TFE_OpAddInput(set_op, set_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* set_val = TestScalarTensorHandle(ctx, tstring("test_val"));
  TFE_OpAddInput(set_op, set_val, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  // Run set op from task1
  TFE_OpSetDevice(set_op, task1_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  int num_retvals = 0;
  TFE_Execute(set_op, nullptr, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(set_key);
  TFE_DeleteTensorHandle(set_val);
  TFE_DeleteOp(set_op);

  TFE_Op* get_op = TFE_NewOp(ctx, "TestGetConfigKeyValue", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* get_key = TestScalarTensorHandle(ctx, tstring("test_key"));
  TFE_OpAddInput(get_op, get_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_TensorHandle* retvals[1];
  num_retvals = 1;
  // Run get op from task2
  TFE_OpSetDevice(get_op, task2_name, status);
  TFE_Execute(get_op, retvals, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const tstring& get_val = *static_cast<tstring*>(TF_TensorData(t));
  EXPECT_EQ(get_val, "test_val")
      << strings::StrCat("Expecting value test_val but got ", get_val);
  TFE_DeleteTensorHandle(get_key);
  TFE_DeleteTensorHandle(retvals[0]);
  TF_DeleteTensor(t);
  TFE_DeleteOp(get_op);

  const string& set_fdef = SetConfigKeyValueFn();
  TFE_ContextAddFunctionDef(ctx, set_fdef.data(), set_fdef.size(), status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Op* set_fn = TFE_NewOp(ctx, "SetConfigKeyValueFn", status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  set_key = TestScalarTensorHandle(ctx, tstring("test_fn_key"));
  TFE_OpAddInput(set_fn, set_key, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  set_val = TestScalarTensorHandle(ctx, tstring("test_fn_val"));
  TFE_OpAddInput(set_fn, set_val, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  // Run set fn on task2
  TFE_OpSetDevice(set_fn, task2_name, status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  num_retvals = 0;
  TFE_Execute(set_fn, nullptr, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(set_key);
  TFE_DeleteTensorHandle(set_val);
  TFE_DeleteOp(set_fn);

  const string& get_fdef = GetConfigKeyValueFn();
  TFE_ContextAddFunctionDef(ctx, get_fdef.data(), get_fdef.size(), status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Op* get_fn = TFE_NewOp(ctx, "GetConfigKeyValueFn", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  get_key = TestScalarTensorHandle(ctx, tstring("test_fn_key"));
  TFE_OpAddInput(get_fn, get_key, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);
  TFE_TensorHandle* fn_retvals[1];
  num_retvals = 1;
  // Run get fn on task1
  TFE_OpSetDevice(get_fn, task2_name, status);
  TFE_Execute(get_fn, fn_retvals, &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  t = TFE_TensorHandleResolve(fn_retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const tstring& get_fn_val = *static_cast<tstring*>(TF_TensorData(t));
  EXPECT_EQ(get_fn_val, "test_fn_val")
      << strings::StrCat("Expecting value test_fn_val but got ", get_fn_val);
  TFE_DeleteTensorHandle(get_key);
  TFE_DeleteTensorHandle(fn_retvals[0]);
  TF_DeleteTensor(t);
  TFE_DeleteOp(get_fn);

  // Since we created async executor, op status is eventually reported at
  // the sync barrier.
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);

  // Grpc servers do not support clean down.
  w0.release();
  w1.release();
  w2.release();
}

INSTANTIATE_TEST_SUITE_P(CAPI, SingleClientCoordinationServiceTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> arg) {
                           return arg.param ? "use_worker0_as_client"
                                            : "use_remote_client";
                         });

}  // namespace
}  // namespace tensorflow
