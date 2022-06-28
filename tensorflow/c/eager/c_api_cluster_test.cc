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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc() {
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

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace {

using ::tensorflow::string;

void ReplaceTaskInServerDef(tensorflow::ServerDef* server_def, int task_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_0(mht_0_v, 202, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "ReplaceTaskInServerDef");

  tensorflow::JobDef* job_def = server_def->mutable_cluster()->mutable_job(0);
  int port = tensorflow::testing::PickUnusedPortOrDie();
  job_def->mutable_tasks()->at(task_index) =
      tensorflow::strings::StrCat("localhost:", port);
}

void CheckTFE_TensorHandleHasFloats(TFE_TensorHandle* handle,
                                    const std::vector<float>& expected_values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_1(mht_1_v, 213, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "CheckTFE_TensorHandleHasFloats");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* t = TFE_TensorHandleResolve(handle, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  std::unique_ptr<float[]> actual_values(new float[expected_values.size()]);
  EXPECT_EQ(sizeof(float) * expected_values.size(), TF_TensorByteSize(t));
  memcpy(actual_values.get(), TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);

  for (int i = 0; i < expected_values.size(); i++) {
    EXPECT_EQ(expected_values[i], actual_values[i])
        << "Mismatch in expected values at (zero-based) index " << i;
  }
}

void CheckRemoteMatMulExecutesOK(TFE_Context* ctx,
                                 const char* remote_device_name,
                                 const char* local_device_name) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("remote_device_name: \"" + (remote_device_name == nullptr ? std::string("nullptr") : std::string((char*)remote_device_name)) + "\"");
   mht_2_v.push_back("local_device_name: \"" + (local_device_name == nullptr ? std::string("nullptr") : std::string((char*)local_device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_2(mht_2_v, 236, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "CheckRemoteMatMulExecutesOK");

  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* h0_task0 = TestMatrixTensorHandle(ctx);

  TFE_Op* matmul = MatMulOp(ctx, h0_task0, h0_task0);
  TFE_OpSetDevice(matmul, remote_device_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  auto* retval_task0 =
      TFE_TensorHandleCopyToDevice(retvals[0], ctx, local_device_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  CheckTFE_TensorHandleHasFloats(retval_task0, {7, 10, 15, 22});

  TFE_DeleteTensorHandle(retval_task0);
  TFE_DeleteTensorHandle(h0_task0);
  TFE_DeleteTensorHandle(retvals[0]);

  TFE_DeleteOp(matmul);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);
  TF_DeleteStatus(status);
}

// Read the value of variable `var` and save it into `out_value`.
void ReadVariable(TFE_Context* ctx, TFE_TensorHandle* var,
                  TFE_TensorHandle** out_value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_3(mht_3_v, 273, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "ReadVariable");

  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, "ReadVariableOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_retvals = 1;
  TFE_Execute(op, out_value, &num_retvals, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(op);
  TF_DeleteStatus(status);
}

void TestRemoteExecuteChangeServerDef(bool async) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_4(mht_4_v, 290, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "TestRemoteExecuteChangeServerDef");

  tensorflow::ServerDef server_def = GetServerDef(2);

  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);

  std::unique_ptr<tensorflow::GrpcServer> worker_server;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  const char local_device_name[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();

  // Update the server def with a new set of names (worker instead of
  // localhost).
  tensorflow::ServerDef updated_server_def = GetServerDef("worker", 2);
  serialized = updated_server_def.SerializeAsString();

  updated_server_def.set_task_index(1);
  tensorflow::Status s = tensorflow::GrpcServer::Create(
      updated_server_def, tensorflow::Env::Default(), &worker_server);
  ASSERT_TRUE(s.ok()) << s.error_message();
  ASSERT_TRUE(worker_server->Start().ok());

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Create a new tensor_handle.
  TFE_TensorHandle* h0_task0_new = TestMatrixTensorHandle(ctx);

  // Check that copying it to the old remote device (named localhost) fails.
  TFE_TensorHandleCopyToDevice(h0_task0_new, ctx, remote_device_name, status);
  EXPECT_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Copying and executing on the new remote device works.
  const char new_remote_device_name[] =
      "/job:worker/replica:0/task:1/device:CPU:0";
  const char new_local_device_name[] =
      "/job:worker/replica:0/task:0/device:CPU:0";

  auto* h0_task1_new = TFE_TensorHandleCopyToDevice(
      h0_task0_new, ctx, new_remote_device_name, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteTensorHandle(h0_task0_new);
  TFE_DeleteTensorHandle(h0_task1_new);

  CheckRemoteMatMulExecutesOK(ctx, new_remote_device_name,
                              new_local_device_name);

  TFE_ExecutorWaitForAllPendingNodes(executor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteExecutor(executor);

  TF_DeleteStatus(status);

  TFE_DeleteContext(ctx);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, RemoteExecuteChangeServerDef) {
  TestRemoteExecuteChangeServerDef(false);
}
TEST(CAPI, RemoteExecuteChangeServerDefAsync) {
  TestRemoteExecuteChangeServerDef(true);
}

void TestRemoteExecuteUpdateServerDef(bool async) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_5(mht_5_v, 387, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "TestRemoteExecuteUpdateServerDef");

  tensorflow::ServerDef server_def = GetServerDef(2);
  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const char local_device_name[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_ContextUpdateServerDef(ctx, 0, serialized.data(), serialized.size(),
                             status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, RemoteExecuteUpdateServerDef) {
  TestRemoteExecuteUpdateServerDef(false);
}

TEST(CAPI, RemoteExecuteUpdateServerDefAsync) {
  TestRemoteExecuteUpdateServerDef(true);
}

void TestRemoteExecuteUpdateServerDefResourceAccess(bool async) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_6(mht_6_v, 438, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "TestRemoteExecuteUpdateServerDefResourceAccess");

  tensorflow::ServerDef server_def = GetServerDef(2);
  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const char dev0_name[] = "/job:localhost/replica:0/task:0/device:CPU:0";
  const char dev1_name[] = "/job:localhost/replica:0/task:1/device:CPU:0";

  TFE_TensorHandle* var_handle0 = TestVariable(ctx, 1.0, dev0_name);
  EXPECT_NE(var_handle0, nullptr);
  TFE_TensorHandle* var_handle1 = TestVariable(ctx, 2.0, dev1_name);
  EXPECT_NE(var_handle1, nullptr);

  TFE_TensorHandle* value_handle = nullptr;
  ReadVariable(ctx, var_handle1, &value_handle);
  CheckTFE_TensorHandleHasFloats(value_handle, {2});
  TFE_DeleteTensorHandle(value_handle);

  // Start a new worker to replace task:1
  ReplaceTaskInServerDef(&server_def, 1);
  server_def.set_task_index(1);
  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  // Update server def to replace the remote device with the device info on the
  // new worker (different incarnation ID).
  server_def.set_task_index(0);
  string serialized_update = server_def.SerializeAsString();
  TFE_ContextUpdateServerDef(ctx, 0, serialized_update.data(),
                             serialized_update.size(), status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // The device of var_handle0 is local device which is the same before and
  // after cluster update. Remove resource with valid device should succeed.
  TFE_Op* op = TFE_NewOp(ctx, "DestroyResourceOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, var_handle0, status);
  TFE_OpSetDevice(op, dev0_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(op);

  // The device of var_handle1 is remote device, which was replaced during
  // cluster update. Removing resource with invalid device should fail
  // gracefully (i.e., with error status) instead of crashing with segfaults.
  op = TFE_NewOp(ctx, "DestroyResourceOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, var_handle1, status);
  TFE_OpSetDevice(op, dev1_name, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  EXPECT_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(op);

  TFE_DeleteTensorHandle(var_handle0);
  TFE_DeleteTensorHandle(var_handle1);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
}

TEST(CAPI, TestRemoteExecuteUpdateServerDefResourceAccess) {
  TestRemoteExecuteUpdateServerDefResourceAccess(false);
}

TEST(CAPI, TestRemoteExecuteUpdateServerDefResourceAccessAsync) {
  TestRemoteExecuteUpdateServerDefResourceAccess(true);
}

void TestRemoteExecuteUpdateServerDefWithFailures(bool async) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_7(mht_7_v, 537, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "TestRemoteExecuteUpdateServerDefWithFailures");

  // Fail fast on GetStatus requests so we can get errors instead of timeout
  // when updating cluster with non-exsitent worker
  tensorflow::setenv("GRPC_FAIL_FAST", "TRUE", /*overwrite=*/1);

  tensorflow::ServerDef server_def = GetServerDef(2);
  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server)
                  .ok());
  ASSERT_TRUE(worker_server->Start().ok());

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  const char local_device_name[] =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  const char remote_device_name[] =
      "/job:localhost/replica:0/task:1/device:CPU:0";
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  // Adding a non-existent remote worker to cluster def. This should cause the
  // UpdateServerDef call to fail.
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->mutable_job(0);
  int port = tensorflow::testing::PickUnusedPortOrDie();
  job_def->mutable_tasks()->insert(
      {2, tensorflow::strings::StrCat("localhost:", port)});
  server_def.set_task_index(0);
  string serialized_update = server_def.SerializeAsString();
  TFE_ContextUpdateServerDef(ctx, 0, serialized_update.data(),
                             serialized_update.size(), status);
  EXPECT_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Even after the prevoiusly failed cluster update, another update and op
  // execution should work fine as long as the provided server_def is valid.
  TFE_ContextUpdateServerDef(ctx, 0, serialized.data(), serialized.size(),
                             status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CheckRemoteMatMulExecutesOK(ctx, remote_device_name, local_device_name);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  // TODO(b/136478427): Figure out how to correctly shut the server down.
  worker_server.release();
  tensorflow::unsetenv("GRPC_FAIL_FAST");
}

TEST(CAPI, RemoteExecuteUpdateServerDefWithFailures) {
  TestRemoteExecuteUpdateServerDefWithFailures(false);
}

TEST(CAPI, RemoteExecuteUpdateServerDefWithFailuresAsync) {
  TestRemoteExecuteUpdateServerDefWithFailures(true);
}

void TestConnectToCluster(bool keep_localhost_for_first_connect) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_cluster_testDTcc mht_8(mht_8_v, 608, "", "./tensorflow/c/eager/c_api_cluster_test.cc", "TestConnectToCluster");

  // Fail fast on GetStatus requests so we can get errors instead of timeout
  // when updating cluster with non-exsitent worker
  tensorflow::setenv("GRPC_FAIL_FAST", "TRUE", /*overwrite=*/1);

  const string first_name =
      keep_localhost_for_first_connect ? "localhost" : "abc";
  tensorflow::ServerDef server_def = GetServerDef(first_name, 1);

  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  const string dev0_name = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_TensorHandle* var_handle0 = TestVariable(ctx, 1.0, dev0_name);
  EXPECT_NE(var_handle0, nullptr);

  tensorflow::Status status2;
  EXPECT_EQ(tensorflow::unwrap(var_handle0)->DeviceName(&status2), dev0_name);

  // Rename local device
  // This server def has the task index set to 0.
  string serialized = server_def.SerializeAsString();
  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const string dev1_name =
      absl::StrCat("/job:", first_name, "/replica:0/task:0/device:CPU:0");
  TFE_TensorHandle* var_handle1 = TestVariable(ctx, 2.0, dev1_name);
  EXPECT_NE(var_handle1, nullptr);
  EXPECT_EQ(tensorflow::unwrap(var_handle1)->DeviceName(&status2), dev1_name);

  // Another renaming of local device
  const string second_name = "def";
  server_def.set_job_name(second_name);
  server_def.mutable_cluster()->mutable_job(0)->set_name(second_name);
  (*server_def.mutable_cluster()->mutable_job(0)->mutable_tasks())[0] =
      absl::StrCat(second_name, ":",
                   tensorflow::testing::PickUnusedPortOrDie());

  serialized = server_def.SerializeAsString();
  TFE_ContextSetServerDef(ctx, 0, serialized.data(), serialized.size(), status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const string dev2_name = "/job:def/replica:0/task:0/device:CPU:0";
  TFE_TensorHandle* var_handle2 = TestVariable(ctx, 2.0, dev2_name);
  EXPECT_NE(var_handle2, nullptr);
  EXPECT_EQ(tensorflow::unwrap(var_handle2)->DeviceName(&status2), dev2_name);

  TFE_DeleteTensorHandle(var_handle0);
  TFE_DeleteTensorHandle(var_handle1);
  TFE_DeleteTensorHandle(var_handle2);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);

  tensorflow::unsetenv("GRPC_FAIL_FAST");
}

TEST(CAPI, ConnectToClusterLocalhostFirst) { TestConnectToCluster(false); }

TEST(CAPI, ConnectToClusterRenameFirst) { TestConnectToCluster(true); }

}  // namespace
