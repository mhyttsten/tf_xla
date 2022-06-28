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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc() {
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
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace {

void StartWorkers(int cluster_size,
                  std::function<void(TFE_Context* ctx, TF_Status* status,
                                     int worker_id, int cluster_size)>
                      fn) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_test.cc", "StartWorkers");

  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size, /*num_virtual_gpus=*/2);
  // Enable coordination service for propagating remote device attributess
  auto* config = server_def.mutable_default_session_config()
                     ->mutable_experimental()
                     ->mutable_coordination_config();
  config->set_service_type("standalone");
  config->set_service_leader("/job:worker/replica:0/task:0");

  // The blocking counter makes sure that worker/0 thread (leader that starts
  // the coordination service) does not exit early while other workers are still
  // interacting with the coordination service.
  tensorflow::BlockingCounter counter(cluster_size);
  auto worker_thread_fn = [&](int worker_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_test.cc", "lambda");

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

    tensorflow::SessionOptions options;
    options.config = server_def_copy.default_session_config();
    opts->session_options.options = options;
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    fn(ctx, status, worker_id, cluster_size);
    counter.DecrementCount();
    counter.Wait();

    // Since we created an async EagerContext, wait for all pending operations
    // to finish before deleting the context.
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };

  std::vector<std::thread> worker_threads;
  for (int i = 0; i < cluster_size; ++i) {
    worker_threads.emplace_back([i, worker_thread_fn] { worker_thread_fn(i); });
  }
  for (auto i = 0; i < cluster_size; ++i) {
    worker_threads[i].join();
  }
}

TEST(CAPI, MultiClientCollectiveOps) {
  auto fn = [](TFE_Context* ctx, TF_Status* status, int worker_id,
               int cluster_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc mht_2(mht_2_v, 277, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_test.cc", "lambda");

    TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
    TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float result[4] = {0};
    EXPECT_EQ(sizeof(result), TF_TensorByteSize(t));
    memcpy(&result[0], TF_TensorData(t), TF_TensorByteSize(t));
    TF_DeleteTensor(t);
    EXPECT_EQ(2.0, result[0]);
    EXPECT_EQ(4.0, result[1]);
    EXPECT_EQ(6.0, result[2]);
    EXPECT_EQ(8.0, result[3]);

    TFE_DeleteTensorHandle(in);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteOp(allreduce);
  };
  StartWorkers(2, fn);
}

TEST(CAPI, MultiClientRemoteDevices) {
  auto fn = [](TFE_Context* ctx, TF_Status* status, int worker_id,
               int cluster_size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc mht_3(mht_3_v, 308, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_test.cc", "lambda");

    std::vector<tensorflow::DeviceAttributes> device_attrs;
    tensorflow::EagerContext* context =
        tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
    context->ListDevices(&device_attrs);
    std::vector<std::string> device_names;
    for (const auto& device_attr : device_attrs) {
      device_names.push_back(device_attr.name());
    }

    bool has_gpu_devices = false;
    std::string unused_gpu_device_name;
    if (GetDeviceName(ctx, &unused_gpu_device_name, "GPU")) {
      has_gpu_devices = true;
    }

    std::vector<std::string> expected_device_names;
    for (int i = 0; i < cluster_size; ++i) {
      expected_device_names.push_back(tensorflow::strings::StrCat(
          "/job:worker/replica:0/task:", i, "/device:CPU:0"));
      if (has_gpu_devices) {
        expected_device_names.push_back(tensorflow::strings::StrCat(
            "/job:worker/replica:0/task:", i, "/device:GPU:0"));
        expected_device_names.push_back(tensorflow::strings::StrCat(
            "/job:worker/replica:0/task:", i, "/device:GPU:1"));
      }
    }

    EXPECT_THAT(device_names,
                testing::UnorderedElementsAreArray(expected_device_names));
  };
  StartWorkers(3, fn);
}

TEST(CAPI, MultiClientSendRecv) {
  auto fn = [](TFE_Context* ctx, TF_Status* status, int worker_id,
               int cluster_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSintegration_testPSc_api_multi_client_testDTcc mht_4(mht_4_v, 347, "", "./tensorflow/core/distributed_runtime/integration_test/c_api_multi_client_test.cc", "lambda");

    // Test with GPUs if present (based on test configuration) and CPUs
    // otherwise.
    auto send_device = "/job:worker/replica:0/task:0/device:CPU:0";
    auto recv_device = "/job:worker/replica:0/task:1/device:CPU:0";
    std::string unused_gpu_device_name;
    if (GetDeviceName(ctx, &unused_gpu_device_name, "GPU")) {
      send_device = "/job:worker/replica:0/task:0/device:GPU:0";
      recv_device = "/job:worker/replica:0/task:1/device:GPU:0";
    }

    std::vector<tensorflow::DeviceAttributes> device_attrs;
    tensorflow::EagerContext* context =
        tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
    context->ListDevices(&device_attrs);

    tensorflow::uint64 send_device_incarnation = 0;
    for (const auto& device_attr : device_attrs) {
      if (device_attr.name() == send_device) {
        send_device_incarnation = device_attr.incarnation();
        break;
      }
    }

    if (worker_id == 0) {
      TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
      const std::string& op_name =
          tensorflow::str_util::StrContains(send_device, "GPU") ? "Send"
                                                                : "_HostSend";
      TFE_Op* sendop = SendOp(ctx, in, op_name, send_device, recv_device,
                              send_device_incarnation);
      TFE_TensorHandle* retvals[1];
      int num_retvals = 1;
      TFE_Execute(sendop, &retvals[0], &num_retvals, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      TFE_DeleteOp(sendop);
      TFE_DeleteTensorHandle(in);
    } else {
      const std::string& op_name =
          tensorflow::str_util::StrContains(send_device, "GPU") ? "Recv"
                                                                : "_HostRecv";
      TFE_Op* recvop = RecvOp(ctx, op_name, send_device, recv_device,
                              send_device_incarnation);
      TFE_TensorHandle* retvals[1];
      int num_retvals = 1;
      TFE_Execute(recvop, &retvals[0], &num_retvals, status);
      TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
      TF_DeleteTensor(t);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      TFE_DeleteTensorHandle(retvals[0]);
      TFE_DeleteOp(recvop);
    }
  };
  StartWorkers(2, fn);
}

}  // namespace
