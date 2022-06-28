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
class MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_remote_testDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_remote_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_remote_testDTcc() {
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

#include <array>
#include <string>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/parallel_device/parallel_device.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/platform/test.h"

tensorflow::ServerDef GetServerDef(const std::string& job_name, int num_tasks) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSparallel_devicePSparallel_device_remote_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/c/eager/parallel_device/parallel_device_remote_test.cc", "GetServerDef");

  tensorflow::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = tensorflow::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, tensorflow::strings::StrCat("localhost", ":", port)});
  }
  return server_def;
}

namespace tensorflow {
namespace parallel_device {

TEST(PARALLEL_DEVICE, TestRemoteBasic) {
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  tensorflow::ServerDef server_def = GetServerDef("worker", 3);

  // This server def has the task index set to 0.
  std::string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server1;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());

  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::GrpcServer> worker_server2;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());

  TFE_ContextSetServerDef(context.get(), 0, serialized.data(),
                          serialized.size(), status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  BasicTestsForTwoDevices(context.get(),
                          "/job:worker/replica:0/task:1/device:CPU:0",
                          "/job:worker/replica:0/task:2/device:CPU:0");

  worker_server1.release();
  worker_server2.release();
}

TEST(PARALLEL_DEVICE, TestAsyncCopyOff) {
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  tensorflow::ServerDef server_def = GetServerDef("worker", 3);

  // This server def has the task index set to 0.
  std::string serialized = server_def.SerializeAsString();

  server_def.set_task_index(1);
  std::unique_ptr<tensorflow::GrpcServer> worker_server1;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server1)
                  .ok());
  ASSERT_TRUE(worker_server1->Start().ok());

  server_def.set_task_index(2);
  std::unique_ptr<tensorflow::GrpcServer> worker_server2;
  ASSERT_TRUE(tensorflow::GrpcServer::Create(
                  server_def, tensorflow::Env::Default(), &worker_server2)
                  .ok());
  ASSERT_TRUE(worker_server2->Start().ok());

  TFE_ContextSetServerDef(context.get(), 0, serialized.data(),
                          serialized.size(), status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const char* first_device = "/job:worker/replica:0/task:1/device:CPU:0";
  const char* second_device = "/job:worker/replica:0/task:2/device:CPU:0";
  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{first_device, second_device};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TensorHandlePtr value_one(FloatTensorHandle(3., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TensorHandlePtr value_two(FloatTensorHandle(-2., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::array<TFE_TensorHandle*, 2> in_components{value_one.get(),
                                                 value_two.get()};
  TensorHandlePtr combined_value = CreatePerDeviceValues(
      context.get(), in_components, device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Loop to make synchronization failures more deterministic
  for (int i = 0; i < 100; ++i) {
    TensorHandlePtr multiply_result(
        Multiply(context.get(), combined_value.get(), combined_value.get(),
                 status.get()));
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TensorHandlePtr, 2> out_components;
    ExtractPerDeviceValues(context.get(), multiply_result.get(),
                           &out_components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    ExpectScalarEq<float>(out_components[0].get(), 9.);
    ExpectScalarEq<float>(out_components[1].get(), 4.);
  }

  worker_server1.release();
  worker_server2.release();
}
}  // namespace parallel_device
}  // namespace tensorflow
