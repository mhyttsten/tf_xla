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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"

namespace tensorflow {
namespace {
using ::testing::_;
using ::testing::DoAll;
using ::testing::InvokeArgument;
using ::testing::SetArgPointee;
using ::testing::WithArgs;

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;
  // MOCK_METHOD does not work on Windows build, using deprecated MOCK_METHOD3
  // instead.
  MOCK_METHOD3(GetKeyValueAsync, void(const GetKeyValueRequest*,
                                      GetKeyValueResponse*, StatusCallback));
  MOCK_METHOD4(RegisterTaskAsync, void(CallOptions*, const RegisterTaskRequest*,
                                       RegisterTaskResponse*, StatusCallback));
  MOCK_METHOD4(ShutdownTaskAsync, void(CallOptions*, const ShutdownTaskRequest*,
                                       ShutdownTaskResponse*, StatusCallback));
  MOCK_METHOD3(ResetTaskAsync, void(const ResetTaskRequest*, ResetTaskResponse*,
                                    StatusCallback));
  MOCK_METHOD3(ReportErrorToServiceAsync,
               void(const ReportErrorToServiceRequest*,
                    ReportErrorToServiceResponse*, StatusCallback));

#define UNIMPLEMENTED(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(errors::Unimplemented(#method "Async"));                     \
  }

  UNIMPLEMENTED(WaitForAllTasks);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(DeleteKeyValue);
  UNIMPLEMENTED(Barrier);
  UNIMPLEMENTED(CancelBarrier);
#undef UNIMPLEMENTED
  void HeartbeatAsync(CallOptions* call_opts, const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      StatusCallback done) override {
    done(errors::Unimplemented("HeartbeatAsync"));
  }
  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc mht_0(mht_0_v, 248, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent_test.cc", "ReportErrorToTaskAsync");

    done(errors::Unimplemented("ReportErrorToTaskAsync"));
  }
};

class CoordinationServiceAgentTest : public ::testing::Test {
 public:
  void SetUp() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc mht_1(mht_1_v, 258, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent_test.cc", "SetUp");

    ON_CALL(*client_, RegisterTaskAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(Status::OK()));
    ON_CALL(*client_, ShutdownTaskAsync(_, _, _, _))
        .WillByDefault(InvokeArgument<3>(Status::OK()));
    ON_CALL(*client_, ReportErrorToServiceAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(Status::OK()));
    ON_CALL(*GetClient(), ResetTaskAsync(_, _, _))
        .WillByDefault(InvokeArgument<2>(Status::OK()));
  }

  // Should be called after mocking service responses, before testing the agent.
  void InitializeAgent() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc mht_2(mht_2_v, 273, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent_test.cc", "InitializeAgent");

    CoordinationServiceConfig config;
    config.set_service_leader("test_leader");
    TF_EXPECT_OK(agent_->Initialize(
        Env::Default(), /*job_name=*/"test_job",
        /*task_id=*/0, config, std::move(client_),
        /*error_fn=*/[](Status s) {
          LOG(ERROR) << "Coordination agent is set to error: " << s;
        }));
  }

  TestCoordinationClient* GetClient() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_agent_testDTcc mht_3(mht_3_v, 287, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_agent_test.cc", "GetClient");

    // InitializeAgent() transfers ownership of the coordination client.
    CHECK(client_ != nullptr)
        << "GetClient() was called after InitializeAgent()";
    return client_.get();
  }

 protected:
  std::unique_ptr<CoordinationServiceAgent> agent_ =
      CreateCoordinationServiceAgent();
  std::unique_ptr<TestCoordinationClient> client_ =
      std::make_unique<TestCoordinationClient>();
};

TEST_F(CoordinationServiceAgentTest, GetKeyValue_Simple_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(Status::OK())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key);

  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie(), test_value);
}

TEST_F(CoordinationServiceAgentTest, GetKeyValue_WithTimeout_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock server response: set key-value pair and invoke done callback.
  GetKeyValueResponse mocked_response;
  auto kv = mocked_response.mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _))
      .WillByDefault(DoAll(SetArgPointee<1>(mocked_response),
                           InvokeArgument<2>(Status::OK())));
  // Initialize coordination agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie(), test_value);
}

TEST_F(CoordinationServiceAgentTest, GetKeyValue_Timeout_ReturnError) {
  const std::string& test_key = "test_key";
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(1));

  EXPECT_EQ(result.status().code(), error::DEADLINE_EXCEEDED);
}

TEST_F(CoordinationServiceAgentTest,
       GetKeyValue_DelayedResponse_TimeoutWithoutMemoryError) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  auto client = std::make_unique<TestCoordinationClient>();
  GetKeyValueResponse* owned_response;
  StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _))
      .WillByDefault(WithArgs<1, 2>(
          [&](GetKeyValueResponse* response, StatusCallback done) {
            // Copy method arguments to prevent de-allocation before mocking the
            // server callback beyond timeout.
            owned_response = response;
            owned_done = done;
          }));
  // Initialize coordination service agent.
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(3));
  EXPECT_EQ(result.status().code(), error::DEADLINE_EXCEEDED);

  // Delayed server response: set key-value response, and invoke done callback.
  auto kv = owned_response->mutable_kv();
  kv->set_key(test_key);
  kv->set_value(test_value);
  owned_done(Status::OK());
  // No explicit test, but used to verify there is no stack-use-after-return
  // or other memory-related errors.
}

TEST_F(CoordinationServiceAgentTest,
       GetKeyValue_DelayedResponseBeforeTimeout_Success) {
  const std::string& test_key = "test_key";
  const std::string& test_value = "test_value";
  // Mock delayed server response before timeout: set key-value pair and invoke
  // done callback.
  auto client = std::make_unique<TestCoordinationClient>();
  std::unique_ptr<Thread> async_thread;
  GetKeyValueResponse* owned_response;
  StatusCallback owned_done;
  ON_CALL(*GetClient(), GetKeyValueAsync(_, _, _))
      // Setup async callback to insert key-value after a brief delay (5s)
      // before timeout (10s).
      .WillByDefault(WithArgs<1, 2>(
          [&](GetKeyValueResponse* response, StatusCallback done) {
            // Copy method arguments to prevent de-allocation before
            //  triggering this async callback.
            owned_response = response;
            owned_done = done;
            async_thread = absl::WrapUnique(Env::Default()->StartThread(
                ThreadOptions(), "async_thread", [&]() {
                  // Set brief delay.
                  absl::SleepFor(absl::Seconds(5));
                  // Set key-value response, and invoke done callback.
                  auto kv = owned_response->mutable_kv();
                  kv->set_key(test_key);
                  kv->set_value(test_value);
                  owned_done(Status::OK());
                }));
          }));
  InitializeAgent();

  auto result = agent_->GetKeyValue(test_key, /*timeout=*/absl::Seconds(10));

  TF_EXPECT_OK(result.status());
  EXPECT_EQ(result.ValueOrDie(), test_value);
}

TEST_F(CoordinationServiceAgentTest, NotAllowedToConnectAfterShuttingDown) {
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());

  TF_EXPECT_OK(agent_->Shutdown());
  Status status = agent_->Connect();

  // Not allowed to connect after shutting down.
  EXPECT_TRUE(errors::IsFailedPrecondition(status));
}

TEST_F(CoordinationServiceAgentTest, ShutdownInErrorShouldReturnError) {
  // Connect coordination agent and set it to error.
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  TF_EXPECT_OK(agent_->ReportError(errors::Internal("Test Error.")));

  // Shutdown should return error.
  Status s = agent_->Shutdown();

  EXPECT_TRUE(errors::IsFailedPrecondition(s));
}

TEST_F(CoordinationServiceAgentTest, Reset_ConnectedButNotInError_Fail) {
  // Connect agent.
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());

  auto status = agent_->Reset();

  // Fails because agent is not in ERROR state.
  EXPECT_TRUE(errors::IsFailedPrecondition(status));
}

TEST_F(CoordinationServiceAgentTest, ConnectAfterResetError) {
  // Connect coordination agent and set it to error.
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  TF_EXPECT_OK(agent_->ReportError(errors::Internal("Test Error.")));

  // Reset error.
  TF_EXPECT_OK(agent_->Reset());
  // Agent should be able to reconnect to the service after resetting.
  TF_EXPECT_OK(agent_->Connect());
}

TEST_F(CoordinationServiceAgentTest, ResetCanBeRetried) {
  // Mock reset error failing for the first time.
  EXPECT_CALL(*GetClient(), ResetTaskAsync(_, _, _))
      .WillOnce(InvokeArgument<2>(errors::Internal("Reset error")))
      .WillOnce(InvokeArgument<2>(Status::OK()));
  // Connect coordination agent and set it to error.
  InitializeAgent();
  TF_EXPECT_OK(agent_->Connect());
  TF_EXPECT_OK(agent_->ReportError(errors::Internal("Test Error.")));

  // Reset error fails for the first time.
  Status reset_status = agent_->Reset();
  EXPECT_TRUE(errors::IsInternal(reset_status));

  // Agent should be able to attempt resetting again.
  TF_EXPECT_OK(agent_->Reset());
  // Agent should be able to reconnect to the service after resetting.
  TF_EXPECT_OK(agent_->Connect());
}
}  // namespace
}  // namespace tensorflow
