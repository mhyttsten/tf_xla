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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/coordination/coordination_service.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.pb.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {
namespace {
using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

constexpr absl::Duration kHeartbeatTimeout = absl::Seconds(2);
constexpr absl::Duration kShutdownBarrierTimeout = absl::Seconds(1);
constexpr char kCoordinationServiceType[] = "standalone";

class TestCoordinationClient : public CoordinationClient {
 public:
  TestCoordinationClient() = default;

  Status GetStatus() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "GetStatus");

    mutex_lock l(mu_);
    return status_;
  }

  void RegisterTaskAsync(CallOptions* opts, const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         StatusCallback done) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_1(mht_1_v, 233, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "RegisterTaskAsync");

    done(Status::OK());
  }

  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "ReportErrorToTaskAsync");

    mutex_lock l(mu_);
    status_ = Status(static_cast<errors::Code>(request->error_code()),
                     request->error_message());
    done(Status::OK());
  }

#define UNIMPLEMENTED(method)                                         \
  void method##Async(const method##Request* request,                  \
                     method##Response* response, StatusCallback done) \
      override {                                                      \
    done(errors::Unimplemented(#method "Async"));                     \
  }

  UNIMPLEMENTED(WaitForAllTasks);
  UNIMPLEMENTED(ResetTask);
  UNIMPLEMENTED(ReportErrorToService);
  UNIMPLEMENTED(InsertKeyValue);
  UNIMPLEMENTED(GetKeyValue);
  UNIMPLEMENTED(DeleteKeyValue);
  UNIMPLEMENTED(Barrier);
  UNIMPLEMENTED(CancelBarrier);
#undef UNIMPLEMENTED
  void HeartbeatAsync(CallOptions* call_opts, const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      StatusCallback done) override {
    done(errors::Unimplemented("HeartbeatAsync"));
  }
  void ShutdownTaskAsync(CallOptions* call_opts,
                         const ShutdownTaskRequest* request,
                         ShutdownTaskResponse* response,
                         StatusCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_3(mht_3_v, 277, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "ShutdownTaskAsync");

    done(errors::Unimplemented("ShutdownTaskAsync"));
  }

 private:
  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);
};

class TestCoordinationClientCache : public CoordinationClientCache {
 public:
  void AddTask(const std::string& target, CoordinationClient* client) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_4(mht_4_v, 292, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "AddTask");

    clients_.emplace(target, client);
  }

  CoordinationClient* GetClient(const string& target) override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "GetClient");

    auto it = clients_.find(target);
    if (it == clients_.end()) return nullptr;
    return it->second;
  }

  std::unique_ptr<CoordinationClient> GetOwnedClient(
      const string& target) override {
    LOG(ERROR) << "GetOwnedClient is not supported.";
    return nullptr;
  }

 private:
  std::unordered_map<std::string, CoordinationClient*> clients_;
};

class CoordinationBarrierTest : public ::testing::Test {
 protected:
  CoordinationBarrierTest() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_6(mht_6_v, 321, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "CoordinationBarrierTest");

    // Set up fake cluster with 3 tasks.
    const int num_tasks = 3;
    const ServerDef& server_def = GetMultiClientServerDef("worker", num_tasks);
    auto client_cache = std::make_unique<TestCoordinationClientCache>();
    for (int i = 0; i < num_tasks; ++i) {
      CoordinatedTask task;
      task.set_job_name("worker");
      task.set_task_id(i);

      auto client = std::make_unique<TestCoordinationClient>();
      client_cache->AddTask(absl::StrCat("/job:worker/replica:0/task:", i),
                            client.get());

      tasks_.push_back(task);
      clients_.push_back(std::move(client));
    }

    coord_service_ = CoordinationServiceInterface::EnableCoordinationService(
        kCoordinationServiceType, Env::Default(), server_def,
        std::move(client_cache));
    // Register the tasks.
    for (int i = 0; i < num_tasks; ++i) {
      Status s = coord_service_->RegisterTask(tasks_[i], /*incarnation=*/0);
      if (!s.ok()) {
        LOG(FATAL) << "RegisterTask() failed in CoordinationBarrierTest(): "
                   << s;
      }
    }
  }

  CoordinationServiceInterface* GetCoordinationService() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_7(mht_7_v, 355, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "GetCoordinationService");

    return coord_service_.get();
  }
  CoordinatedTask GetTask(int i) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_8(mht_8_v, 361, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "GetTask");
 return tasks_[i]; }

 private:
  std::unique_ptr<CoordinationServiceInterface> coord_service_;
  std::vector<CoordinatedTask> tasks_;
  std::vector<std::unique_ptr<TestCoordinationClient>> clients_;
};

// Sets up coordination service that expects 2 worker tasks.
class CoordinateTwoTasksTest : public ::testing::Test {
 protected:
  CoordinateTwoTasksTest() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_9(mht_9_v, 375, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "CoordinateTwoTasksTest");

    task_0_.set_job_name("worker");
    task_0_.set_task_id(0);
    task_1_.set_job_name("worker");
    task_1_.set_task_id(1);
  }

  // Set up coordination service.
  void EnableCoordinationService(bool has_service_to_client_connection = true,
                                 bool enable_shutdown_barrier = false) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_10(mht_10_v, 387, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "EnableCoordinationService");

    ServerDef server_def = GetMultiClientServerDef("worker", /*num_tasks=*/2);
    auto client_cache = std::make_unique<TestCoordinationClientCache>();
    if (has_service_to_client_connection) {
      client_cache->AddTask("/job:worker/replica:0/task:0", &client_0_);
      client_cache->AddTask("/job:worker/replica:0/task:1", &client_1_);
    } else {
      client_cache = nullptr;
    }
    auto coord_config = server_def.mutable_default_session_config()
                            ->mutable_experimental()
                            ->mutable_coordination_config();
    coord_config->set_service_type(kCoordinationServiceType);
    coord_config->set_heartbeat_timeout_in_ms(kHeartbeatTimeout /
                                              absl::Milliseconds(1));
    if (enable_shutdown_barrier) {
      coord_config->set_shutdown_barrier_timeout_in_ms(kShutdownBarrierTimeout /
                                                       absl::Milliseconds(1));
    }
    // Init service.
    coord_service_ = CoordinationServiceInterface::EnableCoordinationService(
        kCoordinationServiceType, Env::Default(), server_def,
        std::move(client_cache));
  }

  CoordinatedTask task_0_;
  const uint64_t incarnation_0_ = random::New64();
  TestCoordinationClient client_0_;
  CoordinatedTask task_1_;
  const uint64_t incarnation_1_ = random::New64();
  TestCoordinationClient client_1_;
  std::unique_ptr<CoordinationServiceInterface> coord_service_;
};

// Construct fake device protos.
DeviceAttributes CreateTestTfDevice(absl::string_view name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_11(mht_11_v, 426, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "CreateTestTfDevice");

  DeviceAttributes device;
  device.set_name(name);
  device.set_device_type("CPU");
  return device;
}

xla::DeviceProto CreateTestXlaDevice(absl::string_view name,
                                     const int local_id) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_12(mht_12_v, 438, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "CreateTestXlaDevice");

  xla::DeviceProto device;
  device.set_name(name);
  device.set_local_device_ordinal(local_id);
  return device;
}

TEST_F(CoordinateTwoTasksTest, TestStandaloneService) {
  EnableCoordinationService();
  // Not specified in server def.
  CoordinatedTask task_2;
  task_2.set_job_name("worker");
  task_2.set_task_id(2);

  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  absl::Notification wait_for_all;
  coord_service_->WaitForAllTasks(task_0_, {}, [&](Status s) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_13(mht_13_v, 457, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

    TF_ASSERT_OK(s);
    wait_for_all.Notify();
  });
  // Not all tasks have registered, so must not be notified here.
  ASSERT_FALSE(wait_for_all.HasBeenNotified());
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  coord_service_->WaitForAllTasks(task_1_, {},
                                  [&](Status s) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_14(mht_14_v, 468, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 TF_ASSERT_OK(s); });
  // All tasks have registered.
  wait_for_all.WaitForNotification();

  TF_ASSERT_OK(coord_service_->RecordHeartbeat(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RecordHeartbeat(task_1_, incarnation_1_));
  EXPECT_TRUE(
      errors::IsInvalidArgument(coord_service_->RecordHeartbeat(task_2, 0)));

  // Sending heartbeat with incarnation mismatch leads to Aborted error.
  EXPECT_TRUE(errors::IsAborted(coord_service_->RecordHeartbeat(task_1_, 0)));
  EXPECT_TRUE(errors::IsAborted(coord_service_->RecordHeartbeat(task_1_, 0)));
  // Error is propagated to other tasks.
  EXPECT_TRUE(errors::IsAborted(client_0_.GetStatus()));
}

TEST(CoordinationServiceTest, TestCoordinatedJobs) {
  ServerDef server_def = GetMultiClientServerDef("chief", 1);
  CoordinatedTask chief;
  chief.set_job_name("chief");
  chief.set_task_id(0);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  CoordinatedTask evaluator;
  evaluator.set_job_name("evaluator");
  evaluator.set_task_id(0);

  // Add a worker job with 2 tasks
  ClusterDef* cluster_def = server_def.mutable_cluster();
  JobDef* job_def = cluster_def->add_job();
  job_def->set_name("worker");
  job_def->mutable_tasks()->insert({0, "dummy address"});
  job_def->mutable_tasks()->insert({1, "dummy address"});

  // Add an evaluator job with 1 task
  job_def = cluster_def->add_job();
  job_def->set_name("evaluator");
  job_def->mutable_tasks()->insert({0, "dummy address"});

  CoordinationServiceConfig* configs =
      server_def.mutable_default_session_config()
          ->mutable_experimental()
          ->mutable_coordination_config();
  configs->mutable_coordinated_jobs()->Add("chief");
  configs->mutable_coordinated_jobs()->Add("worker");

  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  TestCoordinationClient ci;
  client_cache->AddTask("/job:chief/replica:0/task:0", &ci);
  TestCoordinationClient wi0;
  client_cache->AddTask("/job:worker/replica:0/task:0", &wi0);
  TestCoordinationClient wi1;
  client_cache->AddTask("/job:worker/replica:0/task:1", &wi1);
  TestCoordinationClient ei;
  client_cache->AddTask("/job:evaluator/replica:0/task:0", &ei);
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));

  // Each coordinated task registers and waits for other tasks.
  absl::Notification register_chief;
  TF_ASSERT_OK(coord_service->RegisterTask(chief, /*incarnation=*/0));
  coord_service->WaitForAllTasks(chief, {}, [&](Status s) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_15(mht_15_v, 538, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

    TF_ASSERT_OK(s);
    register_chief.Notify();
  });
  absl::Notification register_task0;
  TF_ASSERT_OK(coord_service->RegisterTask(task_0, /*incarnation=*/0));
  coord_service->WaitForAllTasks(task_0, {}, [&](Status s) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_16(mht_16_v, 547, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

    TF_ASSERT_OK(s);
    register_task0.Notify();
  });
  absl::Notification register_task1;
  TF_ASSERT_OK(coord_service->RegisterTask(task_1, /*incarnation=*/0));
  coord_service->WaitForAllTasks(task_1, {}, [&](Status s) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_17(mht_17_v, 556, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

    TF_ASSERT_OK(s);
    register_task1.Notify();
  });
  // All tasks in the coordinated jobs have registered.
  register_chief.WaitForNotification();
  register_task0.WaitForNotification();
  register_task1.WaitForNotification();

  // Registering the evaluator task is unexpected
  Status status = coord_service->RegisterTask(evaluator, /*incarnation=*/0);
  EXPECT_TRUE(errors::IsInvalidArgument(status)) << status;
}

TEST_F(CoordinateTwoTasksTest, TestTaskHeartbeatTimeout) {
  EnableCoordinationService();
  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));

  // No heartbeat for a while, leader considers the task as stale.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  EXPECT_TRUE(errors::IsUnavailable(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
  EXPECT_TRUE(errors::IsUnavailable(
      coord_service_->RecordHeartbeat(task_1_, incarnation_1_)));
}

TEST_F(CoordinateTwoTasksTest,
       HeartbeatTimeoutWithoutServerToClientConnection) {
  EnableCoordinationService(/*has_service_to_client_connection=*/false);
  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));

  // No heartbeat for a while, leader consider the task as stale.
  // Service stops and disconnects both tasks.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(2 * kHeartbeatTimeout));
  // Unexpected heartbeat from unregistered tasks since service state has been
  // reset.
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_1_, incarnation_1_)));
}

TEST_F(CoordinateTwoTasksTest, TestTaskRestart) {
  EnableCoordinationService();
  TF_ASSERT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_ASSERT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));

  // Simulate task restart scenario: trying to register to cluster again.
  Status s =
      coord_service_->RegisterTask(task_1_, /*incarnation=*/random::New64());
  EXPECT_TRUE(errors::IsAborted(s)) << s;
  // Aborted error is also propagated to other tasks in cluster.
  EXPECT_TRUE(errors::IsAborted(client_0_.GetStatus()))
      << client_0_.GetStatus();
}

TEST_F(CoordinateTwoTasksTest, TestSetGetValues) {
  EnableCoordinationService();

  // Simple key
  TF_ASSERT_OK(coord_service_->InsertKeyValue("key0", "value0"));
  // Unix file like key path
  TF_ASSERT_OK(coord_service_->InsertKeyValue("/path", "value"));
  TF_ASSERT_OK(coord_service_->InsertKeyValue("/path/to/key1", "value1"));
  // Key with redundant slashes
  TF_ASSERT_OK(coord_service_->InsertKeyValue("path/to//key2/", "value2"));
  // Error when repeatedly inserting the same key
  EXPECT_TRUE(errors::IsAlreadyExists(
      coord_service_->InsertKeyValue("/path/to/key1/", "value2")));

  // Get simple key
  auto ret = coord_service_->GetKeyValue("key0");
  TF_ASSERT_OK(ret.status());
  EXPECT_EQ(ret.ValueOrDie(), "value0");
  // Get key with redundant slashes
  ret = coord_service_->GetKeyValue("path//to///key1////");
  EXPECT_EQ(ret.ValueOrDie(), "value1");

  // Delete single key-value
  TF_ASSERT_OK(coord_service_->DeleteKeyValue("key0"));
  // Get key that is not available
  absl::Notification n;
  coord_service_->GetKeyValueAsync(
      "key0", [&](const StatusOr<std::string>& status_or_value) {
        ret = status_or_value;
        n.Notify();
      });
  EXPECT_FALSE(n.HasBeenNotified());
  // Insert the previously deleted key again
  TF_ASSERT_OK(coord_service_->InsertKeyValue("key0", "value0_new"));
  n.WaitForNotification();
  EXPECT_EQ(ret.ValueOrDie(), "value0_new");

  // Delete key-values recursively
  TF_ASSERT_OK(coord_service_->DeleteKeyValue("/path"));
  // Get key that is not available
  absl::Notification n2;
  coord_service_->GetKeyValueAsync(
      "/path/to/key1",
      [&](const StatusOr<std::string>& status_or_value) { n2.Notify(); });
  EXPECT_FALSE(n2.HasBeenNotified());
}

}  // namespace

// Verify that coordination service can gather each task's device info and
// propagate the aggregated cluster device info correctly.
TEST(CoordinationServiceTest, ListClusterDevices_TfDevice) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 3);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  CoordinatedTask task_2;
  task_2.set_job_name("worker");
  task_2.set_task_id(2);
  Status status = Status::OK();
  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));
  absl::Notification n;
  // Map fake devices to each task.
  CoordinationServiceDeviceInfo local_devices_0;
  CoordinationServiceDeviceInfo local_devices_1;
  CoordinationServiceDeviceInfo local_devices_2;
  *local_devices_0.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task0_device0");
  *local_devices_0.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task0_device1");
  *local_devices_1.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task1_device0");
  *local_devices_2.mutable_tf()->mutable_devices()->Add() =
      CreateTestTfDevice("task2_device0");

  // Each task sends its device info.
  CoordinationServiceDeviceInfo cluster_devices;
  coord_service->WaitForAllTasks(task_0, local_devices_0,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_1, local_devices_1,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_2, local_devices_2, [&](Status s) {
    TF_ASSERT_OK(s);
    // Gather the cluster device info.
    cluster_devices = coord_service->ListClusterDevices();
    n.Notify();
  });
  n.WaitForNotification();

  CoordinationServiceDeviceInfo expected_cluster_devices;
  auto expected_devices =
      expected_cluster_devices.mutable_tf()->mutable_devices();
  expected_devices->Add(local_devices_0.mutable_tf()->devices().begin(),
                        local_devices_0.mutable_tf()->devices().end());
  expected_devices->Add(local_devices_1.mutable_tf()->devices().begin(),
                        local_devices_1.mutable_tf()->devices().end());
  expected_devices->Add(local_devices_2.mutable_tf()->devices().begin(),
                        local_devices_2.mutable_tf()->devices().end());
  EXPECT_THAT(cluster_devices, IgnoringRepeatedFieldOrdering(
                                   EqualsProto(expected_cluster_devices)));
}

TEST(CoordinationServiceTest, ListClusterDevices_XlaDevice) {
  const ServerDef& server_def = GetMultiClientServerDef("worker", 3);
  CoordinatedTask task_0;
  task_0.set_job_name("worker");
  task_0.set_task_id(0);
  CoordinatedTask task_1;
  task_1.set_job_name("worker");
  task_1.set_task_id(1);
  CoordinatedTask task_2;
  task_2.set_job_name("worker");
  task_2.set_task_id(2);
  Status status = Status::OK();
  auto client_cache = std::make_unique<TestCoordinationClientCache>();
  std::unique_ptr<CoordinationServiceInterface> coord_service =
      CoordinationServiceInterface::EnableCoordinationService(
          kCoordinationServiceType, Env::Default(), server_def,
          std::move(client_cache));
  absl::Notification n;
  // Map fake devices to each task.
  CoordinationServiceDeviceInfo local_devices_0;
  CoordinationServiceDeviceInfo local_devices_1;
  CoordinationServiceDeviceInfo local_devices_2;
  xla::LocalTopologyProto local_0;
  xla::LocalTopologyProto local_1;
  xla::LocalTopologyProto local_2;
  local_0.set_node_id(0);
  local_1.set_node_id(1);
  local_2.set_node_id(2);
  *local_0.add_devices() = CreateTestXlaDevice("task0_device0", 0);
  *local_0.add_devices() = CreateTestXlaDevice("task0_device1", 1);
  *local_1.add_devices() = CreateTestXlaDevice("task1_device0", 0);
  *local_2.add_devices() = CreateTestXlaDevice("task2_device0", 0);
  *local_devices_0.mutable_xla()->mutable_devices()->add_nodes() = local_0;
  *local_devices_1.mutable_xla()->mutable_devices()->add_nodes() = local_1;
  *local_devices_2.mutable_xla()->mutable_devices()->add_nodes() = local_2;

  // Each task sends its device info.
  CoordinationServiceDeviceInfo cluster_devices;
  coord_service->WaitForAllTasks(task_0, local_devices_0,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_1, local_devices_1,
                                 [&](Status s) { TF_ASSERT_OK(s); });
  coord_service->WaitForAllTasks(task_2, local_devices_2, [&](Status s) {
    TF_ASSERT_OK(s);
    // Gather the cluster device info.
    cluster_devices = coord_service->ListClusterDevices();
    n.Notify();
  });
  n.WaitForNotification();

  CoordinationServiceDeviceInfo expected_cluster_devices;
  local_0.mutable_devices(0)->set_global_device_id(0);
  local_0.mutable_devices(1)->set_global_device_id(1);
  local_1.mutable_devices(0)->set_global_device_id(2);
  local_2.mutable_devices(0)->set_global_device_id(3);
  *expected_cluster_devices.mutable_xla()->mutable_devices()->add_nodes() =
      local_0;
  *expected_cluster_devices.mutable_xla()->mutable_devices()->add_nodes() =
      local_1;
  *expected_cluster_devices.mutable_xla()->mutable_devices()->add_nodes() =
      local_2;
  EXPECT_THAT(cluster_devices, IgnoringRepeatedFieldOrdering(
                                   EqualsProto(expected_cluster_devices)));
}

TEST_F(CoordinationBarrierTest, Barrier) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;
  absl::Notification n_0;
  absl::Notification n_1;
  absl::Notification n_2;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n_0](Status s) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_18(mht_18_v, 805, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_0 = s;
                                           n_0.Notify();
                                         });
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(1),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_1, &n_1](Status s) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_19(mht_19_v, 814, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_1 = s;
                                           n_1.Notify();
                                         });
  // Make sure barrier has not been exited prematurely.
  EXPECT_FALSE(n_0.HasBeenNotified());
  EXPECT_FALSE(n_1.HasBeenNotified());
  EXPECT_FALSE(n_2.HasBeenNotified());

  // Last task calls the barrier.
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(2),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_2, &n_2](Status s) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_20(mht_20_v, 829, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_2 = s;
                                           n_2.Notify();
                                         });

  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_TRUE(n_1.HasBeenNotified());
  EXPECT_TRUE(n_2.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinationBarrierTest, BarrierWithSubsetOfTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  absl::Notification n_0;
  absl::Notification n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0, &n_0](Status s) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_21(mht_21_v, 856, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_0 = s;
        n_0.Notify();
      });
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_1, &n_1](Status s) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_22(mht_22_v, 866, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_1 = s;
        n_1.Notify();
      });

  // All listed tasks passed the barrier.
  EXPECT_TRUE(n_0.HasBeenNotified());
  EXPECT_TRUE(n_1.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
}

TEST_F(CoordinationBarrierTest, BarrierWithMismatchedTasks) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0](Status s) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_23(mht_23_v, 890, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_0 = s; });
  // task_1's barrier call specified a conflicting set of tasks (task_2 instead
  // of task_0).
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{GetTask(1), GetTask(2)},
      [&barrier_status_1](Status s) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_24(mht_24_v, 899, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_1 = s; });

  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_0));
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_1));
}

TEST_F(CoordinationBarrierTest, BarrierByNonParticipatingTask) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  absl::Notification n_0;
  absl::Notification n_1;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0](Status s) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_25(mht_25_v, 919, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_0 = s; });
  // Task 2 unexpectedly calls a barrier that it is not participating in.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(2),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_1](Status s) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_26(mht_26_v, 927, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_1 = s; });

  // Barrier should fail for all tasks with the unexpected call.
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_0));
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_1));
}

TEST_F(CoordinationBarrierTest, BarrierByNonClusterTask) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  absl::Notification n_0;
  CoordinatedTask unspecified_task;
  unspecified_task.set_job_name("task_from_another_cluster");
  unspecified_task.set_task_id(2);

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), unspecified_task},
      [&barrier_status_0, &n_0](Status s) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_27(mht_27_v, 949, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_0 = s;
        n_0.Notify();
      });
  n_0.WaitForNotification();

  // Barrier should fail with the unexpected participating task argument.
  EXPECT_TRUE(errors::IsInvalidArgument(barrier_status_0));
}

TEST_F(CoordinationBarrierTest, BarrierTimeout) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  Status barrier_status_0;
  absl::Notification n_0;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n_0](Status s) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_28(mht_28_v, 970, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_0 = s;
                                           n_0.Notify();
                                         });

  // Block until user-specified timeout.
  n_0.WaitForNotification();
  EXPECT_TRUE(errors::IsDeadlineExceeded(barrier_status_0));
}

TEST_F(CoordinationBarrierTest, BarrierReturnsPreviousError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(1);
  Status barrier_status_0;
  Status barrier_status_1;
  absl::Notification n_0;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n_0](Status s) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_29(mht_29_v, 992, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_0 = s;
                                           n_0.Notify();
                                         });
  TF_ASSERT_OK(GetCoordinationService()->ReportTaskError(
      GetTask(0), errors::Internal("test_error")));
  // Block until barrier has failed due to task error.
  n_0.WaitForNotification();
  // Same response should be returned immediately.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status_1](Status s) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_30(mht_30_v, 1007, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_1 = s; });

  EXPECT_TRUE(errors::IsInternal(barrier_status_0));
  EXPECT_TRUE(errors::IsInternal(barrier_status_1));
}

TEST_F(CoordinationBarrierTest, BarrierCancelled) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{},
      [&barrier_status](Status s) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_31(mht_31_v, 1024, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status = s; });
  Status cancelled_status =
      GetCoordinationService()->CancelBarrier(barrier_id, GetTask(0));

  EXPECT_TRUE(errors::IsCancelled(barrier_status));
  TF_EXPECT_OK(cancelled_status);
}

TEST_F(CoordinationBarrierTest, CancelNonExistentBarrier) {
  std::string wrong_barrier_id = "wrong_barrier_id";

  // Cancel barrier should fail if non-existent id is specified.
  Status cancelled_status =
      GetCoordinationService()->CancelBarrier(wrong_barrier_id, GetTask(0));

  EXPECT_TRUE(errors::IsNotFound(cancelled_status));
}

TEST_F(CoordinationBarrierTest, CancelAfterBarrierHasPassed) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{},
      [&barrier_status_0](Status s) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_32(mht_32_v, 1055, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_0 = s; });
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status_1](Status s) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_33(mht_33_v, 1062, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_1 = s; });
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(2),
      /*participating_tasks=*/{},
      [&barrier_status_2](Status s) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_34(mht_34_v, 1069, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status_2 = s; });
  // Cancel barrier should fail if barrier has already been passed.
  Status cancelled_status =
      GetCoordinationService()->CancelBarrier(barrier_id, GetTask(0));

  EXPECT_TRUE(errors::IsFailedPrecondition(cancelled_status));
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinationBarrierTest, PassedBarrierReturnsImmediately) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;
  Status barrier_status_repeat;
  absl::Notification n0;
  absl::Notification n1;
  absl::Notification n2;
  absl::Notification n_repeat;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_0, &n0](Status s) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_35(mht_35_v, 1097, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_0 = s;
                                           n0.Notify();
                                         });
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(1),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_1, &n1](Status s) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_36(mht_36_v, 1106, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_1 = s;
                                           n1.Notify();
                                         });
  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(2),
                                         /*participating_tasks=*/{},
                                         [&barrier_status_2, &n2](Status s) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_37(mht_37_v, 1115, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status_2 = s;
                                           n2.Notify();
                                         });
  // Repeated call should return the same result.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status_repeat, &n_repeat](Status s) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_38(mht_38_v, 1126, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_repeat = s;
        n_repeat.Notify();
      });

  EXPECT_TRUE(n0.HasBeenNotified());
  EXPECT_TRUE(n1.HasBeenNotified());
  EXPECT_TRUE(n2.HasBeenNotified());
  EXPECT_TRUE(n_repeat.HasBeenNotified());
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
  TF_EXPECT_OK(barrier_status_repeat);
}

TEST_F(CoordinationBarrierTest, BarrierFailsIfTaskIsAlreadyInError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  // Set task 0 to error state.
  TF_ASSERT_OK(GetCoordinationService()->ReportTaskError(
      GetTask(0), errors::Internal("test_error")));
  Status barrier_status;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{},
      [&barrier_status](Status s) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_39(mht_39_v, 1155, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");
 barrier_status = s; });

  EXPECT_TRUE(errors::IsInternal(barrier_status));
}

TEST_F(CoordinationBarrierTest, BarrierFailsUponTaskError) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  absl::Notification n0;
  Status barrier_status;

  GetCoordinationService()->BarrierAsync(barrier_id, timeout, GetTask(0),
                                         /*participating_tasks=*/{},
                                         [&barrier_status, &n0](Status s) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_40(mht_40_v, 1171, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

                                           barrier_status = s;
                                           n0.Notify();
                                         });
  TF_ASSERT_OK(GetCoordinationService()->ReportTaskError(
      GetTask(0), errors::Internal("test_error")));
  n0.WaitForNotification();

  EXPECT_TRUE(errors::IsInternal(barrier_status));
}

TEST_F(CoordinationBarrierTest,
       BarrierStillBlocksIfSameTaskCallsOngoingBarrierRepeatedly) {
  const std::string barrier_id = "barrier_id";
  absl::Duration timeout = absl::Seconds(5);
  Status barrier_status_0;
  Status barrier_status_1;
  Status barrier_status_2;
  absl::Notification n_0;
  absl::Notification n_1;
  absl::Notification n_2;

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_0, &n_0](Status s) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_41(mht_41_v, 1199, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_0 = s;
        n_0.Notify();
      });
  // Duplicate call.
  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(0),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_1, &n_1](Status s) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_42(mht_42_v, 1210, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_1 = s;
        n_1.Notify();
      });
  // All listed tasks passed the barrier.
  EXPECT_FALSE(n_0.HasBeenNotified());
  EXPECT_FALSE(n_1.HasBeenNotified());

  GetCoordinationService()->BarrierAsync(
      barrier_id, timeout, GetTask(1),
      /*participating_tasks=*/{GetTask(0), GetTask(1)},
      [&barrier_status_2, &n_2](Status s) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_43(mht_43_v, 1224, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status_2 = s;
        n_2.Notify();
      });
  TF_EXPECT_OK(barrier_status_0);
  TF_EXPECT_OK(barrier_status_1);
  TF_EXPECT_OK(barrier_status_2);
}

TEST_F(CoordinateTwoTasksTest, ResetAndRegisterAgain) {
  EnableCoordinationService();
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  TF_EXPECT_OK(coord_service_->ResetTask(task_0_));

  // Task should be allowed to register again after being reset.
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
}

TEST_F(CoordinateTwoTasksTest, Reset_HeartbeatsAreAcceptedForAGracePeriod) {
  EnableCoordinationService();
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  TF_EXPECT_OK(coord_service_->ResetTask(task_0_));
  // Heartbeat should be allowed for a short grace period after reset.
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(task_0_, incarnation_0_));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
}

TEST_F(CoordinateTwoTasksTest, Reset_FailsOngoingBarrier) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  Status barrier_status;
  absl::Notification barrier_n;
  coord_service_->BarrierAsync(
      "ongoing_barrier", absl::InfiniteDuration(), task_0_,
      /*participating_tasks=*/{}, [&barrier_status, &barrier_n](Status s) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_44(mht_44_v, 1270, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status = s;
        barrier_n.Notify();
      });

  TF_EXPECT_OK(coord_service_->ResetTask(task_0_));

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_TRUE(errors::IsInternal(barrier_status)) << barrier_status;
}

TEST_F(CoordinateTwoTasksTest, Shutdown_HeartbeatsAreAcceptedForAGracePeriod) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(task_0_, [&n](Status s) {
    TF_EXPECT_OK(s);
    n.Notify();
  });
  n.WaitForNotification();

  // Heartbeat should be allowed for a short grace period after shutdown.
  TF_EXPECT_OK(coord_service_->RecordHeartbeat(task_0_, incarnation_0_));

  // Heartbeat failure should be triggered for disconnected task after grace
  // period.
  Env::Default()->SleepForMicroseconds(
      absl::ToInt64Microseconds(3 * kHeartbeatTimeout));
  EXPECT_TRUE(errors::IsInvalidArgument(
      coord_service_->RecordHeartbeat(task_0_, incarnation_0_)));
}

TEST_F(CoordinateTwoTasksTest, Shutdown_FailsOngoingBarrier) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/false);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  Status barrier_status;
  absl::Notification barrier_n;
  coord_service_->BarrierAsync(
      "ongoing_barrier", absl::InfiniteDuration(), task_0_,
      /*participating_tasks=*/{}, [&barrier_status, &barrier_n](Status s) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScoordinationPScoordination_service_testDTcc mht_45(mht_45_v, 1316, "", "./tensorflow/core/distributed_runtime/coordination/coordination_service_test.cc", "lambda");

        barrier_status = s;
        barrier_n.Notify();
      });

  absl::Notification shutdown_n;
  coord_service_->ShutdownTaskAsync(task_0_, [&shutdown_n](Status s) {
    TF_EXPECT_OK(s);
    shutdown_n.Notify();
  });
  shutdown_n.WaitForNotification();

  // Ongoing barrier should fail with error after shutdown.
  EXPECT_TRUE(barrier_n.HasBeenNotified());
  EXPECT_TRUE(errors::IsInternal(barrier_status)) << barrier_status;
}

TEST_F(CoordinateTwoTasksTest, ShutdownWithBarrier_BarrierSucceeds) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  Status barrier_status;
  Status barrier_status_2;

  coord_service_->ShutdownTaskAsync(
      task_0_, [&barrier_status](Status s) { barrier_status = s; });
  coord_service_->ShutdownTaskAsync(
      task_1_, [&barrier_status_2](Status s) { barrier_status_2 = s; });

  TF_EXPECT_OK(barrier_status);
  TF_EXPECT_OK(barrier_status_2);

  // Confirm that both tasks have disconnected.
  // Note: this should not happen in prod where RegisterTask() is called after
  // Shutdown(), which is prevented by agent-side logic.
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
}

TEST_F(CoordinateTwoTasksTest,
       ShutdownWithBarrier_BarrierFails_TaskDisconnectsOtherTaskIsAlerted) {
  EnableCoordinationService(/*has_service_to_client_connection=*/true,
                            /*enable_shutdown_barrier=*/true);
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));
  TF_EXPECT_OK(coord_service_->RegisterTask(task_1_, incarnation_1_));
  Status barrier_status;

  absl::Notification n;
  coord_service_->ShutdownTaskAsync(task_0_, [&n, &barrier_status](Status s) {
    barrier_status = s;
    n.Notify();
  });
  // Block until barrier times out.
  n.WaitForNotification();

  EXPECT_TRUE(errors::IsDeadlineExceeded(barrier_status)) << barrier_status;
  // Confirm that task_0_ has disconnected.
  // Note: this should not happen in prod where RegisterTask() is called after
  // Shutdown(), which is prevented by agent-side logic.
  TF_EXPECT_OK(coord_service_->RegisterTask(task_0_, incarnation_0_));

  // Other task is alerted that shutdown has been initiated without it.
  Status other_task_status = client_1_.GetStatus();
  EXPECT_TRUE(errors::IsInternal(other_task_status)) << other_task_status;
}
}  // namespace tensorflow
