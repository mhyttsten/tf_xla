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
class MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"

#include <atomic>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

#define NUM_DEVS 3

class CollectiveParamResolverLocalTest : public ::testing::Test {
 protected:
  CollectiveParamResolverLocalTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "CollectiveParamResolverLocalTest");

    ConfigProto cp;
    SessionOptions options;
    task_name_ = "/job:localhost/replica:0/task:0";
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name_, &devices));
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    drl_.reset(new DeviceResolverLocal(device_mgr_.get()));
    ResetParamResolver(ConfigProto());
  }

  void ResetParamResolver(const ConfigProto& config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_1(mht_1_v, 228, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "ResetParamResolver");

    prl_.reset(new CollectiveParamResolverLocal(
        config, device_mgr_.get(), drl_.get(), /*nccl_communicator*/ nullptr,
        task_name_));
  }

  void RunCompleteDefaultRanking(
      CollGroupParams group, const std::vector<int32>& gpu_ring_order,
      const std::vector<string>& expected_device_order) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "RunCompleteDefaultRanking");

    ConfigProto config;
    if (!gpu_ring_order.empty()) {
      config.mutable_gpu_options()
          ->mutable_experimental()
          ->set_collective_ring_order(absl::StrJoin(gpu_ring_order, ","));
    }
    ResetParamResolver(config);
    prl_->CompleteDefaultRanking(&group);
    std::vector<string> actual_device_order;
    for (const CollGroupMember& member : group.members) {
      actual_device_order.push_back(member.device.name());
    }
    EXPECT_EQ(actual_device_order, expected_device_order);
  }

  DeviceAttributes GetDeviceAttributes(const string& device_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "GetDeviceAttributes");

    Device* device = nullptr;
    TF_CHECK_OK(device_mgr_->LookupDevice(device_name, &device));
    return device->attributes();
  }

  string task_name_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
  std::unique_ptr<CollectiveParamResolverLocal> prl_;
};

TEST_F(CollectiveParamResolverLocalTest, CompleteDefaultRanking) {
  constexpr int kNumGpus = 8;
  CollGroupParams group;
  group.device_type = DeviceType("GPU");
  group.num_tasks = 1;
  group.group_size = kNumGpus;
  std::unordered_set<int> clique1 = {0, 1, 6, 7};
  for (int gpu_idx = 0; gpu_idx < kNumGpus; ++gpu_idx) {
    CollGroupMember member;
    member.task = "/job:localhost/replica:0/task:0";
    member.device.set_name(strings::StrCat(
        "/job:localhost/replica:0/task:0/device:GPU:", gpu_idx));
    // Build localities so that 0,1,6,7 and 2,3,4,5 form 2 strongly connected
    // components.  Across components, connect 3 and 7.
    for (int link_idx = 0; link_idx < kNumGpus; ++link_idx) {
      if (gpu_idx == link_idx) continue;
      bool gpu_in_clique1 = clique1.find(gpu_idx) != clique1.end();
      bool link_in_clique1 = clique1.find(link_idx) != clique1.end();
      if ((gpu_in_clique1 && link_in_clique1) ||
          (!gpu_in_clique1 && !link_in_clique1)) {
        LocalLinks* links = member.device.mutable_locality()->mutable_links();
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(link_idx);
        ilink->set_strength(2);
      } else if ((gpu_idx == 3 && link_idx == 7) ||
                 (gpu_idx == 7 && link_idx == 3)) {
        LocalLinks* links = member.device.mutable_locality()->mutable_links();
        InterconnectLink* ilink = links->add_link();
        ilink->set_device_id(link_idx);
        ilink->set_strength(1);
      }
    }
    group.members.push_back(member);
  }
  RunCompleteDefaultRanking(group, {1, 3, 5, 7, 6, 4, 2, 0},
                            {
                                "/job:localhost/replica:0/task:0/device:GPU:1",
                                "/job:localhost/replica:0/task:0/device:GPU:3",
                                "/job:localhost/replica:0/task:0/device:GPU:5",
                                "/job:localhost/replica:0/task:0/device:GPU:7",
                                "/job:localhost/replica:0/task:0/device:GPU:6",
                                "/job:localhost/replica:0/task:0/device:GPU:4",
                                "/job:localhost/replica:0/task:0/device:GPU:2",
                                "/job:localhost/replica:0/task:0/device:GPU:0",
                            });
  RunCompleteDefaultRanking(group, {7, 6, 5, 4, 3, 2, 1, 0},
                            {
                                "/job:localhost/replica:0/task:0/device:GPU:7",
                                "/job:localhost/replica:0/task:0/device:GPU:6",
                                "/job:localhost/replica:0/task:0/device:GPU:5",
                                "/job:localhost/replica:0/task:0/device:GPU:4",
                                "/job:localhost/replica:0/task:0/device:GPU:3",
                                "/job:localhost/replica:0/task:0/device:GPU:2",
                                "/job:localhost/replica:0/task:0/device:GPU:1",
                                "/job:localhost/replica:0/task:0/device:GPU:0",
                            });
  // With no gpu_ring_order passed, automatic link detection should kick in.
  // Starting at dev 0, the best order would be: 0,1,6,7,3,2,4,5
  RunCompleteDefaultRanking(group, {},
                            {
                                "/job:localhost/replica:0/task:0/device:GPU:0",
                                "/job:localhost/replica:0/task:0/device:GPU:1",
                                "/job:localhost/replica:0/task:0/device:GPU:6",
                                "/job:localhost/replica:0/task:0/device:GPU:7",
                                "/job:localhost/replica:0/task:0/device:GPU:3",
                                "/job:localhost/replica:0/task:0/device:GPU:2",
                                "/job:localhost/replica:0/task:0/device:GPU:4",
                                "/job:localhost/replica:0/task:0/device:GPU:5",
                            });
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsReduction1Task) {
  CollectiveParams* cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    cps[i] = new CollectiveParams();
    CollectiveParams* cp = cps[i];
    cp->group.group_key = 1;
    cp->group.group_size = 3;
    cp->group.device_type = DeviceType("CPU");
    cp->group.num_tasks = 1;
    cp->instance.instance_key = 7;
    cp->instance.type = REDUCTION_COLLECTIVE;
    cp->instance.data_type = DataType(DT_FLOAT);
    cp->instance.shape = TensorShape({5});
    cp->instance.impl_details.subdiv_offsets.push_back(0);
    cp->is_source = false;
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
                                nullptr /*CancellationManager*/,
                                [&statuses, &note, i](const Status& s) {
                                  statuses[i] = s;
                                  note[i].Notify();
                                });
    });
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    note[i].WaitForNotification();
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    TF_ASSERT_OK(statuses[i]);
    ASSERT_EQ(cps[i]->group.members.size(), 3);
    for (int j = 0; j < NUM_DEVS; ++j) {
      EXPECT_EQ(
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", j),
          cps[i]->group.members[j].device.name());
      EXPECT_TRUE(cps[i]->group.members[j].is_local);
    }
    EXPECT_EQ(cps[i]->instance.impl_details.subdiv_source_rank.size(), 0);
    EXPECT_FALSE(cps[i]->is_source);
    EXPECT_EQ(cps[i]->default_rank, i);
    EXPECT_TRUE(cps[i]->group.same_num_devices_per_task);
    cps[i]->Unref();
  }
}

void InitializeCollectiveParamsForBroadcast(int instance_key, int device_idx,
                                            bool is_source,
                                            CollectiveParams* cp) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_4(mht_4_v, 395, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "InitializeCollectiveParamsForBroadcast");

  cp->group.group_key = 1;
  cp->group.group_size = 3;
  cp->group.device_type = DeviceType("CPU");
  cp->group.num_tasks = 1;
  cp->instance.instance_key = instance_key;
  cp->instance.type = BROADCAST_COLLECTIVE;
  cp->instance.data_type = DataType(DT_FLOAT);
  cp->instance.shape = TensorShape({5});
  cp->instance.impl_details.subdiv_offsets.push_back(0);
  cp->is_source = is_source;
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcast1Task) {
  constexpr int kInstanceKey = 5;
  CollectiveParams* cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    cps[i] = new CollectiveParams();
    CollectiveParams* cp = cps[i];
    InitializeCollectiveParamsForBroadcast(kInstanceKey, i, i == 1, cp);
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
                                nullptr /*CancellationManager*/,
                                [&statuses, &note, i](const Status& s) {
                                  statuses[i] = s;
                                  note[i].Notify();
                                });
    });
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    note[i].WaitForNotification();
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    TF_ASSERT_OK(statuses[i]);
    ASSERT_EQ(cps[i]->group.members.size(), 3);
    for (int j = 0; j < NUM_DEVS; ++j) {
      EXPECT_EQ(
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", j),
          cps[i]->group.members[j].device.name());
      EXPECT_TRUE(cps[i]->group.members[j].is_local);
    }
    EXPECT_EQ(cps[i]->is_source, (i == 1));
    EXPECT_EQ(cps[i]->default_rank, i);
    EXPECT_TRUE(cps[i]->group.same_num_devices_per_task);
    cps[i]->Unref();
  }
}

// If we don't mark any participant in a broadcast as the source, we essentially
// create a collective group with only broadcast recvs.  In that case, we should
// get an internal error from param resolution.
TEST_F(CollectiveParamResolverLocalTest, CompleteParamsBroadcastForgotSender) {
  constexpr int kInstanceKey = 8;
  CollectiveParams* cps[NUM_DEVS];
  Status statuses[NUM_DEVS];
  Notification note[NUM_DEVS];
  for (int i = 0; i < NUM_DEVS; ++i) {
    cps[i] = new CollectiveParams();
    CollectiveParams* cp = cps[i];
    InitializeCollectiveParamsForBroadcast(kInstanceKey, i, false, cp);
    Env::Default()->SchedClosure([this, i, cp, &note, &statuses]() {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
                                nullptr /*CancellationManager*/,
                                [&statuses, &note, i](const Status& s) {
                                  statuses[i] = s;
                                  note[i].Notify();
                                });
    });
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    note[i].WaitForNotification();
  }
  for (int i = 0; i < NUM_DEVS; ++i) {
    EXPECT_EQ(statuses[i].code(), error::INTERNAL);
    EXPECT_EQ(statuses[i].error_message(),
              strings::StrCat(
                  "Instance ", kInstanceKey,
                  " found no source for broadcast.  This could mean that there"
                  " were group_size=",
                  NUM_DEVS, " BcastRecvs but no BcastSend."));
    cps[i]->Unref();
  }
}

CollectiveParams* MakeCollectiveParams(int group_key, int instance_key,
                                       bool is_source) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_5(mht_5_v, 489, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "MakeCollectiveParams");

  auto* cp = new CollectiveParams();
  cp->group.group_key = group_key;
  cp->group.group_size = NUM_DEVS;
  cp->group.device_type = DeviceType("CPU");
  cp->group.num_tasks = 1;
  cp->instance.instance_key = instance_key;
  // CompleteInstanceLocal only waits for the group for broadcasts.
  // Testing with broadcasts yields better coverage.
  cp->instance.type = BROADCAST_COLLECTIVE;
  cp->is_source = is_source;
  return cp;
}

TEST_F(CollectiveParamResolverLocalTest, AbortPendingGroup) {
  CancellationManager cancel_mgr;
  std::vector<CollectiveParams*> cp(NUM_DEVS - 1);
  BlockingCounter start(NUM_DEVS - 1);
  BlockingCounter done(NUM_DEVS - 1);
  for (int i = 0; i < NUM_DEVS - 1; ++i) {
    Env::Default()->SchedClosure([this, i, &cancel_mgr, &cp, &start, &done] {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      cp[i] = MakeCollectiveParams(/*group_key*/ 100, /*instance_key*/ 100,
                                   /*is_source*/ i == 0);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i], &cancel_mgr,
                                [&done, cp = cp[i]](const Status& s) {
                                  EXPECT_EQ(s.code(), error::ABORTED);
                                  EXPECT_EQ(s.error_message(), "__aborted__");
                                  done.DecrementCount();
                                  cp->Unref();
                                });
      start.DecrementCount();
    });
  }
  start.Wait();
  prl_->StartAbort(Status(error::ABORTED, "__aborted__"));
  done.Wait();
}

TEST_F(CollectiveParamResolverLocalTest, AbortPendingInstance) {
  CancellationManager cancel_mgr;
  std::vector<CollectiveParams*> cp(NUM_DEVS);
  int group_key = 100;
  int instance_key = 100;
  // First do a normal CompleteParamsAsync to complete the group;
  {
    BlockingCounter done(NUM_DEVS);
    for (int i = 0; i < NUM_DEVS; ++i) {
      Env::Default()->SchedClosure([this, group_key, instance_key, i,
                                    &cancel_mgr, &cp, &done] {
        string device =
            strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
        cp[i] = MakeCollectiveParams(group_key, instance_key,
                                     /*is_source*/ i == 0);
        prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i],
                                  &cancel_mgr,
                                  [&done, cp = cp[i]](const Status& s) {
                                    EXPECT_EQ(s.code(), error::OK);
                                    done.DecrementCount();
                                    cp->Unref();
                                  });
      });
    }
    done.Wait();
  }
  BlockingCounter start(NUM_DEVS - 1);
  BlockingCounter done(NUM_DEVS - 1);
  for (int i = 0; i < NUM_DEVS - 1; ++i) {
    Env::Default()->SchedClosure([this, group_key, instance_key, i, &cancel_mgr,
                                  &cp, &start, &done] {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      cp[i] = MakeCollectiveParams(group_key, instance_key + 1,
                                   /*is_source*/ i == 0);
      prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i], &cancel_mgr,
                                [&done, cp = cp[i]](const Status& s) {
                                  EXPECT_EQ(s.code(), error::ABORTED);
                                  EXPECT_EQ(s.error_message(), "__aborted__");
                                  done.DecrementCount();
                                  cp->Unref();
                                });
      start.DecrementCount();
    });
  }
  start.Wait();
  prl_->StartAbort(Status(error::ABORTED, "__aborted__"));
  done.Wait();
}

TEST_F(CollectiveParamResolverLocalTest, CompleteParamsAfterAbortion) {
  CancellationManager cancel_mgr;
  int group_key = 100;
  int instance_key = 100;
  // First do a normal CompleteParamsAsync to complete the group;
  {
    std::vector<CollectiveParams*> cp(NUM_DEVS);
    BlockingCounter done(NUM_DEVS);
    for (int i = 0; i < NUM_DEVS; ++i) {
      Env::Default()->SchedClosure([this, group_key, instance_key, i,
                                    &cancel_mgr, &cp, &done] {
        string device =
            strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
        cp[i] = MakeCollectiveParams(group_key, instance_key,
                                     /*is_source*/ i == 0);
        prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp[i],
                                  &cancel_mgr,
                                  [&done, cp = cp[i]](const Status& s) {
                                    EXPECT_EQ(s.code(), error::OK);
                                    done.DecrementCount();
                                    cp->Unref();
                                  });
      });
    }
    done.Wait();
  }
  prl_->StartAbort(Status(error::ABORTED, "__aborted__"));

  auto complete_params = [this, &cancel_mgr](int group_key, int instance_key) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePScollective_param_resolver_local_testDTcc mht_6(mht_6_v, 610, "", "./tensorflow/core/common_runtime/collective_param_resolver_local_test.cc", "lambda");

    string device = "/job:localhost/replica:0/task:0/device:CPU:0";
    Notification done;
    auto* cp = MakeCollectiveParams(group_key, instance_key,
                                    /*is_source*/ true);
    core::ScopedUnref unref(cp);
    prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp, &cancel_mgr,
                              [&done](const Status& s) {
                                EXPECT_EQ(s.code(), error::ABORTED);
                                EXPECT_EQ(s.error_message(), "__aborted__");
                                done.Notify();
                              });
    done.WaitForNotification();
  };
  // It should error without waiting for the all following combinations:
  // - existing group, existing instance
  complete_params(group_key, instance_key);
  // - existing group, new instance
  complete_params(group_key, instance_key + 1);
  // - new group, new instance
  complete_params(group_key + 1, instance_key + 1);
}

TEST_F(CollectiveParamResolverLocalTest, AbortNormalCompleteParamsAsync) {
  // The concurrent nature makes it hard to test abortion, which can happen at
  // any moment. We don't have good options to inject control points into the
  // code to explicitly test every possible scenarios, so we run the test for
  // many times to have a better chance to cover different cases.
  CancellationManager cancel_mgr;
  std::atomic<int64_t> num_ok{0};
  for (int cnt = 0; cnt < 100; ++cnt) {
    // Launching threads that keep doing CompleteInstanceLocal.
    BlockingCounter done(NUM_DEVS);
    for (int i = 0; i < NUM_DEVS; ++i) {
      string device =
          strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i);
      Env::Default()->SchedClosure(
          [this, i, device, &num_ok, &cancel_mgr, &done] {
            int key = 100;
            while (true) {
              Status status;
              Notification n;
              auto* cp =
                  MakeCollectiveParams(/* group_key*/ key, /*instance_key*/ key,
                                       /*is_source*/ i == 0);
              prl_->CompleteParamsAsync(GetDeviceAttributes(device), cp,
                                        &cancel_mgr,
                                        [&status, &n](const Status& s) {
                                          status = s;
                                          n.Notify();
                                        });
              n.WaitForNotification();
              cp->Unref();
              // The status should be either OK or the aborted status.
              if (!status.ok()) {
                EXPECT_EQ(status.code(), error::ABORTED);
                EXPECT_EQ(status.error_message(), "__aborted__");
                done.DecrementCount();
                return;
              }
              ++num_ok;
              ++key;
            }
          });
    }
    // Introduce a random delay up to 50ms, so that we're more likely to abort
    // on different code points each time.
    int64_t delay_ms = random::New64() % 50000;
    Env::Default()->SleepForMicroseconds(delay_ms);
    prl_->StartAbort(Status(error::ABORTED, "__aborted__"));
    done.Wait();
    ResetParamResolver(ConfigProto());
  }
  // There should be at least a few successes, otherwise the delay may be too
  // short and may not cover certain stages of param resolution.
  EXPECT_GT(num_ok.load(), 50);
}

}  // namespace tensorflow
