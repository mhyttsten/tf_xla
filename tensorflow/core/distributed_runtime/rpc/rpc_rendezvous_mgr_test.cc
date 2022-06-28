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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// string -> Tensor<string>
Tensor V(const string& content) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("content: \"" + content + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "V");

  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<tstring>()() = content;
  return tensor;
}

// Tensor<string> -> string
string V(const Tensor& tensor) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "V");

  CHECK_EQ(tensor.dtype(), DT_STRING);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<tstring>()();
}

Rendezvous::ParsedKey MakeKey(const string& s) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_2(mht_2_v, 224, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "MakeKey");

  Rendezvous::ParsedKey key;
  CHECK(Rendezvous::ParseKey(s, &key).ok());
  return key;
}

namespace {
// A dummy worker interface implementation that simply triggers the callback
// with OK status for RecvTensor request.
class DummyWorker : public TestWorkerInterface {
 public:
  void RecvTensorAsync(CallOptions* opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "RecvTensorAsync");

    SchedClosure([done = std::move(done)]() {
      // Simulate a random delay for RPC. This is needed to fill the entire
      // object buffer in `RpcRecvTensorFreeList` and trigger the destruction of
      // RPC call objects.
      const int64_t t_us = random::New64() % 100 * 1000;
      Env::Default()->SleepForMicroseconds(t_us);
      done(Status::OK());
    });
  }
};

// Fake cache implementation for WorkerEnv.
class DummyWorkerCache : public WorkerCacheInterface {
  void ListWorkers(std::vector<string>* workers) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_4(mht_4_v, 256, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "ListWorkers");
}
  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_5(mht_5_v, 262, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "ListWorkersInJob");
}
  WorkerInterface* GetOrCreateWorker(const string& target) override {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_6(mht_6_v, 267, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "GetOrCreateWorker");

    if (dummy_remote_worker_ == nullptr) {
      // Ownership transferred to WorkerFreeList
      dummy_remote_worker_ = new DummyWorker;
    }
    return dummy_remote_worker_;
  }
  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_7(mht_7_v, 278, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "GetEagerClientCache");

    return errors::Unimplemented("Unimplemented.");
  }
  Status GetCoordinationClientCache(
      std::unique_ptr<CoordinationClientCache>* coord_client_cache) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_8(mht_8_v, 285, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "GetCoordinationClientCache");

    return errors::Unimplemented("Unimplemented.");
  }
  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_9(mht_9_v, 293, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "GetDeviceLocalityNonBlocking");

    return false;
  }
  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_10(mht_10_v, 301, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "GetDeviceLocalityAsync");
}

 private:
  DummyWorker* dummy_remote_worker_ = nullptr;
};

static Device* CreateDevice(const char* type, const char* name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("type: \"" + (type == nullptr ? std::string("nullptr") : std::string((char*)type)) + "\"");
   mht_11_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_11(mht_11_v, 312, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "CreateDevice");

  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_12(mht_12_v, 318, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_13(mht_13_v, 322, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_14(mht_14_v, 326, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "GetAllocator");
 return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

static DeviceMgr* CreateDeviceMgr() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_15(mht_15_v, 337, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "CreateDeviceMgr");

  std::unique_ptr<Device> d0(
      CreateDevice("CPU", "/job:mnist/replica:1/task:2/cpu:1"));
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  return new StaticDeviceMgr(std::move(devices));
}
}  // namespace

class RpcRendezvousMgrTest : public ::testing::Test {
 protected:
  RpcRendezvousMgrTest()
      : cache_(new DummyWorkerCache),
        worker_session_("rpc_session", "/job:mnist/replica:1/task:2",
                        std::unique_ptr<WorkerCacheInterface>(cache_),
                        std::unique_ptr<DeviceMgr>(CreateDeviceMgr()),
                        std::unique_ptr<GraphMgr>(), nullptr),
        rmgr_(&env) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_16(mht_16_v, 357, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "RpcRendezvousMgrTest");

    env.env = Env::Default();
  }

  DummyWorkerCache* cache_;  // Managed by worker_session.
  WorkerEnv env;

  WorkerSession worker_session_;
  RpcRendezvousMgr rmgr_;
};

TEST_F(RpcRendezvousMgrTest, LocalSendRecv) {
  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    RemoteRendezvous* rendez = rmgr_.Find(step_id);
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
  }
  {
    Tensor val(DT_FLOAT);
    bool val_dead = false;
    TF_ASSERT_OK(rmgr_.RecvLocal(step_id, key, &val, &val_dead));
    EXPECT_EQ(V(val), "peach");
  }
  rmgr_.Cleanup(step_id);
}

TEST_F(RpcRendezvousMgrTest, LocalAbort) {
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {  // Explicit Abort().
    const int64_t step_id = 123;
    RemoteRendezvous* rendez = rmgr_.Find(step_id);
    core::ScopedUnref unref(rendez);
    SchedClosure([this, rendez]() {
      env.env->SleepForMicroseconds(100 * 1000);
      rendez->StartAbort(errors::Aborted(""));
    });
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    EXPECT_TRUE(errors::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
  {  // Cleanup causes Abort().
    const int64_t step_id = 321;
    RemoteRendezvous* rendez = rmgr_.Find(step_id);
    core::ScopedUnref unref(rendez);
    SchedClosure([this, step_id]() {
      env.env->SleepForMicroseconds(100 * 1000);
      rmgr_.Cleanup(step_id);
    });
    Tensor val(DT_STRING);
    bool val_dead = false;
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    EXPECT_TRUE(errors::IsAborted(rendez->Recv(key, args, &val, &val_dead)));
  }
}

TEST_F(RpcRendezvousMgrTest, LocalCancel) {
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  auto* cm = new CancellationManager();
  const int64_t step_id = 123;
  RemoteRendezvous* rendez = rmgr_.Find(step_id);
  core::ScopedUnref unref(rendez);
  Notification n;
  SchedClosure([this, cm, &n]() {
    env.env->SleepForMicroseconds(100 * 1000);
    cm->StartCancel();
    n.Notify();
  });
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  TF_ASSERT_OK(rendez->Initialize(&worker_session_));
  EXPECT_TRUE(errors::IsCancelled(rendez->Recv(key, args, &val, &val_dead)));
  n.WaitForNotification();
  delete cm;
}

TEST_F(RpcRendezvousMgrTest, CancelAfterReceived) {
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  auto* cm = new CancellationManager();
  const int64_t step_id = 123;
  RemoteRendezvous* rendez = rmgr_.Find(step_id);
  core::ScopedUnref unref(rendez);
  Notification n;
  SchedClosure([this, rendez, key, cm, &n]() {
    env.env->SleepForMicroseconds(100 * 1000);
    TF_ASSERT_OK(rendez->Send(key, Rendezvous::Args(), V("peach"), false));
    cm->StartCancel();
    n.Notify();
  });
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  TF_ASSERT_OK(rendez->Initialize(&worker_session_));
  TF_ASSERT_OK(rendez->Recv(key, args, &val, &val_dead));
  EXPECT_EQ(V(val), "peach");
  n.WaitForNotification();
  delete cm;
}

namespace {
class DummyDeviceContext : public DeviceContext {
 public:
  explicit DummyDeviceContext(int stream_id) : stream_id_(stream_id) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_17(mht_17_v, 479, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "DummyDeviceContext");
}
  ~DummyDeviceContext() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_18(mht_18_v, 483, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "~DummyDeviceContext");
}
  int stream_id() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSrpc_rendezvous_mgr_testDTcc mht_19(mht_19_v, 487, "", "./tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr_test.cc", "stream_id");
 return stream_id_; }

 private:
  const int stream_id_;
};
}  // namespace

TEST_F(RpcRendezvousMgrTest, TransferDummyDeviceContext) {
  DummyDeviceContext* dc = new DummyDeviceContext(123);

  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    RemoteRendezvous* rendez = rmgr_.Find(step_id);
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;
    args.device_context = dc;
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    TF_ASSERT_OK(rendez->Send(key, args, V("peach"), false));
  }
  {
    Notification n;
    rmgr_.RecvLocalAsync(
        step_id, key,
        [&n](const Status& s, const Rendezvous::Args send_args,
             const Rendezvous::Args recv_args, const Tensor& val,
             bool is_dead) {
          auto send_dev_context =
              static_cast<DummyDeviceContext*>(send_args.device_context);
          CHECK_EQ(123, send_dev_context->stream_id());
          CHECK_EQ(V(val), "peach");
          n.Notify();
        });
    n.WaitForNotification();
  }
  rmgr_.Cleanup(step_id);
  dc->Unref();
}

TEST_F(RpcRendezvousMgrTest, RemoteRecvOne) {
  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:worker/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    RemoteRendezvous* rendez = rmgr_.Find(step_id);
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;

    Tensor val(DT_STRING);
    bool val_dead = false;

    TF_ASSERT_OK(rendez->Recv(key, args, &val, &val_dead));
  }
  rmgr_.Cleanup(step_id);
}

TEST_F(RpcRendezvousMgrTest, RemoteRecvAsyncMany) {
  const int64_t step_id = 123;
  const Rendezvous::ParsedKey key = MakeKey(Rendezvous::CreateKey(
      "/job:worker/replica:1/task:2/cpu:0", 7890,
      "/job:mnist/replica:1/task:2/cpu:1", "foo", FrameAndIter(0, 0)));
  {
    RemoteRendezvous* rendez = rmgr_.Find(step_id);
    TF_ASSERT_OK(rendez->Initialize(&worker_session_));
    core::ScopedUnref unref(rendez);
    Rendezvous::Args args;

    // Send a large number of async RPC requests to fill up the buffer in
    // `RpcRecvTensorFreeList`, in order to test deleting RPC call objects.
    int num_requests = 10000;
    Tensor val(DT_STRING);
    mutex mu_;
    Status status = Status::OK();
    BlockingCounter counter(num_requests);

    for (int i = 0; i < num_requests; i++) {
      rendez->RecvAsync(
          key, args,
          [&mu_, &status, &counter](const Status& s, const Rendezvous::Args&,
                                    const Rendezvous::Args&, const Tensor&,
                                    const bool) {
            {
              mutex_lock l(mu_);
              status.Update(s);
            }
            counter.DecrementCount();
          });
    }
    counter.Wait();
    TF_ASSERT_OK(status);
  }
  rmgr_.Cleanup(step_id);
}

}  // namespace tensorflow
