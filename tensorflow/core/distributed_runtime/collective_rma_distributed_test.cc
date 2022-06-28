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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/collective_rma_distributed.h"

#include "google/protobuf/any.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/transport_options.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

// The only interesting method on CollectiveRemoteAccessDistributed
// that's not on CollectiveRemoteAccessLocal is RecvFromPeer which
// issues a RecvBufAsync call against a WorkerInterface.  That's all
// that's tested here.  Note that RecvFromPeer can do a
// DeviceResolverInterface::GetDeviceLocalityAsync call in preparation
// for the RecvBufAsync.

namespace tensorflow {
namespace {

class FakeAllocator : public Allocator {
 public:
  string Name() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "Name");
 return "fake"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "AllocateRaw");

    return port::AlignedMalloc(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_2(mht_2_v, 229, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "DeallocateRaw");
 return port::AlignedFree(ptr); }
};

static std::unique_ptr<Device> NewDevice(const string& type, const string& name,
                                         Allocator* allocator) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr, Allocator* allocator)
        : Device(nullptr, attr), allocator_(allocator) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_3(mht_3_v, 240, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "FakeDevice");
}
    Status Sync() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "Sync");
 return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_5(mht_5_v, 248, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "GetAllocator");
 return allocator_; }

   private:
    Allocator* const allocator_;
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  attr.mutable_locality()->set_numa_node(3);  // a non-default value
  attr.set_incarnation(random::New64());
  return absl::make_unique<FakeDevice>(attr, allocator);
}

static int64_t kStepId = 123;

class FakeWorker : public TestWorkerInterface {
 public:
  FakeWorker(const string& name, DeviceMgr* dev_mgr,
             DeviceResolverDistributed* dres, bool is_failed,
             bool set_tensor_in_extra)
      : name_(name),
        device_mgr_(dev_mgr),
        device_resolver_(dres),
        buf_rendezvous_(kStepId, dev_mgr),
        is_failed_(is_failed),
        set_tensor_in_extra_(set_tensor_in_extra) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_6(mht_6_v, 277, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "FakeWorker");
}

  // Direct access to a BufRendezvous that holds whatever the remote
  // worker is supposed to have.
  BufRendezvous* buf_rendezvous() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_7(mht_7_v, 284, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "buf_rendezvous");
 return &buf_rendezvous_; }

  void GetStatusAsync(CallOptions* opts, const GetStatusRequest* request,
                      GetStatusResponse* response, bool fail_fast,
                      StatusCallback done) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_8(mht_8_v, 291, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "GetStatusAsync");

    if (is_failed_) {
      done(errors::Unavailable("peer down"));
      return;
    }
    std::vector<DeviceAttributes> dev_attr;
    device_mgr_->ListDeviceAttributes(&dev_attr);
    for (const auto& da : dev_attr) {
      *response->add_device_attributes() = da;
    }
    done(Status::OK());
  }

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_9(mht_9_v, 308, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "RecvBufAsync");

    if (is_failed_) {
      done(errors::Unavailable("peer down"));
      return;
    }
    opts->SetCancelCallback([this]() {
      // Within this test the call is satisfied by a process-local
      // BufRendezvous table. In real application the BufRendezvous
      // would be on the other side of a network hop, so call
      // BufRendezvous::StartAbort() from a separate thread to be
      // more consistent with that situation and avoid mutex deadlock.
      SchedClosure([this]() {
        Env::Default()->SleepForMicroseconds(100);
        buf_rendezvous_.StartAbort(errors::Internal("Cancelled"));
      });
    });
    VLOG(2) << "ConsumeBuf key=" << request->buf_rendezvous_key()
            << " src_device=" << request->src_device()
            << " src_incarnation=" << request->src_incarnation();
    buf_rendezvous_.ConsumeBuf(
        request->buf_rendezvous_key(), request->src_device(),
        request->src_incarnation(),
        [this, opts, request, response, done](const Status& status,
                                              BufRendezvous::Hook* h) {
          Status s = status;
          if (s.ok()) {
            opts->ClearCancelCallback();
            int64_t num_bytes = h->prod_value->TotalBytes();

            if (set_tensor_in_extra_) {
              // Since this is not really RDMA into pre-allocated memory send
              // the bytes in the response.
              RecvBufRespExtra extra;
              extra.add_tensor_content(string(
                  reinterpret_cast<const char*>(DMAHelper::base(h->prod_value)),
                  num_bytes));
              response->mutable_transport_options()->PackFrom(extra);
            } else {
              if (request->num_bytes() != num_bytes) {
                s = errors::Internal("Tensor Size Mismatch.");
              } else {
                memcpy(reinterpret_cast<void*>(request->buf_ptr()),
                       DMAHelper::base(h->prod_value), num_bytes);
              }
            }
          }
          done(s);
          if (h) BufRendezvous::DoneWithHook(h);
        },
        nullptr /*cancellation_manager*/);
  }

 private:
  string name_;
  DeviceMgr* device_mgr_;
  DeviceResolverDistributed* device_resolver_;
  BufRendezvous buf_rendezvous_;
  bool is_failed_;
  const bool set_tensor_in_extra_;
};

class FakeCache : public TestWorkerCache {
 public:
  // Override the Locality methods to actually pass through to the
  // worker.
  bool GetDeviceLocalityNonBlocking(const string& device,
                                    DeviceLocality* locality) override {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_10(mht_10_v, 378, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "GetDeviceLocalityNonBlocking");

    return false;
  }

  void GetDeviceLocalityAsync(const string& device, DeviceLocality* locality,
                              StatusCallback done) override {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_11(mht_11_v, 387, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "GetDeviceLocalityAsync");

    string task_name;
    string dev_part;
    if (!DeviceNameUtils::SplitDeviceName(device, &task_name, &dev_part)) {
      done(errors::Internal("failed to parse device name"));
      return;
    }
    auto it = workers_.find(task_name);
    if (it == workers_.end()) {
      done(errors::Internal("failed to find worker ", task_name));
      return;
    }
    WorkerInterface* wi = it->second;
    GetStatusRequest req;
    GetStatusResponse resp;
    Status status = wi->GetStatus(&req, &resp);
    if (!status.ok()) {
      done(status);
      return;
    }
    for (const auto& it : resp.device_attributes()) {
      if (it.name() == device) {
        *locality = it.locality();
        done(Status::OK());
        return;
      }
    }
    done(errors::Internal("device not found: ", device));
  }
};

enum TEST_PARAM_DEVICE_TYPE {
  TEST_PARAM_DEVICE_TYPE_CPU = 0,
  TEST_PARAM_DEVICE_TYPE_GPU,
};

enum TEST_PARAM_TENSOR_LOC {
  TEST_PARAM_TENSOR_LOC_AT_BUF_PTR = 0,
  TEST_PARAM_TENSOR_LOC_IN_EXTRA,
};

class CollRMADistTest
    : public ::testing::TestWithParam<
          std::tuple<TEST_PARAM_DEVICE_TYPE, TEST_PARAM_TENSOR_LOC>> {
 protected:
  CollRMADistTest()
      : work_queue_(
            std::make_shared<UnboundedWorkQueue>(Env::Default(), "test")) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_12(mht_12_v, 437, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "CollRMADistTest");
}

  ~CollRMADistTest() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_13(mht_13_v, 442, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "~CollRMADistTest");

    for (DeviceMgr* dm : device_mgrs_) {
      delete dm;
    }
    for (auto it : dev_resolvers_) {
      delete it.second;
    }
    for (FakeWorker* w : workers_) {
      delete w;
    }
  }

  void SetUp() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_14(mht_14_v, 457, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "SetUp");

    const int num_workers = 2;
    const int num_devices = 1;
    string device_type = "CPU";
    string dev0_worker_name;
    for (int w = 0; w < num_workers; ++w) {
      string name = strings::StrCat("/job:worker/replica:0/task:", w);
      if (w == 0) {
        dev0_worker_name = name;
      }
      DefineWorker(name, device_type, num_devices);
    }
    // All tests simulate requests from worker 0 to worker 1.
    rma_.reset(new CollectiveRemoteAccessDistributed(
        device_mgrs_[0], dev_resolvers_[dev0_worker_name], work_queue_, &wc_,
        kStepId, "/job:worker/replica:0/task:0"));

    const int kNumElts = 8;
    expected_value_ = Tensor(DT_FLOAT, {kNumElts});
    to_tensor_ = Tensor(DT_FLOAT, {kNumElts});
    large_response_ = Tensor(DT_FLOAT, {2 * kNumElts});
    auto exp_alias = expected_value_.flat<float>();
    auto to_alias = to_tensor_.flat<float>();
    auto large_response_alias = large_response_.flat<float>();
    for (int i = 0; i < kNumElts; ++i) {
      exp_alias(i) = i;
      to_alias(i) = -1;
    }
    for (int i = 0; i < 2 * kNumElts; ++i) {
      large_response_alias(i) = -2;
    }
  }

  // Populates all device resolvers with device attributes of the cluster. This
  // should be called in the beginning of all tests unless you would like to
  // simulate a situation that is before parameter resolution.
  void ResolveDeviceAttributes() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_15(mht_15_v, 496, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "ResolveDeviceAttributes");

    for (auto& dev_resolver_item : dev_resolvers_) {
      DeviceResolverDistributed* dev_resolver = dev_resolver_item.second;
      for (const auto& item : dev_by_task_) {
        TF_CHECK_OK(dev_resolver->UpdateDeviceAttributes(item.second));
      }
    }
  }

  void DefineWorker(const string& worker_name, const string& device_type,
                    int num_devices, bool is_failed = false) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("worker_name: \"" + worker_name + "\"");
   mht_16_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_16(mht_16_v, 511, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "DefineWorker");

    std::vector<std::unique_ptr<Device>> devices;
    for (int i = 0; i < num_devices; ++i) {
      devices.push_back(NewDevice(
          device_type,
          strings::StrCat(worker_name, "/device:", device_type, ":", i),
          &fake_allocator_));
    }
    DeviceMgr* dev_mgr = new StaticDeviceMgr(std::move(devices));
    device_mgrs_.push_back(dev_mgr);
    std::vector<DeviceAttributes>* dv = &dev_by_task_[worker_name];
    dv->clear();
    for (auto d : dev_mgr->ListDevices()) {
      dv->push_back(d->attributes());
    }
    DeviceResolverDistributed* dev_res = new DeviceResolverDistributed(dev_mgr);
    dev_resolvers_[worker_name] = dev_res;
    FakeWorker* fw =
        new FakeWorker(worker_name, dev_mgr, dev_res, is_failed,
                       /*set_tensor_in_extra=*/
                       std::get<TEST_PARAM_TENSOR_LOC>(GetParam()) ==
                           TEST_PARAM_TENSOR_LOC_IN_EXTRA);

    workers_.push_back(fw);
    wc_.AddWorker(worker_name, fw);
  }

  void RestartWorker(const string& worker_name, const string& device_type,
                     int num_devices, bool is_failed = false) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("worker_name: \"" + worker_name + "\"");
   mht_17_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_17(mht_17_v, 544, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "RestartWorker");

    auto it = dev_resolvers_.find(worker_name);
    if (it != dev_resolvers_.end()) {
      delete it->second;
      dev_resolvers_.erase(it);
    }
    // After restarting a worker, the other workers already have the device
    // attributes of the old worker. We don't broadcast device attributes of the
    // new worker to mimic the real world.
    DefineWorker(worker_name, device_type, num_devices, is_failed);
  }

  void ValidateResultTensor() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_18(mht_18_v, 559, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "ValidateResultTensor");

    ASSERT_EQ(expected_value_.NumElements(), to_tensor_.NumElements());
    for (int i = 0; i < to_tensor_.NumElements(); ++i) {
      EXPECT_FLOAT_EQ(expected_value_.flat<float>()(i),
                      to_tensor_.flat<float>()(i));
    }
  }

  void ValidateResultTensorUnchanged() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_19(mht_19_v, 570, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "ValidateResultTensorUnchanged");

    for (int i = 0; i < to_tensor_.NumElements(); ++i) {
      EXPECT_FLOAT_EQ(-1, to_tensor_.flat<float>()(i));
    }
  }

  void MaybeSetGPUDevice(Device* dst_device) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePScollective_rma_distributed_testDTcc mht_20(mht_20_v, 579, "", "./tensorflow/core/distributed_runtime/collective_rma_distributed_test.cc", "MaybeSetGPUDevice");

    if (std::get<TEST_PARAM_DEVICE_TYPE>(GetParam()) ==
        TEST_PARAM_DEVICE_TYPE_GPU) {
      dst_device->set_tensorflow_accelerator_device_info(&gpu_device_info_);
    }
  }

  FakeCache wc_;
  CancellationManager cm_;
  std::vector<DeviceMgr*> device_mgrs_;
  std::unordered_map<string, DeviceResolverDistributed*> dev_resolvers_;
  std::unordered_map<string, std::vector<DeviceAttributes>> dev_by_task_;
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  std::vector<FakeWorker*> workers_;
  std::unique_ptr<CollectiveRemoteAccessDistributed> rma_;
  mutex mu_;
  int num_done_ TF_GUARDED_BY(mu_);
  condition_variable done_;
  Tensor expected_value_;
  Tensor large_response_;
  Tensor to_tensor_;
  CallOptions opts_;
  DeviceLocality device_locality_;
  AllocatorAttributes alloc_attr_;
  FakeAllocator fake_allocator_;
  DeviceBase::AcceleratorDeviceInfo gpu_device_info_;
};

TEST_P(CollRMADistTest, ProdFirstOK) {
  ResolveDeviceAttributes();
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string kBufKey = "fake_buf_key";
  wi->buf_rendezvous()->ProvideBuf(
      kBufKey, nullptr /*device*/, nullptr /*dev_ctx*/, &expected_value_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      },
      nullptr /*cancellation_manager*/);
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  DeviceContext* to_device_ctx = nullptr;
  MaybeSetGPUDevice(dst_device);
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      nullptr /*cancellation_manager*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  consumer_note.WaitForNotification();
  TF_EXPECT_OK(consumer_status);
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensor();
}

TEST_P(CollRMADistTest, ConsFirstOK) {
  ResolveDeviceAttributes();
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string kBufKey = "fake_buf_key";
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  MaybeSetGPUDevice(dst_device);
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      nullptr /*cancellation_manager*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  wi->buf_rendezvous()->ProvideBuf(
      kBufKey, nullptr /*device*/, nullptr /*dev_ctx*/, &expected_value_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      },
      nullptr /*cancellation_manager*/);
  consumer_note.WaitForNotification();
  TF_EXPECT_OK(consumer_status);
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensor();
}

TEST_P(CollRMADistTest, ConsFirstAbort) {
  ResolveDeviceAttributes();
  Notification consumer_note;
  Status consumer_status;
  const string kBufKey = "fake_buf_key";
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  MaybeSetGPUDevice(dst_device);
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      nullptr /*cancellation_manager*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  rma_->StartAbort(errors::Internal("Deliberate Failure"));
  consumer_note.WaitForNotification();
  EXPECT_EQ(consumer_status.error_message(), "Cancelled");
}

TEST_P(CollRMADistTest, ResponseTooLarge) {
  ResolveDeviceAttributes();
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string kBufKey = "fake_buf_key";
  wi->buf_rendezvous()->ProvideBuf(
      kBufKey, nullptr /*device*/, nullptr /*dev_ctx*/, &large_response_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      },
      nullptr /*cancellation_manager*/);
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  DeviceContext* to_device_ctx = nullptr;
  MaybeSetGPUDevice(dst_device);
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      kBufKey, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      nullptr /*cancellation_manager*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  consumer_note.WaitForNotification();
  EXPECT_THAT(consumer_status.error_message(),
              ::testing::HasSubstr("Tensor Size Mismatch"));
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensorUnchanged();
}

TEST_P(CollRMADistTest, WorkerRestart) {
  ResolveDeviceAttributes();
  Notification consumer_note;
  Notification producer_note;
  Status consumer_status;
  Status producer_status;
  FakeWorker* wi = workers_[1];
  const string buf_key = "fake_buf_key";
  Device* dst_device = nullptr;
  string dev_name = "CPU:0";
  TF_EXPECT_OK(device_mgrs_[0]->LookupDevice(dev_name, &dst_device));
  MaybeSetGPUDevice(dst_device);
  DeviceContext* to_device_ctx = nullptr;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      buf_key, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      nullptr /*cancellation_manager*/,
      [&consumer_status, &consumer_note](const Status& s) {
        consumer_status = s;
        consumer_note.Notify();
      });
  wi->buf_rendezvous()->ProvideBuf(
      buf_key, nullptr /*device*/, nullptr /*dev_ctx*/, &expected_value_,
      AllocatorAttributes(),
      [&producer_note, &producer_status](const Status& s) {
        producer_status.Update(s);
        producer_note.Notify();
      },
      nullptr /*cancellation_manager*/);
  consumer_note.WaitForNotification();
  TF_EXPECT_OK(consumer_status);
  producer_note.WaitForNotification();
  TF_EXPECT_OK(producer_status);
  ValidateResultTensor();

  // Restart task 1 and check that recv from task 1 to task 0 fails.
  RestartWorker("/job:worker/replica:0/task:1", "CPU", /*num_devices*/ 1);
  Notification post_restart_note;
  rma_->RecvFromPeer(
      "/job:worker/replica:0/task:1/device:" + dev_name,  // peer_dev
      "/job:worker/replica:0/task:1",                     // peer_task
      false,                                              // peer_is_local
      buf_key, dst_device, to_device_ctx, alloc_attr_, &to_tensor_,
      device_locality_, 0 /*dev_to_dev_stream_index*/,
      nullptr /*cancellation_manager*/,
      [&consumer_status, &post_restart_note](const Status& s) {
        consumer_status = s;
        post_restart_note.Notify();
      });
  post_restart_note.WaitForNotification();
  EXPECT_TRUE(errors::IsFailedPrecondition(consumer_status));
}

TEST_P(CollRMADistTest, CheckHealthOKWithCachedAttr) {
  ResolveDeviceAttributes();
  Status check_health_status;
  Notification check_health_done;
  rma_->CheckPeerHealth(
      "/job:worker/replica:0/task:1", /*timeout_in_ms=*/0,
      [&check_health_status, &check_health_done](const Status s) {
        check_health_status = s;
        check_health_done.Notify();
      });
  check_health_done.WaitForNotification();
  TF_EXPECT_OK(check_health_status);
}

TEST_P(CollRMADistTest, CheckHealthOKWithoutCachedAttr) {
  Status check_health_status;
  Notification check_health_done;
  rma_->CheckPeerHealth(
      "/job:worker/replica:0/task:1", /*timeout_in_ms=*/0,
      [&check_health_status, &check_health_done](const Status s) {
        check_health_status = s;
        check_health_done.Notify();
      });
  check_health_done.WaitForNotification();
  EXPECT_TRUE(check_health_status.ok());
}

TEST_P(CollRMADistTest, CheckHealthRestarted) {
  ResolveDeviceAttributes();
  RestartWorker("/job:worker/replica:0/task:1", "CPU", /*num_devices*/ 1);

  Status check_health_status;
  Notification check_health_done;
  rma_->CheckPeerHealth(
      "/job:worker/replica:0/task:1", /*timeout_in_ms=*/0,
      [&check_health_status, &check_health_done](const Status s) {
        check_health_status = s;
        check_health_done.Notify();
      });
  check_health_done.WaitForNotification();
  EXPECT_TRUE(errors::IsFailedPrecondition(check_health_status));
}

TEST_P(CollRMADistTest, CheckHealthFailedPeer) {
  ResolveDeviceAttributes();
  RestartWorker("/job:worker/replica:0/task:1", "CPU", /*num_devices*/ 1,
                /*is_failed*/ true);

  Status check_health_status;
  Notification check_health_done;
  rma_->CheckPeerHealth(
      "/job:worker/replica:0/task:1", /*timeout_in_ms=*/0,
      [&check_health_status, &check_health_done](const Status s) {
        check_health_status = s;
        check_health_done.Notify();
      });
  check_health_done.WaitForNotification();
  EXPECT_TRUE(errors::IsUnavailable(check_health_status));
}

TEST_P(CollRMADistTest, CheckHealthRestartedWithDifferentDevices) {
  ResolveDeviceAttributes();
  RestartWorker("/job:worker/replica:0/task:1", "GPU", /*num_devices*/ 1);
  Status check_health_status;
  Notification check_health_done;
  rma_->CheckPeerHealth(
      "/job:worker/replica:0/task:1", /*timeout_in_ms=*/0,
      [&check_health_status, &check_health_done](const Status s) {
        check_health_status = s;
        check_health_done.Notify();
      });
  check_health_done.WaitForNotification();
  EXPECT_TRUE(errors::IsFailedPrecondition(check_health_status));
}

INSTANTIATE_TEST_SUITE_P(
    TensorInBufPtrOrExtra, CollRMADistTest,
    ::testing::Combine(::testing::Values(TEST_PARAM_TENSOR_LOC_AT_BUF_PTR,
                                         TEST_PARAM_TENSOR_LOC_IN_EXTRA),
                       ::testing::Values(TEST_PARAM_DEVICE_TYPE_CPU,
                                         TEST_PARAM_DEVICE_TYPE_GPU)));

}  // namespace
}  // namespace tensorflow
