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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/remote_device.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

const char* const kSession = "remote_session";

class RemoteDeviceTest : public ::testing::Test {
 protected:
  string remote_name_;
  std::unique_ptr<WorkerCacheInterface> worker_cache_;
  WorkerInterface* wi_;
  std::vector<Device*> devices_;
  std::unique_ptr<test::TestCluster> cluster_;
  std::unique_ptr<GrpcWorkerEnv> grpc_worker_env_;

  RemoteDeviceTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/distributed_runtime/remote_device_test.cc", "RemoteDeviceTest");

    SessionOptions options;
    (*options.config.mutable_device_count())["CPU"] = 2;
    TF_CHECK_OK(test::TestCluster::MakeTestCluster(options, 1, &cluster_));
    const string& hostport = cluster_->targets()[0];
    GrpcChannelSpec spec;
    TF_CHECK_OK(spec.AddHostPortsJob("localhost", {hostport}));
    ChannelCreationFunction channel_func =
        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
    std::shared_ptr<GrpcChannelCache> channel_cache(
        NewGrpcChannelCache(spec, channel_func));
    grpc_worker_env_.reset(CreateGrpcWorkerEnv());
    worker_cache_.reset(
        NewGrpcWorkerCache(channel_cache, grpc_worker_env_.get()));
    remote_name_ = "/job:localhost/replica:0/task:0";
    wi_ = worker_cache_->GetOrCreateWorker(remote_name_);
  }

  ~RemoteDeviceTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/distributed_runtime/remote_device_test.cc", "~RemoteDeviceTest");

    worker_cache_->ReleaseWorker(remote_name_, wi_);
  }

  void SetUp() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc mht_2(mht_2_v, 241, "", "./tensorflow/core/distributed_runtime/remote_device_test.cc", "SetUp");

    Notification n;
    NewRemoteDevices(Env::Default(), worker_cache_.get(), remote_name_,
                     [&n, this](const Status& s, std::vector<Device*>* found) {
                       TF_CHECK_OK(s);
                       devices_ = *found;
                       n.Notify();
                     });
    n.WaitForNotification();
    EXPECT_EQ(devices_.size(), 2);
    std::sort(devices_.begin(), devices_.end(), [](Device* a, Device* b) {
      return a->name().compare(b->name()) < 0;
    });
  }

  void TearDown() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSremote_device_testDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/distributed_runtime/remote_device_test.cc", "TearDown");

    for (auto d : devices_) delete d;
  }
};

TEST_F(RemoteDeviceTest, GetStatus) {
  // We know what the testlib's fake server does.
  EXPECT_EQ(devices_[0]->name(),
            strings::StrCat(remote_name_, "/device:CPU:0"));
  EXPECT_EQ(devices_[0]->attributes().device_type(),
            DeviceType(DEVICE_CPU).type());
  EXPECT_EQ(devices_[0]->attributes().memory_limit(), 256 << 20);
  EXPECT_EQ(devices_[1]->name(),
            strings::StrCat(remote_name_, "/device:CPU:1"));
  EXPECT_EQ(devices_[1]->attributes().memory_limit(), 256 << 20);
}

}  // namespace tensorflow
