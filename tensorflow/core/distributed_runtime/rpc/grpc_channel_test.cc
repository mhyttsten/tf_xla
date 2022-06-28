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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_testDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
#define IsSameAddrSp DeviceNameUtils::IsSameAddressSpace

TEST(GrpcChannelTest, IsSameAddressSpace) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_channel_testDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/distributed_runtime/rpc/grpc_channel_test.cc", "TEST");

  // Same.
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10/cpu:0",
                           "/job:mnist/replica:10/task:10/cpu:1"));
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10/cpu:0",
                           "/job:mnist/replica:10/task:10/device:GPU:2"));
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10",
                           "/job:mnist/replica:10/task:10/device:GPU:2"));
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10/cpu:1",
                           "/job:mnist/replica:10/task:10"));

  // Different.
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:10/task:9/cpu:0",
                            "/job:mnist/replica:10/task:10/cpu:0"));
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:9/task:10/cpu:0",
                            "/job:mnist/replica:10/task:10/cpu:0"));
  EXPECT_FALSE(IsSameAddrSp("/job:MNIST/replica:10/task:10/cpu:0",
                            "/job:mnist/replica:10/task:10/cpu:0"));

  // Invalid names.
  EXPECT_FALSE(IsSameAddrSp("random_invalid_target", "random_invalid_target"));
  EXPECT_FALSE(IsSameAddrSp("/job:/replica:10/task:10/cpu:0",
                            "/job:/replica:10/task:10/cpu:1"));
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:xx/task:10/cpu:0",
                            "/job:mnist/replica:xx/task:10/cpu:1"));
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:10/task:yy/cpu:0",
                            "/job:mnist/replica:10/task:yy/cpu:1"));
}

TEST(GrpcChannelTest, HostPorts) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(spec.AddHostPortsJob(
      "mnist", {"a:1", "b:2", "c:3", "d:4", "e:5", "f:6"}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, RPCOptions()));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:6"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:1/task:0"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    auto a_1_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");
    auto a_1_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");

    auto d_4_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");
    auto d_4_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");

    auto e_5_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");
    auto e_5_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");

    EXPECT_EQ(a_1_1.get(), a_1_2.get());
    EXPECT_EQ(d_4_1.get(), d_4_2.get());
    EXPECT_EQ(e_5_1.get(), e_5_2.get());

    EXPECT_NE(a_1_1.get(), d_4_2.get());
    EXPECT_NE(a_1_1.get(), e_5_2.get());
    EXPECT_NE(d_4_1.get(), e_5_2.get());
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    EXPECT_EQ(
        std::vector<string>(
            {"/job:mnist/replica:0/task:0", "/job:mnist/replica:0/task:1",
             "/job:mnist/replica:0/task:2", "/job:mnist/replica:0/task:3",
             "/job:mnist/replica:0/task:4", "/job:mnist/replica:0/task:5"}),
        workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(
        std::vector<string>(
            {"/job:mnist/replica:0/task:0", "/job:mnist/replica:0/task:1",
             "/job:mnist/replica:0/task:2", "/job:mnist/replica:0/task:3",
             "/job:mnist/replica:0/task:4", "/job:mnist/replica:0/task:5"}),
        workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, HostPortsMultiChannelPerTarget) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(spec.AddHostPortsJob("mnist", {"a:1", "b:2", "c:3"}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  RPCOptions rpc_options;
  rpc_options.set_num_channels_per_target(4);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, rpc_options));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:3"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:1/task:0"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    std::vector<SharedGrpcChannelPtr> a_1_channels, b_2_channels, c_3_channels;

    for (int i = 0; i < 10; i++) {
      a_1_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:0"));
      b_2_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:1"));
      c_3_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:2"));
    }

    // Same channel every 4 calls.
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(a_1_channels[i].get(), a_1_channels[i + 4].get());
      EXPECT_EQ(b_2_channels[i].get(), b_2_channels[i + 4].get());
      EXPECT_EQ(c_3_channels[i].get(), c_3_channels[i + 4].get());
    }

    // Other channels not equal
    for (int i = 0; i < 6; i++) {
      for (int j = 1; j < 4; j++) {
        EXPECT_NE(a_1_channels[i].get(), a_1_channels[i + j].get());
        EXPECT_NE(b_2_channels[i].get(), b_2_channels[i + j].get());
        EXPECT_NE(c_3_channels[i].get(), c_3_channels[i + j].get());
      }
    }

    // Cross Channels never equal
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        EXPECT_NE(a_1_channels[i].get(), b_2_channels[j].get());
        EXPECT_NE(a_1_channels[i].get(), c_3_channels[j].get());
        EXPECT_NE(b_2_channels[i].get(), c_3_channels[j].get());
      }
    }
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:1",
                                   "/job:mnist/replica:0/task:2"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:1",
                                   "/job:mnist/replica:0/task:2"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, HostPortsMultiGrpcMultiChannelPerTarget) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(spec.AddHostPortsJob("mnist", {"a:1", "b:2", "c:3"}));
  TF_EXPECT_OK(spec.AddHostPortsJob("mnist2", {"a:1", "b:2", "c:3"}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  RPCOptions rpc_options;
  rpc_options.set_num_channels_per_target(4);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, rpc_options));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:3"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:1/task:0"));
  EXPECT_NE(nullptr, cc->FindWorkerChannel("/job:mnist2/replica:0/task:0"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    std::vector<SharedGrpcChannelPtr> a_1_channels, b_2_channels, c_3_channels;

    for (int i = 0; i < 10; i++) {
      a_1_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:0"));
      b_2_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:1"));
      c_3_channels.push_back(
          cc->FindWorkerChannel("/job:mnist2/replica:0/task:0"));
    }

    // Same channel every 4 calls.
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(a_1_channels[i].get(), a_1_channels[i + 4].get());
      EXPECT_EQ(b_2_channels[i].get(), b_2_channels[i + 4].get());
      EXPECT_EQ(c_3_channels[i].get(), c_3_channels[i + 4].get());
    }

    // Other channels not equal
    for (int i = 0; i < 6; i++) {
      for (int j = 1; j < 4; j++) {
        EXPECT_NE(a_1_channels[i].get(), a_1_channels[i + j].get());
        EXPECT_NE(b_2_channels[i].get(), b_2_channels[i + j].get());
        EXPECT_NE(c_3_channels[i].get(), c_3_channels[i + j].get());
      }
    }

    // Cross Channels never equal
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        EXPECT_NE(a_1_channels[i].get(), b_2_channels[j].get());
        EXPECT_NE(a_1_channels[i].get(), c_3_channels[j].get());
        EXPECT_NE(b_2_channels[i].get(), c_3_channels[j].get());
      }
    }
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    EXPECT_EQ(
        std::vector<string>(
            {"/job:mnist/replica:0/task:0", "/job:mnist/replica:0/task:1",
             "/job:mnist/replica:0/task:2", "/job:mnist2/replica:0/task:0",
             "/job:mnist2/replica:0/task:1", "/job:mnist2/replica:0/task:2"}),
        workers);
  }

  {
    std::vector<string> workers, workers2;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:1",
                                   "/job:mnist/replica:0/task:2"}),
              workers);
    cc->ListWorkersInJob("mnist2", &workers2);
    EXPECT_EQ(std::vector<string>({"/job:mnist2/replica:0/task:0",
                                   "/job:mnist2/replica:0/task:1",
                                   "/job:mnist2/replica:0/task:2"}),
              workers2);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, SparseHostPorts) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(
      spec.AddHostPortsJob("mnist", {{0, "a:1"}, {3, "d:4"}, {4, "e:5"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, RPCOptions()));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:1"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:2"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:5"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    auto a_1_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");
    auto a_1_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");

    LOG(WARNING) << " Getting task 3";
    auto d_4_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");
    auto d_4_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");

    LOG(WARNING) << " Getting task 4";
    auto e_5_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");
    auto e_5_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");

    EXPECT_EQ(a_1_1.get(), a_1_2.get());
    EXPECT_EQ(d_4_1.get(), d_4_2.get());
    EXPECT_EQ(e_5_1.get(), e_5_2.get());

    EXPECT_NE(a_1_1.get(), d_4_2.get());
    EXPECT_NE(a_1_1.get(), e_5_2.get());
    EXPECT_NE(d_4_1.get(), e_5_2.get());
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    std::sort(workers.begin(), workers.end());
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:3",
                                   "/job:mnist/replica:0/task:4"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:3",
                                   "/job:mnist/replica:0/task:4"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, NewHostPortGrpcChannelValidation) {
  SharedGrpcChannelPtr mock_ptr;

  EXPECT_TRUE(NewHostPortGrpcChannel("127.0.0.1:2222", /*rpc_options=*/nullptr,
                                     &mock_ptr)
                  .ok());
  EXPECT_TRUE(NewHostPortGrpcChannel("example.com:2222",
                                     /*rpc_options=*/nullptr, &mock_ptr)
                  .ok());
  EXPECT_TRUE(NewHostPortGrpcChannel("fqdn.example.com.:2222",
                                     /*rpc_options=*/nullptr, &mock_ptr)
                  .ok());
  EXPECT_TRUE(NewHostPortGrpcChannel("[2002:a9c:258e::]:2222",
                                     /*rpc_options=*/nullptr, &mock_ptr)
                  .ok());
  EXPECT_TRUE(
      NewHostPortGrpcChannel("[::]:2222", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());

  EXPECT_FALSE(NewHostPortGrpcChannel("example.com/abc:2222",
                                      /*rpc_options=*/nullptr, &mock_ptr)
                   .ok());
  EXPECT_FALSE(NewHostPortGrpcChannel("127.0.0.1:2222/",
                                      /*rpc_options=*/nullptr, &mock_ptr)
                   .ok());
  EXPECT_FALSE(NewHostPortGrpcChannel(
                   "example.com/abc:", /*rpc_options=*/nullptr, &mock_ptr)
                   .ok());
  EXPECT_FALSE(
      NewHostPortGrpcChannel("[::]/:2222", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());
  EXPECT_FALSE(
      NewHostPortGrpcChannel("[::]:2222/", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());
  EXPECT_FALSE(
      NewHostPortGrpcChannel("[::]:", /*rpc_options=*/nullptr, &mock_ptr).ok());

  EXPECT_TRUE(
      NewHostPortGrpcChannel("/bns/example", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());
}

}  // namespace tensorflow
