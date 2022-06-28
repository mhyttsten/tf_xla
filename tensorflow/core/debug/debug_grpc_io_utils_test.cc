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
class MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc() {
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

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/debug/debug_grpc_testlib.h"
#include "tensorflow/core/debug/debug_io_utils.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

class GrpcDebugTest : public ::testing::Test {
 protected:
  struct ServerData {
    int port;
    string url;
    std::unique_ptr<test::TestEventListenerImpl> server;
    std::unique_ptr<thread::ThreadPool> thread_pool;
  };

  void SetUp() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "SetUp");

    ClearEnabledWatchKeys();
    SetUpInProcessServer(&server_data_, 0);
  }

  void TearDown() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "TearDown");
 TearDownInProcessServer(&server_data_); }

  void SetUpInProcessServer(ServerData* server_data,
                            int64_t server_start_delay_micros) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "SetUpInProcessServer");

    server_data->port = testing::PickUnusedPortOrDie();
    server_data->url = strings::StrCat("grpc://localhost:", server_data->port);
    server_data->server.reset(new test::TestEventListenerImpl());

    server_data->thread_pool.reset(
        new thread::ThreadPool(Env::Default(), "test_server", 1));
    server_data->thread_pool->Schedule(
        [server_data, server_start_delay_micros]() {
          Env::Default()->SleepForMicroseconds(server_start_delay_micros);
          server_data->server->RunServer(server_data->port);
        });
  }

  void TearDownInProcessServer(ServerData* server_data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "TearDownInProcessServer");

    server_data->server->StopServer();
    server_data->thread_pool.reset();
  }

  void ClearEnabledWatchKeys() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_4(mht_4_v, 247, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "ClearEnabledWatchKeys");
 DebugGrpcIO::ClearEnabledWatchKeys(); }

  const int64_t GetChannelConnectionTimeoutMicros() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_5(mht_5_v, 252, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "GetChannelConnectionTimeoutMicros");

    return DebugGrpcIO::channel_connection_timeout_micros_;
  }

  void SetChannelConnectionTimeoutMicros(const int64_t timeout) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_6(mht_6_v, 259, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "SetChannelConnectionTimeoutMicros");

    DebugGrpcIO::channel_connection_timeout_micros_ = timeout;
  }

  ServerData server_data_;
};

TEST_F(GrpcDebugTest, ConnectionTimeoutWorks) {
  // Use a short timeout so the test won't take too long.
  const int64_t kOriginalTimeoutMicros = GetChannelConnectionTimeoutMicros();
  const int64_t kShortTimeoutMicros = 500 * 1000;
  SetChannelConnectionTimeoutMicros(kShortTimeoutMicros);
  ASSERT_EQ(kShortTimeoutMicros, GetChannelConnectionTimeoutMicros());

  const string& kInvalidGrpcUrl =
      strings::StrCat("grpc://localhost:", testing::PickUnusedPortOrDie());
  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;
  Status publish_status = DebugIO::PublishDebugTensor(
      DebugNodeKey("/job:localhost/replica:0/task:0/cpu:0", "foo_tensor", 0,
                   "DebugIdentity"),
      tensor, Env::Default()->NowMicros(), {kInvalidGrpcUrl});
  SetChannelConnectionTimeoutMicros(kOriginalTimeoutMicros);
  TF_ASSERT_OK(DebugIO::CloseDebugURL(kInvalidGrpcUrl));

  ASSERT_FALSE(publish_status.ok());
  const string expected_error_msg = strings::StrCat(
      "Failed to connect to gRPC channel at ", kInvalidGrpcUrl.substr(7),
      " within a timeout of ", kShortTimeoutMicros / 1e6, " s");
  ASSERT_NE(string::npos,
            publish_status.error_message().find(expected_error_msg));
}

TEST_F(GrpcDebugTest, ConnectionToDelayedStartingServerWorks) {
  ServerData server_data;
  // Server start will be delayed for 1 second.
  SetUpInProcessServer(&server_data, 1 * 1000 * 1000);

  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  Status publish_status = DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data.url});
  ASSERT_TRUE(publish_status.ok());
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data.url));

  ASSERT_EQ(1, server_data.server->node_names.size());
  ASSERT_EQ(1, server_data.server->output_slots.size());
  ASSERT_EQ(1, server_data.server->debug_ops.size());
  EXPECT_EQ(kDebugNodeKey.device_name, server_data.server->device_names[0]);
  EXPECT_EQ(kDebugNodeKey.node_name, server_data.server->node_names[0]);
  EXPECT_EQ(kDebugNodeKey.output_slot, server_data.server->output_slots[0]);
  EXPECT_EQ(kDebugNodeKey.debug_op, server_data.server->debug_ops[0]);
  TearDownInProcessServer(&server_data);
}

TEST_F(GrpcDebugTest, SendSingleDebugTensorViaGrpcTest) {
  Tensor tensor(DT_FLOAT, TensorShape({1, 1}));
  tensor.flat<float>()(0) = 42.0;
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  TF_ASSERT_OK(DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data_.url}));
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data_.url));

  // Verify that the expected debug tensor sending happened.
  ASSERT_EQ(1, server_data_.server->node_names.size());
  ASSERT_EQ(1, server_data_.server->output_slots.size());
  ASSERT_EQ(1, server_data_.server->debug_ops.size());
  EXPECT_EQ(kDebugNodeKey.device_name, server_data_.server->device_names[0]);
  EXPECT_EQ(kDebugNodeKey.node_name, server_data_.server->node_names[0]);
  EXPECT_EQ(kDebugNodeKey.output_slot, server_data_.server->output_slots[0]);
  EXPECT_EQ(kDebugNodeKey.debug_op, server_data_.server->debug_ops[0]);
}

TEST_F(GrpcDebugTest, SendDebugTensorWithLargeStringAtIndex0ViaGrpcTest) {
  Tensor tensor(DT_STRING, TensorShape({1, 1}));
  tensor.flat<tstring>()(0) = string(5000 * 1024, 'A');
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  const Status status = DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data_.url});
  ASSERT_FALSE(status.ok());
  ASSERT_NE(status.error_message().find("string value at index 0 from debug "
                                        "node foo_tensor:0:DebugIdentity does "
                                        "not fit gRPC message size limit"),
            string::npos);
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data_.url));
}

TEST_F(GrpcDebugTest, SendDebugTensorWithLargeStringAtIndex1ViaGrpcTest) {
  Tensor tensor(DT_STRING, TensorShape({1, 2}));
  tensor.flat<tstring>()(0) = "A";
  tensor.flat<tstring>()(1) = string(5000 * 1024, 'A');
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "foo_tensor", 0, "DebugIdentity");
  const Status status = DebugIO::PublishDebugTensor(
      kDebugNodeKey, tensor, Env::Default()->NowMicros(), {server_data_.url});
  ASSERT_FALSE(status.ok());
  ASSERT_NE(status.error_message().find("string value at index 1 from debug "
                                        "node foo_tensor:0:DebugIdentity does "
                                        "not fit gRPC message size limit"),
            string::npos);
  TF_ASSERT_OK(DebugIO::CloseDebugURL(server_data_.url));
}

TEST_F(GrpcDebugTest, SendMultipleDebugTensorsSynchronizedViaGrpcTest) {
  const int32_t kSends = 4;

  // Prepare the tensors to sent.
  std::vector<Tensor> tensors;
  for (int i = 0; i < kSends; ++i) {
    Tensor tensor(DT_INT32, TensorShape({1, 1}));
    tensor.flat<int>()(0) = i * i;
    tensors.push_back(tensor);
  }

  thread::ThreadPool* tp =
      new thread::ThreadPool(Env::Default(), "grpc_debug_test", kSends);

  mutex mu;
  Notification all_done;
  int tensor_count TF_GUARDED_BY(mu) = 0;
  std::vector<Status> statuses TF_GUARDED_BY(mu);

  const std::vector<string> urls({server_data_.url});

  // Set up the concurrent tasks of sending Tensors via an Event stream to the
  // server.
  auto fn = [this, &mu, &tensor_count, &tensors, &statuses, &all_done,
             &urls]() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_io_utils_testDTcc mht_7(mht_7_v, 393, "", "./tensorflow/core/debug/debug_grpc_io_utils_test.cc", "lambda");

    int this_count;
    {
      mutex_lock l(mu);
      this_count = tensor_count++;
    }

    // Different concurrent tasks will send different tensors.
    const uint64 wall_time = Env::Default()->NowMicros();
    Status publish_status = DebugIO::PublishDebugTensor(
        DebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                     strings::StrCat("synchronized_node_", this_count), 0,
                     "DebugIdentity"),
        tensors[this_count], wall_time, urls);

    {
      mutex_lock l(mu);
      statuses.push_back(publish_status);
      if (this_count == kSends - 1 && !all_done.HasBeenNotified()) {
        all_done.Notify();
      }
    }
  };

  // Schedule the concurrent tasks.
  for (int i = 0; i < kSends; ++i) {
    tp->Schedule(fn);
  }

  // Wait for all client tasks to finish.
  all_done.WaitForNotification();
  delete tp;

  // Close the debug gRPC stream.
  Status close_status = DebugIO::CloseDebugURL(server_data_.url);
  ASSERT_TRUE(close_status.ok());

  // Check all statuses from the PublishDebugTensor calls().
  for (const Status& status : statuses) {
    TF_ASSERT_OK(status);
  }

  // One prep tensor plus kSends concurrent tensors are expected.
  ASSERT_EQ(kSends, server_data_.server->node_names.size());
  for (size_t i = 0; i < server_data_.server->node_names.size(); ++i) {
    std::vector<string> items =
        str_util::Split(server_data_.server->node_names[i], '_');
    int tensor_index;
    strings::safe_strto32(items[2], &tensor_index);

    ASSERT_EQ(TensorShape({1, 1}),
              server_data_.server->debug_tensors[i].shape());
    ASSERT_EQ(tensor_index * tensor_index,
              server_data_.server->debug_tensors[i].flat<int>()(0));
  }
}

TEST_F(GrpcDebugTest, SendDebugTensorsThroughMultipleRoundsUsingGrpcGating) {
  // Prepare the tensor to send.
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "test_namescope/test_node", 0,
                                   "DebugIdentity");
  Tensor tensor(DT_INT32, TensorShape({1, 1}));
  tensor.flat<int>()(0) = 42;

  const std::vector<string> urls({server_data_.url});
  for (int i = 0; i < 3; ++i) {
    server_data_.server->ClearReceivedDebugData();
    const uint64 wall_time = Env::Default()->NowMicros();

    // On the 1st send (i == 0), gating is disabled, so data should be sent.
    // On the 2nd send (i == 1), gating is enabled, and the server has enabled
    //   the watch key in the previous send, so data should be sent.
    // On the 3rd send (i == 2), gating is enabled, but the server has disabled
    //   the watch key in the previous send, so data should not be sent.
    const bool enable_gated_grpc = (i != 0);
    TF_ASSERT_OK(DebugIO::PublishDebugTensor(kDebugNodeKey, tensor, wall_time,
                                             urls, enable_gated_grpc));

    server_data_.server->RequestDebugOpStateChangeAtNextStream(
        i == 0 ? EventReply::DebugOpStateChange::READ_ONLY
               : EventReply::DebugOpStateChange::DISABLED,
        kDebugNodeKey);

    // Close the debug gRPC stream.
    Status close_status = DebugIO::CloseDebugURL(server_data_.url);
    ASSERT_TRUE(close_status.ok());

    // Check dumped files according to the expected gating results.
    if (i < 2) {
      ASSERT_EQ(1, server_data_.server->node_names.size());
      ASSERT_EQ(1, server_data_.server->output_slots.size());
      ASSERT_EQ(1, server_data_.server->debug_ops.size());
      EXPECT_EQ(kDebugNodeKey.device_name,
                server_data_.server->device_names[0]);
      EXPECT_EQ(kDebugNodeKey.node_name, server_data_.server->node_names[0]);
      EXPECT_EQ(kDebugNodeKey.output_slot,
                server_data_.server->output_slots[0]);
      EXPECT_EQ(kDebugNodeKey.debug_op, server_data_.server->debug_ops[0]);
    } else {
      ASSERT_EQ(0, server_data_.server->node_names.size());
    }
  }
}

TEST_F(GrpcDebugTest, SendDebugTensorsThroughMultipleRoundsUnderReadWriteMode) {
  // Prepare the tensor to send.
  const DebugNodeKey kDebugNodeKey("/job:localhost/replica:0/task:0/cpu:0",
                                   "test_namescope/test_node", 0,
                                   "DebugIdentity");
  Tensor tensor(DT_INT32, TensorShape({1, 1}));
  tensor.flat<int>()(0) = 42;

  const std::vector<string> urls({server_data_.url});
  for (int i = 0; i < 3; ++i) {
    server_data_.server->ClearReceivedDebugData();
    const uint64 wall_time = Env::Default()->NowMicros();

    // On the 1st send (i == 0), gating is disabled, so data should be sent.
    // On the 2nd send (i == 1), gating is enabled, and the server has enabled
    //   the watch key in the previous send (READ_WRITE), so data should be
    //   sent. In this iteration, the server response with a EventReply proto to
    //   unblock the debug node.
    // On the 3rd send (i == 2), gating is enabled, but the server has disabled
    //   the watch key in the previous send, so data should not be sent.
    const bool enable_gated_grpc = (i != 0);
    TF_ASSERT_OK(DebugIO::PublishDebugTensor(kDebugNodeKey, tensor, wall_time,
                                             urls, enable_gated_grpc));

    server_data_.server->RequestDebugOpStateChangeAtNextStream(
        i == 0 ? EventReply::DebugOpStateChange::READ_WRITE
               : EventReply::DebugOpStateChange::DISABLED,
        kDebugNodeKey);

    // Close the debug gRPC stream.
    Status close_status = DebugIO::CloseDebugURL(server_data_.url);
    ASSERT_TRUE(close_status.ok());

    // Check dumped files according to the expected gating results.
    if (i < 2) {
      ASSERT_EQ(1, server_data_.server->node_names.size());
      ASSERT_EQ(1, server_data_.server->output_slots.size());
      ASSERT_EQ(1, server_data_.server->debug_ops.size());
      EXPECT_EQ(kDebugNodeKey.device_name,
                server_data_.server->device_names[0]);
      EXPECT_EQ(kDebugNodeKey.node_name, server_data_.server->node_names[0]);
      EXPECT_EQ(kDebugNodeKey.output_slot,
                server_data_.server->output_slots[0]);
      EXPECT_EQ(kDebugNodeKey.debug_op, server_data_.server->debug_ops[0]);
    } else {
      ASSERT_EQ(0, server_data_.server->node_names.size());
    }
  }
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnEmptyEnabledSet) {
  ASSERT_FALSE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity",
                                            {"grpc://localhost:3333"}));

  // file:// debug URLs are not subject to grpc gating.
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen(
      "foo:0:DebugIdentity", {"grpc://localhost:3333", "file:///tmp/tfdbg_1"}));
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnNonEmptyEnabledSet) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kGrpcUrl2 = "grpc://localhost:3334";

  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, "foo:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, "bar:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);

  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:1:DebugIdentity", {kGrpcUrl1}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:1:DebugNumericSummary", {kGrpcUrl1}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("qux:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl1}));

  // Wrong grpc:// debug URLs.
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl2}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl2}));

  // file:// debug URLs are not subject to grpc gating.
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("qux:0:DebugIdentity",
                                           {"file:///tmp/tfdbg_1", kGrpcUrl1}));
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnMultipleEmptyEnabledSets) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kGrpcUrl2 = "grpc://localhost:3334";
  const string kGrpcUrl3 = "grpc://localhost:3335";

  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, "foo:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl2, "bar:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);

  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl2}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl2}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl1}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity", {kGrpcUrl3}));
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity", {kGrpcUrl3}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity",
                                           {kGrpcUrl1, kGrpcUrl2}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity",
                                           {kGrpcUrl1, kGrpcUrl2}));
  ASSERT_TRUE(DebugIO::IsDebugNodeGateOpen("foo:0:DebugIdentity",
                                           {kGrpcUrl1, kGrpcUrl3}));
  ASSERT_FALSE(DebugIO::IsDebugNodeGateOpen("bar:0:DebugIdentity",
                                            {kGrpcUrl1, kGrpcUrl3}));
}

TEST_F(GrpcDebugTest, TestGateDebugNodeOnNonEmptyEnabledSetAndEmptyURLs) {
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      "grpc://localhost:3333", "foo:0:DebugIdentity",
      EventReply::DebugOpStateChange::READ_ONLY);

  std::vector<string> debug_urls_1;
  ASSERT_FALSE(
      DebugIO::IsDebugNodeGateOpen("foo:1:DebugIdentity", debug_urls_1));
}

TEST_F(GrpcDebugTest, TestGateCopyNodeOnEmptyEnabledSet) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kWatch1 = "foo:0:DebugIdentity";

  ASSERT_FALSE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, false)}));

  // file:// debug URLs are not subject to grpc gating.
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec("foo:0:DebugIdentity", kGrpcUrl1, true),
       DebugWatchAndURLSpec("foo:0:DebugIdentity", "file:///tmp/tfdbg_1",
                            false)}));
}

TEST_F(GrpcDebugTest, TestGateCopyNodeOnNonEmptyEnabledSet) {
  const string kGrpcUrl1 = "grpc://localhost:3333";
  const string kGrpcUrl2 = "grpc://localhost:3334";
  const string kWatch1 = "foo:0:DebugIdentity";
  const string kWatch2 = "foo:1:DebugIdentity";
  DebugGrpcIO::SetDebugNodeKeyGrpcState(
      kGrpcUrl1, kWatch1, EventReply::DebugOpStateChange::READ_ONLY);

  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true)}));

  ASSERT_FALSE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl2, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl2, false)}));

  ASSERT_FALSE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch2, kGrpcUrl1, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch2, kGrpcUrl1, false)}));

  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true),
       DebugWatchAndURLSpec(kWatch1, kGrpcUrl2, true)}));
  ASSERT_TRUE(DebugIO::IsCopyNodeGateOpen(
      {DebugWatchAndURLSpec(kWatch1, kGrpcUrl1, true),
       DebugWatchAndURLSpec(kWatch2, kGrpcUrl2, true)}));
}

}  // namespace tensorflow
