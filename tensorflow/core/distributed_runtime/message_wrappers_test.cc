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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/message_wrappers.h"

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace {

Tensor TensorA() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "TensorA");

  Tensor a_tensor(DT_INT32, TensorShape({2, 2}));
  test::FillValues<int32>(&a_tensor, {3, 2, -1, 0});
  return a_tensor;
}

Tensor TensorB() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "TensorB");

  Tensor b_tensor(DT_INT32, TensorShape({1, 2}));
  test::FillValues<int32>(&b_tensor, {1, 2});
  return b_tensor;
}

void BuildRunStepRequest(MutableRunStepRequestWrapper* request) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "BuildRunStepRequest");

  request->set_session_handle("handle");
  request->set_partial_run_handle("partial_handle");
  request->add_feed("feed_a:0", TensorA());
  request->add_feed("feed_b:0", TensorB());
  request->add_fetch("fetch_x:0");
  request->add_fetch("fetch_y:0");
  request->add_target("target_i");
  request->add_target("target_j");
  request->mutable_options()->set_timeout_in_ms(37);
}

void CheckRunStepRequest(const RunStepRequestWrapper& request) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_3(mht_3_v, 230, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "CheckRunStepRequest");

  EXPECT_EQ("handle", request.session_handle());
  EXPECT_EQ("partial_handle", request.partial_run_handle());
  EXPECT_EQ(2, request.num_feeds());
  EXPECT_EQ("feed_a:0", request.feed_name(0));
  EXPECT_EQ("feed_b:0", request.feed_name(1));
  Tensor val;
  TF_EXPECT_OK(request.FeedValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(request.FeedValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);

  EXPECT_EQ(2, request.num_fetches());
  EXPECT_EQ("fetch_x:0", request.fetch_name(0));
  EXPECT_EQ("fetch_y:0", request.fetch_name(1));
  EXPECT_EQ("target_i", request.target_name(0));
  EXPECT_EQ("target_j", request.target_name(1));
  EXPECT_EQ(37, request.options().timeout_in_ms());
}

void BuildRunGraphRequest(const RunStepRequestWrapper& run_step_request,
                          MutableRunGraphRequestWrapper* run_graph_request) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_4(mht_4_v, 254, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "BuildRunGraphRequest");

  run_graph_request->set_graph_handle("graph_handle");
  run_graph_request->set_step_id(13);
  run_graph_request->mutable_exec_opts()->set_record_timeline(true);
  TF_EXPECT_OK(run_graph_request->AddSendFromRunStepRequest(run_step_request, 0,
                                                            "send_0"));
  TF_EXPECT_OK(run_graph_request->AddSendFromRunStepRequest(run_step_request, 1,
                                                            "send_1"));
  run_graph_request->add_recv_key("recv_2");
  run_graph_request->add_recv_key("recv_3");
  run_graph_request->set_is_partial(true);
}

void CheckRunGraphRequest(const RunGraphRequestWrapper& request) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_5(mht_5_v, 270, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "CheckRunGraphRequest");

  EXPECT_EQ("graph_handle", request.graph_handle());
  EXPECT_EQ(13, request.step_id());
  EXPECT_FALSE(request.exec_opts().record_costs());
  EXPECT_TRUE(request.exec_opts().record_timeline());
  EXPECT_FALSE(request.exec_opts().record_partition_graphs());
  EXPECT_EQ(2, request.num_sends());
  Tensor val;
  TF_EXPECT_OK(request.SendValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(request.SendValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);
  EXPECT_TRUE(request.is_partial());
  EXPECT_FALSE(request.is_last_partial_run());
}

void BuildRunGraphResponse(MutableRunGraphResponseWrapper* run_graph_response) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_6(mht_6_v, 289, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "BuildRunGraphResponse");

  run_graph_response->AddRecv("recv_2", TensorA());
  run_graph_response->AddRecv("recv_3", TensorB());
  run_graph_response->mutable_step_stats()->add_dev_stats()->set_device(
      "/cpu:0");
  run_graph_response->mutable_cost_graph()->add_node()->set_name("cost_node");
  GraphDef graph_def;
  graph_def.mutable_versions()->set_producer(1234);
  graph_def.mutable_versions()->set_min_consumer(1234);
  run_graph_response->AddPartitionGraph(graph_def);
}

void CheckRunGraphResponse(MutableRunGraphResponseWrapper* response) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_7(mht_7_v, 304, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "CheckRunGraphResponse");

  ASSERT_EQ(2, response->num_recvs());
  EXPECT_EQ("recv_2", response->recv_key(0));
  EXPECT_EQ("recv_3", response->recv_key(1));
  Tensor val;
  TF_EXPECT_OK(response->RecvValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(response->RecvValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);
  ASSERT_EQ(1, response->mutable_step_stats()->dev_stats_size());
  EXPECT_EQ("/cpu:0", response->mutable_step_stats()->dev_stats(0).device());
  ASSERT_EQ(1, response->mutable_cost_graph()->node_size());
  EXPECT_EQ("cost_node", response->mutable_cost_graph()->node(0).name());
  ASSERT_EQ(1, response->num_partition_graphs());
  EXPECT_EQ(1234, response->mutable_partition_graph(0)->versions().producer());
  EXPECT_EQ(1234,
            response->mutable_partition_graph(0)->versions().min_consumer());
}

void BuildRunStepResponse(MutableRunGraphResponseWrapper* run_graph_response,
                          MutableRunStepResponseWrapper* run_step_response) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_8(mht_8_v, 327, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "BuildRunStepResponse");

  TF_EXPECT_OK(run_step_response->AddTensorFromRunGraphResponse(
      "fetch_x:0", run_graph_response, 0));
  TF_EXPECT_OK(run_step_response->AddTensorFromRunGraphResponse(
      "fetch_y:0", run_graph_response, 1));
  *run_step_response->mutable_metadata()->mutable_step_stats() =
      *run_graph_response->mutable_step_stats();
  protobuf::RepeatedPtrField<GraphDef>* partition_graph_defs =
      run_step_response->mutable_metadata()->mutable_partition_graphs();
  for (size_t i = 0; i < run_graph_response->num_partition_graphs(); i++) {
    partition_graph_defs->Add()->Swap(
        run_graph_response->mutable_partition_graph(i));
  }
}

void CheckRunStepResponse(const MutableRunStepResponseWrapper& response) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSmessage_wrappers_testDTcc mht_9(mht_9_v, 345, "", "./tensorflow/core/distributed_runtime/message_wrappers_test.cc", "CheckRunStepResponse");

  ASSERT_EQ(2, response.num_tensors());
  EXPECT_EQ("fetch_x:0", response.tensor_name(0));
  EXPECT_EQ("fetch_y:0", response.tensor_name(1));
  Tensor val;
  TF_EXPECT_OK(response.TensorValue(0, &val));
  test::ExpectTensorEqual<int32>(TensorA(), val);
  TF_EXPECT_OK(response.TensorValue(1, &val));
  test::ExpectTensorEqual<int32>(TensorB(), val);
  ASSERT_EQ(1, response.metadata().step_stats().dev_stats_size());
  EXPECT_EQ("/cpu:0", response.metadata().step_stats().dev_stats(0).device());
  ASSERT_EQ(1, response.metadata().partition_graphs_size());
  EXPECT_EQ(1234,
            response.metadata().partition_graphs(0).versions().producer());
  EXPECT_EQ(1234,
            response.metadata().partition_graphs(0).versions().min_consumer());
}

TEST(MessageWrappers, RunStepRequest_Basic) {
  InMemoryRunStepRequest in_memory_request;
  BuildRunStepRequest(&in_memory_request);
  CheckRunStepRequest(in_memory_request);

  MutableProtoRunStepRequest proto_request;
  BuildRunStepRequest(&proto_request);
  CheckRunStepRequest(proto_request);

  CheckRunStepRequest(ProtoRunStepRequest(&in_memory_request.ToProto()));
  CheckRunStepRequest(ProtoRunStepRequest(&proto_request.ToProto()));
}

TEST(MessageWrappers, RunGraphRequest_Basic) {
  InMemoryRunStepRequest in_memory_run_step_request;
  BuildRunStepRequest(&in_memory_run_step_request);

  MutableProtoRunStepRequest mutable_proto_run_step_request;
  BuildRunStepRequest(&mutable_proto_run_step_request);

  ProtoRunStepRequest proto_run_step_request(
      &mutable_proto_run_step_request.ToProto());

  // Client -(in memory)-> Master -(in memory)-> Worker.
  {
    InMemoryRunGraphRequest request;
    BuildRunGraphRequest(in_memory_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(mutable proto)-> Master -(in memory)-> Worker.
  {
    InMemoryRunGraphRequest request;
    BuildRunGraphRequest(mutable_proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(proto)-> Master -(in memory)-> Worker.
  {
    InMemoryRunGraphRequest request;
    BuildRunGraphRequest(proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(in memory)-> Master -(mutable proto)-> Worker.
  {
    MutableProtoRunGraphRequest request;
    BuildRunGraphRequest(in_memory_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(mutable proto)-> Master -(mutable proto)-> Worker.
  {
    MutableProtoRunGraphRequest request;
    BuildRunGraphRequest(mutable_proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }

  // Client -(proto)-> Master -(mutable proto)-> Worker.
  {
    MutableProtoRunGraphRequest request;
    BuildRunGraphRequest(proto_run_step_request, &request);
    CheckRunGraphRequest(request);
    CheckRunGraphRequest(ProtoRunGraphRequest(&request.ToProto()));
  }
}

TEST(MessageWrappers, RunGraphResponse_Basic) {
  InMemoryRunGraphResponse in_memory_response;
  BuildRunGraphResponse(&in_memory_response);
  CheckRunGraphResponse(&in_memory_response);

  OwnedProtoRunGraphResponse owned_proto_response;
  BuildRunGraphResponse(&owned_proto_response);
  CheckRunGraphResponse(&owned_proto_response);

  RunGraphResponse response_proto;
  NonOwnedProtoRunGraphResponse non_owned_proto_response(&response_proto);
  BuildRunGraphResponse(&non_owned_proto_response);
  CheckRunGraphResponse(&non_owned_proto_response);
}

TEST(MessageWrappers, RunStepResponse_Basic) {
  {
    // Worker -(in memory)-> Master -(in memory)-> Client.
    InMemoryRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    InMemoryRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(in memory)-> Master -(owned proto)-> Client.
    InMemoryRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    OwnedProtoRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(in memory)-> Master -(non-owned proto)-> Client.
    InMemoryRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    RunStepResponse response_proto;
    NonOwnedProtoRunStepResponse response(&response_proto);
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(owned proto)-> Master -(in memory)-> Client.
    OwnedProtoRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    InMemoryRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(owned proto)-> Master -(owned proto)-> Client.
    OwnedProtoRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    OwnedProtoRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(owned proto)-> Master -(non-owned proto)-> Client.
    OwnedProtoRunGraphResponse run_graph_response;
    BuildRunGraphResponse(&run_graph_response);
    RunStepResponse response_proto;
    NonOwnedProtoRunStepResponse response(&response_proto);
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(non-owned proto)-> Master -(in memory)-> Client.
    RunGraphResponse run_graph_response_proto;
    NonOwnedProtoRunGraphResponse run_graph_response(&run_graph_response_proto);
    BuildRunGraphResponse(&run_graph_response);
    InMemoryRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(non-owned proto)-> Master -(owned proto)-> Client.
    RunGraphResponse run_graph_response_proto;
    NonOwnedProtoRunGraphResponse run_graph_response(&run_graph_response_proto);
    BuildRunGraphResponse(&run_graph_response);
    OwnedProtoRunStepResponse response;
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }

  {
    // Worker -(non-owned proto)-> Master -(non-owned proto)-> Client.
    RunGraphResponse run_graph_response_proto;
    NonOwnedProtoRunGraphResponse run_graph_response(&run_graph_response_proto);
    BuildRunGraphResponse(&run_graph_response);
    RunStepResponse response_proto;
    NonOwnedProtoRunStepResponse response(&response_proto);
    BuildRunStepResponse(&run_graph_response, &response);
    CheckRunStepResponse(response);
  }
}

}  // namespace
}  // namespace tensorflow
