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
class MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc {
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
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc() {
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

#include "tensorflow/core/debug/debug_grpc_testlib.h"

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/debug/debugger_event_metadata.pb.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

namespace test {

::grpc::Status TestEventListenerImpl::SendEvents(
    ::grpc::ServerContext* context,
    ::grpc::ServerReaderWriter<::tensorflow::EventReply, ::tensorflow::Event>*
        stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/debug/debug_grpc_testlib.cc", "TestEventListenerImpl::SendEvents");

  Event event;

  while (stream->Read(&event)) {
    if (event.has_log_message()) {
      debug_metadata_strings.push_back(event.log_message().message());
      stream->Write(EventReply());
    } else if (!event.graph_def().empty()) {
      encoded_graph_defs.push_back(event.graph_def());
      stream->Write(EventReply());
    } else if (event.has_summary()) {
      const Summary::Value& val = event.summary().value(0);

      std::vector<string> name_items =
          tensorflow::str_util::Split(val.node_name(), ':');

      const string node_name = name_items[0];
      const string debug_op = name_items[2];

      const TensorProto& tensor_proto = val.tensor();
      Tensor tensor(tensor_proto.dtype());
      if (!tensor.FromProto(tensor_proto)) {
        return ::grpc::Status::CANCELLED;
      }

      // Obtain the device name, which is encoded in JSON.
      third_party::tensorflow::core::debug::DebuggerEventMetadata metadata;
      if (val.metadata().plugin_data().plugin_name() != "debugger") {
        // This plugin data was meant for another plugin.
        continue;
      }
      auto status = tensorflow::protobuf::util::JsonStringToMessage(
          val.metadata().plugin_data().content(), &metadata);
      if (!status.ok()) {
        // The device name could not be determined.
        continue;
      }

      device_names.push_back(metadata.device());
      node_names.push_back(node_name);
      output_slots.push_back(metadata.output_slot());
      debug_ops.push_back(debug_op);
      debug_tensors.push_back(tensor);

      // If the debug node is currently in the READ_WRITE mode, send an
      // EventReply to 1) unblock the execution and 2) optionally modify the
      // value.
      const DebugNodeKey debug_node_key(metadata.device(), node_name,
                                        metadata.output_slot(), debug_op);
      if (write_enabled_debug_node_keys_.find(debug_node_key) !=
          write_enabled_debug_node_keys_.end()) {
        stream->Write(EventReply());
      }
    }
  }

  {
    mutex_lock l(states_mu_);
    for (size_t i = 0; i < new_states_.size(); ++i) {
      EventReply event_reply;
      EventReply::DebugOpStateChange* change =
          event_reply.add_debug_op_state_changes();

      // State changes will take effect in the next stream, i.e., next debugged
      // Session.run() call.
      change->set_state(new_states_[i]);
      const DebugNodeKey& debug_node_key = debug_node_keys_[i];
      change->set_node_name(debug_node_key.node_name);
      change->set_output_slot(debug_node_key.output_slot);
      change->set_debug_op(debug_node_key.debug_op);
      stream->Write(event_reply);

      if (new_states_[i] == EventReply::DebugOpStateChange::READ_WRITE) {
        write_enabled_debug_node_keys_.insert(debug_node_key);
      } else {
        write_enabled_debug_node_keys_.erase(debug_node_key);
      }
    }

    debug_node_keys_.clear();
    new_states_.clear();
  }

  return ::grpc::Status::OK;
}

void TestEventListenerImpl::ClearReceivedDebugData() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc mht_1(mht_1_v, 293, "", "./tensorflow/core/debug/debug_grpc_testlib.cc", "TestEventListenerImpl::ClearReceivedDebugData");

  debug_metadata_strings.clear();
  encoded_graph_defs.clear();
  device_names.clear();
  node_names.clear();
  output_slots.clear();
  debug_ops.clear();
  debug_tensors.clear();
}

void TestEventListenerImpl::RequestDebugOpStateChangeAtNextStream(
    const EventReply::DebugOpStateChange::State new_state,
    const DebugNodeKey& debug_node_key) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc mht_2(mht_2_v, 308, "", "./tensorflow/core/debug/debug_grpc_testlib.cc", "TestEventListenerImpl::RequestDebugOpStateChangeAtNextStream");

  mutex_lock l(states_mu_);

  debug_node_keys_.push_back(debug_node_key);
  new_states_.push_back(new_state);
}

void TestEventListenerImpl::RunServer(const int server_port) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc mht_3(mht_3_v, 318, "", "./tensorflow/core/debug/debug_grpc_testlib.cc", "TestEventListenerImpl::RunServer");

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("localhost:", server_port),
                           ::grpc::InsecureServerCredentials());
  builder.RegisterService(this);
  std::unique_ptr<::grpc::Server> server = builder.BuildAndStart();

  while (!stop_requested_.load()) {
    Env::Default()->SleepForMicroseconds(200 * 1000);
  }
  server->Shutdown();
  stopped_.store(true);
}

void TestEventListenerImpl::StopServer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc mht_4(mht_4_v, 335, "", "./tensorflow/core/debug/debug_grpc_testlib.cc", "TestEventListenerImpl::StopServer");

  stop_requested_.store(true);
  while (!stopped_.load()) {
  }
}

bool PollTillFirstRequestSucceeds(const string& server_url,
                                  const size_t max_attempts) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("server_url: \"" + server_url + "\"");
   MHTracer_DTPStensorflowPScorePSdebugPSdebug_grpc_testlibDTcc mht_5(mht_5_v, 346, "", "./tensorflow/core/debug/debug_grpc_testlib.cc", "PollTillFirstRequestSucceeds");

  const int kSleepDurationMicros = 100 * 1000;
  size_t n_attempts = 0;
  bool success = false;

  // Try a number of times to send the Event proto to the server, as it may
  // take the server a few seconds to start up and become responsive.
  Tensor prep_tensor(DT_FLOAT, TensorShape({1, 1}));
  prep_tensor.flat<float>()(0) = 42.0f;

  while (n_attempts++ < max_attempts) {
    const uint64 wall_time = Env::Default()->NowMicros();
    Status publish_s = DebugIO::PublishDebugTensor(
        DebugNodeKey("/job:localhost/replica:0/task:0/cpu:0", "prep_node", 0,
                     "DebugIdentity"),
        prep_tensor, wall_time, {server_url});
    Status close_s = DebugIO::CloseDebugURL(server_url);

    if (publish_s.ok() && close_s.ok()) {
      success = true;
      break;
    } else {
      Env::Default()->SleepForMicroseconds(kSleepDurationMicros);
    }
  }

  return success;
}

}  // namespace test

}  // namespace tensorflow
