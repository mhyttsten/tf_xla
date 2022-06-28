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
class MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc {
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
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc() {
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

#include "tensorflow/cc/client/client_session.h"

#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class ClientSession::Impl {
 private:
  friend class ClientSession;

  Impl(Session* session, std::shared_ptr<Graph> graph)
      : session_(session), graph_(std::move(graph)) {}

  static SessionOptions MakeDefaultSessionOptions(const string& target);
  Status MaybeExtendGraph() const;

  std::unique_ptr<Session> session_;
  std::shared_ptr<Graph> graph_;

  mutable mutex mu_;
  mutable int last_num_graph_nodes_ TF_GUARDED_BY(mu_) = 0;
};

ClientSession::ClientSession(const Scope& scope, const string& target)
    : ClientSession(scope, Impl::MakeDefaultSessionOptions(target)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_0(mht_0_v, 218, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::ClientSession");
}

ClientSession::ClientSession(const Scope& scope) : ClientSession(scope, "") {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_1(mht_1_v, 223, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::ClientSession");
}

ClientSession::ClientSession(const Scope& scope,
                             const SessionOptions& session_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_2(mht_2_v, 229, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::ClientSession");

  Session* new_session;
  Status status = NewSession(session_options, &new_session);
  TF_CHECK_OK(status) << status;
  impl_.reset(new Impl(new_session, scope.graph_as_shared_ptr()));
  CHECK_NOTNULL(impl()->session_.get());
}

// Define destructor here so we can forward declare `Impl` in client_session.h.
// If we define a dtor in the header file or use the default dtor,
// unique_ptr<Impl> needs the complete type.
ClientSession::~ClientSession() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_3(mht_3_v, 243, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::~ClientSession");
}

SessionOptions ClientSession::Impl::MakeDefaultSessionOptions(
    const string& target) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_4(mht_4_v, 250, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Impl::MakeDefaultSessionOptions");

  SessionOptions options;
  options.env = Env::Default();
  options.target = target;
  return options;
}

Status ClientSession::Run(const std::vector<Output>& fetch_outputs,
                          std::vector<Tensor>* outputs) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_5(mht_5_v, 261, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Run");

  return Run(FeedType{}, fetch_outputs, {}, outputs);
}

Status ClientSession::Run(const FeedType& inputs,
                          const std::vector<Output>& fetch_outputs,
                          std::vector<Tensor>* outputs) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_6(mht_6_v, 270, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Run");

  return Run(inputs, fetch_outputs, {}, outputs);
}

Status ClientSession::Run(const FeedType& inputs,
                          const std::vector<Output>& fetch_outputs,
                          const std::vector<Operation>& run_outputs,
                          std::vector<Tensor>* outputs) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_7(mht_7_v, 280, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Run");

  return Run(RunOptions(), inputs, fetch_outputs, run_outputs, outputs,
             nullptr);
}

Status ClientSession::Impl::MaybeExtendGraph() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_8(mht_8_v, 288, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Impl::MaybeExtendGraph");

  mutex_lock l(mu_);
  int num_nodes = graph_->num_node_ids();
  if (num_nodes > last_num_graph_nodes_) {
    GraphDef graph_def;
    graph_->ToGraphDefSubRange(&graph_def, last_num_graph_nodes_);
    last_num_graph_nodes_ = num_nodes;
    return session_->Extend(graph_def);
  }
  return Status::OK();
}

Status ClientSession::Run(const RunOptions& run_options, const FeedType& inputs,
                          const std::vector<Output>& fetch_outputs,
                          const std::vector<Operation>& run_outputs,
                          std::vector<Tensor>* outputs,
                          RunMetadata* run_metadata) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_9(mht_9_v, 307, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Run");

  std::vector<std::pair<string, Tensor>> feeds;
  for (auto const& feed : inputs) {
    TF_RETURN_IF_ERROR(feed.second.status);
    feeds.emplace_back(feed.first.name(), feed.second.tensor);
  }
  std::vector<string> output_tensor_names;
  output_tensor_names.reserve(fetch_outputs.size());
  for (auto const& output : fetch_outputs) {
    output_tensor_names.push_back(output.name());
  }
  std::vector<string> target_node_names;
  target_node_names.reserve(run_outputs.size());
  for (auto const& output : run_outputs) {
    target_node_names.push_back(output.node()->name());
  }
  TF_RETURN_IF_ERROR(impl()->MaybeExtendGraph());
  return impl()->session_->Run(run_options, feeds, output_tensor_names,
                               target_node_names, outputs, run_metadata);
}

Status ClientSession::Run(
    const RunOptions& run_options, const FeedType& inputs,
    const std::vector<Output>& fetch_outputs,
    const std::vector<Operation>& run_outputs, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata,
    const thread::ThreadPoolOptions& threadpool_options) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_10(mht_10_v, 336, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::Run");

  std::vector<std::pair<string, Tensor>> feeds;
  for (auto const& feed : inputs) {
    TF_RETURN_IF_ERROR(feed.second.status);
    feeds.emplace_back(feed.first.name(), feed.second.tensor);
  }
  std::vector<string> output_tensor_names;
  output_tensor_names.reserve(fetch_outputs.size());
  for (auto const& output : fetch_outputs) {
    output_tensor_names.push_back(output.name());
  }
  std::vector<string> target_node_names;
  target_node_names.reserve(run_outputs.size());
  for (auto const& output : run_outputs) {
    target_node_names.push_back(output.node()->name());
  }
  TF_RETURN_IF_ERROR(impl()->MaybeExtendGraph());
  return impl()->session_->Run(run_options, feeds, output_tensor_names,
                               target_node_names, outputs, run_metadata,
                               threadpool_options);
}

Status ClientSession::MakeCallable(const CallableOptions& callable_options,
                                   CallableHandle* out_handle) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_11(mht_11_v, 362, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::MakeCallable");

  TF_RETURN_IF_ERROR(impl()->MaybeExtendGraph());
  return impl()->session_->MakeCallable(callable_options, out_handle);
}

Status ClientSession::RunCallable(CallableHandle handle,
                                  const std::vector<Tensor>& feed_tensors,
                                  std::vector<Tensor>* fetch_tensors,
                                  RunMetadata* run_metadata) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_12(mht_12_v, 373, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::RunCallable");

  return impl()->session_->RunCallable(handle, feed_tensors, fetch_tensors,
                                       run_metadata);
}

Status ClientSession::RunCallable(CallableHandle handle,
                                  const std::vector<Tensor>& feed_tensors,
                                  std::vector<Tensor>* fetch_tensors,
                                  RunMetadata* run_metadata,
                                  const thread::ThreadPoolOptions& options) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_13(mht_13_v, 385, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::RunCallable");

  return impl()->session_->RunCallable(handle, feed_tensors, fetch_tensors,
                                       run_metadata, options);
}

Status ClientSession::ReleaseCallable(CallableHandle handle) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSccPSclientPSclient_sessionDTcc mht_14(mht_14_v, 393, "", "./tensorflow/cc/client/client_session.cc", "ClientSession::ReleaseCallable");

  return impl()->session_->ReleaseCallable(handle);
}

}  // end namespace tensorflow
