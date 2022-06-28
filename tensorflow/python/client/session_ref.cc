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
class MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc() {
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
#include "tensorflow/python/client/session_ref.h"

#include <stdlib.h>
#include <memory>
#include <utility>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/protobuf/replay_log.pb.h"

namespace tensorflow {

namespace {

// Scope helper to track active calls and manage session lifetime.
// SessionRef blocks closing until all active calls complete or are cancelled.
struct RunCounter {
  std::shared_ptr<Session> session;
  uint64* value;
  mutex* m;
  condition_variable* cv;

  explicit RunCounter(std::shared_ptr<Session> s, uint64* v, mutex* m,
                      condition_variable* cv)
      : session(std::move(s)), value(v), m(m), cv(cv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_0(mht_0_v, 211, "", "./tensorflow/python/client/session_ref.cc", "RunCounter");

    mutex_lock l(*m);
    ++*value;
  }

  ~RunCounter() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_1(mht_1_v, 219, "", "./tensorflow/python/client/session_ref.cc", "~RunCounter");

    mutex_lock l(*m);
    if (--*value == 0) {
      cv->notify_all();
    }
  }
};

std::string SessionToHandle(Session* session) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_2(mht_2_v, 230, "", "./tensorflow/python/client/session_ref.cc", "SessionToHandle");

  return strings::Printf("%llu", static_cast<unsigned long long>(
                                     reinterpret_cast<uintptr_t>(session)));
}

// The Session interface has many methods of the form:
//
// X(a, b);
// X(RunOptions, a, b);
//
// Not all sessions support the second case (with an empty RunOptions()).
// We use this variable as a sentinel to dispatch to the correct call.
RunOptions* kEmptyRunOptions() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_3(mht_3_v, 245, "", "./tensorflow/python/client/session_ref.cc", "kEmptyRunOptions");

  static RunOptions* options = new RunOptions();
  return options;
}

}  // namespace

// Run the given session operation, recording start and end timestamps.
// If the operation returns a bad status, return after flushing the current
// log request.  This should be run _after_ all request information has been
// added to the current op.
#define RUN_WITH_TIMESTAMP(OpName, ...)              \
  op.set_start_time_us(Env::Default()->NowMicros()); \
  Status status = session->OpName(__VA_ARGS__);      \
  op.set_end_time_us(Env::Default()->NowMicros());   \
  if (!status.ok()) {                                \
    Flush(op).IgnoreError();                         \
    return status;                                   \
  }

// Records requests (and optionally responses) performed against a session.
// The resulting replay log can be used with the `tf_replay` tool to replicate
// the operations against a simulated environment, without requiring the
// original code or cluster setup.
//
// Session logging by setting the TF_REPLAY_LOG_FILE environment variable.
class SessionLogger {
 public:
  SessionLogger() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_4(mht_4_v, 276, "", "./tensorflow/python/client/session_ref.cc", "SessionLogger");

    std::string log_name = getenv("TF_REPLAY_LOG_FILE");
    LOG(INFO) << "Constructing new session logger for " << log_name;
    TF_CHECK_OK(
        Env::Default()->RecursivelyCreateDir(string(io::Dirname(log_name))));
    Env::Default()->DeleteFile(log_name).IgnoreError();

    TF_CHECK_OK(Env::Default()->NewWritableFile(log_name, &log_file_));
    log_writer_ = absl::make_unique<io::RecordWriter>(log_file_.get());
  }

  ~SessionLogger() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_5(mht_5_v, 290, "", "./tensorflow/python/client/session_ref.cc", "~SessionLogger");

    log_writer_->Close().IgnoreError();
    log_writer_.release();
    log_file_->Close().IgnoreError();
  }

  Status RecordNewSession(Session* session) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_6(mht_6_v, 299, "", "./tensorflow/python/client/session_ref.cc", "RecordNewSession");

    ReplayOp op;
    NewReplaySession* req = op.mutable_new_replay_session();
    req->set_session_handle(SessionToHandle(session));
    return Flush(op);
  }

  Status RecordRun(Session* session,
                   const std::vector<std::pair<string, Tensor> >& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_7(mht_7_v, 313, "", "./tensorflow/python/client/session_ref.cc", "RecordRun");

    return RecordRun(session, *kEmptyRunOptions(), inputs, output_tensor_names,
                     target_node_names, outputs, nullptr);
  }

  Status RecordRun(Session* session, const RunOptions& run_options,
                   const std::vector<std::pair<string, Tensor> >& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs, RunMetadata* run_metadata) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_8(mht_8_v, 325, "", "./tensorflow/python/client/session_ref.cc", "RecordRun");

    ReplayOp op;
    RunStepRequest* req = op.mutable_run_step();
    RunStepResponse* resp = op.mutable_run_step_response();

    req->set_session_handle(SessionToHandle(session));
    *req->mutable_options() = run_options;

    for (const auto& it : inputs) {
      NamedTensorProto* feed = req->add_feed();
      feed->set_name(it.first);
      it.second.AsProtoField(feed->mutable_tensor());
    }

    // Build an index from fetch tensor name to first index in
    // output_tensor_names.
    std::unordered_map<string, int> output_name_to_offset;
    for (int i = 0, end = output_tensor_names.size(); i < end; ++i) {
      const string& name = output_tensor_names[i];
      if (output_name_to_offset.insert(std::make_pair(name, i)).second) {
        req->add_fetch(name);
      }
    }
    for (const string& target : target_node_names) {
      req->add_target(target);
    }

    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Run, inputs, output_tensor_names, target_node_names,
                         outputs);
    } else {
      RUN_WITH_TIMESTAMP(Run, run_options, inputs, output_tensor_names,
                         target_node_names, outputs, run_metadata);
    }

    for (size_t i = 0; i < outputs->size(); ++i) {
      const Tensor& tensor = (*outputs)[i];
      NamedTensorProto* tproto = resp->add_tensor();
      tensor.AsProtoField(tproto->mutable_tensor());
      tproto->set_name(output_tensor_names[i]);
    }

    if (run_metadata) {
      *resp->mutable_metadata() = *run_metadata;
    }

    return Flush(op);
  }

  Status RecordCreate(Session* session, const GraphDef& graph) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_9(mht_9_v, 377, "", "./tensorflow/python/client/session_ref.cc", "RecordCreate");

    return RecordCreate(session, *kEmptyRunOptions(), graph);
  }

  // N.B. RunOptions is not stored (it has no entry in CreateRequest)
  Status RecordCreate(Session* session, const RunOptions& run_options,
                      const GraphDef& graph) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_10(mht_10_v, 386, "", "./tensorflow/python/client/session_ref.cc", "RecordCreate");

    ReplayOp op;
    CreateSessionRequest* req = op.mutable_create_session();
    *req->mutable_graph_def() = graph;

    CreateSessionResponse* resp = op.mutable_create_session_response();
    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Create, graph);
    } else {
      RUN_WITH_TIMESTAMP(Create, run_options, graph);
    }
    resp->set_session_handle(SessionToHandle(session));
    return Flush(op);
  }

  Status RecordExtend(Session* session, const GraphDef& graph) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_11(mht_11_v, 404, "", "./tensorflow/python/client/session_ref.cc", "RecordExtend");

    return RecordExtend(session, *kEmptyRunOptions(), graph);
  }

  // N.B. RunOptions is not stored (it has no entry in ExtendRequest)
  Status RecordExtend(Session* session, const RunOptions& run_options,
                      const GraphDef& graph) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_12(mht_12_v, 413, "", "./tensorflow/python/client/session_ref.cc", "RecordExtend");

    ReplayOp op;
    ExtendSessionRequest* req = op.mutable_extend_session();
    op.mutable_extend_session_response();
    req->set_session_handle(SessionToHandle(session));
    *req->mutable_graph_def() = graph;
    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Extend, graph);
    } else {
      RUN_WITH_TIMESTAMP(Extend, run_options, graph);
    }

    return Flush(op);
  }

  Status RecordClose(Session* session) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_13(mht_13_v, 431, "", "./tensorflow/python/client/session_ref.cc", "RecordClose");

    return RecordClose(session, *kEmptyRunOptions());
  }

  // N.B. RunOptions is not stored (it has no entry in CloseRequest)
  Status RecordClose(Session* session, const RunOptions& run_options) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_14(mht_14_v, 439, "", "./tensorflow/python/client/session_ref.cc", "RecordClose");

    ReplayOp op;
    CloseSessionRequest* req = op.mutable_close_session();
    req->set_session_handle(SessionToHandle(session));
    op.mutable_close_session_response();
    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Close);
    } else {
      RUN_WITH_TIMESTAMP(Close, run_options);
    }
    return Flush(op);
  }

  Status RecordListDevices(Session* session,
                           std::vector<DeviceAttributes>* response) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_15(mht_15_v, 456, "", "./tensorflow/python/client/session_ref.cc", "RecordListDevices");

    ReplayOp op;
    ListDevicesRequest* req = op.mutable_list_devices();
    ListDevicesResponse* resp = op.mutable_list_devices_response();
    req->set_session_handle(SessionToHandle(session));
    RUN_WITH_TIMESTAMP(ListDevices, response);

    // TODO(power) -- local vs remote device distinction is lost here!
    *resp->mutable_local_device() = {response->begin(), response->end()};
    return Flush(op);
  }

  Status RecordPRunSetup(Session* session,
                         const std::vector<string>& input_names,
                         const std::vector<string>& output_names,
                         const std::vector<string>& target_nodes,
                         string* handle) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_16(mht_16_v, 475, "", "./tensorflow/python/client/session_ref.cc", "RecordPRunSetup");

    ReplayOp op;
    PartialRunSetupRequest* req = op.mutable_partial_run_setup();
    req->set_session_handle(SessionToHandle(session));
    for (auto& input : input_names) {
      req->add_feed(input);
    }
    for (auto& output : output_names) {
      req->add_fetch(output);
    }
    for (auto& target : target_nodes) {
      req->add_target(target);
    }
    RUN_WITH_TIMESTAMP(PRunSetup, input_names, output_names, target_nodes,
                       handle);
    op.mutable_partial_run_setup_response()->set_partial_run_handle(*handle);
    return Flush(op);
  }

  Status RecordPRun(Session* session, const string& handle,
                    const std::vector<std::pair<string, Tensor> >& inputs,
                    const std::vector<string>& output_names,
                    std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_17(mht_17_v, 501, "", "./tensorflow/python/client/session_ref.cc", "RecordPRun");

    ReplayOp op;
    RunStepRequest* req = op.mutable_run_step();
    RunStepResponse* resp = op.mutable_run_step_response();
    req->set_session_handle(SessionToHandle(session));

    // Mark this step as a partial run for replay.
    req->set_partial_run_handle(handle);
    for (auto& input : inputs) {
      auto* feed = req->add_feed();
      feed->set_name(input.first);
      input.second.AsProtoField(feed->mutable_tensor());
    }

    for (auto& output : output_names) {
      req->add_fetch(output);
    }

    RUN_WITH_TIMESTAMP(PRun, handle, inputs, output_names, outputs);

    for (size_t i = 0; i < outputs->size(); ++i) {
      const Tensor& tensor = (*outputs)[i];
      NamedTensorProto* tproto = resp->add_tensor();
      tensor.AsProtoField(tproto->mutable_tensor());
      tproto->set_name(output_names[i]);
    }

    return Flush(op);
  }

  Status RecordMakeCallable(Session* session,
                            const CallableOptions& callable_options,
                            Session::CallableHandle* handle) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_18(mht_18_v, 536, "", "./tensorflow/python/client/session_ref.cc", "RecordMakeCallable");

    ReplayOp op;
    MakeCallableRequest* req = op.mutable_make_callable();
    req->set_session_handle(SessionToHandle(session));
    *req->mutable_options() = callable_options;

    RUN_WITH_TIMESTAMP(MakeCallable, callable_options, handle);

    MakeCallableResponse* resp = op.mutable_make_callable_response();
    resp->set_handle(*handle);

    return Flush(op);
  }

  Status RecordRunCallable(Session* session, Session::CallableHandle handle,
                           const std::vector<Tensor>& feed_tensors,
                           std::vector<Tensor>* fetch_tensors,
                           RunMetadata* run_metadata) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_19(mht_19_v, 556, "", "./tensorflow/python/client/session_ref.cc", "RecordRunCallable");

    ReplayOp op;
    RunCallableRequest* req = op.mutable_run_callable();
    req->set_session_handle(SessionToHandle(session));
    req->set_handle(handle);
    for (auto& tensor : feed_tensors) {
      tensor.AsProtoField(req->add_feed());
    }
    RUN_WITH_TIMESTAMP(RunCallable, handle, feed_tensors, fetch_tensors,
                       run_metadata);

    RunCallableResponse* resp = op.mutable_run_callable_response();
    if (run_metadata) {
      *resp->mutable_metadata() = *run_metadata;
    }
    for (const Tensor& tensor : *fetch_tensors) {
      tensor.AsProtoTensorContent(resp->add_fetch());
    }
    return Flush(op);
  }

  Status RecordReleaseCallable(Session* session,
                               Session::CallableHandle handle) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_20(mht_20_v, 581, "", "./tensorflow/python/client/session_ref.cc", "RecordReleaseCallable");

    ReplayOp op;
    ReleaseCallableRequest* req = op.mutable_release_callable();
    req->set_session_handle(SessionToHandle(session));
    req->set_handle(handle);
    RUN_WITH_TIMESTAMP(ReleaseCallable, handle);
    return Flush(op);
  }

 private:
  Status Flush(const ReplayOp& op) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_21(mht_21_v, 594, "", "./tensorflow/python/client/session_ref.cc", "Flush");

    mutex_lock l(log_mutex_);

    string buf;
    op.SerializeToString(&buf);
    TF_RETURN_IF_ERROR(log_writer_->WriteRecord(buf));

    // TODO(b/116624106): Not all file-systems respect calls to `Sync()`
    return log_file_->Sync();
  }

  std::unique_ptr<WritableFile> log_file_;
  std::unique_ptr<io::RecordWriter> log_writer_;
  mutex log_mutex_;
};

static SessionLogger* global_session_logger() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_22(mht_22_v, 613, "", "./tensorflow/python/client/session_ref.cc", "global_session_logger");

  static SessionLogger* logger = new SessionLogger();
  return logger;
}

SessionRef::SessionRef(Session* session) : session_(session) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_23(mht_23_v, 621, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::SessionRef");

  if (getenv("TF_REPLAY_LOG_FILE") != nullptr) {
    logger_ = global_session_logger();
    logger_->RecordNewSession(this->session_.get()).IgnoreError();
  } else {
    logger_ = nullptr;
  }
}

SessionRef::~SessionRef() = default;

Status SessionRef::CheckNotClosed() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_24(mht_24_v, 635, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::CheckNotClosed");

  mutex_lock l(run_lock_);
  if (session_ == nullptr) return errors::Cancelled("Session has been closed.");
  return ::tensorflow::Status::OK();
}

// If logging is active, log the start and end time of the operation along with
// the request and response.
#define LOG_AND_RUN_OPERATION(OpName, ...)                          \
  TF_RETURN_IF_ERROR(CheckNotClosed());                             \
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_); \
  if (!logger_) {                                                   \
    return rc.session->OpName(__VA_ARGS__);                         \
  }                                                                 \
  return logger_->Record##OpName(rc.session.get(), __VA_ARGS__);

Status SessionRef::Run(const RunOptions& run_options,
                       const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs,
                       RunMetadata* run_metadata) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_25(mht_25_v, 659, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Run");

  LOG_AND_RUN_OPERATION(Run, run_options, inputs, output_tensor_names,
                        target_node_names, outputs, run_metadata);
}

Status SessionRef::Run(const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_26(mht_26_v, 670, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Run");

  LOG_AND_RUN_OPERATION(Run, inputs, output_tensor_names, target_node_names,
                        outputs);
}

Status SessionRef::Create(const GraphDef& graph) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_27(mht_27_v, 678, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Create");

  LOG_AND_RUN_OPERATION(Create, graph);
}

Status SessionRef::Create(const RunOptions& run_options,
                          const GraphDef& graph) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_28(mht_28_v, 686, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Create");

  LOG_AND_RUN_OPERATION(Create, run_options, graph);
}

Status SessionRef::Extend(const RunOptions& run_options,
                          const GraphDef& graph) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_29(mht_29_v, 694, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Extend");

  LOG_AND_RUN_OPERATION(Extend, run_options, graph);
}

Status SessionRef::Extend(const GraphDef& graph) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_30(mht_30_v, 701, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Extend");

  LOG_AND_RUN_OPERATION(Extend, graph);
}

Status SessionRef::ListDevices(std::vector<DeviceAttributes>* response) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_31(mht_31_v, 708, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::ListDevices");

  LOG_AND_RUN_OPERATION(ListDevices, response);
}

Status SessionRef::PRunSetup(const std::vector<string>& input_names,
                             const std::vector<string>& output_names,
                             const std::vector<string>& target_nodes,
                             string* handle) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_32(mht_32_v, 718, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::PRunSetup");

  LOG_AND_RUN_OPERATION(PRunSetup, input_names, output_names, target_nodes,
                        handle);
}

Status SessionRef::PRun(const string& handle,
                        const std::vector<std::pair<string, Tensor> >& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_33(mht_33_v, 730, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::PRun");

  LOG_AND_RUN_OPERATION(PRun, handle, inputs, output_names, outputs);
}

Status SessionRef::MakeCallable(const CallableOptions& callable_options,
                                CallableHandle* out_handle) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_34(mht_34_v, 738, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::MakeCallable");

  LOG_AND_RUN_OPERATION(MakeCallable, callable_options, out_handle);
}

Status SessionRef::RunCallable(CallableHandle handle,
                               const std::vector<Tensor>& feed_tensors,
                               std::vector<Tensor>* fetch_tensors,
                               RunMetadata* run_metadata) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_35(mht_35_v, 748, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::RunCallable");

  LOG_AND_RUN_OPERATION(RunCallable, handle, feed_tensors, fetch_tensors,
                        run_metadata);
}

Status SessionRef::ReleaseCallable(CallableHandle handle) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_36(mht_36_v, 756, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::ReleaseCallable");

  {
    mutex_lock l(run_lock_);
    if (session_ == nullptr) {
      // Session already closed. Do nothing.
      return Status::OK();
    }
  }
  LOG_AND_RUN_OPERATION(ReleaseCallable, handle);
}

Status SessionRef::Close(const RunOptions& run_options) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_37(mht_37_v, 770, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Close");

  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(run_lock_);
  Status status;
  if (logger_) {
    status = logger_->RecordClose(session_.get(), run_options);
  } else {
    status = session_->Close(run_options);
  }
  session_.reset();
  while (run_count_ > 0) {
    run_finished_.wait(l);
  }
  return status;
}

Status SessionRef::Close() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSpythonPSclientPSsession_refDTcc mht_38(mht_38_v, 789, "", "./tensorflow/python/client/session_ref.cc", "SessionRef::Close");

  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(run_lock_);
  Status status;
  if (logger_) {
    status = logger_->RecordClose(session_.get());
  } else {
    status = session_->Close();
  }
  session_.reset();
  while (run_count_ > 0) {
    run_finished_.wait(l);
  }
  return status;
}

}  // namespace tensorflow
