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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"

#include <unordered_map>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_master.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

const char* const kSchemePrefix = "grpc://";
const size_t kSchemePrefixLength = strlen(kSchemePrefix);

GrpcSession::GrpcSession(const SessionOptions& options)
    : options_(options), current_graph_version_(-1) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::GrpcSession");
}

GrpcSession::~GrpcSession() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::~GrpcSession");
}

/* static */
Status GrpcSession::Create(const SessionOptions& options,
                           std::unique_ptr<GrpcSession>* out_session) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Create");

  std::unique_ptr<GrpcSession> session(new GrpcSession(options));
  std::unique_ptr<MasterInterface> master;
  // For testing, we enable the client to disable the use of the local
  // master registry, so that the RPC stack is exercised.
  if (!options.config.rpc_options().use_rpc_for_inprocess_master()) {
    master = LocalMaster::Lookup(options.target);
  }
  if (!master) {
    SharedGrpcChannelPtr master_channel;
    TF_RETURN_IF_ERROR(
        NewHostPortGrpcChannel(options.target.substr(kSchemePrefixLength),
                               &options.config.rpc_options(), &master_channel));
    master.reset(NewGrpcMaster(master_channel));
  } else {
    session->is_local_ = true;
  }
  session->SetRemoteMaster(std::move(master));
  *out_session = std::move(session);
  return Status::OK();
}

namespace {
// Re-encodes constant represented in tensor proto into
// tensor_content, which is slightly better (less copies and lower peak
// memory usage) when used with rpc subsystems.
void ReEncodeConsts(GraphDef* gdef) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "ReEncodeConsts");

  for (NodeDef& ndef : *(gdef->mutable_node())) {
    if (ndef.op() == "Const") {
      TensorProto* proto = nullptr;
      for (auto& attr : *ndef.mutable_attr()) {
        if (attr.first == "value") {
          proto = attr.second.mutable_tensor();
        }
      }
      if (proto != nullptr && proto->tensor_content().empty() &&
          proto->ByteSizeLong() > 64) {
        // If the constant is encoded with repeated proto fields and
        // it is moderate large, we re-encode it in tensor_content as
        // a Cord. This is mildly helpful for reducing the peak memory
        // usage on the server side where GraphDef/NodeDef are copied
        // quite often.
        Tensor parsed(proto->dtype());
        if (parsed.FromProto(*proto)) {
          parsed.AsProtoTensorContent(proto);
        }
      }
    }
  }
}
}  // namespace

void GrpcSession::SetHandleAndGraphVersion(string handle,
                                           int64_t graph_version) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_4(mht_4_v, 282, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::SetHandleAndGraphVersion");

  mutex_lock l(mu_);
  handle_ = std::move(handle);
  current_graph_version_ = graph_version;
}

Status GrpcSession::Handle(string* out_handle) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_5(mht_5_v, 291, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Handle");

  mutex_lock l(mu_);
  if (handle_.empty()) {
    return errors::InvalidArgument("A session is not created yet....");
  }
  *out_handle = handle_;
  return Status::OK();
}

Status GrpcSession::CreateImpl(CallOptions* call_options, GraphDef graph) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_6(mht_6_v, 303, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::CreateImpl");

  {
    mutex_lock l(mu_);
    if (!handle_.empty()) {
      return errors::InvalidArgument("A session is alive.");
    }
  }
  CreateSessionRequest req;
  *req.mutable_config() = options_.config;
  req.mutable_graph_def()->Swap(&graph);
  req.set_target(options_.target);
  ReEncodeConsts(req.mutable_graph_def());
  CreateSessionResponse resp;
  Status s = master_->CreateSession(call_options, &req, &resp);
  if (s.ok()) {
    SetHandleAndGraphVersion(resp.session_handle(), resp.graph_version());
  }
  return s;
}

Status GrpcSession::Create(const GraphDef& graph) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_7(mht_7_v, 326, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Create");

  return Create(GraphDef(graph));
}

Status GrpcSession::Create(const RunOptions& run_options,
                           const GraphDef& graph) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_8(mht_8_v, 334, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Create");

  return Create(run_options, GraphDef(graph));
}

Status GrpcSession::Create(GraphDef&& graph) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_9(mht_9_v, 341, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Create");

  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return CreateImpl(&call_options, std::move(graph));
}

Status GrpcSession::Create(const RunOptions& run_options, GraphDef&& graph) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_10(mht_10_v, 350, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Create");

  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return CreateImpl(&call_options, std::move(graph));
}

Status GrpcSession::ExtendImpl(CallOptions* call_options, GraphDef graph) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_11(mht_11_v, 359, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::ExtendImpl");

  bool handle_is_empty;
  {
    mutex_lock l(mu_);
    handle_is_empty = handle_.empty();
  }
  if (handle_is_empty) {
    // Session was uninitialized, so simply initialize the session with 'graph'.
    return Create(std::move(graph));
  }
  mutex_lock l(mu_);
  ExtendSessionRequest req;
  req.set_session_handle(handle_);
  req.mutable_graph_def()->Swap(&graph);
  req.set_current_graph_version(current_graph_version_);
  ExtendSessionResponse resp;
  Status s = master_->ExtendSession(call_options, &req, &resp);
  if (s.ok()) {
    current_graph_version_ = resp.new_graph_version();
  }
  return s;
}

Status GrpcSession::Extend(const GraphDef& graph) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_12(mht_12_v, 385, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Extend");

  return Extend(GraphDef(graph));
}

Status GrpcSession::Extend(const RunOptions& run_options,
                           const GraphDef& graph) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_13(mht_13_v, 393, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Extend");

  return Extend(run_options, GraphDef(graph));
}

Status GrpcSession::Extend(GraphDef&& graph) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_14(mht_14_v, 400, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Extend");

  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return ExtendImpl(&call_options, std::move(graph));
}

Status GrpcSession::Extend(const RunOptions& run_options, GraphDef&& graph) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_15(mht_15_v, 409, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Extend");

  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return ExtendImpl(&call_options, std::move(graph));
}

Status GrpcSession::RunHelper(
    const RunOptions& run_options,
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names, std::vector<Tensor>* outputs,
    RunMetadata* run_metadata, const string& prun_handle) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("prun_handle: \"" + prun_handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_16(mht_16_v, 424, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::RunHelper");

  // Convert to proto
  std::unique_ptr<MutableRunStepRequestWrapper> req(
      master_->CreateRunStepRequest());
  std::unique_ptr<MutableRunStepResponseWrapper> resp(
      master_->CreateRunStepResponse());

  *req->mutable_options() = run_options;

  if (run_options.timeout_in_ms() == 0) {
    req->mutable_options()->set_timeout_in_ms(
        options_.config.operation_timeout_in_ms());
  }

  if (!prun_handle.empty()) {
    req->set_partial_run_handle(prun_handle);
  }

  for (const auto& it : inputs) {
    req->add_feed(it.first, it.second);
  }

  // Support long error messages by storing the error code in the response body.
  req->set_store_errors_in_response_body(true);

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

  CallOptions call_options;
  call_options.SetTimeout(req->options().timeout_in_ms());
  TF_RETURN_IF_ERROR(RunProto(&call_options, req.get(), resp.get()));

  // Look for an extended error returned in the response body.
  if (resp->status_code() != error::Code::OK) {
    return resp->status();
  }

  if (!output_tensor_names.empty()) {
    outputs->resize(output_tensor_names.size());
  }

  // Convert response back to Tensors in the correct order.
  for (size_t i = 0; i < resp->num_tensors(); ++i) {
    auto fetch_it = output_name_to_offset.find(resp->tensor_name(i));
    if (fetch_it == output_name_to_offset.end()) {
      return errors::Internal("Received response for unrequested fetch: ",
                              resp->tensor_name(i));
    }

    Tensor output;
    TF_RETURN_IF_ERROR(resp->TensorValue(i, &output));
    (*outputs)[fetch_it->second] = output;
  }
  // In the unlikely event that output_tensor_names contains duplicates, fill in
  // the duplicate values.
  if (output_name_to_offset.size() != output_tensor_names.size()) {
    for (int i = 0, end = output_tensor_names.size(); i < end; ++i) {
      const string& name = output_tensor_names[i];
      int offset = output_name_to_offset[name];
      if (offset != i) {
        (*outputs)[i] = (*outputs)[offset];
      }
    }
  }

  if (run_metadata) {
    run_metadata->Swap(resp->mutable_metadata());
  }

  return Status::OK();
}

Status GrpcSession::Run(const RunOptions& run_options,
                        const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_tensor_names,
                        const std::vector<string>& target_node_names,
                        std::vector<Tensor>* outputs,
                        RunMetadata* run_metadata) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_17(mht_17_v, 514, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Run");

  return RunHelper(run_options, inputs, output_tensor_names, target_node_names,
                   outputs, run_metadata, /* prun_handle */ "");
}

Status GrpcSession::Run(const std::vector<std::pair<string, Tensor>>& inputs,
                        const std::vector<string>& output_tensor_names,
                        const std::vector<string>& target_node_names,
                        std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_18(mht_18_v, 525, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Run");

  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return Run(run_options, inputs, output_tensor_names, target_node_names,
             outputs, nullptr);
}

Status GrpcSession::RunProto(CallOptions* call_options,
                             MutableRunStepRequestWrapper* req,
                             MutableRunStepResponseWrapper* resp) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_19(mht_19_v, 537, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::RunProto");

  string handle;
  TF_RETURN_IF_ERROR(Handle(&handle));
  req->set_session_handle(handle);
  return master_->RunStep(call_options, req, resp);
}

Status GrpcSession::PRunSetup(const std::vector<string>& input_names,
                              const std::vector<string>& output_names,
                              const std::vector<string>& target_nodes,
                              string* handle) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_20(mht_20_v, 550, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::PRunSetup");

  // Convert to proto
  PartialRunSetupRequest req;
  PartialRunSetupResponse resp;
  CallOptions call_options;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  for (const string& feed : input_names) {
    req.add_feed(feed);
  }
  for (const string& fetch : output_names) {
    req.add_fetch(fetch);
  }
  for (const string& target : target_nodes) {
    req.add_target(target);
  }
  if (!is_local_) req.set_request_id(GetUniqueRequestId());
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->PartialRunSetup(&call_options, &req, &resp));
  *handle = resp.partial_run_handle();
  return Status::OK();
}

Status GrpcSession::PRun(const string& handle,
                         const std::vector<std::pair<string, Tensor>>& inputs,
                         const std::vector<string>& output_names,
                         std::vector<Tensor>* outputs) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("handle: \"" + handle + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_21(mht_21_v, 579, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::PRun");

  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return RunHelper(run_options, inputs, output_names, /* targets */ {}, outputs,
                   /* run_metadata */ nullptr, handle);
}

Status GrpcSession::Close() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_22(mht_22_v, 589, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Close");

  CloseSessionRequest req;
  {
    mutex_lock l(mu_);
    if (handle_.empty()) {
      return Status::OK();
    }
    req.set_session_handle(handle_);
    handle_.clear();
  }
  CloseSessionResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return master_->CloseSession(&call_options, &req, &resp);
}

Status GrpcSession::ListDevices(std::vector<DeviceAttributes>* response) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_23(mht_23_v, 608, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::ListDevices");

  ListDevicesRequest req;
  {
    mutex_lock l(mu_);
    req.set_session_handle(handle_);
  }
  if (req.session_handle().empty()) {
    LOG(WARNING) << "GrpcSession::ListDevices will initialize the session with "
                    "an empty graph and other defaults because the session has "
                    "not yet been created.";
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(Create(graph_def));
    {
      mutex_lock l(mu_);
      req.set_session_handle(handle_);
    }
  }
  ListDevicesResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  Status s = master_->ListDevices(&call_options, &req, &resp);
  if (!s.ok()) {
    LOG(ERROR) << "Could not list devices: " << s;
    return s;
  }

  response->clear();
  response->reserve(resp.local_device_size() + resp.remote_device_size());
  for (const auto& device_attr : resp.local_device()) {
    response->emplace_back(device_attr);
  }
  for (const auto& device_attr : resp.remote_device()) {
    response->emplace_back(device_attr);
  }
  return Status::OK();
}

void GrpcSession::SetRemoteMaster(std::unique_ptr<MasterInterface> master) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_24(mht_24_v, 648, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::SetRemoteMaster");

  master_ = std::move(master);
}

// Static method.
Status GrpcSession::Reset(const SessionOptions& options,
                          const std::vector<string>& containers) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_25(mht_25_v, 657, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::Reset");

  SharedGrpcChannelPtr master_channel;
  TF_RETURN_IF_ERROR(
      NewHostPortGrpcChannel(options.target.substr(kSchemePrefixLength),
                             /*rpc_options=*/nullptr, &master_channel));
  auto master = NewGrpcMaster(master_channel);
  ResetRequest req;
  req.mutable_container()->Reserve(containers.size());
  for (const auto& c : containers) req.add_container(c);
  ResetResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options.config.operation_timeout_in_ms());
  Status ret = master->Reset(&call_options, &req, &resp);
  delete master;
  return ret;
}

Status GrpcSession::MakeCallable(const CallableOptions& callable_options,
                                 CallableHandle* out_handle) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_26(mht_26_v, 678, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::MakeCallable");

  MakeCallableRequest req;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  *req.mutable_options() = callable_options;
  if (!is_local_) req.set_request_id(GetUniqueRequestId());
  MakeCallableResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->MakeCallable(&call_options, &req, &resp));
  *out_handle = resp.handle();
  return Status::OK();
}

Status GrpcSession::RunCallable(CallableHandle handle,
                                const std::vector<Tensor>& feed_tensors,
                                std::vector<Tensor>* fetch_tensors,
                                RunMetadata* run_metadata) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_27(mht_27_v, 697, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::RunCallable");

  RunCallableRequest req;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  req.set_handle(handle);
  if (!is_local_) req.set_request_id(GetUniqueRequestId());
  for (const Tensor& feed : feed_tensors) {
    feed.AsProtoTensorContent(req.mutable_feed()->Add());
  }

  RunCallableResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->RunCallable(&call_options, &req, &resp));
  for (const TensorProto& fetch : resp.fetch()) {
    Tensor fetch_tensor;
    if (!fetch_tensor.FromProto(cpu_allocator(), fetch)) {
      return errors::Internal(
          "Could not parse fetched tensor data in response from master.");
    }
    fetch_tensors->push_back(std::move(fetch_tensor));
  }
  return Status::OK();
}

Status GrpcSession::ReleaseCallable(CallableHandle handle) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_28(mht_28_v, 724, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSession::ReleaseCallable");

  ReleaseCallableRequest req;
  TF_RETURN_IF_ERROR(Handle(req.mutable_session_handle()));
  req.set_handle(handle);
  ReleaseCallableResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return master_->ReleaseCallable(&call_options, &req, &resp);
}

class GrpcSessionFactory : public SessionFactory {
 public:
  bool AcceptsOptions(const SessionOptions& options) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_29(mht_29_v, 739, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "AcceptsOptions");

    return absl::StartsWith(options.target, kSchemePrefix);
  }

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_30(mht_30_v, 747, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "NewSession");

    std::unique_ptr<GrpcSession> session;
    TF_RETURN_IF_ERROR(GrpcSession::Create(options, &session));
    *out_session = session.release();
    return Status::OK();
  }

  // Invokes the session specific static method to reset containers.
  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_31(mht_31_v, 759, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "Reset");

    return GrpcSession::Reset(options, containers);
  }
};

class GrpcSessionRegistrar {
 public:
  GrpcSessionRegistrar() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_sessionDTcc mht_32(mht_32_v, 769, "", "./tensorflow/core/distributed_runtime/rpc/grpc_session.cc", "GrpcSessionRegistrar");

    SessionFactory::Register("GRPC_SESSION", new GrpcSessionFactory());
  }
};
static GrpcSessionRegistrar registrar;

}  // namespace tensorflow
