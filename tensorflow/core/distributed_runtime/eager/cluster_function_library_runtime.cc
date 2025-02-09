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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc() {
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
#include "tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h"

#include <map>
#include <memory>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_execute_node.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace tensorflow {
namespace eager {
namespace {
void StripDefaultAttributesInRegisterFunctionOp(
    RegisterFunctionOp* register_function) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.cc", "StripDefaultAttributesInRegisterFunctionOp");

  StripDefaultAttributes(
      *OpRegistry::Global(),
      register_function->mutable_function_def()->mutable_node_def());
  for (auto& function :
       *register_function->mutable_library()->mutable_function()) {
    StripDefaultAttributes(*OpRegistry::Global(), function.mutable_node_def());
  }
}
}  // namespace

void EagerClusterFunctionLibraryRuntime::Instantiate(
    const string& function_name, const FunctionLibraryDefinition& lib_def,
    AttrSlice attrs, const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::LocalHandle* handle,
    FunctionLibraryRuntime::DoneCallback done) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.cc", "EagerClusterFunctionLibraryRuntime::Instantiate");

  auto target = options.target;
  auto released_op = std::make_unique<EagerOperation>(ctx_);
  Status s =
      released_op->Reset(function_name.c_str(), target.c_str(), true, nullptr);
  if (!s.ok()) {
    done(s);
    return;
  }
  if (!released_op->is_function()) {
    done(errors::Internal(function_name, " is not a function."));
    return;
  }

  VLOG(1) << "CFLR::Instantiate: " << function_name << " on " << target
          << " (this: " << this << ")";
  core::RefCountPtr<eager::EagerClient> eager_client;
  s = ctx_->GetClient(target, &eager_client);
  if (!s.ok()) {
    done(s);
    return;
  }

  if (eager_client == nullptr) {
    done(errors::InvalidArgument("Could not find eager client for target: ",
                                 target));
    return;
  }

  const FunctionLibraryDefinition& func_lib_def =
      options.lib_def ? *options.lib_def : lib_def;

  auto request = std::make_shared<EnqueueRequest>();
  auto response = std::make_shared<EnqueueResponse>();

  request->set_context_id(context_id_);

  RegisterFunctionOp* register_function =
      request->add_queue()->mutable_register_function();
  *register_function->mutable_function_def() =
      *func_lib_def.Find(function_name);
  register_function->set_is_component_function(true);
  *register_function->mutable_library() =
      func_lib_def.ReachableDefinitions(register_function->function_def())
          .ToProto();
  StripDefaultAttributesInRegisterFunctionOp(register_function);

  const absl::optional<std::vector<int>>& ret_indices = options.ret_indices;
  eager_client->EnqueueAsync(
      /*call_opts=*/nullptr, request.get(), response.get(),
      [this, request, response, handle, released_op = released_op.release(),
       target, ret_indices, eager_client = eager_client.get(),
       done](const Status& s) {
        {
          mutex_lock l(mu_);
          *handle = function_data_.size();
          function_data_.emplace_back(target, ret_indices, eager_client,
                                      absl::WrapUnique(released_op));
        }
        done(s);
      });
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc mht_2(mht_2_v, 294, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.cc", "EagerClusterFunctionLibraryRuntime::Run");

  std::vector<FunctionArg> function_args;
  for (const auto& tensor : args) {
    function_args.push_back(tensor);
  }
  std::vector<FunctionRet>* function_rets = new std::vector<FunctionRet>;
  Run(opts, handle, function_args, function_rets,
      [rets, function_rets, done = std::move(done)](const Status& s) {
        Status status = s;
        if (status.ok()) {
          for (const auto& t : *function_rets) {
            if (t.index() == 0) {
              rets->push_back(absl::get<Tensor>(t));
            } else {
              status.Update(
                  errors::Internal("Expect a Tensor as a remote function "
                                   "output but got a TensorShape."));
              break;
            }
          }
        }
        delete function_rets;
        done(status);
      });
}

void EagerClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle,
    gtl::ArraySlice<FunctionArg> args, std::vector<FunctionRet>* rets,
    FunctionLibraryRuntime::DoneCallback done) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc mht_3(mht_3_v, 327, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.cc", "EagerClusterFunctionLibraryRuntime::Run");

  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    DCHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  EagerClient* eager_client = function_data->eager_client.get();
  if (eager_client == nullptr) {
    done(errors::Internal("Could not find eager client"));
    return;
  }

  EagerOperation* op = function_data->op.get();
  if (!op->Inputs().empty()) {
    done(errors::Internal("Inputs should not be set during instantiation."));
    return;
  }

  auto request = std::make_shared<RunComponentFunctionRequest>();
  auto response = std::make_shared<RunComponentFunctionResponse>();
  request->set_context_id(context_id_);
  eager::Operation* remote_op = request->mutable_operation();

  if (function_data->ret_indices.has_value()) {
    for (const int ret_index : function_data->ret_indices.value()) {
      request->add_output_num(ret_index);
    }
  }

  for (const auto& arg : args) {
    if (arg.index() == 0) {
      absl::get<Tensor>(arg).AsProtoTensorContent(
          remote_op->add_op_inputs()->mutable_tensor());
    } else {
      remote_op->add_op_inputs()->mutable_remote_handle()->Swap(
          absl::get<RemoteTensorHandle*>(arg));
    }
  }

  // The remote component function should use the same op_id as its parent
  // multi-device function's in order to get the global unique op_id generated
  // by the master context.
  if (opts.op_id.has_value()) {
    remote_op->set_id(opts.op_id.value());
  } else {
    remote_op->set_id(kInvalidOpId);
  }
  remote_op->set_is_function(true);
  remote_op->set_is_component_function(true);
  remote_op->set_func_step_id(opts.step_id);
  remote_op->set_name(op->Name());
  op->Attrs().FillAttrValueMap(remote_op->mutable_attrs());
  remote_op->set_device(function_data->target);

  CancellationManager* cm = opts.cancellation_manager;
  CancellationToken token = 0;
  auto call_opts = std::make_shared<CallOptions>();
  call_opts->SetTimeout(
      ctx_->session_options().config.operation_timeout_in_ms());
  if (cm != nullptr) {
    token = cm->get_cancellation_token();
    const bool already_cancelled = !cm->RegisterCallback(
        token,
        [call_opts, request, response, done]() { call_opts->StartCancel(); });
    if (already_cancelled) {
      done(errors::Cancelled("EagerClusterFunctionLibraryRuntime::Run"));
      return;
    }
  }

  // Execute component function on remote worker using RunComponentFunction RPC.
  // Different from executing remote functions with Enqueue, this method runs
  // a function on remote worker without tying up a thread (i.e., pure
  // asynchronously).
  eager_client->RunComponentFunctionAsync(
      call_opts.get(), request.get(), response.get(),
      [request, response, rets, call_opts, cm, token,
       done = std::move(done)](const Status& s) {
        if (cm != nullptr) {
          cm->TryDeregisterCallback(token);
        }
        if (!s.ok()) {
          done(s);
          return;
        }
        if (!response->shape().empty() && !response->tensor().empty()) {
          done(errors::Internal(
              "Both shape and tensor are specified in the same response"));
          return;
        }
        for (const auto& shape : response->shape()) {
          rets->push_back(shape);
        }
        for (const auto& tensor_proto : response->tensor()) {
          Tensor t;
          if (t.FromProto(tensor_proto)) {
            rets->push_back(std::move(t));
          } else {
            done(errors::Internal("Could not convert tensor proto: ",
                                  tensor_proto.DebugString()));
            return;
          }
        }
        done(Status::OK());
      });
}

void EagerClusterFunctionLibraryRuntime::CleanUp(
    uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
    FunctionLibraryRuntime::DoneCallback done) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc mht_4(mht_4_v, 441, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.cc", "EagerClusterFunctionLibraryRuntime::CleanUp");

  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    DCHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  EagerClient* eager_client = function_data->eager_client.get();
  if (eager_client == nullptr) {
    done(errors::Internal("Could not find eager client"));
    return;
  }

  auto request = std::make_shared<EnqueueRequest>();
  auto response = std::make_shared<EnqueueResponse>();
  request->set_context_id(context_id_);
  CleanupFunctionOp* cleanup_function =
      request->add_queue()->mutable_cleanup_function();
  cleanup_function->set_step_id(step_id);
  // StreamingEnqueueAsync could be blocking when streaming RPC is disabled.
  // CleanUp() needs to be non-blocking since it would be invoked inside the
  // enqueue done callback of Run(). So we don't use StreamingEnqueueAsync here.
  eager_client->EnqueueAsync(
      /*call_opts=*/nullptr, request.get(), response.get(),
      [request, response, done](const Status& status) { done(status); });
}

DistributedFunctionLibraryRuntime* CreateClusterFLR(
    const uint64 context_id, EagerContext* ctx, WorkerSession* worker_session) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSeagerPScluster_function_library_runtimeDTcc mht_5(mht_5_v, 473, "", "./tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.cc", "CreateClusterFLR");

  return new EagerClusterFunctionLibraryRuntime(
      context_id, ctx, worker_session->remote_device_mgr());
}

}  // namespace eager
}  // namespace tensorflow
