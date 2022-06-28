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
class MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc {
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
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/generic/generic_stub.h"
#include "grpcpp/impl/codegen/client_context.h"
#include "grpcpp/impl/codegen/server_context.h"
#include "grpcpp/impl/codegen/status.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
// Needed for encoding and decoding ResourceDeleter Variant.
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/distribute/experimental/rpc/kernels/grpc_credentials.h"
#include "tensorflow/distribute/experimental/rpc/kernels/grpc_rpc_service.h"
#include "tensorflow/distribute/experimental/rpc/proto/tf_rpc_service.pb.h"

namespace tensorflow {
namespace rpc {

// Register a function to local built in server or RPC server
class RpcServerRegisterOp : public OpKernel {
 public:
  explicit RpcServerRegisterOp(OpKernelConstruction* ctx);

  void Compute(OpKernelContext* ctx) override;

 private:
  NameAttrList func_;
  StructuredValue output_specs_;
  StructuredValue input_specs_;
  TF_DISALLOW_COPY_AND_ASSIGN(RpcServerRegisterOp);
};

// Create a server resource to store registered functions
class RpcServerOp : public OpKernel {
 public:
  explicit RpcServerOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RpcServerOp);
};

// Start GRPC server with registered methods
class RpcServerStartOp : public OpKernel {
 public:
  explicit RpcServerStartOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RpcServerStartOp);
};

// Create a client resource to store registered functions.
class RpcClientOp : public AsyncOpKernel {
 public:
  explicit RpcClientOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::string name_;
  bool list_registered_methods_;
  TF_DISALLOW_COPY_AND_ASSIGN(RpcClientOp);
};

// Remote RPC using client handle passed and returns a future Resource handle to
// get Status and value.
class RpcCallOp : public OpKernel {
 public:
  explicit RpcCallOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RpcCallOp);
};

// Remote Check Status Op waits till the RPC issued by Call Op is finished.
class RpcCheckStatusOp : public AsyncOpKernel {
 public:
  explicit RpcCheckStatusOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RpcCheckStatusOp);
};

// Op to get response output after RPC Call.
class RpcGetValueOp : public AsyncOpKernel {
 public:
  explicit RpcGetValueOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RpcGetValueOp);
};

class DeleteRpcFutureResourceOp : public OpKernel {
 public:
  explicit DeleteRpcFutureResourceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_0(mht_0_v, 316, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "DeleteRpcFutureResourceOp");
}

 protected:
  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_1(mht_1_v, 322, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "Compute");

    const ResourceHandle& handle = ctx->input(0).flat<ResourceHandle>()(0);
    // The resource is guaranteed to exist because the variant tensor
    // wrapping the deleter is provided as an unused input to this op, which
    // guarantees that it has not run yet.
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete(handle));
  }
};

struct FunctionMetadata {
  FunctionLibraryRuntime::Handle handle;
  FunctionLibraryRuntime* lib;
  std::vector<Tensor> captured_inputs;
  StructuredValue input_specs;
  StructuredValue output_specs;
};

class FunctionRegistry {
 public:
  std::string DebugString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_2(mht_2_v, 344, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "DebugString");

    mutex_lock l(mu_);
    std::string debug_string = "Registered methods: [";
    debug_string.append(absl::StrJoin(
        registered_methods_, ", ",
        [](std::string* out, const auto& pair) { return pair.first; }));

    debug_string.append("]");
    return debug_string;
  }

  tensorflow::Status Register(const std::string& method,
                              FunctionLibraryRuntime* lib,
                              FunctionLibraryRuntime::Handle fn_handle,
                              std::vector<Tensor> captured_inputs,
                              const StructuredValue& input_specs,
                              const StructuredValue& output_specs) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_3(mht_3_v, 364, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "Register");

    mutex_lock l(mu_);
    FunctionMetadata fn_metadata;
    fn_metadata.handle = fn_handle;
    fn_metadata.lib = lib;
    fn_metadata.captured_inputs = std::move(captured_inputs);
    fn_metadata.input_specs = input_specs;
    fn_metadata.output_specs = output_specs;
    auto result = registered_methods_.insert(
        std::pair<std::string, FunctionMetadata>(method, fn_metadata));
    if (!result.second) {
      return tensorflow::errors::InvalidArgument(
          absl::StrCat(method, " is already registered."));
    }
    return tensorflow::Status::OK();
  }

  tensorflow::Status LookUp(const std::string& method,
                            FunctionMetadata* output) const {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_4(mht_4_v, 386, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "LookUp");

    mutex_lock l(mu_);
    auto it = registered_methods_.find(method);
    if (it == registered_methods_.end()) {
      return tensorflow::errors::InvalidArgument(
          absl::StrCat(method, " is not registered."));
    }

    *output = it->second;
    return tensorflow::Status::OK();
  }

  const gtl::FlatMap<std::string, FunctionMetadata>& List() const {
    return registered_methods_;
  }

 private:
  mutable mutex mu_;
  gtl::FlatMap<std::string, FunctionMetadata> registered_methods_
      TF_GUARDED_BY(mu_);
};

class RpcServiceImpl : public grpc::RpcService::Service {
 public:
  explicit RpcServiceImpl(const FunctionRegistry& registry)
      : registry_(registry) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_5(mht_5_v, 414, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServiceImpl");
}

  ::grpc::Status Call(::grpc::ServerContext* context,
                      const CallRequest* request,
                      CallResponse* response) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_6(mht_6_v, 421, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "Call");

    const auto& method_name = request->method();

    FunctionLibraryRuntime::Options opts;

    FunctionMetadata fn_metadata;
    auto status = registry_.LookUp(method_name, &fn_metadata);
    FunctionLibraryRuntime::Handle handle = fn_metadata.handle;
    FunctionLibraryRuntime* fn_lib = fn_metadata.lib;
    std::vector<Tensor> captured_inputs =
        std::move(fn_metadata.captured_inputs);

    if (!status.ok()) {
      return ToGrpcStatus(status);
    }

    std::vector<Tensor> args;
    for (const auto& t : request->input_tensors()) {
      Tensor tensor;
      if (tensor.FromProto(t)) {
        args.push_back(std::move(tensor));
      } else {
        return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                              "Failed to parse input tensor from proto.");
      }
    }

    // Add captured args as well
    for (const auto& t : captured_inputs) {
      args.push_back(std::move(t));
    }

    std::vector<Tensor>* rets = new std::vector<Tensor>;
    Notification notification;
    fn_lib->Run(opts, handle, args, rets,
                [rets, response, &notification, &status](const Status& st) {
                  status = st;
                  if (status.ok()) {
                    for (size_t i = 0; i < rets->size(); ++i) {
                      auto t = response->add_output_tensors();
                      (*rets)[i].AsProtoField(t);
                    }
                  }
                  delete rets;
                  notification.Notify();
                });

    notification.WaitForNotification();
    return ToGrpcStatus(status);
  }

  ::grpc::Status List(::grpc::ServerContext* context,
                      const rpc::ListRequest* request,
                      rpc::ListResponse* response) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_7(mht_7_v, 477, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "List");

    auto methods = registry_.List();
    for (auto it : methods) {
      auto* registered_method = response->add_registered_methods();
      registered_method->set_method(it.first);
      *registered_method->mutable_output_specs() = it.second.output_specs;
      *registered_method->mutable_input_specs() = it.second.input_specs;
    }
    return ::grpc::Status(::grpc::Status::OK);
  }

 private:
  const FunctionRegistry& registry_;
};

class RpcServer : public ResourceBase {
 public:
  explicit RpcServer(std::string server_address)
      : server_address_(server_address),
        server_(nullptr),
        server_started_(false) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("server_address: \"" + server_address + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_8(mht_8_v, 501, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServer");

    service_ = std::make_unique<RpcServiceImpl>(registry_);
  }

  ~RpcServer() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_9(mht_9_v, 508, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "~RpcServer");

    if (server_) {
      LOG(INFO) << "Shutting down server listening on: " << server_address_;
      server_->Shutdown();
    }
  }

  std::string DebugString() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_10(mht_10_v, 518, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "DebugString");

    return absl::StrCat("RpcServer resource with ", registry_.DebugString());
  }

  tensorflow::Status Register(const std::string& method,
                              FunctionLibraryRuntime* lib,
                              FunctionLibraryRuntime::Handle fn_handle,
                              std::vector<Tensor> captured_inputs,
                              const StructuredValue& input_specs,
                              const StructuredValue& output_specs) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("method: \"" + method + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_11(mht_11_v, 531, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "Register");

    mutex_lock m(mu_);
    if (server_started_) {
      return tensorflow::errors::FailedPrecondition(
          "All methods must be registered before starting the server. Method "
          "registration after starting the server is not supported.");
    }
    return registry_.Register(method, lib, fn_handle, captured_inputs,
                              input_specs, output_specs);
  }

  void StartServer() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_12(mht_12_v, 545, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "StartServer");

    mutex_lock l(mu_);
    ::grpc::ServerBuilder builder;
    std::shared_ptr<::grpc::ServerCredentials> creds =
        GetDefaultServerCredentials();
    builder.AddListeningPort(server_address_, creds);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on: " << server_address_;
    server_started_ = true;
  }

 private:
  FunctionRegistry registry_;
  std::unique_ptr<RpcServiceImpl> service_;
  std::string server_address_;
  std::unique_ptr<::grpc::Server> server_;
  bool server_started_ TF_GUARDED_BY(mu_);
  mutex mu_;
};

class GrpcPollingThread {
 public:
  explicit GrpcPollingThread(std::string thread_name) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("thread_name: \"" + thread_name + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_13(mht_13_v, 572, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "GrpcPollingThread");

    // Thread name can only have alpha numeric characters. Remove special
    // characters from input thread_name.
    thread_name.erase(
        std::remove_if(thread_name.begin(), thread_name.end(),
                       [](auto const c) -> bool { return !std::isalnum(c); }),
        thread_name.end());
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), absl::StrCat("GrpcPollingThread", thread_name),
        [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
          }
        }));
  }

  ~GrpcPollingThread() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_14(mht_14_v, 594, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "~GrpcPollingThread");

    completion_queue_.Shutdown();
    thread_.reset();
  }

  ::grpc::CompletionQueue* completion_queue() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_15(mht_15_v, 602, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "completion_queue");
 return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};

class RpcClient : public ResourceBase {
 public:
  explicit RpcClient(std::string address, std::string resource_name,
                     int64 timeout_in_ms)
      : server_address_(address),
        thread_(resource_name),
        timeout_in_ms_(timeout_in_ms) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("address: \"" + address + "\"");
   mht_16_v.push_back("resource_name: \"" + resource_name + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_16(mht_16_v, 620, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcClient");

    std::shared_ptr<::grpc::ChannelCredentials> creds =
        GetDefaultChannelCredentials();

    channel_ = ::grpc::CreateChannel(address, creds);

    stub_ = std::make_unique<::grpc::GenericStub>(channel_);
    cq_ = thread_.completion_queue();
    callback_threadpool_ = std::make_unique<thread::ThreadPool>(
        Env::Default(), ThreadOptions(), "RPC_Client_threadpool", 5,
        /*low_latency_hint=*/false, /*allocator=*/nullptr);
  }

  std::string DebugString() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_17(mht_17_v, 636, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "DebugString");

    return absl::StrCat("Rpc client for address: ", server_address_);
  }

  void CallAsync(const std::string& method_name,
                 const std::vector<Tensor>& inputs, CallResponse* response,
                 StatusCallback callback, int64 timeout_in_ms) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("method_name: \"" + method_name + "\"");
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_18(mht_18_v, 646, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "CallAsync");

    CallRequest request;
    request.set_method(method_name);
    for (const auto& t : inputs) {
      t.AsProtoField(request.add_input_tensors());
    }
    ::grpc::ClientContext context;
    // Use per call timeout if specified, otherwise use default client timeout.
    int64 timeout = timeout_in_ms > 0 ? timeout_in_ms : timeout_in_ms_;
    new RPCState<CallResponse>(
        stub_.get(), cq_, "/tensorflow.rpc.RpcService/Call", request, response,
        /*done=*/std::move(callback),
        /*call_opts=*/nullptr,
        /*threadpool=*/callback_threadpool_.get(),
        /*fail_fast=*/false, /*timeout_in_ms=*/timeout,
        /*max_retries=*/0, /*target=*/nullptr);
  }

  void ListAsync(rpc::ListResponse* response, StatusCallback callback) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_19(mht_19_v, 667, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "ListAsync");

    rpc::ListRequest request;
    ::grpc::ClientContext context;
    // fail_fast=false sets wait_for_ready to true in GRPC call.
    // ListAsync is called during Client creation thus, we want to wait till
    // server is ready for issuing RPC.
    new RPCState<rpc::ListResponse>(
        stub_.get(), cq_, "/tensorflow.rpc.RpcService/List", request, response,
        /*done=*/std::move(callback),
        /*call_opts=*/nullptr,
        /*threadpool=*/callback_threadpool_.get(),
        /*fail_fast=*/false, /*timeout_in_ms=*/timeout_in_ms_,
        /*max_retries=*/0, /*target=*/nullptr);
  }

 private:
  std::shared_ptr<::grpc::Channel> channel_;
  std::string server_address_;
  std::unique_ptr<::grpc::GenericStub> stub_;
  ::grpc::CompletionQueue* cq_;
  GrpcPollingThread thread_;
  std::unique_ptr<thread::ThreadPool> callback_threadpool_;
  int64 timeout_in_ms_;
};

class RpcFutureResource : public ResourceBase {
  typedef std::function<void(const Status&, const CallResponse&)>
      FutureCallBack;

 public:
  RpcFutureResource() : done_(false) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_20(mht_20_v, 700, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcFutureResource");
}
  std::string DebugString() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_21(mht_21_v, 704, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "DebugString");
 return "Wait Resource"; }

  void AddDoneCallback(FutureCallBack cb) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_22(mht_22_v, 709, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "AddDoneCallback");

    mutex_lock l(mu_);
    if (!done_) {
      call_backs_.push_back(cb);
    } else {
      cb(status_, response_);
    }
  }

  void OperationFinished() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_23(mht_23_v, 721, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "OperationFinished");

    mutex_lock l(mu_);
    for (const auto& cb : call_backs_) {
      cb(status_, response_);
    }
    done_ = true;
  }

  void set_status(Status status) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_24(mht_24_v, 732, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "set_status");
 status_.Update(status); }
  Status get_status() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_25(mht_25_v, 736, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "get_status");
 return status_; }
  CallResponse* get_response() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_26(mht_26_v, 740, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "get_response");
 return &response_; }

 private:
  CallResponse response_;
  bool done_ TF_GUARDED_BY(mu_);
  Status status_;
  std::vector<FutureCallBack> call_backs_ TF_GUARDED_BY(mu_);
  mutable mutex mu_;
};

Status ExtractServerAddressFromInput(OpKernelContext* ctx,
                                     std::string* address) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_27(mht_27_v, 754, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "ExtractServerAddressFromInput");

  const Tensor* server_address;
  auto status = ctx->input("server_address", &server_address);
  if (status.ok()) {
    *address = server_address->scalar<tstring>()();
  }
  return status;
}

RpcServerOp::RpcServerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_28(mht_28_v, 766, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServerOp::RpcServerOp");
}

void RpcServerOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_29(mht_29_v, 771, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServerOp::Compute");

  std::string address = "";
  OP_REQUIRES_OK(ctx, ExtractServerAddressFromInput(ctx, &address));

  // Create resource handle
  AllocatorAttributes attr;
  attr.set_on_host(true);

  ResourceHandle resource_handle =
      MakeResourceHandle<RpcServer>(ctx, "rpc_server", address);
  Tensor handle;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
  handle.scalar<ResourceHandle>()() = resource_handle;

  // Create resource
  auto creator = [address](RpcServer** server) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_30(mht_30_v, 790, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "lambda");

    *server = new RpcServer(address);
    return Status::OK();
  };
  core::RefCountPtr<RpcServer> server;
  OP_REQUIRES_OK(ctx, LookupOrCreateResource<RpcServer>(ctx, resource_handle,
                                                        &server, creator));
  ctx->set_output(0, handle);
}

RpcClientOp::RpcClientOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_31(mht_31_v, 803, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcClientOp::RpcClientOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("list_registered_methods", &list_registered_methods_));
}

void RpcClientOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_32(mht_32_v, 812, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcClientOp::ComputeAsync");

  std::string address = "";
  OP_REQUIRES_OK_ASYNC(ctx, ExtractServerAddressFromInput(ctx, &address), done);

  const Tensor* timeout;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input("timeout_in_ms", &timeout), done);
  auto timeout_in_ms = timeout->scalar<int64_t>()();

  // Create resource handle
  AllocatorAttributes attr;
  attr.set_on_host(true);
  auto resource_name = absl::StrCat(name_, address);

  ResourceHandle resource_handle =
      MakeResourceHandle<RpcClient>(ctx, "rpc_client", resource_name);
  Tensor handle;
  OP_REQUIRES_OK_ASYNC(
      ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr),
      done);
  handle.scalar<ResourceHandle>()() = resource_handle;

  // Delete old client handle if exists, to clear old client resource state.
  DeleteResource(ctx, resource_handle).IgnoreError();

  // Create resource
  auto creator = [&address, &resource_name, timeout_in_ms](RpcClient** client) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_33(mht_33_v, 840, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "lambda");

    *client = new RpcClient(address, resource_name, timeout_in_ms);
    return Status::OK();
  };

  core::RefCountPtr<RpcClient> client;
  OP_REQUIRES_OK_ASYNC(
      ctx,
      LookupOrCreateResource<RpcClient>(ctx, resource_handle, &client, creator),
      done);
  ctx->set_output(0, handle);

  if (!list_registered_methods_) {
    Tensor* method_output_t;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(1, TensorShape({}), &method_output_t), done);
    method_output_t->scalar<tstring>()() = "";
    done();
    return;
  }
  auto* response = new ListResponse();
  client->ListAsync(
      response, [ctx, response, done](const Status& status) {
        if (!status.ok()) {
          ctx->SetStatus(status);
        } else {
          Tensor* method_output_signatures_t;
          auto method_output_shape = TensorShape(
              {static_cast<int64_t>(response->registered_methods_size())});
          OP_REQUIRES_OK_ASYNC(
              ctx,
              ctx->allocate_output(1, method_output_shape,
                                   &method_output_signatures_t),
              done);
          auto method_output_signatures =
              method_output_signatures_t->vec<tstring>();
          for (int i = 0; i < response->registered_methods_size(); ++i) {
            method_output_signatures(i) =
                response->registered_methods(i).SerializeAsString();
          }
        }
        delete response;
        done();
      });
}

RpcServerStartOp::RpcServerStartOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_34(mht_34_v, 889, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServerStartOp::RpcServerStartOp");
}

void RpcServerStartOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_35(mht_35_v, 894, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServerStartOp::Compute");

  core::RefCountPtr<RpcServer> server;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &server));

  server->StartServer();
  ctx->SetStatus(Status::OK());
}

RpcServerRegisterOp::RpcServerRegisterOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_36(mht_36_v, 906, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServerRegisterOp::RpcServerRegisterOp");

  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(FunctionLibraryDefinition::kFuncAttr, &func_));
  std::string output_specs_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_specs", &output_specs_string));

  OP_REQUIRES(ctx, output_specs_.ParseFromString(output_specs_string),
              tensorflow::errors::InvalidArgument(
                  "Unable to parse StructuredValue output_spec string: ",
                  output_specs_string));

  std::string input_specs_string;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("input_specs", &input_specs_string));

  OP_REQUIRES(ctx, input_specs_.ParseFromString(input_specs_string),
              tensorflow::errors::InvalidArgument(
                  "Unable to parse StructuredValue output_spec string: ",
                  input_specs_string));
}

void RpcServerRegisterOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_37(mht_37_v, 929, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcServerRegisterOp::Compute");

  FunctionLibraryRuntime* lib = ctx->function_library();
  OP_REQUIRES(ctx, lib != nullptr,
              errors::Internal("No function library is provided"));

  const Tensor* method_name;
  OP_REQUIRES_OK(ctx, ctx->input("method_name", &method_name));

  std::string method = method_name->scalar<tstring>()();

  OpInputList captured_inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list("captured_inputs", &captured_inputs));
  std::vector<Tensor> captured(captured_inputs.begin(), captured_inputs.end());

  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = ctx->device()->name();
  instantiate_opts.lib_def = lib->GetFunctionLibraryDefinition();
  // In case captured inputs are on different device.
  instantiate_opts.is_multi_device_function = true;

  const FunctionDef* fdef =
      lib->GetFunctionLibraryDefinition()->Find(func_.name());
  OP_REQUIRES(ctx, fdef != nullptr,
              errors::Internal("Failed to find function."));
  int num_args = fdef->signature().input_arg_size();

  const int num_non_captured_inputs = num_args - captured.size();
  for (int i = 0; i < num_non_captured_inputs; ++i) {
    instantiate_opts.input_devices.push_back(ctx->device()->name());
  }

  absl::flat_hash_map<string, std::vector<string>> composite_devices;
  for (int i = 0; i < captured.size(); ++i) {
    if (captured[i].dtype() == DT_RESOURCE) {
      instantiate_opts.input_devices.push_back(GetFunctionResourceInputDevice(
          captured[i], num_non_captured_inputs + i, *fdef, &composite_devices));
    } else {
      instantiate_opts.input_devices.push_back(ctx->device()->name());
    }
  }

  for (const auto& it : composite_devices) {
    instantiate_opts.composite_devices[it.first] = &it.second;
  }

  FunctionLibraryRuntime::Handle handle;
  OP_REQUIRES_OK(ctx, lib->Instantiate(func_.name(), AttrSlice(&func_.attr()),
                                       instantiate_opts, &handle));

  core::RefCountPtr<RpcServer> server;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &server));

  OP_REQUIRES_OK(ctx, server->Register(method, lib, handle, std::move(captured),
                                       input_specs_, output_specs_));
}

RpcCallOp::RpcCallOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_38(mht_38_v, 988, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcCallOp::RpcCallOp");
}

void RpcCallOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_39(mht_39_v, 993, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcCallOp::Compute");

  const Tensor* method_name;
  OP_REQUIRES_OK(ctx, ctx->input("method_name", &method_name));
  std::string method = method_name->scalar<tstring>()();

  const Tensor* timeout;
  OP_REQUIRES_OK(ctx, ctx->input("timeout_in_ms", &timeout));
  auto timeout_in_ms = timeout->scalar<int64_t>()();

  OpInputList arguments;
  OP_REQUIRES_OK(ctx, ctx->input_list("args", &arguments));
  std::vector<Tensor> args(arguments.begin(), arguments.end());

  core::RefCountPtr<RpcClient> client;
  OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &client));

  ResourceHandle resource_handle = MakeResourceHandle<RpcFutureResource>(
      ctx, "rpc_future_resource", absl::StrFormat("%d", random::New64()));

  AllocatorAttributes attr;
  attr.set_on_host(true);
  Tensor handle;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_RESOURCE, TensorShape({}), &handle, attr));
  handle.scalar<ResourceHandle>()() = resource_handle;

  // Create resource
  auto creator = [](RpcFutureResource** resource) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_40(mht_40_v, 1023, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "lambda");

    *resource = new RpcFutureResource();
    return Status::OK();
  };
  core::RefCountPtr<RpcFutureResource> future_resource;
  OP_REQUIRES_OK(ctx, LookupOrCreateResource<RpcFutureResource>(
                          ctx, resource_handle, &future_resource, creator));

  Tensor deleter_t;
  OP_REQUIRES_OK(
      ctx, ctx->allocate_temp(DT_VARIANT, TensorShape({}), &deleter_t, attr));
  deleter_t.scalar<Variant>()() =
      ResourceDeleter(resource_handle, ctx->resource_manager());
  ctx->set_output(0, handle);
  ctx->set_output(1, deleter_t);

  CallResponse* response = future_resource->get_response();
  auto* future_resource_ptr = future_resource.release();

  client->CallAsync(
      method, args, response,
      [future_resource_ptr](const Status& status) {
        future_resource_ptr->set_status(status);
        future_resource_ptr->OperationFinished();
        future_resource_ptr->Unref();
      },
      timeout_in_ms);
}

RpcCheckStatusOp::RpcCheckStatusOp(OpKernelConstruction* ctx)
    : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_41(mht_41_v, 1056, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcCheckStatusOp::RpcCheckStatusOp");
}

void RpcCheckStatusOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_42(mht_42_v, 1061, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcCheckStatusOp::ComputeAsync");

  core::RefCountPtr<RpcFutureResource> future_resource;
  auto handle = HandleFromInput(ctx, 0);
  {
    auto status = LookupResource(ctx, handle, &future_resource);
    if (!status.ok()) {
      if (errors::IsNotFound(status)) {
        ctx->SetStatus(tensorflow::errors::NotFound(
            absl::StrCat("Future resource no longer exists. Please make sure "
                         "resource is not already deleted.")));
        done();
        return;
      } else {
        ctx->SetStatus(status);
      }
    }
  }

  future_resource->AddDoneCallback(
      [ctx, done, handle](const Status& status, const CallResponse& response) {
        Tensor error_code(DT_INT64, TensorShape({})),
            error_message(DT_STRING, TensorShape({}));
        error_code.scalar<int64_t>()() = status.code();
        error_message.scalar<tstring>()() = status.error_message();

        ctx->set_output(0, error_code);
        ctx->set_output(1, error_message);

        done();
      });
}

RpcGetValueOp::RpcGetValueOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_43(mht_43_v, 1096, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcGetValueOp::RpcGetValueOp");
}

void RpcGetValueOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPSdistributePSexperimentalPSrpcPSkernelsPSrpc_opsDTcc mht_44(mht_44_v, 1101, "", "./tensorflow/distribute/experimental/rpc/kernels/rpc_ops.cc", "RpcGetValueOp::ComputeAsync");

  core::RefCountPtr<RpcFutureResource> future_resource;
  auto handle = HandleFromInput(ctx, 0);
  {
    auto status = LookupResource(ctx, handle, &future_resource);
    if (!status.ok()) {
      if (errors::IsNotFound(status)) {
        ctx->SetStatus(tensorflow::errors::NotFound(
            absl::StrCat("Future resource no longer exists. Please ensure "
                         "resource is not already deleted.")));
        done();
        return;
      } else {
        ctx->SetStatus(status);
      }
    }
  }

  future_resource->AddDoneCallback(
      [ctx, done, handle](const Status& status, const CallResponse& response) {
        if (!status.ok()) {
          ctx->SetStatus(status);
        } else {
          if (ctx->num_outputs() != response.output_tensors().size()) {
            ctx->SetStatus(tensorflow::errors::InvalidArgument(absl::StrCat(
                "Incorrect number of output types specified.",
                ctx->num_outputs(), " ", response.output_tensors().size())));
          } else {
            int i = 0;
            for (const auto& t_proto : response.output_tensors()) {
              Tensor t;
              if (!t.FromProto(t_proto)) {
                ctx->SetStatus(tensorflow::errors::Internal(
                    absl::StrCat("Invalid Tensor Proto response returned.")));
              }
              ctx->set_output(i++, std::move(t));
            }
          }
        }
        done();
      });
}

REGISTER_OP("RpcServer")
    .Input("server_address: string")
    .Output("server: resource")
    .SetIsStateful();

REGISTER_OP("RpcClient")
    .Attr("shared_name: string = ''")
    .Input("server_address: string")
    .Attr("list_registered_methods: bool = false")
    .Input("timeout_in_ms: int64")  // 0 indicates no timeout.
                                    // Positive value indicates specified
                                    // timeout.
    .Output("client: resource")
    .Output("method_specs: string")
    .SetIsStateful();

REGISTER_OP("RpcServerStart").Input("server: resource").SetIsStateful();

REGISTER_OP("RpcServerRegister")
    .Input("server: resource")
    .Input("method_name: string")
    .Input("captured_inputs: Tin")
    .Attr("Tin: list(type) >=0 = []")
    .Attr("f: func")
    .Attr("input_specs: string = ''")
    .Attr("output_specs: string")
    .SetIsStateful();

REGISTER_OP("DeleteRpcFutureResource")
    .Input("handle: resource")
    .Input("deleter: variant")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("RpcCall")
    .Input("client: resource")
    .Input("method_name: string")
    .Input("args: Tin")
    .Input("timeout_in_ms: int64")
    .Attr("Tin: list(type) >= 0")
    .Output("future: resource")
    .Output("deleter: variant")
    .SetIsStateful();

REGISTER_OP("RpcCheckStatus")
    .Input("status_or: resource")
    .Output("error_code: int64")
    .Output("error: string")
    .SetIsStateful();

REGISTER_OP("RpcGetValue")
    .Input("status_or: resource")
    .Attr("Tout: list(type) >= 0")
    .Output("output: Tout")
    .SetIsStateful();

REGISTER_KERNEL_BUILDER(Name("RpcServer").Device(DEVICE_CPU), RpcServerOp);
REGISTER_KERNEL_BUILDER(Name("RpcClient").Device(DEVICE_CPU), RpcClientOp);
REGISTER_KERNEL_BUILDER(Name("RpcServerStart").Device(DEVICE_CPU),
                        RpcServerStartOp);
REGISTER_KERNEL_BUILDER(Name("RpcServerRegister").Device(DEVICE_CPU),
                        RpcServerRegisterOp);
REGISTER_KERNEL_BUILDER(Name("RpcCall").Device(DEVICE_CPU), RpcCallOp);
REGISTER_KERNEL_BUILDER(Name("RpcCheckStatus").Device(DEVICE_CPU),
                        RpcCheckStatusOp);
REGISTER_KERNEL_BUILDER(Name("RpcGetValue").Device(DEVICE_CPU), RpcGetValueOp);
REGISTER_KERNEL_BUILDER(Name("DeleteRpcFutureResource").Device(DEVICE_CPU),
                        DeleteRpcFutureResourceOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("RpcServerRegister");
}  // namespace rpc
}  // namespace tensorflow
