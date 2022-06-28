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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/collective_param_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"
#include "tensorflow/core/distributed_runtime/graph_mgr.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc_collective_executor_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_cache_wrapper.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_0(mht_0_v, 241, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "UpdateArguments");

    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_1(mht_1_v, 249, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "UpdatePlugins");
}
};

// Define an option subclass in order to enable SO_REUSEPORT for the
// server socket.
class ReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "UpdateArguments");

    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 1);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "UpdatePlugins");
}
};

// static utility function
RendezvousMgrInterface* NewRpcRendezvousMgr(const WorkerEnv* env) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "NewRpcRendezvousMgr");

  return new RpcRendezvousMgr(env);
}

}  // namespace

GrpcServer::GrpcServer(const ServerDef& server_def, Env* env)
    : env_(env), state_(NEW), server_def_(server_def) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_5(mht_5_v, 284, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::GrpcServer");
}

GrpcServer::~GrpcServer() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_6(mht_6_v, 289, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::~GrpcServer");

  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete master_service_;
  delete worker_service_;
  delete eager_service_;

  for (auto& kv : extra_services_) {
    AsyncServiceInterface* service = kv.second;
    delete service;
  }

  // TODO(mrry): Refactor the *Env classes so that it is less fiddly
  // to destroy them.

  // Shut down all outstanding rendezvous.
  delete worker_env_.rendezvous_mgr;

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  if (worker_env_.session_mgr != nullptr) {
    delete worker_env_.session_mgr;  // Deletes graph_mgr's.
  }

  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
}

// Look up the requested host name and port for this task in `server_def`.
Status GrpcServer::GetHostAndPort(const ServerDef& server_def,
                                  string* host_name, int* port) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_7(mht_7_v, 328, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::GetHostAndPort");

  *port = -1;
  *host_name = "localhost";
  for (const auto& job : server_def.cluster().job()) {
    if (job.name() == server_def.job_name()) {
      auto iter = job.tasks().find(server_def.task_index());
      if (iter == job.tasks().end()) {
        return errors::Internal("Task ", server_def.task_index(),
                                " was not defined in job \"",
                                server_def.job_name(), "\"");
      }

      if (server_def.port() != 0) {
        *port = server_def.port();
      } else {
        auto colon_index = iter->second.find_last_of(':');
        if (!strings::safe_strto32(iter->second.substr(colon_index + 1),
                                   port)) {
          return errors::InvalidArgument(
              "Could not parse port for local server from \"", iter->second,
              "\".");
        }

        if (colon_index != string::npos &&
            !iter->second.substr(0, colon_index).empty()) {
          *host_name = iter->second.substr(0, colon_index);
        }
      }
      break;
    }
  }
  if (*port == -1) {
    return errors::Internal("Job \"", server_def.job_name(),
                            "\" was not defined in cluster");
  }

  return Status::OK();
}

Status GrpcServer::Init(const GrpcServerOptions& opts) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_8(mht_8_v, 370, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Init");

  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  // Check parameters before DeviceFactory::AddDevices,
  // otherwise if 'task_index=-1' the program will abort.

  int requested_port;
  TF_RETURN_IF_ERROR(GetHostAndPort(server_def_, &host_name_, &requested_port));

  SessionOptions sess_opts;
  VLOG(3) << "Grpc Server Init Definition: " << server_def_.DebugString();
  ConfigProto config = server_def_.default_session_config();
  sess_opts.config = config;

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0",
                      "/task:", server_def_.task_index());
  if (opts.local_device_mgr == nullptr) {
    std::vector<std::unique_ptr<Device>> devices;
    TF_RETURN_IF_ERROR(
        DeviceFactory::AddDevices(sess_opts, name_prefix, &devices));
    worker_env_.device_mgr = new DynamicDeviceMgr(std::move(devices));
    owned_device_manager_.reset(worker_env_.device_mgr);
  } else {
    worker_env_.device_mgr = opts.local_device_mgr;
    owned_device_manager_.reset(nullptr);
  }
  worker_env_.local_devices = worker_env_.device_mgr->ListDevices();
  master_env_.local_devices = worker_env_.device_mgr->ListDevices();
  worker_env_.rendezvous_mgr = opts.rendezvous_mgr_func == nullptr
                                   ? new RpcRendezvousMgr(&worker_env_)
                                   : opts.rendezvous_mgr_func(&worker_env_);
  string unused;
  string default_worker_name;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // N.B. The order of initialization here is intricate, because we
  // wish to allow `requested_port == 0` (for choosing any port,
  // mostly for testing). Therefore, the construction of the channel
  // and worker caches depends on `bound_port_`, which is not set
  // until we call `builder.BuildAndStart()`. We must create the
  // service objects before calling `builder.BuildAndStart()`, but
  // `master_env_` and `worker_env_` are only partially
  // configured. However, this is not dangerous, because we do not
  // start serving requests until `this->Start()` is called, which
  // happens after this method returns.
  //
  // TODO(mrry): Provide a general mechanism for dynamically setting
  // the identities of tasks in the worker pool after the service is
  // running.
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(strings::StrCat("0.0.0.0:", requested_port),
                           GetServerCredentials(server_def_), &bound_port_);
  builder.SetMaxMessageSize(std::numeric_limits<int32>::max());

  bool reuse_port = false;
  const Status status =
      ReadBoolFromEnvVar("TF_GRPC_REUSE_PORT", false, &reuse_port);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  auto server_build_option =
      reuse_port
          ? std::unique_ptr<::grpc::ServerBuilderOption>(new ReusePortOption)
          : std::unique_ptr<::grpc::ServerBuilderOption>(new NoReusePortOption);
  builder.SetOption(std::move(server_build_option));

  // Allow subclasses to specify more args to pass to the gRPC server.
  MaybeMutateBuilder(&builder, requested_port);
  master_impl_ = CreateMaster(&master_env_);
  master_service_ = NewGrpcMasterService(master_impl_.get(), config, &builder);
  worker_impl_ = opts.worker_func ? opts.worker_func(&worker_env_, config)
                                  : NewGrpcWorker(&worker_env_, config);
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder,
                                         opts.worker_service_options)
                        .release();
  eager_service_ = new eager::GrpcEagerServiceImpl(&worker_env_, &builder);
  thread::ThreadPool* compute_pool = ComputePool(sess_opts);
  coordination_service_ =
      new GrpcCoordinationServiceImpl(compute_pool, &builder);

  profiler_service_ = profiler::CreateProfilerService();
  builder.RegisterService(profiler_service_.get());

  // Add any extra services to be started.
  extra_services_ = ExtraServices(&builder);

  // extra service:
  if (opts.service_func != nullptr) {
    opts.service_func(&worker_env_, &builder);
  }
  server_ = builder.BuildAndStart();

  if (!server_) {
    return errors::Unknown("Could not start gRPC server");
  }
  // Create the execution environment for the GRPC workers cache.
  grpc_worker_env_.reset(CreateGrpcWorkerEnv());

  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  CHECK_NE(nullptr, worker_cache);

  if (opts.collective_mgr_func) {
    worker_env_.collective_executor_mgr.reset(
        opts.collective_mgr_func(config, &worker_env_, worker_cache));
    if (worker_env_.collective_executor_mgr == nullptr) {
      return errors::Internal(
          "collective_mgr_func did not return CollectiveExecutorMgr");
    }
  } else {
    worker_env_.collective_executor_mgr = CreateProdRpcCollectiveExecutorMgr(
        config, worker_env_.device_mgr, MaybeCreateNcclCommunicator(config),
        worker_cache, default_worker_name);
  }

  // Set up worker environment.
  worker_env_.session_mgr = new SessionMgr(
      &worker_env_, SessionMgr::WorkerNameFromServerDef(server_def_),
      std::unique_ptr<WorkerCacheInterface>(worker_cache),
      [this](const ServerDef& server_def, WorkerCacheInterface** worker_cache) {
        WorkerCacheFactoryOptions options(server_def);
        return WorkerCacheFactory(options, worker_cache);
      });
  worker_env_.compute_pool = compute_pool;

  // Finish setting up master environment.
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr =
      worker_env_.collective_executor_mgr.get();
  StatsPublisherFactory stats_factory = opts.stats_factory;
  master_env_.master_session_factory =
      [config, stats_factory](
          SessionOptions options, const MasterEnv* env,
          std::unique_ptr<std::vector<std::unique_ptr<Device>>> remote_devs,
          std::unique_ptr<WorkerCacheInterface> worker_cache,
          std::unique_ptr<DeviceSet> device_set,
          std::vector<string> filtered_worker_list) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_9(mht_9_v, 520, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "lambda");

        options.config.MergeFrom(config);
        return new MasterSession(options, env, std::move(remote_devs),
                                 std::move(worker_cache), std::move(device_set),
                                 std::move(filtered_worker_list),
                                 stats_factory);
      };
  master_env_.worker_cache_factory =
      [this](const WorkerCacheFactoryOptions& options,
             WorkerCacheInterface** worker_cache) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_10(mht_10_v, 532, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "lambda");

        return WorkerCacheFactory(options, worker_cache);
      };

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get(),
                        config.operation_timeout_in_ms());

  return Status::OK();
}

Status GrpcServer::ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                                    GrpcChannelSpec* channel_spec) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_11(mht_11_v, 547, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::ParseChannelSpec");

  for (const auto& job : options.cluster_def->job()) {
    std::map<int, string> host_ports;
    for (const auto& task : job.tasks()) {
      string& host_port = host_ports[task.first];
      if (!host_port.empty()) {
        return errors::InvalidArgument("JobDef for job \"", job.name(),
                                       "\" specified two addresses for task \"",
                                       task.first, "\": ", host_port, " and ",
                                       task.second);
      }
      if (job.name() == *options.job_name && task.first == options.task_index) {
        host_port = strings::StrCat(host_name_, ":", bound_port_);
      } else {
        host_port = task.second;
      }
    }
    TF_RETURN_IF_ERROR(channel_spec->AddHostPortsJob(job.name(), host_ports));
  }
  return Status::OK();
}

Status GrpcServer::WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                      WorkerCacheInterface** worker_cache) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_12(mht_12_v, 573, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::WorkerCacheFactory");

  if (options.job_name == nullptr || options.job_name->empty()) {
    Status s = errors::InvalidArgument(
        "The master (current machine) is not included in the provided "
        "cluster_def. ",
        options.cluster_def->DebugString());
    LOG(WARNING) << s;
    return s;
  }

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(options, &channel_spec));

  if (options.rpc_options == nullptr) {
    return errors::InvalidArgument(
        "rpc_options not set in WorkerCacheFactoryOptions");
  }
  std::shared_ptr<GrpcChannelCache> channel_cache(NewGrpcChannelCache(
      channel_spec, GetChannelCreationFunction(), *options.rpc_options));

  string name_prefix = strings::StrCat("/job:", *options.job_name, "/replica:0",
                                       "/task:", options.task_index);

  const string host_port = channel_cache->TranslateTask(name_prefix);
  int requested_port;

  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strto32(host_port.substr(colon_index + 1),
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            host_port, "\".");
  }
  if (requested_port != bound_port_) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ", bound_port_);
  }
  *worker_cache = NewGrpcWorkerCacheWithLocalWorker(
      channel_cache, grpc_worker_env(), worker_impl(), name_prefix);
  return Status::OK();
}

Status GrpcServer::Start() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_13(mht_13_v, 617, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Start");

  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_master_service",
                            [this] { master_service_->HandleRPCsLoop(); }));
      worker_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_worker_service",
                            [this] { worker_service_->HandleRPCsLoop(); }));
      eager_thread_.reset(
          env_->StartThread(ThreadOptions(), "TF_eager_service",
                            [this] { eager_service_->HandleRPCsLoop(); }));
      coordination_thread_.reset(env_->StartThread(
          ThreadOptions(), "TF_coordination_service",
          [this] { coordination_service_->HandleRPCsLoop(); }));

      for (const auto& kv : extra_services_) {
        const std::string& service_name = kv.first;
        AsyncServiceInterface* service = kv.second;
        std::unique_ptr<Thread> extra_service_thread;
        extra_service_thread.reset(env_->StartThread(
            ThreadOptions(), service_name,
            [service = service] { service->HandleRPCsLoop(); }));
        extra_service_threads_.push_back(std::move(extra_service_thread));
        VLOG(3) << "Started extra service: " << service_name;
      }

      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      return Status::OK();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return Status::OK();
    case STOPPED:
      return errors::FailedPrecondition("Server has stopped.");
    default:
      LOG(FATAL);
  }
}

Status GrpcServer::AddMasterEagerContextToEagerService(
    const tensorflow::uint64 context_id, tensorflow::EagerContext* context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_14(mht_14_v, 663, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::AddMasterEagerContextToEagerService");

  auto* eager_service =
      static_cast<eager::GrpcEagerServiceImpl*>(eager_service_);
  return eager_service->CreateMasterContext(context_id, context);
}

Status GrpcServer::UpdateServerDef(const ServerDef& server_def) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_15(mht_15_v, 672, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::UpdateServerDef");

  mutex_lock l(mu_);
  server_def_ = server_def;
  WorkerCacheInterface* worker_cache;
  WorkerCacheFactoryOptions worker_cache_factory_options(server_def_);
  TF_RETURN_IF_ERROR(
      WorkerCacheFactory(worker_cache_factory_options, &worker_cache));
  if (worker_cache == nullptr) {
    return errors::InvalidArgument(
        "Failed to build worker cache with the provided server def.");
  }
  // Transfer ownership of worker_cache to worker_env_.session_mgr.
  worker_env_.session_mgr->ResetDefaultWorkerCache(worker_cache);

  string default_worker_name;
  string unused;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &default_worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }
  worker_env_.collective_executor_mgr = CreateProdRpcCollectiveExecutorMgr(
      server_def_.default_session_config(), worker_env_.device_mgr,
      MaybeCreateNcclCommunicator(server_def_.default_session_config()),
      worker_cache, default_worker_name);

  master_env_.worker_cache = worker_cache;
  master_env_.collective_executor_mgr =
      worker_env_.collective_executor_mgr.get();
  return Status::OK();
}

// TODO(haoyuzhang): Remove this method once we have a mechanism to directly set
// field inside the RPC coordination service handler.
Status GrpcServer::SetCoordinationServiceAgentInstance(
    CoordinationServiceAgent* agent) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_16(mht_16_v, 709, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::SetCoordinationServiceAgentInstance");

  auto* coord_service =
      static_cast<GrpcCoordinationServiceImpl*>(coordination_service_);
  coord_service->SetCoordinationServiceAgentInstance(agent);
  return Status::OK();
}

Status GrpcServer::StopCoordinationService() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_17(mht_17_v, 719, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::StopCoordinationService");

  // Note: the sequence of events is important here.
  // 1. Agent must be torn down before the service as it needs to notify the
  // service.
  // 2. Remove RPC handlers' access to agent/service first before destructing
  // them within the session manager to prevent data races.
  TF_RETURN_IF_ERROR(SetCoordinationServiceAgentInstance(nullptr));
  worker_env()->session_mgr->TeardownCoordinationServiceAgent();
  coordination_service_->Shutdown();
  worker_env()->session_mgr->TeardownCoordinationService();
  return Status::OK();
}

Status GrpcServer::Stop() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_18(mht_18_v, 735, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Stop");

  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      return errors::Unimplemented(
          "Clean shutdown is not currently implemented");
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

Status GrpcServer::Join() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_19(mht_19_v, 755, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Join");

  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.reset();
      worker_thread_.reset();
      eager_thread_.reset();
      for (auto& thread : extra_service_threads_) {
        thread.reset();
      }
      return Status::OK();
    default:
      LOG(FATAL);
  }
}

const string GrpcServer::target() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_20(mht_20_v, 779, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::target");

  return strings::StrCat("grpc://", host_name_, ":", bound_port_);
}

std::shared_ptr<::grpc::ServerCredentials> GrpcServer::GetServerCredentials(
    const ServerDef& server_def) const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_21(mht_21_v, 787, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::GetServerCredentials");

  return ::grpc::InsecureServerCredentials();
}

ChannelCreationFunction GrpcServer::GetChannelCreationFunction() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_22(mht_22_v, 794, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::GetChannelCreationFunction");

  // We can do this because SparseGrpcChannelCache is robust to nullptr being
  // returned by the channel creation function
  return ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
}

std::unique_ptr<Master> GrpcServer::CreateMaster(MasterEnv* master_env) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_23(mht_23_v, 803, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::CreateMaster");

  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          DeviceMgr* local_device_mgr,
                          std::unique_ptr<ServerInterface>* out_server) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_24(mht_24_v, 813, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Create");

  std::unique_ptr<GrpcServer> ret(
      new GrpcServer(server_def, env == nullptr ? Env::Default() : env));
  GrpcServerOptions options;
  options.rendezvous_mgr_func = NewRpcRendezvousMgr;
  options.local_device_mgr = local_device_mgr;
  Status s = ret->Init(options);
  if (!s.ok()) {
    LOG(ERROR) << s;
    return s;
  }
  *out_server = std::move(ret);
  return Status::OK();
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_25(mht_25_v, 833, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Create");

  return Create(server_def, env, nullptr, out_server);
}

/* static */
Status GrpcServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<GrpcServer>* out_server) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_26(mht_26_v, 842, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServer::Create");

  std::unique_ptr<ServerInterface> server;
  Status s = Create(server_def, env, nullptr, &server);
  if (!s.ok()) {
    return s;
  }
  out_server->reset(dynamic_cast<GrpcServer*>(server.release()));
  return Status::OK();
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_27(mht_27_v, 859, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "AcceptsOptions");

    return server_def.protocol() == "grpc";
  }

  Status NewServer(const ServerDef& server_def, const Options& options,
                   std::unique_ptr<ServerInterface>* out_server) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_28(mht_28_v, 867, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "NewServer");

    return GrpcServer::Create(server_def, Env::Default(),
                              options.local_device_mgr, out_server);
  }
};

// Registers a `ServerFactory` for `GrpcServer` instances.
class GrpcServerRegistrar {
 public:
  GrpcServerRegistrar() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_server_libDTcc mht_29(mht_29_v, 879, "", "./tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc", "GrpcServerRegistrar");

    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory());
  }
};
static GrpcServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
