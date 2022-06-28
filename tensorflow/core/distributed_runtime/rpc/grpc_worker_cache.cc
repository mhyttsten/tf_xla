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
class MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc {
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
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc() {
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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"

#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

class GrpcWorkerCache : public WorkerCachePartial {
 public:
  explicit GrpcWorkerCache(std::shared_ptr<GrpcChannelCache> channel_cache,
                           WorkerInterface* local_worker,
                           const string& local_target,
                           GrpcWorkerEnv* worker_env)
      : local_target_(local_target),
        local_worker_(local_worker),
        channel_cache_(channel_cache),
        worker_env_(worker_env),
        next_round_robin_assignment_(0) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("local_target: \"" + local_target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_0(mht_0_v, 213, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GrpcWorkerCache");
}

  void ListWorkers(std::vector<string>* workers) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "ListWorkers");

    channel_cache_->ListWorkers(workers);
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "ListWorkersInJob");

    channel_cache_->ListWorkersInJob(job_name, workers);
  }

  WorkerInterface* GetOrCreateWorker(const string& target) override {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GetOrCreateWorker");

    if (target == local_target_) {
      return local_worker_;
    } else {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (!channel) {
        return nullptr;
      }
      size_t index = AssignWorkerToThread(target);
      return NewGrpcRemoteWorker(
          channel, worker_env_->GetCompletionQueue(index),
          worker_env_->GetThreadPool(), &logger_, target);
    }
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_4(mht_4_v, 254, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "ReleaseWorker");

    if (target == local_target_) {
      CHECK_EQ(worker, local_worker_)
          << "Releasing a worker that was not returned by this WorkerCache";
    } else {
      WorkerCacheInterface::ReleaseWorker(target, worker);
    }
  }

  Status GetEagerClientCache(
      std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_5(mht_5_v, 267, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GetEagerClientCache");

    eager_client_cache->reset(eager::NewGrpcEagerClientCache(channel_cache_));
    return Status::OK();
  }

  Status GetCoordinationClientCache(std::unique_ptr<CoordinationClientCache>*
                                        coordination_client_cache) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_6(mht_6_v, 276, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GetCoordinationClientCache");

    coordination_client_cache->reset(
        NewGrpcCoordinationClientCache(channel_cache_));
    return Status::OK();
  }

  void SetLogging(bool v) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_7(mht_7_v, 285, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "SetLogging");
 logger_.SetLogging(v); }

  void ClearLogs() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_8(mht_8_v, 290, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "ClearLogs");
 logger_.ClearLogs(); }

  bool RetrieveLogs(int64_t step_id, StepStats* ss) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_9(mht_9_v, 295, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "RetrieveLogs");

    return logger_.RetrieveLogs(step_id, ss);
  }

 private:
  size_t AssignWorkerToThread(const string& target) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("target: \"" + target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_10(mht_10_v, 304, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "AssignWorkerToThread");

    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    mutex_lock lock(assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(target,
                                      (next_round_robin_assignment_++) %
                                          worker_env_->CompletionQueueSize()))
               .first;
    }
    return it->second;
  }

  const string local_target_;
  WorkerInterface* const local_worker_;  // Not owned.
  std::shared_ptr<GrpcChannelCache> channel_cache_;
  WorkerCacheLogger logger_;
  GrpcWorkerEnv* worker_env_;  // Not owned

  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      TF_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ TF_GUARDED_BY(assignment_mu_);
};

}  // namespace

GrpcWorkerEnv::GrpcWorkerEnv(size_t num_completion_queues, size_t num_threads)
    : threadpool_(new thread::ThreadPool(
          Env::Default(), ThreadOptions(), "GrpcWorkerEnvQueues", num_threads,
          /*low_latency_hint=*/false, /*allocator=*/nullptr)),
      threads_(num_completion_queues) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_11(mht_11_v, 340, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GrpcWorkerEnv::GrpcWorkerEnv");
}

GrpcWorkerEnv::~GrpcWorkerEnv() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_12(mht_12_v, 345, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GrpcWorkerEnv::~GrpcWorkerEnv");
 threads_.clear(); }

GrpcWorkerEnv::GrpcWorkerCacheThread::GrpcWorkerCacheThread() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_13(mht_13_v, 350, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GrpcWorkerEnv::GrpcWorkerCacheThread::GrpcWorkerCacheThread");

  thread_.reset(Env::Default()->StartThread(
      ThreadOptions(), "GrpcWorkerEnvPool", [this]() {
        void* tag;
        bool ok;
        while (completion_queue_.Next(&tag, &ok)) {
          GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
          callback_tag->OnCompleted(ok);
        }
      }));
}

GrpcWorkerEnv::GrpcWorkerCacheThread::~GrpcWorkerCacheThread() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_14(mht_14_v, 365, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "GrpcWorkerEnv::GrpcWorkerCacheThread::~GrpcWorkerCacheThread");

  completion_queue_.Shutdown();
  thread_.reset();
}

GrpcWorkerEnv* CreateGrpcWorkerEnv() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_15(mht_15_v, 373, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "CreateGrpcWorkerEnv");

  int num_cpus = port::NumSchedulableCPUs();
  int64_t num_completion_queues;
  Status status = ReadInt64FromEnvVar("TF_GRPC_WORKER_CACHE_QUEUES", 64,
                                      &num_completion_queues);
  if (!status.ok()) {
    LOG(ERROR) << "Error parsing TF_GRPC_WORKER_CACHE_QUEUES: " << status;
  }
  int64_t num_threads;
  status = ReadInt64FromEnvVar("TF_GRPC_WORKER_CACHE_THREADS", num_cpus,
                               &num_threads);
  if (!status.ok()) {
    LOG(ERROR) << "Error parsing TF_GRPC_WORKER_CACHE_THREADS: " << status;
  }
  return new GrpcWorkerEnv(num_completion_queues, num_threads);
}

WorkerCacheInterface* NewGrpcWorkerCache(std::shared_ptr<GrpcChannelCache> cc,
                                         GrpcWorkerEnv* worker_env) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_16(mht_16_v, 394, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "NewGrpcWorkerCache");

  return new GrpcWorkerCache(cc, /*local_worker=*/nullptr, /*local_target=*/"",
                             worker_env);
}

WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(
    std::shared_ptr<GrpcChannelCache> cc, GrpcWorkerEnv* worker_env,
    WorkerInterface* local_worker, const string& local_target) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("local_target: \"" + local_target + "\"");
   MHTracer_DTPStensorflowPScorePSdistributed_runtimePSrpcPSgrpc_worker_cacheDTcc mht_17(mht_17_v, 405, "", "./tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.cc", "NewGrpcWorkerCacheWithLocalWorker");

  return new GrpcWorkerCache(cc, local_worker, local_target, worker_env);
}

}  // namespace tensorflow
