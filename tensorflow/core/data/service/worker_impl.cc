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
class MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/worker_impl.h"

#include <memory>
#include <string>
#include <utility>

#include "grpcpp/create_channel.h"
#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/data/dataset.pb.h"
#include "tensorflow/core/data/service/auto_shard_rewriter.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dispatcher.grpc.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/grpc_util.h"
#include "tensorflow/core/data/service/split_provider.h"
#include "tensorflow/core/data/service/task_runner.h"
#include "tensorflow/core/data/service/utils.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace {

constexpr int64_t kRetryIntervalMicros = 5 * 1000 * 1000;        // 5 seconds.
constexpr int64_t kDefaultHeartBeatIntervalMs = 30 * 1000;       // 30 seconds.
constexpr int64_t kDefaultDispatcherTimeoutMs = 60 * 60 * 1000;  // 1 hour.

using WorkerConfig = experimental::WorkerConfig;

// Moves the element into the response. If the tensor contains a single
// CompressedElement variant, the move will be zero-copy. Otherwise, the tensor
// data will be serialized as TensorProtos.
Status MoveElementToResponse(std::vector<Tensor>&& element,
                             GetElementResponse& resp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_0(mht_0_v, 246, "", "./tensorflow/core/data/service/worker_impl.cc", "MoveElementToResponse");

  if (element.size() != 1 || element[0].dtype() != DT_VARIANT ||
      !TensorShapeUtils::IsScalar(element[0].shape())) {
    for (const auto& component : element) {
      UncompressedElement* uncompressed = resp.mutable_uncompressed();
      component.AsProtoTensorContent(uncompressed->add_components());
    }
    return Status::OK();
  }
  Variant& variant = element[0].scalar<Variant>()();
  CompressedElement* compressed = variant.get<CompressedElement>();
  if (compressed == nullptr) {
    return errors::FailedPrecondition(
        "Expected dataset to produce a CompressedElement variant tensor, but "
        "it produced ",
        variant.TypeName());
  }
  *resp.mutable_compressed() = *compressed;
  return Status::OK();
}

WorkerConfig ApplyWorkerDefaults(const WorkerConfig& config) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_1(mht_1_v, 270, "", "./tensorflow/core/data/service/worker_impl.cc", "ApplyWorkerDefaults");

  WorkerConfig new_config(config);
  if (new_config.heartbeat_interval_ms() == 0) {
    new_config.set_heartbeat_interval_ms(kDefaultHeartBeatIntervalMs);
  }
  if (new_config.dispatcher_timeout_ms() == 0) {
    new_config.set_dispatcher_timeout_ms(kDefaultDispatcherTimeoutMs);
  }
  return new_config;
}
}  // namespace

mutex LocalWorkers::mu_(LINKER_INITIALIZED);
LocalWorkers::AddressToWorkerMap* LocalWorkers::local_workers_ =
    new AddressToWorkerMap();

DataServiceWorkerImpl::DataServiceWorkerImpl(const WorkerConfig& config)
    : config_(ApplyWorkerDefaults(config)), worker_uid_(port::JobUid()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_2(mht_2_v, 290, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::DataServiceWorkerImpl");

  metrics::RecordTFDataServiceWorkerCreated();
}

DataServiceWorkerImpl::~DataServiceWorkerImpl() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_3(mht_3_v, 297, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::~DataServiceWorkerImpl");

  mutex_lock l(mu_);
  cancelled_ = true;
  task_completion_cv_.notify_one();
  heartbeat_cv_.notify_one();
}

Status DataServiceWorkerImpl::Start(const std::string& worker_address,
                                    const std::string& transfer_address) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("worker_address: \"" + worker_address + "\"");
   mht_4_v.push_back("transfer_address: \"" + transfer_address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_4(mht_4_v, 310, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::Start");

  VLOG(3) << "Starting tf.data service worker at address " << worker_address;
  TF_RETURN_IF_ERROR(ValidateWorkerConfig());
  worker_address_ = worker_address;
  transfer_address_ = transfer_address;

  dispatcher_ = absl::make_unique<DataServiceDispatcherClient>(
      config_.dispatcher_address(), config_.protocol());
  TF_RETURN_IF_ERROR(dispatcher_->Initialize());

  Status s = Heartbeat();
  while (!s.ok()) {
    if (!IsPreemptedError(s)) {
      return s;
    }
    LOG(WARNING) << "Failed to register with dispatcher at "
                 << config_.dispatcher_address() << ": " << s;
    Env::Default()->SleepForMicroseconds(kRetryIntervalMicros);
    s = Heartbeat();
  }
  LOG(INFO) << "Worker registered with dispatcher running at "
            << config_.dispatcher_address();
  task_completion_thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "data-service-worker-task-completion",
                                  [this]() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_5(mht_5_v, 337, "", "./tensorflow/core/data/service/worker_impl.cc", "lambda");
 TaskCompletionThread(); }));
  heartbeat_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      {}, "data-service-worker-heartbeat", [this]() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_6(mht_6_v, 342, "", "./tensorflow/core/data/service/worker_impl.cc", "lambda");
 HeartbeatThread(); }));
  mutex_lock l(mu_);
  registered_ = true;
  return Status::OK();
}

void DataServiceWorkerImpl::Stop() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_7(mht_7_v, 351, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::Stop");

  std::vector<std::shared_ptr<Task>> tasks;
  {
    mutex_lock l(mu_);
    cancelled_ = true;
    for (const auto& entry : tasks_) {
      tasks.push_back(entry.second);
    }
  }
  for (auto& task : tasks) {
    StopTask(*task);
  }
  // At this point there are no outstanding requests in this RPC handler.
  // However, requests successfully returned from this RPC handler may still be
  // in progress within the gRPC server. If we shut down the gRPC server
  // immediately, it could cause these requests to fail, e.g. with broken pipe.
  // To mitigate this, we sleep for some time to give the gRPC server time to
  // complete requests.
  Env::Default()->SleepForMicroseconds(config_.shutdown_quiet_period_ms() *
                                       1000);
}

Status DataServiceWorkerImpl::ValidateWorkerConfig() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_8(mht_8_v, 376, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::ValidateWorkerConfig");

  const bool any_tag_is_empty = absl::c_any_of(
      config_.worker_tags(),
      [](const std::string& worker_tag) { return worker_tag.empty(); });
  if (any_tag_is_empty) {
    return errors::FailedPrecondition(
        "Worker tags cannot be empty. Got tags {",
        absl::StrJoin(config_.worker_tags().begin(),
                      config_.worker_tags().end(), ", "),
        "}");
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::GetElementResult(
    const GetElementRequest* request, struct GetElementResult* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_9(mht_9_v, 394, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::GetElementResult");

  Task* task = nullptr;
  {
    mutex_lock l(mu_);
    if (cancelled_) {
      return errors::Cancelled("Worker is shutting down");
    }
    if (!registered_) {
      // We need to reject requests until the worker has registered with the
      // dispatcher, so that we don't return NOT_FOUND for tasks that the worker
      // had before preemption.
      return errors::Unavailable(
          "Worker has not yet registered with dispatcher.");
    }
    auto it = tasks_.find(request->task_id());
    if (it == tasks_.end()) {
      if (deleted_tasks_.contains(request->task_id())) {
        return errors::FailedPrecondition(
            "Got request for local task ", request->task_id(), " of worker ",
            worker_address_, ", which has been deleted. You may be creating ",
            "a duplicate job which has already finished. To fix this, make "
            "sure to create your dataset only once, as opposed to re-creating "
            "it repeatedly inside a loop.");
      }
      if (finished_tasks_.contains(request->task_id())) {
        VLOG(3) << "Task is already finished";
        result->end_of_sequence = true;
        result->skip = false;
        return Status::OK();
      }
      // Perhaps the worker hasn't gotten the task from the dispatcher yet.
      // Return Unavailable so that the client knows to continue retrying.
      return errors::Unavailable("Task ", request->task_id(), " not found");
    }
    task = it->second.get();
    TF_RETURN_IF_ERROR(EnsureTaskInitialized(*task));
    task->outstanding_requests++;
  }
  auto cleanup = gtl::MakeCleanup([&] {
    mutex_lock l(mu_);
    task->outstanding_requests--;
    cv_.notify_all();
  });
  TF_RETURN_IF_ERROR(task->task_runner->GetNext(*request, *result));

  if (result->end_of_sequence) {
    mutex_lock l(mu_);
    VLOG(3) << "Reached end_of_sequence for task " << request->task_id();
    pending_completed_tasks_.insert(request->task_id());
    task_completion_cv_.notify_one();
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::ProcessTask(const ProcessTaskRequest* request,
                                          ProcessTaskResponse* response) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_10(mht_10_v, 452, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::ProcessTask");

  mutex_lock l(mu_);
  const TaskDef& task = request->task();
  VLOG(3) << "Received request to process task " << task.task_id();
  return ProcessTaskInternal(task);
}

Status DataServiceWorkerImpl::ProcessTaskInternal(const TaskDef& task_def)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_11(mht_11_v, 463, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::ProcessTaskInternal");

  std::shared_ptr<Task>& task = tasks_[task_def.task_id()];
  if (task) {
    VLOG(1) << "Received request to process already-processed task "
            << task->task_def.task_id();
    return Status::OK();
  }
  task = absl::make_unique<Task>(task_def);
  VLOG(3) << "Began processing for task " << task_def.task_id()
          << " with processing mode "
          << task_def.processing_mode_def().DebugString();
  return Status::OK();
}

Status DataServiceWorkerImpl::EnsureTaskInitialized(
    DataServiceWorkerImpl::Task& task) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_12(mht_12_v, 481, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::EnsureTaskInitialized");

  if (task.task_def.worker_address() != worker_address_) {
    return errors::Internal(absl::Substitute(
        "Dispatcher's worker address $0 does not match worker's address $1.",
        task.task_def.worker_address(), worker_address_));
  }

  mutex_lock l(task.mu);
  if (task.initialized) {
    return Status::OK();
  }
  TF_ASSIGN_OR_RETURN(DatasetDef dataset_def, GetDatasetDef(task.task_def));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Dataset> dataset,
                      MakeDataset(dataset_def, task.task_def));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<standalone::Iterator> iterator,
                      MakeDatasetIterator(*dataset, task.task_def));
  auto task_iterator = absl::make_unique<StandaloneTaskIterator>(
      std::move(dataset), std::move(iterator));
  TF_RETURN_IF_ERROR(TaskRunner::Create(
      config_, task.task_def, std::move(task_iterator), task.task_runner));

  task.initialized = true;
  VLOG(3) << "Created iterator for task " << task.task_def.task_id();
  return Status::OK();
}

StatusOr<DatasetDef> DataServiceWorkerImpl::GetDatasetDef(
    const TaskDef& task_def) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_13(mht_13_v, 511, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::GetDatasetDef");

  switch (task_def.dataset_case()) {
    case TaskDef::kDatasetDef:
      return task_def.dataset_def();
    case TaskDef::kPath: {
      DatasetDef def;
      Status s = ReadDatasetDef(task_def.path(), def);
      if (!s.ok()) {
        LOG(INFO) << "Failed to read dataset from " << task_def.path() << ": "
                  << s << ". Falling back to reading from dispatcher.";
        TF_RETURN_IF_ERROR(
            dispatcher_->GetDatasetDef(task_def.dataset_id(), def));
      }
      return def;
    }
    case TaskDef::DATASET_NOT_SET:
      return errors::Internal("Unrecognized dataset case: ",
                              task_def.dataset_case());
  }
}

StatusOr<std::unique_ptr<standalone::Dataset>>
DataServiceWorkerImpl::MakeDataset(const DatasetDef& dataset_def,
                                   const TaskDef& task_def) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_14(mht_14_v, 537, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::MakeDataset");

  TF_ASSIGN_OR_RETURN(AutoShardRewriter auto_shard_rewriter,
                      AutoShardRewriter::Create(task_def));
  // `ApplyAutoShardRewrite` does nothing if auto-sharding is disabled.
  TF_ASSIGN_OR_RETURN(
      GraphDef rewritten_graph,
      auto_shard_rewriter.ApplyAutoShardRewrite(dataset_def.graph()));
  std::unique_ptr<standalone::Dataset> dataset;
  TF_RETURN_IF_ERROR(standalone::Dataset::FromGraph(
      standalone::Dataset::Params(), rewritten_graph, &dataset));
  return dataset;
}

StatusOr<std::unique_ptr<standalone::Iterator>>
DataServiceWorkerImpl::MakeDatasetIterator(standalone::Dataset& dataset,
                                           const TaskDef& task_def) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_15(mht_15_v, 555, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::MakeDatasetIterator");

  std::unique_ptr<standalone::Iterator> iterator;
  if (IsNoShard(task_def.processing_mode_def()) ||
      IsStaticShard(task_def.processing_mode_def())) {
    TF_RETURN_IF_ERROR(dataset.MakeIterator(&iterator));
    return iterator;
  }

  if (IsDynamicShard(task_def.processing_mode_def())) {
    std::vector<std::unique_ptr<SplitProvider>> split_providers;
    split_providers.reserve(task_def.num_split_providers());
    for (int i = 0; i < task_def.num_split_providers(); ++i) {
      split_providers.push_back(absl::make_unique<DataServiceSplitProvider>(
          config_.dispatcher_address(), config_.protocol(), task_def.job_id(),
          i, config_.dispatcher_timeout_ms()));
    }
    TF_RETURN_IF_ERROR(
        dataset.MakeIterator(std::move(split_providers), &iterator));
    return iterator;
  }

  return errors::InvalidArgument("Unrecognized processing mode: ",
                                 task_def.processing_mode_def().DebugString());
}

void DataServiceWorkerImpl::StopTask(Task& task) TF_LOCKS_EXCLUDED(mu_) {
  {
    mutex_lock l(task.mu);
    task.initialized = true;
  }
  if (task.task_runner) {
    task.task_runner->Cancel();
  }
  mutex_lock l(mu_);
  while (task.outstanding_requests > 0) {
    cv_.wait(l);
  }
}

Status DataServiceWorkerImpl::GetElement(const GetElementRequest* request,
                                         GetElementResponse* response) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_16(mht_16_v, 598, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::GetElement");

  VLOG(3) << "Received GetElement request for task " << request->task_id();
  struct GetElementResult result;
  TF_RETURN_IF_ERROR(GetElementResult(request, &result));
  response->set_end_of_sequence(result.end_of_sequence);
  response->set_skip_task(result.skip);
  if (!response->end_of_sequence() && !response->skip_task()) {
    TF_RETURN_IF_ERROR(
        MoveElementToResponse(std::move(result.components), *response));
    VLOG(3) << "Producing an element for task " << request->task_id();
  }
  return Status::OK();
}

Status DataServiceWorkerImpl::GetWorkerTasks(
    const GetWorkerTasksRequest* request, GetWorkerTasksResponse* response) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_17(mht_17_v, 616, "", "./tensorflow/core/data/service/worker_impl.cc", "DataServiceWorkerImpl::GetWorkerTasks");

  mutex_lock l(mu_);
  for (const auto& it : tasks_) {
    Task* task = it.second.get();
    TaskInfo* task_info = response->add_tasks();
    task_info->set_worker_address(worker_address_);
    task_info->set_task_id(task->task_def.task_id());
    task_info->set_job_id(task->task_def.job_id());
  }
  return Status::OK();
}

void DataServiceWorkerImpl::TaskCompletionThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && pending_completed_tasks_.empty()) {
        task_completion_cv_.wait(l);
      }
      if (cancelled_) {
        VLOG(3) << "Task completion thread shutting down";
        return;
      }
    }
    Status s = SendTaskUpdates();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send task updates to dispatcher: " << s;
      mutex_lock l(mu_);
      if (!cancelled_) {
        task_completion_cv_.wait_for(
            l, std::chrono::microseconds(kRetryIntervalMicros));
      }
    }
  }
}

Status DataServiceWorkerImpl::SendTaskUpdates() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<TaskProgress> task_progress;
  {
    mutex_lock l(mu_);
    VLOG(3) << "Sending " << pending_completed_tasks_.size()
            << " task updates to dispatcher";
    task_progress.reserve(pending_completed_tasks_.size());
    for (int task_id : pending_completed_tasks_) {
      task_progress.emplace_back();
      task_progress.back().set_task_id(task_id);
      task_progress.back().set_completed(true);
    }
  }

  TF_RETURN_IF_ERROR(dispatcher_->WorkerUpdate(worker_address_, task_progress));
  mutex_lock l(mu_);
  for (const auto& update : task_progress) {
    pending_completed_tasks_.erase(update.task_id());
  }
  VLOG(3) << "Sent " << task_progress.size() << " task updates ";
  return Status::OK();
}

void DataServiceWorkerImpl::HeartbeatThread() TF_LOCKS_EXCLUDED(mu_) {
  while (true) {
    int64_t next_heartbeat_micros =
        Env::Default()->NowMicros() + (config_.heartbeat_interval_ms() * 1000);
    {
      mutex_lock l(mu_);
      while (!cancelled_ &&
             Env::Default()->NowMicros() < next_heartbeat_micros) {
        int64_t time_to_wait_micros =
            next_heartbeat_micros - Env::Default()->NowMicros();
        heartbeat_cv_.wait_for(l,
                               std::chrono::microseconds(time_to_wait_micros));
      }
      if (cancelled_) {
        VLOG(3) << "Heartbeat thread shutting down";
        return;
      }
      if (!registered_) {
        VLOG(1) << "Not performing heartbeat; worker is not yet registered";
        continue;
      }
    }
    Status s = Heartbeat();
    if (!s.ok()) {
      LOG(WARNING) << "Failed to send heartbeat to dispatcher: " << s;
    }
  }
}

Status DataServiceWorkerImpl::Heartbeat() TF_LOCKS_EXCLUDED(mu_) {
  std::vector<int64_t> current_tasks;
  {
    mutex_lock l(mu_);
    for (const auto& task : tasks_) {
      current_tasks.push_back(task.first);
    }
  }
  WorkerHeartbeatRequest request;
  request.set_worker_address(worker_address_);
  request.set_transfer_address(transfer_address_);
  *request.mutable_worker_tags() = config_.worker_tags();
  request.set_worker_uid(worker_uid_);
  *request.mutable_current_tasks() = {current_tasks.begin(),
                                      current_tasks.end()};
  TF_ASSIGN_OR_RETURN(WorkerHeartbeatResponse response,
                      dispatcher_->WorkerHeartbeat(request));

  std::vector<std::shared_ptr<Task>> tasks_to_delete;
  {
    mutex_lock l(mu_);
    for (const auto& task : response.new_tasks()) {
      VLOG(1) << "Received new task from dispatcher with id " << task.task_id();
      if (deleted_tasks_.contains(task.task_id())) {
        continue;
      }
      Status s = ProcessTaskInternal(task);
      if (!s.ok() && !errors::IsAlreadyExists(s)) {
        LOG(WARNING) << "Failed to start processing task " << task.task_id()
                     << ": " << s;
      }
    }
    tasks_to_delete.reserve(response.tasks_to_delete_size());
    for (int64_t task_id : response.tasks_to_delete()) {
      VLOG(3) << "Deleting task " << task_id
              << " at the request of the dispatcher";
      if (!tasks_.contains(task_id)) {
        continue;
      }
      tasks_to_delete.push_back(std::move(tasks_[task_id]));
      tasks_.erase(task_id);
      finished_tasks_.insert(task_id);
    }
  }
  for (const auto& task : tasks_to_delete) {
    StopTask(*task);
  }
  return Status::OK();
}

void DataServiceWorkerImpl::DeleteLocalTask(const TaskInfo& task_info)
    TF_LOCKS_EXCLUDED(mu_) {
  std::shared_ptr<Task> task;
  {
    mutex_lock l(mu_);
    auto it = tasks_.find(task_info.task_id());
    if (it == tasks_.end() || !it->second) {
      return;
    }
    task = std::move(it->second);
    tasks_.erase(task_info.task_id());
    pending_completed_tasks_.insert(task_info.task_id());
    deleted_tasks_.insert(task_info.task_id());
  }

  VLOG(2) << "Delete local task " << task_info.task_id() << " from worker "
          << worker_address_ << " at the request of the client.";
  StopTask(*task);
}

void LocalWorkers::Add(absl::string_view worker_address,
                       std::shared_ptr<DataServiceWorkerImpl> worker) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_18(mht_18_v, 779, "", "./tensorflow/core/data/service/worker_impl.cc", "LocalWorkers::Add");

  DCHECK(worker != nullptr) << "Adding a nullptr local worker is disallowed.";
  VLOG(1) << "Register local worker at address " << worker_address;
  mutex_lock l(mu_);
  (*local_workers_)[worker_address] = worker;
}

std::shared_ptr<DataServiceWorkerImpl> LocalWorkers::Get(
    absl::string_view worker_address) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_19(mht_19_v, 791, "", "./tensorflow/core/data/service/worker_impl.cc", "LocalWorkers::Get");

  tf_shared_lock l(mu_);
  AddressToWorkerMap::const_iterator it = local_workers_->find(worker_address);
  if (it == local_workers_->end()) {
    return nullptr;
  }
  return it->second;
}

bool LocalWorkers::Empty() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_20(mht_20_v, 803, "", "./tensorflow/core/data/service/worker_impl.cc", "LocalWorkers::Empty");

  tf_shared_lock l(mu_);
  return local_workers_->empty();
}

void LocalWorkers::Remove(absl::string_view worker_address) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("worker_address: \"" + std::string(worker_address.data(), worker_address.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePSworker_implDTcc mht_21(mht_21_v, 812, "", "./tensorflow/core/data/service/worker_impl.cc", "LocalWorkers::Remove");

  VLOG(1) << "Remove local worker at address " << worker_address;
  mutex_lock l(mu_);
  local_workers_->erase(worker_address);
}

}  // namespace data
}  // namespace tensorflow
