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
class MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc() {
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
#include "tensorflow/core/data/service/task_runner.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/logging_utils.h"
#include "tensorflow/core/data/service/multi_trainer_cache.h"
#include "tensorflow/core/data/service/thread_safe_buffer.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/data/standalone.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {
// Time to wait before skipping a round if data still isn't available.
const int64_t kWaitBeforeSkipUs = 100 * 1000;  // 100ms.

}  // namespace

StandaloneTaskIterator::StandaloneTaskIterator(
    std::unique_ptr<standalone::Dataset> dataset,
    std::unique_ptr<standalone::Iterator> iterator)
    : dataset_(std::move(dataset)), iterator_(std::move(iterator)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_0(mht_0_v, 221, "", "./tensorflow/core/data/service/task_runner.cc", "StandaloneTaskIterator::StandaloneTaskIterator");
}

Status StandaloneTaskIterator::GetNext(std::vector<Tensor>& element,
                                       bool& end_of_sequence) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/data/service/task_runner.cc", "StandaloneTaskIterator::GetNext");

  return iterator_->GetNext(&element, &end_of_sequence);
}

int64_t StandaloneTaskIterator::Cardinality() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/data/service/task_runner.cc", "StandaloneTaskIterator::Cardinality");

  return dataset_->Get()->Cardinality();
}

Status TaskRunner::Create(const experimental::WorkerConfig& worker_config,
                          const TaskDef& task_def,
                          std::unique_ptr<TaskIterator> iterator,
                          std::unique_ptr<TaskRunner>& out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_3(mht_3_v, 244, "", "./tensorflow/core/data/service/task_runner.cc", "TaskRunner::Create");

  if (task_def.optional_num_consumers_case() == TaskDef::kNumConsumers) {
    int64_t cardinality = iterator->Cardinality();
    if (cardinality != kInfiniteCardinality &&
        cardinality != kUnknownCardinality) {
      return errors::FailedPrecondition(
          "Round robin reads require that the input dataset has infinite "
          "cardinality, but the dataset has cardinality ",
          cardinality,
          ". Consider adding a `.repeat()` transformation to the dataset.");
    }
    out = absl::make_unique<RoundRobinTaskRunner>(std::move(iterator),
                                                  task_def.num_consumers(),
                                                  task_def.worker_address());
  } else {
    out =
        absl::make_unique<FirstComeFirstServedTaskRunner>(std::move(iterator));
  }
  return Status::OK();
}

FirstComeFirstServedTaskRunner::FirstComeFirstServedTaskRunner(
    std::unique_ptr<TaskIterator> iterator)
    : iterator_(std::move(iterator)), buffer_(/*buffer_size=*/1) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_4(mht_4_v, 270, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::FirstComeFirstServedTaskRunner");

  RunPrefetchThread();
}

FirstComeFirstServedTaskRunner::~FirstComeFirstServedTaskRunner() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::~FirstComeFirstServedTaskRunner");
 Cancel(); }

Status FirstComeFirstServedTaskRunner::GetNext(const GetElementRequest& req,
                                               GetElementResult& result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_6(mht_6_v, 283, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::GetNext");

  return GetNext(result);
}

Status FirstComeFirstServedTaskRunner::GetNext(GetElementResult& result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_7(mht_7_v, 290, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::GetNext");

  TF_ASSIGN_OR_RETURN(result, buffer_.Pop());
  return Status::OK();
}

Status FirstComeFirstServedTaskRunner::PrefetchFn() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_8(mht_8_v, 298, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::PrefetchFn");

  while (true) {
    TF_RETURN_IF_ERROR(buffer_.Push(GetNextFromInputIterator()));
  }
  return Status::OK();
}

void FirstComeFirstServedTaskRunner::RunPrefetchThread() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_9(mht_9_v, 308, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::RunPrefetchThread");

  auto prefetch_fn = [this] {
    Status status = PrefetchFn();
    if (!status.ok()) {
      buffer_.Cancel(status);
    }
  };
  prefetch_thread_ = absl::WrapUnique(Env::Default()->StartThread(
      /*thread_options=*/{}, /*name=*/"tf_data_service_fcfs_prefetch_thread",
      prefetch_fn));
}

StatusOr<GetElementResult>
FirstComeFirstServedTaskRunner::GetNextFromInputIterator()
    TF_LOCKS_EXCLUDED(mu_) {
  GetElementResult result;
  std::vector<Tensor> element;
  bool end_of_task;
  result.skip = false;
  {
    mutex_lock l(mu_);
    TF_RETURN_IF_ERROR(iterator_->GetNext(element, end_of_task));
    result.end_of_sequence = end_of_task;
    result.element_index = element_index_++;
  }
  if (!end_of_task) {
    result.components = std::move(element);
  }
  return result;
}

void FirstComeFirstServedTaskRunner::Cancel() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_10(mht_10_v, 342, "", "./tensorflow/core/data/service/task_runner.cc", "FirstComeFirstServedTaskRunner::Cancel");

  VLOG(2) << "Cancelling tf.data service FCFS task.";
  buffer_.Cancel(errors::Cancelled("tf.data service FCFS task is cancelled."));
}

CachingTaskRunner::CachingTaskRunner(std::unique_ptr<TaskIterator> iterator,
                                     size_t max_cache_size_bytes)
    : fcfs_task_runner_(std::move(iterator)),
      cache_(max_cache_size_bytes,
             absl::make_unique<GetElementResultSequence>(fcfs_task_runner_)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_11(mht_11_v, 354, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::CachingTaskRunner");

  LOG(INFO) << "Initialized tf.data service multi-trainer cache with "
            << FormatBytes(max_cache_size_bytes) << " of memory.";
}

CachingTaskRunner::~CachingTaskRunner() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_12(mht_12_v, 362, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::~CachingTaskRunner");
 Cancel(); }

Status CachingTaskRunner::GetNext(const GetElementRequest& req,
                                  GetElementResult& result) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_13(mht_13_v, 368, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::GetNext");

  TF_ASSIGN_OR_RETURN(std::shared_ptr<const GetElementResult> element,
                      cache_.Get(req.trainer_id()));
  result = element->Copy();
  return Status::OK();
}

CachingTaskRunner::GetElementResultSequence::GetElementResultSequence(
    FirstComeFirstServedTaskRunner& fcfs_task_runner)
    : fcfs_task_runner_(fcfs_task_runner) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_14(mht_14_v, 380, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::GetElementResultSequence::GetElementResultSequence");
}

StatusOr<GetElementResult>
CachingTaskRunner::GetElementResultSequence::GetNext() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_15(mht_15_v, 386, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::GetElementResultSequence::GetNext");

  GetElementResult result;
  TF_RETURN_IF_ERROR(fcfs_task_runner_.GetNext(result));
  return result;
}

size_t CachingTaskRunner::GetElementResultSequence::GetElementSizeBytes(
    const GetElementResult& element) const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_16(mht_16_v, 396, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::GetElementResultSequence::GetElementSizeBytes");

  return element.EstimatedMemoryUsageBytes();
}

void CachingTaskRunner::Cancel() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_17(mht_17_v, 403, "", "./tensorflow/core/data/service/task_runner.cc", "CachingTaskRunner::Cancel");

  VLOG(2) << "Cancelling tf.data service multi-trainer cache task.";
  if (!cache_.IsCancelled()) {
    cache_.Cancel(errors::Cancelled(
        "tf.data service multi-trainer cache task is cancelled."));
  }
  fcfs_task_runner_.Cancel();
}

RoundRobinTaskRunner::RoundRobinTaskRunner(
    std::unique_ptr<TaskIterator> iterator, int64_t num_consumers,
    string worker_address)
    : num_consumers_(num_consumers),
      worker_address_(worker_address),
      buffer_(num_consumers_),
      prefetch_thread_(std::move(iterator), num_consumers_) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("worker_address: \"" + worker_address + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_18(mht_18_v, 422, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::RoundRobinTaskRunner");

  VLOG(1) << "Creating task runner for distributing data round-robin to "
          << num_consumers << " consumers";
}

Status RoundRobinTaskRunner::ValidateRequest(const GetElementRequest& req) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_19(mht_19_v, 430, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::ValidateRequest");

  if (req.consumer_index() < 0 || req.round_index() < 0) {
    return errors::FailedPrecondition(
        "RoundRobinTaskRunner needs to know the consumer index and element "
        "index of each request.");
  }
  if (req.consumer_index() >= num_consumers_) {
    return errors::FailedPrecondition(
        "Requesting data for consumer index ", req.consumer_index(),
        ", but the task is configured for only ", num_consumers_, " consumers");
  }
  return Status::OK();
}

Status RoundRobinTaskRunner::PrepareFullRound(int64_t wait_us)
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_20(mht_20_v, 448, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::PrepareFullRound");

  VLOG(1) << worker_address_ << ": Preparing full round for round "
          << current_round_;
  // This was the last request to arrive, time to start a new round.
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(wait_us, buffer_));
  round_skipped_ = buffer_.empty();
  new_round_cv_.notify_all();
  return Status::OK();
}

Status RoundRobinTaskRunner::PreparePartialRound()
    TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_21(mht_21_v, 462, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::PreparePartialRound");

  VLOG(1) << worker_address_ << ": Starting partial round " << first_round_
          << " for " << requests_[first_round_].size() << " consumers";
  current_round_ = first_round_;
  new_round_cv_.notify_all();
  // Indicates that we need a partial round to get consumers back in sync.
  auto next_round_request = *(requests_[first_round_ + 1].begin()->second);
  if (next_round_request.skipped_previous_round()) {
    VLOG(1) << "Skipping partial round";
    round_skipped_ = true;
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(prefetch_thread_.FillBuffer(/*wait_us=*/-1, buffer_));
  round_skipped_ = false;
  return Status::OK();
}

Status RoundRobinTaskRunner::PrepareRound(const GetElementRequest& req) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_22(mht_22_v, 482, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::PrepareRound");

  mutex_lock l(mu_);
  first_round_ = std::min(first_round_, req.round_index());
  absl::flat_hash_map<int64_t, const GetElementRequest*>& round =
      requests_[req.round_index()];
  round[req.consumer_index()] = &req;
  auto cleanup = gtl::MakeCleanup([&]() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    requests_[req.round_index()].erase(req.consumer_index());
  });
  if (current_round_ < req.round_index() && round.size() == num_consumers_) {
    current_round_ = req.round_index();
    int64_t wait_us = kWaitBeforeSkipUs;
    if (!req.allow_skip()) {
      wait_us = -1;
    }
    TF_RETURN_IF_ERROR(PrepareFullRound(wait_us));
  }
  if (current_round_ < 0 &&
      requests_[first_round_].size() + requests_[first_round_ + 1].size() ==
          num_consumers_) {
    TF_RETURN_IF_ERROR(PreparePartialRound());
  }
  while (!cancelled_ && current_round_ < req.round_index()) {
    TF_RETURN_IF_ERROR(prefetch_thread_.GetStatus());
    new_round_cv_.wait(l);
  }
  if (current_round_ < req.round_index() && cancelled_) {
    return errors::Cancelled("Worker is shutting down.");
  }
  if (current_round_ != req.round_index()) {
    return errors::FailedPrecondition(
        "Consumer ", req.consumer_index(), " requested data for round ",
        req.round_index(), ", but the current round has already reached ",
        current_round_,
        ". This may indicate that the consumer was restarted with the same job "
        "name.`");
  }
  return prefetch_thread_.GetStatus();
}

Status RoundRobinTaskRunner::GetNext(const GetElementRequest& req,
                                     GetElementResult& result) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_23(mht_23_v, 526, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::GetNext");

  TF_RETURN_IF_ERROR(ValidateRequest(req));
  result.end_of_sequence = false;
  VLOG(2) << worker_address_ << ": Received request from consumer index "
          << req.consumer_index() << " for round " << req.round_index();
  TF_RETURN_IF_ERROR(PrepareRound(req));
  tf_shared_lock l(mu_);
  result.skip = round_skipped_;
  if (round_skipped_) {
    VLOG(1) << worker_address_ << ": Buffer not ready, skipping round "
            << current_round_ << " for consumer " << req.consumer_index();
    return Status::OK();
  }
  auto& buffer_result = buffer_[req.consumer_index()];
  result.element_index = buffer_result->index;
  std::vector<Tensor> element;
  for (auto& component : buffer_result->components) {
    element.push_back(tensor::DeepCopy(component));
  }
  if (VLOG_IS_ON(2)) {
    int64_t size = 0;
    for (auto& component : element) {
      size += component.TotalBytes();
    }
    VLOG(2) << worker_address_ << ": Returning element " << result.element_index
            << " to consumer " << req.consumer_index() << " for round "
            << req.round_index() << ". element size " << size;
  }
  result.components = std::move(element);
  return Status::OK();
}

void RoundRobinTaskRunner::Cancel() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_24(mht_24_v, 561, "", "./tensorflow/core/data/service/task_runner.cc", "RoundRobinTaskRunner::Cancel");

  mutex_lock l(mu_);
  cancelled_ = true;
  new_round_cv_.notify_all();
}

PrefetchThread::PrefetchThread(std::unique_ptr<TaskIterator> iterator,
                               int64_t round_size)
    : iterator_(std::move(iterator)), round_size_(round_size) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_25(mht_25_v, 572, "", "./tensorflow/core/data/service/task_runner.cc", "PrefetchThread::PrefetchThread");

  thread_ = absl::WrapUnique(
      Env::Default()->StartThread({}, "round-robin-prefetch", [&] { Run(); }));
}

PrefetchThread::~PrefetchThread() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_26(mht_26_v, 580, "", "./tensorflow/core/data/service/task_runner.cc", "PrefetchThread::~PrefetchThread");

  mutex_lock l(mu_);
  cancelled_ = true;
  cv_.notify_all();
}

void PrefetchThread::Run() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_27(mht_27_v, 589, "", "./tensorflow/core/data/service/task_runner.cc", "PrefetchThread::Run");

  while (true) {
    {
      mutex_lock l(mu_);
      while (!cancelled_ && buffer_.size() >= round_size_) {
        cv_.wait(l);
      }
      if (cancelled_) {
        return;
      }
    }
    std::vector<Tensor> element;
    bool end_of_sequence;
    Status s = iterator_->GetNext(element, end_of_sequence);
    if (!s.ok()) {
      mutex_lock l(mu_);
      status_ = s;
      cv_.notify_all();
      return;
    }
    if (end_of_sequence) {
      mutex_lock l(mu_);
      status_ = errors::FailedPrecondition(
          "Encountered end of sequence on a round-robin read iterator. "
          "Please ensure that the dataset used for round-robin reading has "
          "infinite cardinality, e.g. by adding a .repeat() transformation "
          "at the end.");
      cv_.notify_all();
      return;
    }
    mutex_lock l(mu_);
    buffer_.push_back(absl::make_unique<Element>(std::move(element), index_++));
    cv_.notify_all();
  }
}

Status PrefetchThread::FillBuffer(int64_t wait_us,
                                  std::vector<std::unique_ptr<Element>>& out) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_28(mht_28_v, 629, "", "./tensorflow/core/data/service/task_runner.cc", "PrefetchThread::FillBuffer");

  int64_t start_us = Env::Default()->NowMicros();
  out.clear();
  mutex_lock l(mu_);
  while (buffer_.size() < round_size_ && !cancelled_ && status_.ok()) {
    int64_t remaining_us = start_us + wait_us - Env::Default()->NowMicros();
    if (wait_us >= 0 && remaining_us <= 0) {
      break;
    }
    cv_.wait_for(l, std::chrono::microseconds(remaining_us));
  }
  TF_RETURN_IF_ERROR(status_);
  if (cancelled_) {
    return errors::Cancelled("Prefetch thread cancelled");
  }
  if (buffer_.size() < round_size_) {
    DCHECK_GE(wait_us, 0);
    return Status::OK();
  }
  for (auto& elem : buffer_) {
    out.push_back(std::move(elem));
  }
  buffer_.clear();
  cv_.notify_all();
  return Status::OK();
}

Status PrefetchThread::GetStatus() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdataPSservicePStask_runnerDTcc mht_29(mht_29_v, 659, "", "./tensorflow/core/data/service/task_runner.cc", "PrefetchThread::GetStatus");

  mutex_lock l(mu_);
  return status_;
}
}  // namespace data
}  // namespace tensorflow
