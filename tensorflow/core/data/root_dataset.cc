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
class MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc() {
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

#include "tensorflow/core/data/root_dataset.h"

#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/framework/model.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kDatasetType[] = "Root";

constexpr char kAlgorithm[] = "algorithm";
constexpr char kCpuBudget[] = "cpu_budget";
constexpr char kExperiments[] = "experiments";
constexpr char kInjectPrefetchEligibleOpt[] = "inject_prefetch_eligible";
constexpr char kIntraOpParallelism[] = "intra_op_parallelism";
constexpr char kMemBandwidth[] = "mem_bw_used_megabytes_per_sec";
constexpr char kPrivateThreadpoolSize[] = "threadpool_size";
constexpr char kRamBudget[] = "ram_budget_megabytes";
constexpr char kRamUsage[] = "ram_usage_megabytes";
constexpr char kMaxBufferBytes[] = "max_buffered_megabytes";

// If value `x` matches `y`, returns default value `z`. Otherwise, return `x`.
inline int64_t value_or_default(int64_t x, int64_t y, int64_t z) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/data/root_dataset.cc", "value_or_default");

  return x == y ? z : x;
}

void SetRootDatasetParams(const Options& options, RootDataset::Params* params) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/data/root_dataset.cc", "SetRootDatasetParams");

  if (ShouldConfigureMaxIntraOpParallelism(options)) {
    params->max_intra_op_parallelism =
        options.threading_options().max_intra_op_parallelism();
  }
  if (ShouldUsePrivateThreadPool(options)) {
    params->private_threadpool_size =
        options.threading_options().private_threadpool_size();
  }
  params->autotune = ShouldUseAutotuning(options);
  if (params->autotune) {
    params->autotune_algorithm =
        options.autotune_options().optional_autotune_algorithm_case() ==
                AutotuneOptions::kAutotuneAlgorithm
            ? options.autotune_options().autotune_algorithm()
            : model::AutotuneAlgorithm::DEFAULT;
    params->autotune_cpu_budget = value_or_default(
        options.autotune_options().cpu_budget(), 0, GetCpuBudget());
    params->autotune_ram_budget =
        value_or_default(options.autotune_options().ram_budget(), 0,
                         model::kRamBudgetShare * port::AvailableRam());
  }
}

void AddTraceMetadata(const RootDataset::Params& params,
                      TraceMeMetadata* trace_metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/data/root_dataset.cc", "AddTraceMetadata");

  if (params.autotune) {
    trace_metadata->push_back(std::make_pair(
        kAlgorithm, model::AutotuneAlgorithm_Name(params.autotune_algorithm)));
    trace_metadata->push_back(std::make_pair(
        kCpuBudget, strings::Printf("%lld", static_cast<long long>(
                                                params.autotune_cpu_budget))));
    trace_metadata->push_back(std::make_pair(
        kRamBudget,
        strings::Printf("%lld", static_cast<long long>(
                                    params.autotune_ram_budget / 1.0e6))));
  }
  if (params.max_intra_op_parallelism >= 0) {
    trace_metadata->push_back(std::make_pair(
        kIntraOpParallelism,
        strings::Printf("%lld", static_cast<long long>(value_or_default(
                                    params.max_intra_op_parallelism, 0,
                                    port::MaxParallelism())))));
  }
  if (params.private_threadpool_size >= 0) {
    trace_metadata->push_back(std::make_pair(
        kPrivateThreadpoolSize,
        strings::Printf("%lld", static_cast<long long>(value_or_default(
                                    params.private_threadpool_size, 0,
                                    port::MaxParallelism())))));
  }
  auto experiments = GetExperiments();
  if (!experiments.empty()) {
    trace_metadata->push_back(
        std::make_pair(kExperiments, absl::StrJoin(experiments, " ")));
  }
}
}  // namespace

// static
Status RootDataset::FromOptions(const DatasetBase* input,
                                DatasetBase** output) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_3(mht_3_v, 292, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::FromOptions");

  Params params;
  SetRootDatasetParams(input->options(), &params);
  *output = new RootDataset(input, params);
  (*output)->Initialize(/*metadata=*/{});
  return Status::OK();
}

Status RootDataset::FromOptions(core::RefCountPtr<DatasetBase> input,
                                DatasetBase** output) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_4(mht_4_v, 304, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::FromOptions");

  Params params;
  SetRootDatasetParams(input->options(), &params);
  *output = new RootDataset(std::move(input), params);
  (*output)->Initialize(/*metadata=*/{});
  return Status::OK();
}

class RootDataset::Iterator : public DatasetIterator<RootDataset> {
 public:
  explicit Iterator(const Params& params)
      : DatasetIterator<RootDataset>(params) {
    if (dataset()->params_.autotune) {
      model_ = std::make_shared<model::Model>();
    }
    if (dataset()->params_.max_intra_op_parallelism >= 0) {
      max_intra_op_parallelism_ =
          value_or_default(dataset()->params_.max_intra_op_parallelism, 0,
                           port::MaxParallelism());
    }
    if (dataset()->params_.private_threadpool_size >= 0) {
      threadpool_size_ =
          value_or_default(dataset()->params_.private_threadpool_size, 0,
                           port::MaxParallelism());
      thread_pool_ = absl::make_unique<thread::ThreadPool>(
          Env::Default(), ThreadOptions{}, "data_private_threadpool",
          threadpool_size_);
    }
    cancellation_manager_ = absl::make_unique<CancellationManager>();
  }

  ~Iterator() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_5(mht_5_v, 338, "", "./tensorflow/core/data/root_dataset.cc", "~Iterator");
 cancellation_manager_->StartCancel(); }

  Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_6(mht_6_v, 343, "", "./tensorflow/core/data/root_dataset.cc", "Initialize");

    return dataset()->input_->MakeIterator(IteratorContext(CreateParams(ctx)),
                                           this, prefix(), &input_impl_);
  }

  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_7(mht_7_v, 352, "", "./tensorflow/core/data/root_dataset.cc", "GetNextInternal");

    if (dataset()->params_.autotune) {
      TF_RETURN_IF_ERROR(EnsureModelThreadStarted(ctx));
    }
    return input_impl_->GetNext(IteratorContext(CreateParams(ctx)), out_tensors,
                                end_of_sequence);
  }

 protected:
  std::shared_ptr<model::Node> CreateNode(
      IteratorContext* ctx, model::Node::Args args) const override {
    return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
  }

  Status SaveInternal(SerializationContext* ctx,
                      IteratorStateWriter* writer) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_8(mht_8_v, 370, "", "./tensorflow/core/data/root_dataset.cc", "SaveInternal");

    TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
    return Status::OK();
  }

  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_9(mht_9_v, 379, "", "./tensorflow/core/data/root_dataset.cc", "RestoreInternal");

    TF_RETURN_IF_ERROR(
        RestoreInput(IteratorContext(CreateParams(ctx)), reader, input_impl_));
    return Status::OK();
  }

  TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_10(mht_10_v, 388, "", "./tensorflow/core/data/root_dataset.cc", "GetTraceMeMetadata");

    tensorflow::data::TraceMeMetadata traceme_metadata =
        dataset()->traceme_metadata_;
    const int64_t mem_bw = port::GetMemoryBandwidthInfo().bw_used;
    if (mem_bw != INT64_MAX) {
      traceme_metadata.push_back(std::make_pair(
          kMemBandwidth,
          strings::Printf("%lld", static_cast<long long>(mem_bw))));
    }
    const auto memory_info = port::GetMemoryInfo();
    const auto memory_usage = memory_info.total - memory_info.free;
    traceme_metadata.push_back(std::make_pair(
        kRamUsage,
        strings::Printf("%lld out of %lld (%.2f%%)",
                        static_cast<long long>(memory_usage / 1.0e6),
                        static_cast<long long>(memory_info.total / 1.0e6),
                        static_cast<double>(memory_usage) /
                            static_cast<double>(memory_info.total))));
    if (model_node() != nullptr) {
      traceme_metadata.push_back(std::make_pair(
          kMaxBufferBytes,
          strings::Printf(
              "%lld", static_cast<long long>(
                          model_node()->TotalMaximumBufferedBytes() / 1.0e6))));
    }
    return traceme_metadata;
  }

 private:
  IteratorContext::Params CreateParams(IteratorContext* ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_11(mht_11_v, 420, "", "./tensorflow/core/data/root_dataset.cc", "CreateParams");

    IteratorContext::Params params(ctx);
    if (dataset()->params_.autotune) {
      params.model = model_;
    }
    if (dataset()->params_.private_threadpool_size >= 0) {
      params.runner = [pool = thread_pool_.get()](std::function<void()> c) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_12(mht_12_v, 429, "", "./tensorflow/core/data/root_dataset.cc", "lambda");

        pool->Schedule(std::move(c));
      };
      params.runner_threadpool_size = threadpool_size_;
    }
    if (dataset()->params_.max_intra_op_parallelism >= 0) {
      params.runner =
          RunnerWithMaxParallelism(params.runner, max_intra_op_parallelism_);
    }
    params.options = &dataset()->options();
    return params;
  }

  Status EnsureModelThreadStarted(IteratorContext* ctx) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_13(mht_13_v, 445, "", "./tensorflow/core/data/root_dataset.cc", "EnsureModelThreadStarted");

    mutex_lock l(mu_);
    if (!model_thread_) {
      model_thread_ = ctx->StartThread("tf_data_model", [this]() {
        Status status =
            model_->OptimizeLoop(dataset()->params_.autotune_algorithm,
                                 dataset()->params_.autotune_cpu_budget,
                                 dataset()->params_.autotune_ram_budget,
                                 cancellation_manager_.get());
        if (!status.ok()) {
          LOG(WARNING) << "Optimization loop failed: " << status.ToString();
        }
      });
    }
    return Status::OK();
  }

  std::shared_ptr<model::Model> model_ = nullptr;
  // Controls cancellation of `model_thread_`. Must be ordered before
  // `model_thread_` so that `model_thread_` is destroyed first.
  std::unique_ptr<CancellationManager> cancellation_manager_;
  mutex mu_;
  std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
  int64_t max_intra_op_parallelism_;
  int64_t threadpool_size_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // Must be ordered last as its execution may depend on other members.
  std::unique_ptr<IteratorBase> input_impl_;
};

RootDataset::RootDataset(const DatasetBase* input, const Params& params)
    : DatasetBase(DatasetContext({name_utils::OpName(kDatasetType),
                                  name_utils::OpName(kDatasetType)})),
      input_(input),
      params_(std::move(params)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_14(mht_14_v, 483, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::RootDataset");

  AddTraceMetadata(params_, &traceme_metadata_);
}

RootDataset::RootDataset(core::RefCountPtr<DatasetBase> input,
                         const Params& params)
    : DatasetBase(DatasetContext({name_utils::OpName(kDatasetType),
                                  name_utils::OpName(kDatasetType)})),
      params_(std::move(params)) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_15(mht_15_v, 494, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::RootDataset");

  owned_input_ = std::move(input);
  input_ = owned_input_.get();
  AddTraceMetadata(params_, &traceme_metadata_);
}

RootDataset::~RootDataset() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_16(mht_16_v, 503, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::~RootDataset");
}

std::unique_ptr<IteratorBase> RootDataset::MakeIteratorInternal(
    const string& prefix) const {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_17(mht_17_v, 510, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::MakeIteratorInternal");

  return absl::make_unique<Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

const DataTypeVector& RootDataset::output_dtypes() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_18(mht_18_v, 518, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::output_dtypes");

  return input_->output_dtypes();
}

const std::vector<PartialTensorShape>& RootDataset::output_shapes() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_19(mht_19_v, 525, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::output_shapes");

  return input_->output_shapes();
}

string RootDataset::DebugString() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_20(mht_20_v, 532, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::DebugString");

  return name_utils::DatasetDebugString(kDatasetType);
}

int64_t RootDataset::CardinalityInternal() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_21(mht_21_v, 539, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::CardinalityInternal");

  return input_->Cardinality();
}

int64_t RootDataset::CardinalityInternal(CardinalityOptions options) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_22(mht_22_v, 546, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::CardinalityInternal");

  return input_->Cardinality(options);
}

Status RootDataset::Get(OpKernelContext* ctx, int64 index,
                        std::vector<Tensor>* out_tensors) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_23(mht_23_v, 554, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::Get");

  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(this->InputDatasets(&inputs));
  return inputs[0]->Get(ctx, index, out_tensors);
}

Status RootDataset::InputDatasets(
    std::vector<const DatasetBase*>* inputs) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_24(mht_24_v, 564, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::InputDatasets");

  inputs->push_back(input_);
  return Status::OK();
}

Status RootDataset::CheckExternalState() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_25(mht_25_v, 572, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::CheckExternalState");

  return input_->CheckExternalState();
}

Status RootDataset::AsGraphDefInternal(SerializationContext* ctx,
                                       DatasetGraphDefBuilder* b,
                                       Node** output) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_26(mht_26_v, 581, "", "./tensorflow/core/data/root_dataset.cc", "RootDataset::AsGraphDefInternal");

  return errors::Unimplemented("RootDataset does not support serialization.");
}

#if !defined(IS_MOBILE_PLATFORM)
Status FinalizeDataset(OpKernelContext* ctx, const DatasetBase* input,
                       DatasetBase** output) {
  const Options& options = input->options();
  absl::flat_hash_set<tstring> optimizations_enabled;
  absl::flat_hash_set<tstring> optimizations_disabled;
  absl::flat_hash_set<tstring> optimizations_default;
  GetOptimizations(options, &optimizations_enabled, &optimizations_disabled,
                   &optimizations_default);
  // Disable `enable_gradient_descent` as it assumes presence of ModelDatasetOp.
  optimizations_disabled.insert("enable_gradient_descent");
  if (!port::JobName().empty()) {
    // Enable kInjectPrefetchEligibleOpt that does not modify the graph and is
    // used to check whether the `inject_prefetch` optimization would modify the
    // graph.
    optimizations_enabled.insert(kInjectPrefetchEligibleOpt);
  }

  auto experiments = GetExperiments();
  LogAndRecordExperiments(experiments);
  auto optimizations =
      SelectOptimizations(experiments, optimizations_enabled,
                          optimizations_disabled, optimizations_default);
  if (optimizations.empty()) {
    return RootDataset::FromOptions(input, output);
  }

  auto optimization_configs = CreateGraphRewriteConfigs(options);
  auto config_factory = [&optimizations, &optimization_configs]() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdataPSroot_datasetDTcc mht_27(mht_27_v, 616, "", "./tensorflow/core/data/root_dataset.cc", "lambda");

    return CreateRewriterConfig(optimizations, optimization_configs);
  };
  core::RefCountPtr<DatasetBase> rewritten_output;
  Status s = RewriteDataset(ctx, input, std::move(config_factory),
                            /*record_fingerprint=*/true, &rewritten_output);

  *output = rewritten_output.get();
  bool rewritten = (*output != input);
  if (errors::IsDeadlineExceeded(s)) {
    // Ignore DeadlineExceeded as it implies that the attempted rewrite took too
    // long which should not prevent further computation.
    LOG(WARNING) << s.ToString();
  } else if (!s.ok()) {
    return s;
  }
  if (!rewritten) {
    return RootDataset::FromOptions(input, output);
  } else {
    return RootDataset::FromOptions(std::move(rewritten_output), output);
  }
  return Status::OK();
}

#else   // !IS_MOBILE_PLATFORM
Status FinalizeDataset(OpKernelContext* ctx, const DatasetBase* input,
                       DatasetBase** output) {
  return RootDataset::FromOptions(input, output);
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace data
}  // namespace tensorflow
