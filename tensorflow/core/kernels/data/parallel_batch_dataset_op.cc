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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/parallel_batch_dataset_op.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const ParallelBatchDatasetOp::kDatasetType;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kInputDataset;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kBatchSize;
/* static */ constexpr const char* const
    ParallelBatchDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kDropRemainder;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kParallelCopy;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kOutputShapes;
/* static */ constexpr const char* const ParallelBatchDatasetOp::kDeterministic;

namespace {

constexpr char kBatchResultsSize[] = "batch_results_size";
constexpr char kTFDataParallelBatch[] = "tf_data_parallel_batch";
constexpr char kBatchResults[] = "batch_results";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kNumElements[] = "num_elements";
constexpr char kCallFinished[] = "call_finished";
constexpr char kOutputAllocated[] = "output_allocated";
constexpr char kStatus[] = "status";

}  // namespace

class ParallelBatchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t batch_size, int64_t num_parallel_calls,
          bool drop_remainder, bool parallel_copy, const DatasetBase* input,
          DeterminismPolicy deterministic)
      : DatasetBase(DatasetContext(ctx)),
        batch_size_(batch_size),
        // Dataset batch is sometimes used to stack all elements in the
        // dataset. In such cases, a very large batch size (e.g., INT32_MAX)
        // is passed with drop_remainder set to false. Avoid OOM in such case
        // by limiting `reserve()` size by 2**16.
        reserve_size_(drop_remainder ? batch_size
                                     : std::min<int64_t>(batch_size, 1 << 16)),
        num_parallel_calls_(num_parallel_calls),
        drop_remainder_(drop_remainder),
        parallel_copy_(parallel_copy),
        input_(input),
        deterministic_(deterministic),
        traceme_metadata_(
            {{"autotune",
              num_parallel_calls == model::kAutotune ? "true" : "false"},
             {"batch_size",
              strings::Printf("%lld", static_cast<long long>(batch_size))},
             {"drop_remainder", drop_remainder ? "true" : "false"},
             {"parallel_copy", parallel_copy ? "true" : "false"}}) {
    input_->Ref();

    const auto& input_shapes = input_->output_shapes();
    output_shapes_.reserve(input_shapes.size());
    for (const auto& input_shape : input_shapes) {
      if (drop_remainder_ || input_->Cardinality() == kInfiniteCardinality) {
        output_shapes_.emplace_back(
            PartialTensorShape({batch_size_}).Concatenate(input_shape));
      } else {
        output_shapes_.emplace_back(
            PartialTensorShape({-1}).Concatenate(input_shape));
      }
    }
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_0(mht_0_v, 278, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_1(mht_1_v, 289, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_2(mht_2_v, 296, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "output_shapes");

    return output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_3(mht_3_v, 303, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.set_args(batch_size_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_4(mht_4_v, 312, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "CardinalityInternal");

    int64_t n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / batch_size_ + (n % batch_size_ == 0 || drop_remainder_ ? 0 : 1);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_5(mht_5_v, 323, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_6(mht_6_v, 331, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_7(mht_7_v, 341, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "AsGraphDefInternal");

    // Input: input_dataset
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    // Input: batch_size
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));

    // Input: num_parallel_calls
    Node* num_parallel_calls = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_parallel_calls_, &num_parallel_calls));

    // Input: drop_remainder
    Node* drop_remainder = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder));

    std::vector<std::pair<StringPiece, AttrValue>> attrs;
    // Attr: parallel_copy
    AttrValue parallel_copy_attr;
    b->BuildAttrValue(parallel_copy_, &parallel_copy_attr);
    attrs.emplace_back(kParallelCopy, parallel_copy_attr);

    // Attr: deterministic
    AttrValue deterministic_attr;
    b->BuildAttrValue(deterministic_.String(), &deterministic_attr);
    attrs.emplace_back(kDeterministic, deterministic_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {input_graph_node, batch_size, num_parallel_calls, drop_remainder},
        attrs, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          num_parallel_calls_(std::make_shared<model::SharedState>(
              params.dataset->num_parallel_calls_, mu_, cond_var_)),
          deterministic_(params.dataset->deterministic_.IsDeterministic() ||
                         params.dataset->deterministic_.IsDefault()) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_8(mht_8_v, 389, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "Iterator");
}

    ~Iterator() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_9(mht_9_v, 394, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "~Iterator");

      CancelThreads(/*wait=*/true);
      if (deregister_fn_) deregister_fn_();
    }

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_10(mht_10_v, 402, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "Initialize");

      mutex_lock l(*mu_);
      interleave_depth_ = ctx->interleave_depth();

      if (num_parallel_calls_->value == model::kAutotune) {
        // If we copy elements in the same batch in parallel, to be safe, we
        // initialize the parallelism to be 1.
        if (dataset()->parallel_copy_) {
          num_parallel_calls_->value = 1;
        } else {
          num_parallel_calls_->value = GetAutotuneDefaultParallelism(ctx);
        }
      }
      cancellation_manager_ = absl::make_unique<CancellationManager>();
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(),
          [this]() { CancelThreads(/*wait=*/false); }, &deregister_fn_));
      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      return dataset()->input_->MakeIterator(IteratorContext(params), this,
                                             prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_11(mht_11_v, 430, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "GetNextInternal");

      std::shared_ptr<BatchResult> result;
      {
        mutex_lock l(*mu_);
        EnsureRunnerThreadStarted(ctx);
        while (ShouldWait(&result)) {
          RecordStop(ctx);
          cond_var_->wait(l);
          RecordStart(ctx);
        }
        if (cancelled_) {
          return errors::Cancelled("Iterator was cancelled");
        }
      }

      profiler::TraceMe traceme([&] {
        return profiler::TraceMeEncode("ParallelBatchConsume",
                                       {{"element_id", result->uid}});
      });
      mutex_lock l(result->mu);
      // Deallocate tensors allocated for the output.
      auto cleanup =
          gtl::MakeCleanup([result]() TF_EXCLUSIVE_LOCKS_REQUIRED(
                               &BatchResult::mu) { result->output.clear(); });
      if (result->output_allocated) {
        RecordBufferDequeue(ctx, result->output);
      }
      TF_RETURN_IF_ERROR(
          ProcessBatch(dataset()->batch_size_, result->num_elements,
                       dataset()->drop_remainder_, result->status, ctx,
                       out_tensors, end_of_sequence, &result->output));
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args),
          /*ratio=*/dataset()->batch_size_, /*memory_ratio=*/1.0,
          {model::MakeParameter("parallelism", num_parallel_calls_, /*min=*/1,
                                /*max=*/ctx->runner_threadpool_size())});
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_12(mht_12_v, 478, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "SaveInternal");

      mutex_lock l(*mu_);
      // Wait for all in-flight calls to complete.
      while (num_calls_ > 0) {
        cond_var_->wait(l);
      }
      DCHECK_EQ(num_calls_, 0);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kBatchResultsSize),
                                             batch_results_.size()));
      for (size_t i = 0; i < batch_results_.size(); ++i) {
        TF_RETURN_IF_ERROR(WriteBatchResult(writer, i));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_13(mht_13_v, 498, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "RestoreInternal");

      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      int64_t batch_results_size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kBatchResultsSize),
                                            &batch_results_size));
      DCHECK(batch_results_.empty());
      for (int i = 0; i < batch_results_size; ++i) {
        TF_RETURN_IF_ERROR(ReadBatchResult(ctx, reader, i));
      }
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_14(mht_14_v, 514, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "GetTraceMeMetadata");

      int64_t parallelism = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = num_parallel_calls_->value;
        mu_->unlock();
      }
      auto result = dataset()->traceme_metadata_;
      result.push_back(
          std::make_pair("deterministic", deterministic_ ? "true" : "false"));
      result.push_back(std::make_pair(
          "parallelism",
          parallelism == -1
              ? kTraceInfoUnavailable
              : strings::Printf("%lld", static_cast<long long>(parallelism))));
      result.push_back(std::make_pair(
          "interleave_depth",
          strings::Printf("%lld", static_cast<long long>(interleave_depth_))));
      return result;
    }

    // BatchResult encapsulates the output batch.
    struct BatchResult {
      explicit BatchResult()
          : end_of_input(false),
            num_elements(0),
            status(Status::OK()),
            call_finished(false),
            output_allocated(false),
            uid(tensorflow::EnvTime::NowNanos()) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_15(mht_15_v, 547, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "BatchResult");
}

      mutex mu;
      bool end_of_input TF_GUARDED_BY(mu);
      int64_t num_elements TF_GUARDED_BY(mu);
      std::vector<Tensor> output TF_GUARDED_BY(mu);
      Status status TF_GUARDED_BY(mu);
      bool call_finished TF_GUARDED_BY(&Iterator::mu_);
      bool output_allocated TF_GUARDED_BY(mu);
      const int64_t uid = -1;
    };

    void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                       const std::shared_ptr<BatchResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      mutex_lock l(*mu_);
      num_calls_--;
      result->call_finished = true;
      cond_var_->notify_all();
    }

    // The function fetches elements from input dataset sequentially and then
    // executes the batching for different batches in parallel using the context
    // runner.
    void CallBatching(std::shared_ptr<IteratorContext> ctx,
                      const std::shared_ptr<BatchResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      profiler::TraceMe traceme([&] {
        return profiler::TraceMeEncode("ParallelBatchProduce",
                                       {{"element_id", result->uid}});
      });

      if (!input_impl_) {
        CallCompleted(ctx, result);
        return;
      }

      // Each row of `batch_elements` is a tuple of tensors from the input
      // iterator.
      auto batch_elements =
          std::make_shared<std::vector<std::vector<Tensor>>>();
      batch_elements->reserve(dataset()->reserve_size_);

      bool end_of_input = false;
      for (int i = 0; i < dataset()->batch_size_ && !end_of_input; ++i) {
        std::vector<Tensor> batch_element_tuple;
        Status status = input_impl_->GetNext(ctx.get(), &batch_element_tuple,
                                             &end_of_input);
        {
          mutex_lock l(result->mu);
          result->end_of_input = result->end_of_input || end_of_input;
          result->status.Update(status);
          if (result->end_of_input || !result->status.ok()) break;
        }
        if (!end_of_input) {
          batch_elements->emplace_back(std::move(batch_element_tuple));
          mutex_lock l(result->mu);
          result->num_elements++;
        } else {
          input_impl_.reset();
        }
      }

      if (batch_elements->empty()) {
        CallCompleted(ctx, result);
        return;
      }

      auto copy_elements_fn = [this, ctx, result, batch_elements]() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_16(mht_16_v, 618, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "lambda");

        Status status;
        {
          mutex_lock l(result->mu);
          auto allocation_callback =
              [this, ctx, result]()
                  TF_EXCLUSIVE_LOCKS_REQUIRED(&BatchResult::mu) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_17(mht_17_v, 627, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "lambda");

                    result->output_allocated = true;
                    RecordBufferEnqueue(ctx.get(), result->output);
                    return Status::OK();
                  };
          status = CopyBatch(CopyBatchParams(ctx.get()), *batch_elements,
                             dataset()->parallel_copy_,
                             std::move(allocation_callback), &result->output);
          result->status.Update(status);
        }
        CallCompleted(ctx, result);
        return status;
      };

      (*ctx->runner())(copy_elements_fn);
    }

    void CancelThreads(bool wait) TF_LOCKS_EXCLUDED(mu_) {
      cancellation_manager_->StartCancel();
      mutex_lock l(*mu_);
      cancelled_ = true;
      cond_var_->notify_all();
      // Wait for all in-flight calls to complete.
      while (wait && num_calls_ > 0) {
        cond_var_->wait(l);
      }
    }

    void EnsureRunnerThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_18(mht_18_v, 659, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "EnsureRunnerThreadStarted");

      if (!runner_thread_) {
        auto ctx_copy = std::make_shared<IteratorContext>(*ctx);
        runner_thread_ = ctx->StartThread(
            kTFDataParallelBatch,
            std::bind(&Iterator::RunnerThread, this, ctx_copy));
      }
    }

    void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
        TF_LOCKS_EXCLUDED(*mu_) {
      std::vector<std::shared_ptr<BatchResult>> new_calls;
      RecordStart(ctx.get());
      auto stop_cleanup =
          gtl::MakeCleanup([this, &ctx]() { RecordStop(ctx.get()); });
      {
        tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
        new_calls.reserve(num_parallel_calls_->value);
      }
      auto busy = [this]() TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_19(mht_19_v, 681, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "lambda");

        int64_t num_parallel_calls = num_parallel_calls_->value;
        return num_calls_ >= num_parallel_calls ||
               batch_results_.size() >= num_parallel_calls;
      };
      while (true) {
        {
          mutex_lock l(*mu_);
          while (!cancelled_ && busy()) {
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
          }

          if (cancelled_) {
            return;
          }

          while (!busy()) {
            batch_results_.push_back(std::make_shared<BatchResult>());
            new_calls.emplace_back(batch_results_.back());
            num_calls_++;
          }
        }
        for (const auto& call : new_calls) {
          CallBatching(ctx, call);
        }
        new_calls.clear();
      }
    }

    // Determines whether the caller needs to wait for a result. Upon returning
    // false, `result` will point to the result.
    bool ShouldWait(std::shared_ptr<BatchResult>* result)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_20(mht_20_v, 718, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "ShouldWait");

      if (cancelled_) {
        return false;
      }
      if (!deterministic_) {
        // Iterate through in-flight results and return the first one that is
        // found to be available and not end-of-input. If the first result (in
        // order) is end-of-input, we know that all earlier iterations have
        // already been completed, so it is safe to return that result for the
        // caller to process end of iteration.
        bool find_batch;
        for (auto it = batch_results_.begin(); it != batch_results_.end();
             ++it) {
          if (!(*it)->call_finished) continue;
          find_batch = (it == batch_results_.begin());
          if (!find_batch) {
            tf_shared_lock l((*it)->mu);
            find_batch = !(*it)->end_of_input;
          }
          if (find_batch) {
            std::swap(*result, *it);
            batch_results_.erase(it);
            cond_var_->notify_all();
            return false;
          }
        }
      } else if (!batch_results_.empty() &&
                 batch_results_.front()->call_finished) {
        std::swap(*result, batch_results_.front());
        batch_results_.pop_front();
        cond_var_->notify_all();
        return false;
      }
      return true;
    }

    Status ReadBatchResult(IteratorContext* ctx, IteratorStateReader* reader,
                           size_t index) TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_21(mht_21_v, 758, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "ReadBatchResult");

      batch_results_.push_back(std::make_shared<BatchResult>());
      std::shared_ptr<BatchResult> result = batch_results_.back();
      string batch_prefix = strings::StrCat(kBatchResults, "_", index);
      mutex_lock l(result->mu);
      result->end_of_input = reader->Contains(
          full_name(strings::StrCat(batch_prefix, "_", kEndOfInput)));
      TF_RETURN_IF_ERROR(reader->ReadScalar(
          full_name(strings::StrCat(batch_prefix, "_", kNumElements)),
          &result->num_elements));
      result->call_finished = reader->Contains(
          full_name(strings::StrCat(batch_prefix, "_", kCallFinished)));
      result->output_allocated = reader->Contains(
          full_name(strings::StrCat(batch_prefix, "_", kOutputAllocated)));

      TF_RETURN_IF_ERROR(ReadBatch(ctx, reader, dataset()->batch_size_,
                                   prefix(), batch_prefix, &result->output));
      TF_RETURN_IF_ERROR(ReadStatus(prefix(),
                                    strings::StrCat(batch_prefix, "_", kStatus),
                                    reader, &result->status));
      if (result->output_allocated) {
        RecordBufferEnqueue(ctx, result->output);
      }
      return Status::OK();
    }

    Status WriteBatchResult(IteratorStateWriter* writer, size_t index)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_22(mht_22_v, 788, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "WriteBatchResult");

      std::shared_ptr<BatchResult> result = batch_results_[index];
      string batch_prefix = strings::StrCat(kBatchResults, "_", index);
      mutex_lock l(result->mu);
      if (result->end_of_input) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(batch_prefix, "_", kEndOfInput)), ""));
      }
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          full_name(strings::StrCat(batch_prefix, "_", kNumElements)),
          result->num_elements));
      if (result->call_finished) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(batch_prefix, "_", kCallFinished)), ""));
      }
      if (result->output_allocated) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(batch_prefix, "_", kOutputAllocated)),
            ""));
      }

      TF_RETURN_IF_ERROR(WriteBatch(dataset()->batch_size_,
                                    result->num_elements, prefix(),
                                    batch_prefix, writer, &result->output));
      TF_RETURN_IF_ERROR(
          WriteStatus(prefix(), strings::StrCat(batch_prefix, "_", kStatus),
                      result->status, writer));
      return Status::OK();
    }

    // Used for coordination between the main thread and the runner thread.
    const std::shared_ptr<mutex> mu_;
    // Used for coordination between the main thread and the runner thread. In
    // particular, the runner thread should only schedule new calls when the
    // number of in-flight calls is less than the user specified level of
    // parallelism and there are slots available in the `invocation_results_`
    // buffer.
    const std::shared_ptr<condition_variable> cond_var_;
    // Identifies the maximum number of parallel calls.
    const std::shared_ptr<model::SharedState> num_parallel_calls_;
    const bool deterministic_;

    // Controls cancellation of `input_impl_`. Must be ordered before
    // `input_impl_` so that `input_impl_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    // Counts the number of outstanding calls for this batch.
    int64_t num_calls_ TF_GUARDED_BY(*mu_) = 0;
    std::unique_ptr<IteratorBase> input_impl_;
    // Buffer for storing the (intermediate) batch results. Whenever a non-empty
    // batch result is added to or removed from `batch_results_`, call
    // `RecordBufferEnqueue` or `RecordBufferDequeue` respectively.
    //
    // TODO(xiaojies): improve the accuracy of the condition used for
    // determining when to record allocated bytes.
    std::deque<std::shared_ptr<BatchResult>> batch_results_ TF_GUARDED_BY(*mu_);
    // Background thread used for coordinating input processing.
    std::unique_ptr<Thread> runner_thread_ TF_GUARDED_BY(*mu_);
    // Determines whether the transformation has been cancelled.
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;

    // Records the number of ParallelInterleave operations in the path from the
    // root node to this node (not including this node) in the input pipeline
    // tree. We record the interleave depth so that it can be included in the
    // trace metadata.
    int64 interleave_depth_ = -1;
  };

  const int64_t batch_size_;
  const int64_t reserve_size_;
  const int64_t num_parallel_calls_;
  const bool drop_remainder_;
  const bool parallel_copy_;
  const DatasetBase* const input_;
  std::vector<PartialTensorShape> output_shapes_;
  const DeterminismPolicy deterministic_;
  const TraceMeMetadata traceme_metadata_;
};

ParallelBatchDatasetOp::ParallelBatchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_23(mht_23_v, 873, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "ParallelBatchDatasetOp::ParallelBatchDatasetOp");

  if (ctx->HasAttr(kDeterministic)) {
    std::string deterministic;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDeterministic, &deterministic));
    OP_REQUIRES_OK(
        ctx, DeterminismPolicy::FromString(deterministic, &deterministic_));
  }
  if (ctx->HasAttr(kParallelCopy)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kParallelCopy, &parallel_copy_));
  }
}

void ParallelBatchDatasetOp::MakeDataset(OpKernelContext* ctx,
                                         DatasetBase* input,
                                         DatasetBase** output) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_batch_dataset_opDTcc mht_24(mht_24_v, 890, "", "./tensorflow/core/kernels/data/parallel_batch_dataset_op.cc", "ParallelBatchDatasetOp::MakeDataset");

  int64_t batch_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("Batch size must be greater than zero."));

  int64_t num_parallel_calls = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kNumParallelCalls,
                                                   &num_parallel_calls));

  bool drop_remainder = false;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<bool>(ctx, kDropRemainder, &drop_remainder));

  *output = new Dataset(ctx, batch_size, num_parallel_calls, drop_remainder,
                        parallel_copy_, input, deterministic_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ParallelBatchDataset").Device(DEVICE_CPU),
                        ParallelBatchDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
