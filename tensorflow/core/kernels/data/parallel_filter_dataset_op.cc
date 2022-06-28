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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/parallel_filter_dataset_op.h"

#include <deque>
#include <utility>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ParallelFilterDatasetOp::kDatasetType;
/* static */ constexpr const char* const ParallelFilterDatasetOp::kInputDataset;
/* static */ constexpr const char* const
    ParallelFilterDatasetOp::kOtherArguments;
/* static */ constexpr const char* const
    ParallelFilterDatasetOp::kNumParallelCalls;
/* static */ constexpr const char* const ParallelFilterDatasetOp::kPredicate;
/* static */ constexpr const char* const
    ParallelFilterDatasetOp::kDeterministic;
/* static */ constexpr const char* const ParallelFilterDatasetOp::kTarguments;
/* static */ constexpr const char* const ParallelFilterDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ParallelFilterDatasetOp::kOutputShapes;

constexpr char kComponent[] = "component";
constexpr char kReturnValues[] = "return_values";
constexpr char kPredicateValues[] = "predicate_values";
constexpr char kInvocationResults[] = "invocation_results";
constexpr char kSize[] = "size";
constexpr char kEndOfInput[] = "end_of_input";
constexpr char kErrorCode[] = "code";
constexpr char kErrorMessage[] = "error_message";

class ParallelFilterDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          int64_t num_parallel_calls, DeterminismPolicy deterministic,
          std::unique_ptr<CapturedFunction> captured_func)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        num_parallel_calls_(num_parallel_calls),
        deterministic_(deterministic),
        captured_func_(std::move(captured_func)) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_0(mht_0_v, 243, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_4(mht_4_v, 275, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_5(mht_5_v, 283, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "CheckExternalState");

    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    Node* num_parallel_calls = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(static_cast<int32>(num_parallel_calls_),
                                    &num_parallel_calls));
    AttrValue deterministic_attr;
    b->BuildAttrValue(deterministic_.String(), &deterministic_attr);
    AttrValue predicate_attr;
    b->BuildAttrValue(captured_func_->func(), &predicate_attr);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {{0, input_graph_node}}, {{1, other_arguments}},
                      {{kDeterministic, deterministic_attr},
                       {kPredicate, predicate_attr},
                       {kTarguments, other_arguments_types_attr}},
                      output));
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
                         params.dataset->deterministic_.IsDefault()),
          autotune_(params.dataset->num_parallel_calls_ == model::kAutotune) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_7(mht_7_v, 334, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "Iterator");
}

    ~Iterator() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_8(mht_8_v, 339, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "~Iterator");

      CancelThreads(/*wait=*/true);
      input_impl_.reset();
      if (deregister_fn_) deregister_fn_();
    }

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_9(mht_9_v, 348, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "Initialize");

      mutex_lock l(*mu_);
      interleave_depth_ = ctx->interleave_depth();
      if (num_parallel_calls_->value == model::kAutotune) {
        num_parallel_calls_->value = GetAutotuneDefaultParallelism(ctx);
      }
      cancellation_manager_ = absl::make_unique<CancellationManager>();
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(),
          [this]() { CancelThreads(/*wait=*/false); }, &deregister_fn_));
      IteratorContext::Params params(ctx);
      params.cancellation_manager = cancellation_manager_.get();
      TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
          IteratorContext(params), this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_10(mht_10_v, 371, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "GetNextInternal");

      std::shared_ptr<InvocationResult> result;
      {
        mutex_lock l(*mu_);
        EnsureThreadsStarted(ctx);
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
        return profiler::TraceMeEncode("ParallelFilterConsume",
                                       {{"element_id", result->uid}});
      });
      return ProcessResult(ctx, result, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncUnknownRatioNode(
          std::move(args),
          {model::MakeParameter("parallelism", num_parallel_calls_, /*min=*/1,
                                /*max=*/ctx->runner_threadpool_size())});
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_11(mht_11_v, 405, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "SaveInternal");

      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      mutex_lock l(*mu_);
      // Wait for all in-flight calls to complete.
      while (num_calls_ > 0) {
        cond_var_->wait(l);
      }
      if (num_calls_ != 0) {
        return errors::FailedPrecondition(
            "Unexpected outstanding calls encountered.");
      }
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(absl::StrCat(prefix(), "::", kInvocationResults),
                              kSize, invocation_results_.size()));
      for (size_t i = 0; i < invocation_results_.size(); i++) {
        const auto& result = *(invocation_results_[i]);
        std::string element_prefix =
            absl::StrCat(prefix(), "::", kInvocationResults, "::", i);
        TF_RETURN_IF_ERROR(
            WriteStatusLocked(writer, element_prefix, result.status));
        TF_RETURN_IF_ERROR(WriteComponentsLocked(
            writer, absl::StrCat(element_prefix, "::", kReturnValues),
            result.return_values));
        TF_RETURN_IF_ERROR(WriteComponentsLocked(
            writer, absl::StrCat(element_prefix, "::", kPredicateValues),
            result.predicate_values));
        if (result.end_of_input) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(element_prefix, kEndOfInput, ""));
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_12(mht_12_v, 445, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "RestoreInternal");

      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      int64_t invocation_results_size;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(absl::StrCat(prefix(), "::", kInvocationResults),
                             kSize, &invocation_results_size));
      DCHECK(invocation_results_.empty());
      for (size_t i = 0; i < invocation_results_size; i++) {
        invocation_results_.push_back(std::make_shared<InvocationResult>());
        auto& result = *invocation_results_.back();
        std::string element_prefix =
            absl::StrCat(prefix(), "::", kInvocationResults, "::", i);
        TF_RETURN_IF_ERROR(
            ReadStatusLocked(reader, element_prefix, &result.status));
        TF_RETURN_IF_ERROR(ReadComponentsLocked(
            ctx, reader, absl::StrCat(element_prefix, "::", kReturnValues),
            &result.return_values));
        TF_RETURN_IF_ERROR(ReadComponentsLocked(
            ctx, reader, absl::StrCat(element_prefix, "::", kPredicateValues),
            &result.predicate_values));
        result.end_of_input = reader->Contains(element_prefix, kEndOfInput);
        RecordBufferEnqueue(ctx, result.return_values);
        result.notification.Notify();
      }
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_13(mht_13_v, 476, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "GetTraceMeMetadata");

      int64_t parallelism = -1;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        parallelism = num_parallel_calls_->value;
        mu_->unlock();
      }
      data::TraceMeMetadata result;
      result.push_back(
          std::make_pair("autotune", autotune_ ? "true" : "false"));
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

   private:
    struct InvocationResult {
      InvocationResult() : uid(tensorflow::EnvTime::NowNanos()) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_14(mht_14_v, 505, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "InvocationResult");
}

      Notification notification;
      Status status;
      std::vector<Tensor> return_values;
      std::vector<Tensor> predicate_values;
      bool end_of_input = false;
      const int64_t uid;
    };

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

    void EnsureThreadsStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_15(mht_15_v, 530, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "EnsureThreadsStarted");

      if (!runner_thread_) {
        auto ctx_copy = std::make_shared<IteratorContext>(*ctx);
        runner_thread_ = ctx->StartThread(
            "tf_data_parallel_filter",
            std::bind(&Iterator::RunnerThread, this, ctx_copy));
      }
    }

    void CallCompleted(const std::shared_ptr<IteratorContext>& ctx,
                       const std::shared_ptr<InvocationResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      mutex_lock l(*mu_);
      num_calls_--;
      result->notification.Notify();
      cond_var_->notify_all();
    }

    void CallFunction(const std::shared_ptr<IteratorContext>& ctx,
                      const std::shared_ptr<InvocationResult>& result)
        TF_LOCKS_EXCLUDED(*mu_) {
      profiler::TraceMe traceme([&] {
        return profiler::TraceMeEncode("ParallelFilterProduce",
                                       {{"element_id", result->uid}});
      });
      // Get the next input element.
      std::vector<Tensor> input_element;
      result->status = input_impl_->GetNext(ctx.get(), &input_element,
                                            &result->end_of_input);
      if (result->end_of_input || !result->status.ok()) {
        CallCompleted(ctx, result);
        return;
      }
      result->return_values = input_element;
      auto done = [this, ctx, result](Status status) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_16(mht_16_v, 567, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "lambda");

        result->status.Update(status);
        // Callback is not a predicate function, set the error status of this
        // result.
        if (status.ok() && (result->predicate_values.size() != 1 ||
                            result->predicate_values[0].dtype() != DT_BOOL ||
                            result->predicate_values[0].NumElements() != 1)) {
          result->status.Update(errors::InvalidArgument(
              "Filter predicate `predicate` must return a scalar bool."));
        }
        RecordBufferEnqueue(ctx.get(), result->return_values);
        CallCompleted(ctx, result);
      };

      // Apply the map function on `input_element`, storing the result in
      // `result->return_values`, and invoking `done` when finished.
      if (dataset()->captured_func_->use_inter_op_parallelism()) {
        instantiated_captured_func_->RunAsync(
            ctx.get(), std::move(input_element), &result->predicate_values,
            std::move(done), model_node());
      } else {
        // In this case, the function will be executed using single-threaded
        // executor. We schedule it using `ctx->runner()` to enable concurrent
        // application of the function over different input elements.
        auto fn = std::bind(
            [this, ctx, result](std::vector<Tensor> input_element) {
              return instantiated_captured_func_->Run(
                  ctx.get(), std::move(input_element),
                  &result->predicate_values, model_node());
            },
            std::move(input_element));
        (*ctx->runner())(
            [this, ctx, fn = std::move(fn), done = std::move(done)]() {
              Status s;
              // Check whether we are already recording to prevent invalid
              // nesting of `RecordStart` calls.
              if (IsRecording(ctx.get())) {
                s = fn();
              } else {
                RecordStart(ctx.get());
                s = fn();
                RecordStop(ctx.get());
              }
              done(s);
            });
      }
    }

    Status ProcessResult(IteratorContext* ctx,
                         const std::shared_ptr<InvocationResult>& result,
                         std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) TF_LOCKS_EXCLUDED(*mu_) {
      if (!result->end_of_input && result->status.ok()) {
        *out_tensors = std::move(result->return_values);
        RecordBufferDequeue(ctx, *out_tensors);
        *end_of_sequence = false;
        return Status::OK();
      }
      if (errors::IsOutOfRange(result->status)) {
        // `predicate` may deliberately raise `errors::OutOfRange` to indicate
        // that we should terminate the iteration early.
        return errors::InvalidArgument(
            "Function invocation produced OutOfRangeError: ",
            result->status.error_message());
      }
      *end_of_sequence = result->end_of_input;
      return result->status;
    }

    void RunnerThread(const std::shared_ptr<IteratorContext>& ctx)
        TF_LOCKS_EXCLUDED(*mu_) {
      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      std::vector<std::shared_ptr<InvocationResult>> new_calls;
      {
        tf_shared_lock l(*mu_);  // mu_ == num_parallel_calls_->mu
        new_calls.reserve(num_parallel_calls_->value);
      }
      auto busy = [this]() TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) -> bool {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_17(mht_17_v, 648, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "lambda");

        int64_t num_parallel_calls = num_parallel_calls_->value;
        return num_calls_ >= num_parallel_calls ||
               invocation_results_.size() >= num_parallel_calls;
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
            invocation_results_.push_back(std::make_shared<InvocationResult>());
            new_calls.push_back(invocation_results_.back());
            num_calls_++;
          }
          cond_var_->notify_all();
        }
        for (const auto& call : new_calls) {
          CallFunction(ctx, call);
        }
        new_calls.clear();
      }
    }

    // Determines whether the caller needs to wait for a result. Upon returning
    // false, `result` will point to the result and the result is fully
    // resolved, i.e. the predicate computation is finished.
    bool ShouldWait(std::shared_ptr<InvocationResult>* result)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_18(mht_18_v, 685, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "ShouldWait");

      if (cancelled_) {
        return false;
      }
      auto PredicateReady = [](const InvocationResult* result) -> bool {
        return result->status.ok() && !result->end_of_input;
      };
      auto GetPredicateValue = [](const InvocationResult* result) -> bool {
        return result->predicate_values[0].scalar<bool>()();
      };
      // Remove results from the front of the queue that are filtered. A result
      // is filtered if all of the following conditions are true:
      // 1. processing has finished, i.e. notification is received.
      // 2. status is OK to indicate that predicate evaluation succeeded.
      // 3. it is not the end of input.
      // 4. the predicate evaluates to false.
      while (!invocation_results_.empty() &&
             invocation_results_.front()->notification.HasBeenNotified() &&
             PredicateReady(invocation_results_.front().get()) &&
             !GetPredicateValue(invocation_results_.front().get())) {
        invocation_results_.pop_front();
        // A buffer is freed, notify all so that a new call can start.
        cond_var_->notify_all();
      }
      if (!deterministic_) {
        // Iterate through in-flight results and return the first one that is
        // found to be available and not end-of-input. If the first result (in
        // order) is end-of-input, we know that all earlier iterations have
        // already been completed, so it is safe to return that result for the
        // caller to process end of iteration. Available means its processing is
        // done (notified) and it is not filtered.
        for (auto it = invocation_results_.begin();
             it != invocation_results_.end(); ++it) {
          if ((*it)->notification.HasBeenNotified() &&
              (it == invocation_results_.begin() ||
               (PredicateReady(it->get()) && GetPredicateValue(it->get())))) {
            std::swap(*result, *it);
            invocation_results_.erase(it);
            cond_var_->notify_all();
            return false;
          }
        }
      } else {
        if (!invocation_results_.empty() &&
            invocation_results_.front()->notification.HasBeenNotified()) {
          std::swap(*result, invocation_results_.front());
          invocation_results_.pop_front();
          cond_var_->notify_all();
          return false;
        }
      }
      return true;
    }

    Status WriteComponentsLocked(IteratorStateWriter* writer,
                                 const std::string& prefix,
                                 const std::vector<Tensor>& values)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_19(mht_19_v, 746, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "WriteComponentsLocked");

      TF_RETURN_IF_ERROR(writer->WriteScalar(prefix, kSize, values.size()));
      for (size_t j = 0; j < values.size(); j++) {
        TF_RETURN_IF_ERROR(writer->WriteTensor(
            prefix, absl::StrCat(kComponent, "[", j, "]"), values[j]));
      }
      return Status::OK();
    }

    Status ReadComponentsLocked(IteratorContext* ctx,
                                IteratorStateReader* reader,
                                const std::string& prefix,
                                std::vector<Tensor>* values)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_20(mht_20_v, 763, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "ReadComponentsLocked");

      int64_t size;
      TF_RETURN_IF_ERROR(reader->ReadScalar(prefix, kSize, &size));
      size_t num_return_values = static_cast<size_t>(size);
      if (num_return_values != size) {
        return errors::InvalidArgument(prefix, ",", kSize, ": ", size,
                                       " is not a valid value of type size_t.");
      }
      values->reserve(num_return_values);
      for (size_t j = 0; j < num_return_values; j++) {
        values->emplace_back();
        TF_RETURN_IF_ERROR(reader->ReadTensor(
            ctx->flr(), prefix, absl::StrCat(kComponent, "[", j, "]"),
            &values->back()));
      }
      return Status::OK();
    }

    Status WriteStatusLocked(IteratorStateWriter* writer,
                             const std::string& key, const Status& status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_21(mht_21_v, 787, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "WriteStatusLocked");

      TF_RETURN_IF_ERROR(writer->WriteScalar(
          key, kErrorCode, static_cast<int64_t>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(key, kErrorMessage, status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatusLocked(IteratorStateReader* reader, const std::string& key,
                            Status* status) TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_22(mht_22_v, 802, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "ReadStatusLocked");

      int64_t code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(key, kErrorCode, &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(key, kErrorMessage, &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
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
    const bool autotune_;
    // Counts the number of outstanding calls.
    int64_t num_calls_ TF_GUARDED_BY(*mu_) = 0;
    // Controls cancellation of `input_impl_`. Must be ordered before
    // `input_impl_` so that `input_impl_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    // Must be ordered after `cancellation_manager_` so that `input_impl_` is
    // destroyed first.
    std::unique_ptr<IteratorBase> input_impl_;
    // Buffer for storing the invocation results.
    std::deque<std::shared_ptr<InvocationResult>> invocation_results_
        TF_GUARDED_BY(*mu_);
    std::unique_ptr<Thread> runner_thread_ TF_GUARDED_BY(*mu_);
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;

    // Records the number of ParallelInterleave operations in the path from the
    // root node to this node (not including this node) in the input pipeline
    // tree. We record the interleave depth so that it can be included in the
    // trace metadata.
    int64 interleave_depth_ = -1;
  };

  const DatasetBase* const input_;
  const int64_t num_parallel_calls_;
  const DeterminismPolicy deterministic_;
  const std::unique_ptr<CapturedFunction> captured_func_;
};

ParallelFilterDatasetOp::ParallelFilterDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_23(mht_23_v, 865, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "ParallelFilterDatasetOp::ParallelFilterDatasetOp");

  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kPredicate, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES(ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
              errors::InvalidArgument(
                  "predicate function has more than one return value."));
  std::string deterministic;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kDeterministic, &deterministic));
  OP_REQUIRES_OK(ctx,
                 DeterminismPolicy::FromString(deterministic, &deterministic_));
}

void ParallelFilterDatasetOp::MakeDataset(OpKernelContext* ctx,
                                          DatasetBase* input,
                                          DatasetBase** output) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSparallel_filter_dataset_opDTcc mht_24(mht_24_v, 882, "", "./tensorflow/core/kernels/data/parallel_filter_dataset_op.cc", "ParallelFilterDatasetOp::MakeDataset");

  int64_t num_parallel_calls;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument(ctx, kNumParallelCalls, &num_parallel_calls));
  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));
  if (num_parallel_calls == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }
  *output = new Dataset(ctx, input, num_parallel_calls, deterministic_,
                        std::move(captured_func));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("ParallelFilterDataset").Device(DEVICE_CPU),
                        ParallelFilterDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("ParallelFilterDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
