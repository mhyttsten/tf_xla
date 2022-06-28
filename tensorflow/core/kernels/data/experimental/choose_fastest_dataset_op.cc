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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/histogram/histogram.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

static const double kPercentile = 90.0;

class ChooseFastestDatasetOp : public DatasetOpKernel {
 public:
  explicit ChooseFastestDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "ChooseFastestDatasetOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_experiments", &num_experiments_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_1(mht_1_v, 211, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "MakeDataset");

    OpInputList input_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("input_datasets", &input_list));
    OP_REQUIRES(
        ctx, input_list.size() > 1,
        errors::InvalidArgument(
            "ChooseFastestDataset must have at least two input datasets."));

    std::vector<DatasetBase*> inputs;
    inputs.reserve(input_list.size());
    for (const auto& tensor : input_list) {
      DatasetBase* input;
      OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(tensor, &input));
      inputs.push_back(input);
    }

    for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
      OP_REQUIRES(
          ctx, inputs[i]->output_dtypes() == output_types_,
          errors::InvalidArgument(
              "All inputs to ChooseFastestDataset "
              "must have the same output types. Input ",
              i, " has output types: ",
              DataTypeVectorString(inputs[i]->output_dtypes()),
              ". Expected: ", DataTypeVectorString(output_types_), "."));
    }

    // Merge the output shapes of all the input datasets, returning an
    // error if any of them are incompatible.
    for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
      OP_REQUIRES(
          ctx, inputs[i]->output_shapes().size() == output_shapes_.size(),
          errors::InvalidArgument(
              "All inputs to ChooseFastestDataset must have compatible outputs."
              " Input ",
              i, " has ", inputs[i]->output_shapes().size(),
              " components. Expected to have ", output_shapes_.size(),
              " components."));
      for (size_t j = 0, num_components = output_shapes_.size();
           j < num_components; ++j) {
        PartialTensorShape result;
        OP_REQUIRES(ctx,
                    output_shapes_[j]
                        .MergeWith(inputs[i]->output_shapes().at(j), &result)
                        .ok(),
                    errors::InvalidArgument(
                        "All inputs to ChooseFastestDataset must have "
                        "compatible output shapes. Component ",
                        j, " of input ", i,
                        " has shape: ", inputs[i]->output_shapes().at(j),
                        ". Expected to be compatible with shape: ",
                        output_shapes_[j], "."));
        output_shapes_[j] = std::move(result);
      }
    }

    int64_t cardinality = inputs[0]->Cardinality();
    for (size_t i = 1, num_inputs = inputs.size(); i < num_inputs; ++i) {
      if (cardinality == kUnknownCardinality) {
        cardinality = inputs[i]->Cardinality();
      } else {
        OP_REQUIRES(
            ctx,
            inputs[i]->Cardinality() == cardinality ||
                inputs[i]->Cardinality() == kUnknownCardinality,
            errors::InvalidArgument(
                "All inputs to ChooseFastestDataset must have compatible "
                "cardinalities. Input ",
                i, " has cardinality: ", inputs[i]->Cardinality(),
                ", while all prior inputs have cardinality: ", cardinality,
                "."));
      }
    }
    *output = new Dataset(ctx, std::move(inputs), output_types_, output_shapes_,
                          cardinality, num_experiments_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, std::vector<DatasetBase*> inputs,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            int64_t cardinality, int64_t num_experiments)
        : DatasetBase(DatasetContext(ctx)),
          inputs_(std::move(inputs)),
          output_types_(output_types),
          output_shapes_(output_shapes),
          cardinality_(cardinality),
          num_experiments_(num_experiments) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_2(mht_2_v, 303, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "Dataset");

      for (auto input : inputs_) {
        input->Ref();
      }
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_3(mht_3_v, 312, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "~Dataset");

      for (auto input : inputs_) {
        input->Unref();
      }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<ChooseFastestIterator>(
          ChooseFastestIterator::Params{
              this, strings::StrCat(prefix, "::ChooseFastest")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_4(mht_4_v, 328, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "output_dtypes");

      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_5(mht_5_v, 335, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_6(mht_6_v, 342, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "DebugString");

      return "ChooseFastestDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_7(mht_7_v, 349, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "CardinalityInternal");
 return cardinality_; }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_8(mht_8_v, 355, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "InputDatasets");

      for (const auto& input : inputs_) {
        inputs->push_back(input);
      }
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_9(mht_9_v, 365, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "CheckExternalState");

      for (const auto& input : inputs_) {
        TF_RETURN_IF_ERROR(input->CheckExternalState());
      }
      return Status::OK();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_10(mht_10_v, 378, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "AsGraphDefInternal");

      std::vector<Node*> input_nodes;
      input_nodes.reserve(inputs_.size());
      for (const auto& input : inputs_) {
        Node* input_node;
        TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &input_node));
        input_nodes.push_back(input_node);
      }
      AttrValue num_experiments_attr;
      b->BuildAttrValue(num_experiments_, &num_experiments_attr);
      return b->AddDataset(
          this, {}, {std::make_pair(0, input_nodes)},
          {std::make_pair("num_experiments", std::move(num_experiments_attr))},
          output);
    }

   private:
    class ChooseFastestIterator : public DatasetIterator<Dataset> {
     public:
      explicit ChooseFastestIterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            histograms_(dataset()->inputs_.size()) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_11(mht_11_v, 402, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "ChooseFastestIterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_12(mht_12_v, 407, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "Initialize");

        mutex_lock l(mu_);
        input_impls_.resize(dataset()->inputs_.size());
        for (size_t i = 0, num_inputs = dataset()->inputs_.size();
             i < num_inputs; ++i) {
          TF_RETURN_IF_ERROR(dataset()->inputs_[i]->MakeIterator(
              ctx, this, strings::StrCat(prefix(), "[", i, "]"),
              &input_impls_[i]));
        }
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_13(mht_13_v, 424, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);

        // The first num_experiments_ iterations, we fire up a thread for
        // each input that calls its GetNext function and records the time
        // taken. We only return when all the threads have completed.
        if (experiment_counter_ < dataset()->num_experiments_) {
          experiment_counter_++;
          std::vector<ThreadInfo> threads = StartThreads(ctx);
          for (const auto& thread : threads) {
            thread.result->notification.WaitForNotification();
          }

          *out_tensors = std::move(threads[0].result->out_tensors);
          *end_of_sequence = threads[0].result->end_of_sequence;

          if (experiment_counter_ == dataset()->num_experiments_) {
            SelectFastestInputIndex();
          }
          return threads[0].result->status;
        }
        return fastest_input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
      }

      // TODO(rachelim): Save and restore histogram state as well. Currently,
      // if an iterator is saved and restored, the histograms start recording
      // from scratch.
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_14(mht_14_v, 461, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("experiment_counter"),
                                               experiment_counter_));

        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("fastest_index"), fastest_index_));
        if (fastest_index_ != -1) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, fastest_input_impl_));
        } else if (input_impls_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impls_empty"), ""));
        } else {
          for (auto& input_impl : input_impls_) {
            TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl));
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_15(mht_15_v, 485, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("experiment_counter"),
                                              &experiment_counter_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("fastest_index"), &fastest_index_));
        if (fastest_index_ != -1) {
          TF_RETURN_IF_ERROR(dataset()->inputs_[fastest_index_]->MakeIterator(
              ctx, this, strings::StrCat(prefix(), "[", fastest_index_, "]"),
              &fastest_input_impl_));
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, fastest_input_impl_));
        } else if (reader->Contains(full_name("input_impls_empty"))) {
          input_impls_.clear();
        } else {
          DCHECK_EQ(input_impls_.size(), dataset()->inputs_.size());
          for (auto& input_impl : input_impls_) {
            TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl));
          }
        }
        return Status::OK();
      }

     private:
      struct InvocationResult {
        Notification notification;
        Status status;
        bool end_of_sequence;
        std::vector<Tensor> out_tensors;
      };

      struct ThreadInfo {
        std::unique_ptr<InvocationResult> result;
        std::unique_ptr<Thread> thread;
      };

      std::vector<std::unique_ptr<IteratorBase>> input_impls_;
      std::unique_ptr<IteratorBase> fastest_input_impl_;
      // For tracking the time taken for each input's iterations.
      std::vector<histogram::Histogram> histograms_;

      mutex mu_;
      int64_t experiment_counter_ TF_GUARDED_BY(mu_) = 0;
      int64_t fastest_index_ = -1;

      std::vector<ThreadInfo> StartThreads(IteratorContext* ctx)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        std::vector<ThreadInfo> threads(dataset()->inputs_.size());
        for (size_t i = 0, num_inputs = dataset()->inputs_.size();
             i < num_inputs; ++i) {
          threads[i].result = absl::make_unique<InvocationResult>();
          threads[i].thread = ctx->StartThread(
              strings::StrCat("tf_data_merge_", i),
              std::bind(&ChooseFastestIterator::RunnerThread, this, ctx,
                        threads[i].result.get(), i));
        }
        return threads;
      }

      void RunnerThread(IteratorContext* ctx, InvocationResult* result, int i) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_16(mht_16_v, 546, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "RunnerThread");

        RecordStart(ctx);
        auto cleanup = gtl::MakeCleanup([this, ctx]() { RecordStop(ctx); });
        int64_t start = EnvTime::NowNanos();
        Status s = input_impls_[i]->GetNext(ctx, &result->out_tensors,
                                            &result->end_of_sequence);
        histograms_[i].Add(static_cast<double>(EnvTime::NowNanos() - start));

        result->status = s;
        result->notification.Notify();
      }

      // Select the fastest input to use based on the histograms of timings
      // of the completed threads. The input with the best 90th percentile
      // iteration time is selected.
      void SelectFastestInputIndex() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_dataset_opDTcc mht_17(mht_17_v, 564, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_dataset_op.cc", "SelectFastestInputIndex");

        fastest_index_ = 0;

        VLOG(2) << "90.0 percentile iteration time:";
        double best_percentile = histograms_[0].Percentile(kPercentile);
        VLOG(2) << "Branch 0: " << best_percentile;
        for (size_t i = 1, num_inputs = histograms_.size(); i < num_inputs;
             ++i) {
          double percentile = histograms_[i].Percentile(kPercentile);
          VLOG(2) << "Branch " << i << ": " << percentile;
          if (percentile <= best_percentile) {
            best_percentile = percentile;
            fastest_index_ = i;
          }
        }
        VLOG(1) << "Selecting index " << fastest_index_
                << " as the fastest index.";

        fastest_input_impl_ = std::move(input_impls_[fastest_index_]);
        input_impls_.clear();  // Delete the unused iterators.
      }
    };  // class Iterator

    const std::vector<DatasetBase*> inputs_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const int64_t cardinality_;
    const int64_t num_experiments_;
  };  // class Dataset

  int64_t num_experiments_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};  // class ChooseFastestDatasetOp

REGISTER_KERNEL_BUILDER(Name("ChooseFastestDataset").Device(DEVICE_CPU),
                        ChooseFastestDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalChooseFastestDataset").Device(DEVICE_CPU),
    ChooseFastestDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
