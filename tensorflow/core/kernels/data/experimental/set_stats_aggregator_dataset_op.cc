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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc() {
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
#include <memory>

#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class StatsAggregatorWithTagAndPrefix : public StatsAggregator {
 public:
  StatsAggregatorWithTagAndPrefix(
      std::shared_ptr<StatsAggregator> stats_aggregator, const string& tag,
      const string& prefix)
      : wrapped_(stats_aggregator), tag_(tag), prefix_(prefix) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tag: \"" + tag + "\"");
   mht_0_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "StatsAggregatorWithTagAndPrefix");
}

  void AddToHistogram(const string& name, gtl::ArraySlice<double> values,
                      int64_t steps) override {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "AddToHistogram");

    wrapped_->AddToHistogram(TaggedName(name), values, steps);
  }

  void AddScalar(const string& name, float value, int64_t steps) override {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_2(mht_2_v, 222, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "AddScalar");

    wrapped_->AddScalar(TaggedName(name), value, steps);
  }

  void EncodeToProto(Summary* out_summary) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "EncodeToProto");

    wrapped_->EncodeToProto(out_summary);
  }

  void IncrementCounter(const string& name, const string& label,
                        int64_t val) override {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + name + "\"");
   mht_4_v.push_back("label: \"" + label + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "IncrementCounter");

    if (!prefix_.empty()) {
      wrapped_->IncrementCounter(
          strings::StrCat(prefix_, "/", TaggedName(name)), label, val);
    } else {
      wrapped_->IncrementCounter(
          strings::StrCat("/tensorflow/", TaggedName(name)), label, val);
    }
  }

  Status SetSummaryWriter(SummaryWriterInterface* summary_writer) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_5(mht_5_v, 252, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "SetSummaryWriter");

    return wrapped_->SetSummaryWriter(summary_writer);
  }

 private:
  string TaggedName(const string& name) const {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_6(mht_6_v, 261, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "TaggedName");

    if (!tag_.empty()) {
      string tagged_name = strings::StrCat(tag_, stats_utils::kDelimiter, name);
      return tagged_name;
    }
    return name;
  }

  std::shared_ptr<StatsAggregator> wrapped_;
  string tag_;
  string prefix_;
  TF_DISALLOW_COPY_AND_ASSIGN(StatsAggregatorWithTagAndPrefix);
};

class SetStatsAggregatorDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit SetStatsAggregatorDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_7(mht_7_v, 281, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "SetStatsAggregatorDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_8(mht_8_v, 287, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "MakeDataset");

    core::RefCountPtr<StatsAggregatorResource> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &resource));
    tstring tag;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "tag", &tag));
    tstring prefix;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "counter_prefix", &prefix));

    *output =
        new Dataset(ctx, input, ctx->input(1), resource.get(), tag, prefix);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input,
                     const Tensor& resource_handle,
                     StatsAggregatorResource* resource, const string& tag,
                     const string& prefix)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          resource_handle_(resource_handle),
          stats_aggregator_resource_(resource),
          tag_(tag),
          prefix_(prefix) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("tag: \"" + tag + "\"");
   mht_9_v.push_back("prefix: \"" + prefix + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_9(mht_9_v, 317, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "Dataset");

      input_->Ref();
      stats_aggregator_resource_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_10(mht_10_v, 325, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "~Dataset");

      input_->Unref();
      stats_aggregator_resource_->Unref();
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::SetStatsAggregator")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_11(mht_11_v, 339, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "output_dtypes");

      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_12(mht_12_v, 345, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "output_shapes");

      return input_->output_shapes();
    }

    string DebugString() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_13(mht_13_v, 352, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "DebugString");

      return "SetStatsAggregatorDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_14(mht_14_v, 359, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "CardinalityInternal");

      return input_->Cardinality();
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_15(mht_15_v, 367, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_16(mht_16_v, 375, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_17(mht_17_v, 385, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* resource_handle_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddTensor(resource_handle_, &resource_handle_node));
      Node* tag_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(tag_, &tag_node));
      Node* prefix_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(prefix_, &prefix_node));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, resource_handle_node, tag_node, prefix_node},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_18(mht_18_v, 407, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_19(mht_19_v, 412, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "Initialize");

        IteratorContext iter_ctx = ContextWithAggregator(ctx);
        return dataset()->input_->MakeIterator(&iter_ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_20(mht_20_v, 423, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        IteratorContext iter_ctx = ContextWithAggregator(ctx);
        return input_impl_->GetNext(&iter_ctx, out_tensors, end_of_sequence);
      }

      IteratorContext ContextWithAggregator(IteratorContext* ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_21(mht_21_v, 432, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "ContextWithAggregator");

        StatsAggregatorResource* resource =
            dataset()->stats_aggregator_resource_;
        IteratorContext::Params params(ctx);
        params.stats_aggregator = std::shared_ptr<StatsAggregator>(
            new StatsAggregatorWithTagAndPrefix(resource->stats_aggregator(),
                                                dataset()->tag_,
                                                dataset()->prefix_));
        IteratorContext iter_ctx(std::move(params));
        return iter_ctx;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_22(mht_22_v, 455, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        return SaveInput(ctx, writer, input_impl_);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSset_stats_aggregator_dataset_opDTcc mht_23(mht_23_v, 464, "", "./tensorflow/core/kernels/data/experimental/set_stats_aggregator_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        return RestoreInput(ctx, reader, input_impl_);
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const Tensor resource_handle_;
    StatsAggregatorResource* stats_aggregator_resource_;
    tstring tag_;
    tstring prefix_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SetStatsAggregatorDataset").Device(DEVICE_CPU),
                        SetStatsAggregatorDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalSetStatsAggregatorDataset").Device(DEVICE_CPU),
    SetStatsAggregatorDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
