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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc() {
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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

// This op defines a `Dataset` that passes through its input elements and
// records the latency of producing each element in the context's
// `StatsAggregator`.
//
// TODO(mrry): It is likely that many *StatsDatasetOp kernels will have the
// same or similar structure. We should abstract the common boilerplate into
// a base case and/or investigate how to make general-purpose *StatsDatasetOp
// kernels that use TensorFlow functions to represent their logic. For example,
// if the performance were adequate, we might replace this kernel with an
// implementation that executes functions before and after the `GetNext()` call
// on the input, each executing an op that gets the current time and performing
// the subtraction.
class LatencyStatsDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit LatencyStatsDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "LatencyStatsDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "MakeDataset");

    tstring tag;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "tag", &tag));
    *output = new Dataset(ctx, input, std::move(tag));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input, string tag)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          tag_(std::move(tag)) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::LatencyStats")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_4(mht_4_v, 250, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "output_dtypes");

      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_5(mht_5_v, 256, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "output_shapes");

      return input_->output_shapes();
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_6(mht_6_v, 263, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "DebugString");

      return "LatencyStatsDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_7(mht_7_v, 270, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "CardinalityInternal");

      return input_->Cardinality();
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_8(mht_8_v, 278, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_9(mht_9_v, 286, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_10(mht_10_v, 296, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "AsGraphDefInternal");

      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
      Node* tag_node;
      TF_RETURN_IF_ERROR(b->AddScalar(tag_, &tag_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_node, tag_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_11(mht_11_v, 312, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_12(mht_12_v, 317, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_13(mht_13_v, 327, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "GetNextInternal");

        tf_shared_lock l(mu_);
        uint64 start = EnvTime::NowMicros();
        Status s = input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
        uint64 end = EnvTime::NowMicros();
        auto stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator && !*end_of_sequence) {
          int64_t steps = num_elements();
          stats_aggregator->AddToHistogram(
              dataset()->tag_, {static_cast<double>(end - start)}, steps);
        }
        return s;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_14(mht_14_v, 352, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_15(mht_15_v, 362, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const tstring tag_;
  };
};

class BytesProducedStatsDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit BytesProducedStatsDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_16(mht_16_v, 384, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "BytesProducedStatsDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_17(mht_17_v, 390, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "MakeDataset");

    tstring tag;
    OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "tag", &tag));
    *output = new Dataset(ctx, input, std::move(tag));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, const DatasetBase* input, string tag)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          tag_(std::move(tag)) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("tag: \"" + tag + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_18(mht_18_v, 406, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_19(mht_19_v, 413, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(Iterator::Params{
          this, strings::StrCat(prefix, "::BytesProducedStats")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_20(mht_20_v, 424, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "output_dtypes");

      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_21(mht_21_v, 430, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "output_shapes");

      return input_->output_shapes();
    }

    string DebugString() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_22(mht_22_v, 437, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "DebugString");

      return "BytesProducedStatsDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_23(mht_23_v, 444, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "CardinalityInternal");

      return input_->Cardinality();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_24(mht_24_v, 451, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_25(mht_25_v, 461, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "AsGraphDefInternal");

      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
      Node* tag_node;
      TF_RETURN_IF_ERROR(b->AddScalar(tag_, &tag_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_node, tag_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_26(mht_26_v, 477, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_27(mht_27_v, 482, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_28(mht_28_v, 492, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "GetNextInternal");

        tf_shared_lock l(mu_);
        Status s = input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
        auto stats_aggregator = ctx->stats_aggregator();
        if (stats_aggregator && s.ok() && !*end_of_sequence) {
          size_t total_bytes = 0;
          for (const Tensor& t : *out_tensors) {
            total_bytes += t.TotalBytes();
          }
          int64_t steps = num_elements();
          stats_aggregator->AddToHistogram(
              dataset()->tag_, {static_cast<double>(total_bytes)}, steps);
        }
        return s;
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(std::move(args),
                                         /*ratio=*/1);
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_29(mht_29_v, 519, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSstats_dataset_opsDTcc mht_30(mht_30_v, 529, "", "./tensorflow/core/kernels/data/experimental/stats_dataset_ops.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    const tstring tag_;
  };
};

REGISTER_KERNEL_BUILDER(Name("BytesProducedStatsDataset").Device(DEVICE_CPU),
                        BytesProducedStatsDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalBytesProducedStatsDataset").Device(DEVICE_CPU),
    BytesProducedStatsDatasetOp);

REGISTER_KERNEL_BUILDER(Name("LatencyStatsDataset").Device(DEVICE_CPU),
                        LatencyStatsDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalLatencyStatsDataset").Device(DEVICE_CPU),
    LatencyStatsDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
