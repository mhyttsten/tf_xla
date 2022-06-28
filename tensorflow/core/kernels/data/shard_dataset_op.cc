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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/shard_dataset_op.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ShardDatasetOp::kDatasetType;
/* static */ constexpr const char* const ShardDatasetOp::kInputDataset;
/* static */ constexpr const char* const ShardDatasetOp::kNumShards;
/* static */ constexpr const char* const ShardDatasetOp::kIndex;
/* static */ constexpr const char* const ShardDatasetOp::kRequireNonEmpty;
/* static */ constexpr const char* const ShardDatasetOp::kOutputTypes;
/* static */ constexpr const char* const ShardDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kNextIndex[] = "next_index";

class ShardDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t num_shards, int64_t index,
          bool require_non_empty, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)),
        num_shards_(num_shards),
        index_(index),
        input_(input),
        require_non_empty_(require_non_empty),
        traceme_metadata_(
            {{"index", strings::Printf("%lld", static_cast<long long>(index))},
             {"num_shards",
              strings::Printf("%lld", static_cast<long long>(num_shards))}}) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_0(mht_0_v, 228, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.set_args(num_shards_, index_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "CardinalityInternal");

    int64_t n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / num_shards_ + (index_ < n % num_shards_ ? 1 : 0);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_5(mht_5_v, 273, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "CardinalityInternal");

    int64_t n = input_->Cardinality(options);
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return n / num_shards_ + (index_ < n % num_shards_ ? 1 : 0);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_7(mht_7_v, 292, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_8(mht_8_v, 300, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    return input_->Get(ctx, index_ + (num_shards_ * index), out_tensors);
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_9(mht_9_v, 311, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* num_shards = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(num_shards_, &num_shards));
    Node* index = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(index_, &index));

    AttrValue require_non_empty_attr;
    b->BuildAttrValue(require_non_empty_, &require_non_empty_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, num_shards, index},
                      {{kRequireNonEmpty, require_non_empty_attr}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params), next_index_(0) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_10(mht_10_v, 335, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_11(mht_11_v, 340, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "Initialize");

      if (dataset()->num_shards_ == kShardHint) {
        return errors::FailedPrecondition(
            "`tf.data.Dataset.shard(SHARD_HINT, ...)` can only be used in "
            "`tf.distribute.Strategy.experimental_distribute_dataset()` with "
            "`tf.data.experimental.AutoShardPolicy.HINT` policy, or tf.data "
            "service with "
            "`tf.data.experimental.service.ShardingPolicy.HINT` processing "
            "mode.");
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_12(mht_12_v, 358, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);

      *end_of_sequence = false;
      if (!input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      int num_to_skip =
          (dataset()->index_ - next_index_) % dataset()->num_shards_;
      if (num_to_skip < 0) {
        num_to_skip += dataset()->num_shards_;
      }
      int num_skipped;
      TF_RETURN_IF_ERROR(
          input_impl_->Skip(ctx, num_to_skip, end_of_sequence, &num_skipped));
      next_index_ += num_skipped;
      if (*end_of_sequence) {
        input_impl_.reset();
        return Status::OK();
      }

      std::vector<Tensor> result;
      TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &result, end_of_sequence));
      if (*end_of_sequence) {
        input_impl_.reset();
        return Status::OK();
      }
      next_index_++;

      if (dataset()->require_non_empty_ &&
          next_index_ < dataset()->num_shards_) {
        int num_skipped;
        Status s = input_impl_->Skip(ctx, dataset()->num_shards_ - next_index_,
                                     end_of_sequence, &num_skipped);
        if (*end_of_sequence || errors::IsOutOfRange(s)) {
          // `dataset()->require_non_empty_` implies that this transformation
          // was introduced by auto_sharding rewrite, so it's acceptable
          // produce an error message that assumes auto-sharding context.
          return errors::InvalidArgument(
              "Could not apply FILE based sharding: the dataset only has ",
              next_index_, " file(s), which is not enough for the required ",
              dataset()->num_shards_,
              " shards/workers."
              "If you are using datasets with distribution strategy, "
              "consider setting the auto sharding policy to either DATA or "
              "OFF using the `experimental_distribute.auto_shard_policy` option"
              "of `tf.data.Options()`. Or, split your input files into a "
              "larger number of small files such that number of files is "
              "greater than number of shards/workers.");
        } else if (!s.ok()) {
          return s;
        }

        next_index_ = dataset()->num_shards_;
      }

      *out_tensors = std::move(result);
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), dataset()->num_shards_);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_13(mht_13_v, 430, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kNextIndex), next_index_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_14(mht_14_v, 446, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name(kNextIndex), &next_index_));
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_15(mht_15_v, 461, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "GetTraceMeMetadata");

      return dataset()->traceme_metadata_;
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    int64_t next_index_ TF_GUARDED_BY(mu_);
  };

  const int64_t num_shards_;
  const int64_t index_;
  const DatasetBase* const input_;
  const bool require_non_empty_;
  const TraceMeMetadata traceme_metadata_;
};

ShardDatasetOp::ShardDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_16(mht_16_v, 482, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "ShardDatasetOp::ShardDatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kRequireNonEmpty, &require_non_empty_));
}

void ShardDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshard_dataset_opDTcc mht_17(mht_17_v, 490, "", "./tensorflow/core/kernels/data/shard_dataset_op.cc", "ShardDatasetOp::MakeDataset");

  int64_t index = 0;
  int64_t num_shards = 0;

  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kNumShards, &num_shards));
  OP_REQUIRES(
      ctx, num_shards > 0 || num_shards == kShardHint,
      errors::InvalidArgument("Number of shards must be greater than zero "
                              "(currently num_shards = ",
                              num_shards, ")."));

  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kIndex, &index));
  OP_REQUIRES(
      ctx, (index >= 0 && index < num_shards) || num_shards == kShardHint,
      errors::InvalidArgument("Index must be between 0 and ", num_shards - 1,
                              " (currently index = ", index, ")."));

  *output = new Dataset(ctx, num_shards, index, require_non_empty_, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ShardDataset").Device(DEVICE_CPU),
                        ShardDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
