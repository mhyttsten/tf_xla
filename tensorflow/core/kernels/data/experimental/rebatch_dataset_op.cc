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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc() {
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

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

inline int64_t CeilDiv(int64_t dividend, int64_t divisor) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "CeilDiv");

  return (dividend - 1 + divisor) / divisor;
}

constexpr const char* const kDatasetTypeV1 = "Rebatch";
constexpr const char* const kDatasetTypeV2 = "RebatchV2";

class RebatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit RebatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "RebatchDatasetOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "MakeDataset");

    int64_t num_replicas;
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument(ctx, "num_replicas", &num_replicas));
    OP_REQUIRES(
        ctx, num_replicas > 0,
        errors::InvalidArgument("num_replicas must be greater than zero."));
    *output =
        new Dataset(ctx, input, num_replicas, output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            const int64_t num_replicas, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          num_replicas_(num_replicas),
          output_types_(output_types),
          output_shapes_(output_shapes),
          traceme_metadata_(
              {{"num_replicas", strings::Printf("%lld", static_cast<long long>(
                                                            num_replicas))}}) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      name_utils::IteratorPrefixParams params;
      return absl::make_unique<Iterator>(Iterator::Params{
          this, name_utils::IteratorPrefix(kDatasetTypeV1, prefix, params)});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_5(mht_5_v, 264, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "output_dtypes");

      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_6(mht_6_v, 271, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_7(mht_7_v, 278, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "DebugString");

      name_utils::DatasetDebugStringParams params;
      params.set_args(num_replicas_);
      return name_utils::DatasetDebugString(kDatasetTypeV1, params);
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_8(mht_8_v, 288, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_9(mht_9_v, 296, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_10(mht_10_v, 306, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* num_replicas = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(num_replicas_, &num_replicas));
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {input_graph_node, num_replicas}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_11(mht_11_v, 323, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "Iterator");
}

      ~Iterator() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_12(mht_12_v, 328, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "~Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_13(mht_13_v, 333, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_14(mht_14_v, 343, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        *end_of_sequence = false;
        if (slice_number_ % dataset()->num_replicas_ == 0) {
          input_descriptors_.clear();
          std::vector<Tensor> input_tensors;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &input_tensors, end_of_sequence));
          if (*end_of_sequence) {
            return Status::OK();
          }

          input_descriptors_.reserve(input_tensors.size());
          for (int i = 0; i < input_tensors.size(); ++i) {
            if (input_tensors[i].dims() == 0) {
              return errors::InvalidArgument(
                  "Cannot rebatch dataset: All components must have at least "
                  "one dimension. Perhaps your input dataset is not batched? "
                  "Component ",
                  i, " is scalar.");
            }

            int64_t original_batch_dim = input_tensors[i].dim_size(0);
            int64_t interval =
                CeilDiv(original_batch_dim, dataset()->num_replicas_);
            input_descriptors_.push_back(
                {std::move(input_tensors[i]), original_batch_dim, interval});
          }
        }

        out_tensors->reserve(input_descriptors_.size());

        // We slice each component independently because they may have
        // different batch dimensions.
        for (const auto& input_desc : input_descriptors_) {
          int64_t start = input_desc.interval * slice_number_;
          int64_t end = std::min(start + input_desc.interval,
                                 input_desc.original_batch_dim);
          if (start >= end) {
            // We can get here if ceil(original_batch_dim_ / new batch dim) <
            // num_replicas_, i.e. the batch isn't big enough to distribute
            // over num replicas. In this case, we return empty tensors for
            // the remaining iterations that correspond to this batch.
            start = end;
          }
          Tensor slice = input_desc.whole_tensor.Slice(start, end);
          if (slice.IsAligned()) {
            out_tensors->push_back(std::move(slice));
          } else {
            out_tensors->push_back(tensor::DeepCopy(std::move(slice)));
          }
        }
        slice_number_ = (slice_number_ + 1) % dataset()->num_replicas_;
        return Status::OK();
      }

     protected:
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_15(mht_15_v, 404, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("slice_number"), slice_number_));

        if (slice_number_ % dataset()->num_replicas_ != 0) {
          // Save state of input tensors.
          for (int i = 0; i < input_descriptors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("tensors[", i, "]")),
                input_descriptors_[i].whole_tensor));
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_16(mht_16_v, 430, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("slice_number"), &slice_number_));

        input_descriptors_.clear();
        input_descriptors_.resize(dataset()->output_dtypes().size());
        if (slice_number_ % dataset()->num_replicas_ != 0) {
          for (int i = 0; i < input_descriptors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(strings::StrCat("tensors[", i, "]")),
                &input_descriptors_[i].whole_tensor));
            input_descriptors_[i].original_batch_dim =
                input_descriptors_[i].whole_tensor.dim_size(0);
            input_descriptors_[i].interval =
                CeilDiv(input_descriptors_[i].original_batch_dim,
                        dataset()->num_replicas_);
          }
        }
        return Status::OK();
      }

      TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_17(mht_17_v, 460, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "GetTraceMeMetadata");

        return dataset()->traceme_metadata_;
      }

     private:
      // Describes one component of the input.
      struct InputDescriptor {
        InputDescriptor() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_18(mht_18_v, 470, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "InputDescriptor");
}
        InputDescriptor(Tensor&& whole_tensor, int64_t original_batch_dim,
                        int64_t interval)
            : whole_tensor(std::move(whole_tensor)),
              original_batch_dim(original_batch_dim),
              interval(interval) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_19(mht_19_v, 478, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "InputDescriptor");
}

        Tensor whole_tensor;
        int64_t original_batch_dim;
        int64_t interval;
      };

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_;
      std::vector<InputDescriptor> input_descriptors_ TF_GUARDED_BY(mu_);
      int64_t slice_number_ TF_GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const int64_t num_replicas_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const TraceMeMetadata traceme_metadata_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

// This dataset rebatches its input batches into batches of different size(s).
//
// This differs from RebatchDatasetOp. Namely, RebatchDatasetV2 rebatches
// incoming batches into batches whose new sizes are specified by the
// `batch_sizes` argument, while RebatchDataset splits its batches based
// on the (dynamic) input batch size and the given number of splits to make (its
// `num_replicas` argument). When used in tf.distribute, this allows
// RebatchDataset to split batches more correctly when the splits are
// distributed across multiple workers and replicas.
class RebatchDatasetV2Op : public UnaryDatasetOpKernel {
 public:
  explicit RebatchDatasetV2Op(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_20(mht_20_v, 517, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "RebatchDatasetV2Op");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_21(mht_21_v, 527, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "MakeDataset");

    const Tensor* batch_sizes_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("batch_sizes", &batch_sizes_tensor));
    OP_REQUIRES(
        ctx, batch_sizes_tensor->dims() <= 1,
        errors::InvalidArgument("`batch_sizes` must be a scalar or a vector."));

    std::vector<int64_t> batch_sizes;
    batch_sizes.reserve(batch_sizes_tensor->NumElements());
    for (int i = 0; i < batch_sizes_tensor->NumElements(); ++i) {
      batch_sizes.push_back(batch_sizes_tensor->flat<int64_t>()(i));
    }

    bool drop_remainder;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<bool>(ctx, "drop_remainder", &drop_remainder));

    *output = new Dataset(ctx, input, std::move(batch_sizes), drop_remainder,
                          output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::vector<int64_t>&& batch_sizes, bool drop_remainder,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          batch_sizes_(std::move(batch_sizes)),
          drop_remainder_(drop_remainder),
          output_types_(output_types),
          output_shapes_(output_shapes),
          traceme_metadata_(
              {{"batch_sizes", absl::StrJoin(batch_sizes, ",")}}) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_22(mht_22_v, 565, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_23(mht_23_v, 572, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      name_utils::IteratorPrefixParams params;
      return absl::make_unique<Iterator>(Iterator::Params{
          this, name_utils::IteratorPrefix(kDatasetTypeV2, prefix, params)});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_24(mht_24_v, 584, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "output_dtypes");

      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_25(mht_25_v, 591, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_26(mht_26_v, 598, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "DebugString");

      return name_utils::DatasetDebugString(kDatasetTypeV2);
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_27(mht_27_v, 606, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_28(mht_28_v, 614, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_29(mht_29_v, 624, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      Node* batch_sizes = nullptr;
      TF_RETURN_IF_ERROR(b->AddVector(batch_sizes_, &batch_sizes));
      Node* drop_remainder = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder));
      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {input_graph_node, batch_sizes, drop_remainder}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_30(mht_30_v, 643, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "Iterator");
}

      ~Iterator() override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_31(mht_31_v, 648, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "~Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_32(mht_32_v, 653, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_33(mht_33_v, 663, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        if (end_of_sequence_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        *end_of_sequence = false;

        auto desired_batch_size = dataset()->batch_sizes_[batch_sizes_index_];
        // Tracks the size of the current batch as it's built up, possibly from
        // different input tensors.
        int64_t batch_size = 0;

        std::vector<std::vector<Tensor>> slices_to_concatenate;
        // Get slices from input tensors until they make up the whole batch
        // size or we run out of input.
        while (batch_size < desired_batch_size) {
          if (offset_ == -1) {
            // Get new input tensors.
            tensors_.clear();
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(ctx, &tensors_, &end_of_sequence_));
            if (end_of_sequence_) {
              // Break and return partial batch, if any.
              break;
            }
            TF_RETURN_IF_ERROR(ValidateInputTensors());
            offset_ = 0;
          }

          int64_t slice_end =
              std::min(offset_ + desired_batch_size - batch_size,
                       tensors_[0].dim_size(0));

          std::vector<Tensor> slices;
          slices.reserve(tensors_.size());
          for (const auto& tensor : tensors_) {
            slices.push_back(tensor.Slice(offset_, slice_end));
          }
          slices_to_concatenate.push_back(std::move(slices));

          batch_size += (slice_end - offset_);
          offset_ = slice_end;
          if (offset_ == tensors_[0].dim_size(0)) {
            // Exhausted current input tensors, reset.
            offset_ = -1;
          }
        }

        batch_sizes_index_++;
        batch_sizes_index_ %= dataset()->batch_sizes_.size();

        // Return end_of_sequence if GetNext is expected to produce a non-empty
        // batch and there are no more inputs, or if drop_remainder is true and
        // we can't make a full batch.
        if ((batch_size == 0 && desired_batch_size > 0) ||
            (dataset()->drop_remainder_ && batch_size < desired_batch_size)) {
          DCHECK(end_of_sequence_);
          *end_of_sequence = true;
          return Status::OK();
        }

        const size_t num_components = dataset()->output_dtypes().size();
        out_tensors->reserve(num_components);

        // Special case: desired batch size == 0. This may be the case when,
        // with distribution strategies, one of replicas expects an empty batch
        // so that the global batch size adds up correctly.
        if (desired_batch_size == 0) {
          DCHECK_EQ(batch_size, 0);
          DCHECK_EQ(slices_to_concatenate.size(), 0);
          for (int i = 0; i < dataset()->output_dtypes().size(); ++i) {
            if (dataset()->output_shapes()[i].unknown_rank()) {
              // For unknown rank tensors, we just create a empty Tensor since
              // it doesn't matter what shape it is.
              out_tensors->push_back(Tensor(dataset()->output_dtypes()[i]));
            } else {
              auto dim_sizes = dataset()->output_shapes()[i].dim_sizes();

              // The output batch size is always zero since the desired batch
              // size is zero.
              dim_sizes[0] = 0;

              // Handle unknown dimensions by setting any unknown dimensions to
              // zero since there isn't any data anyway.
              for (int j = 1; j < dim_sizes.size(); ++j) {
                if (dim_sizes[j] == -1) dim_sizes[j] = 0;
              }

              TensorShape tensor_shape(dim_sizes);
              out_tensors->push_back(
                  Tensor(dataset()->output_dtypes()[i], tensor_shape));
            }
          }
          return Status::OK();
        }

        // Special case: when there's only one slice, we return the slice
        // directly where possible instead of copying the tensor data.
        if (slices_to_concatenate.size() == 1) {
          auto tensors = std::move(slices_to_concatenate[0]);
          for (size_t i = 0; i < num_components; ++i) {
            // If the slice is aligned, we return it directly.
            if (!tensors[i].IsAligned()) {
              tensors[i] = tensor::DeepCopy(std::move(tensors[i]));
            }
          }
          *out_tensors = std::move(tensors);
          return Status::OK();
        }

        // For each component, concatenate slices into one tensor.
        for (size_t i = 0; i < num_components; ++i) {
          TensorShape component_shape({batch_size});
          TensorShape remaining_shape = slices_to_concatenate[0][i].shape();
          remaining_shape.RemoveDim(0);
          component_shape.AppendShape(remaining_shape);
          out_tensors->emplace_back(ctx->allocator({}),
                                    dataset()->output_dtypes()[i],
                                    component_shape);
          if (!out_tensors->back().IsInitialized()) {
            return errors::ResourceExhausted(
                "Failed to allocate memory for the batch of component ", i);
          }
          int64_t dst_offset = 0;
          for (size_t j = 0; j < slices_to_concatenate.size(); ++j) {
            auto num_slices = slices_to_concatenate[j][i].shape().dim_size(0);
            TF_RETURN_IF_ERROR(batch_util::CopyContiguousSlices(
                slices_to_concatenate[j][i], 0, dst_offset, num_slices,
                &(*out_tensors)[i]));
            dst_offset += num_slices;
          }
        }

        return Status::OK();
      }

     protected:
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_34(mht_34_v, 806, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        } else {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("batch_sizes_index"),
                                               batch_sizes_index_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("offset"), offset_));
        if (offset_ != -1) {
          for (int i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("tensors[", i, "]")), tensors_[i]));
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_35(mht_35_v, 830, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("batch_sizes_index"),
                                              &batch_sizes_index_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("offset"), &offset_));

        tensors_.clear();
        if (offset_ != -1) {
          tensors_.resize(dataset()->output_dtypes().size());
          for (int i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(strings::StrCat("tensors[", i, "]")),
                &tensors_[i]));
          }
        }
        return Status::OK();
      }

      TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_36(mht_36_v, 856, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "GetTraceMeMetadata");

        return dataset()->traceme_metadata_;
      }

     private:
      Status ValidateInputTensors() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSrebatch_dataset_opDTcc mht_37(mht_37_v, 864, "", "./tensorflow/core/kernels/data/experimental/rebatch_dataset_op.cc", "ValidateInputTensors");

        for (size_t i = 0; i < tensors_.size(); ++i) {
          if (tensors_[i].dims() == 0) {
            return errors::InvalidArgument(
                "Input element must have a non-scalar value in each "
                "component.");
          }
          if (tensors_[i].dim_size(0) != tensors_[0].dim_size(0)) {
            return errors::InvalidArgument(
                "Input element must have the same batch size in each "
                "component. Component 0 had size ",
                tensors_[0].dim_size(0), " but component ", i, " had size, ",
                tensors_[i].dim_size(0), ".");
          }
        }
        return Status::OK();
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_;
      // Whether we have reached the end of the input.
      bool end_of_sequence_ TF_GUARDED_BY(mu_) = false;
      // Represents the current input tensor(s).
      std::vector<Tensor> tensors_ TF_GUARDED_BY(mu_);
      // Represents the offset into the current input tensor(s).
      // An offset of -1 indicates that there is no data left in the current
      // slice.
      int64_t offset_ TF_GUARDED_BY(mu_) = -1;
      // Represents the current index into the batch_sizes list.
      int64_t batch_sizes_index_ TF_GUARDED_BY(mu_) = 0;
    };

    const DatasetBase* const input_;
    const std::vector<int64_t> batch_sizes_;
    const bool drop_remainder_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const TraceMeMetadata traceme_metadata_;
  };

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("RebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalRebatchDataset").Device(DEVICE_CPU),
                        RebatchDatasetOp);

REGISTER_KERNEL_BUILDER(Name("RebatchDatasetV2").Device(DEVICE_CPU),
                        RebatchDatasetV2Op);

}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
