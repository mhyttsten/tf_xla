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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc() {
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
#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class UnbatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit UnbatchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "UnbatchDatasetOp");
}

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "MakeDataset");

    *output = new Dataset(ctx, input);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    explicit Dataset(OpKernelContext* ctx, DatasetBase* input)
        : DatasetBase(DatasetContext(ctx)), input_(input) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "Dataset");

      input_->Ref();
      batch_size_ = -1;
      for (const PartialTensorShape& shape : input->output_shapes()) {
        if (!shape.unknown_rank()) {
          if (batch_size_ < 0 && shape.dim_size(0) >= 0) {
            batch_size_ = shape.dim_size(0);
          }
          gtl::InlinedVector<int64_t, 4> partial_dim_sizes;
          for (int i = 1; i < shape.dims(); ++i) {
            partial_dim_sizes.push_back(shape.dim_size(i));
          }
          shapes_.emplace_back(std::move(partial_dim_sizes));
        } else {
          // If the input shape is unknown, the output shape will be unknown.
          shapes_.emplace_back();
        }
      }
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_3(mht_3_v, 242, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Unbatch")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "output_dtypes");

      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "output_shapes");

      return shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_6(mht_6_v, 266, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "DebugString");
 return "UnbatchDatasetOp::Dataset"; }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_7(mht_7_v, 271, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "CardinalityInternal");

      int64_t n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      if (batch_size_ > 0) {
        return n * batch_size_;
      }
      return kUnknownCardinality;
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_8(mht_8_v, 286, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_9(mht_9_v, 294, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_10(mht_10_v, 304, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
      TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node}, output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            current_index_(0),
            current_batch_size_(0),
            shapes_(params.dataset->output_shapes().size()) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_11(mht_11_v, 321, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_12(mht_12_v, 326, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "Initialize");

        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_13(mht_13_v, 336, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return Status::OK();
        }
        *end_of_sequence = false;
        while (!*end_of_sequence) {
          if (current_index_ < current_batch_size_) {
            out_tensors->clear();
            out_tensors->reserve(tensors_.size());
            for (int i = 0; i < tensors_.size(); ++i) {
              // TODO(b/201790899): Investigate why using MaybeCopySubSlice
              // may lead to a memory leak.
              out_tensors->emplace_back(ctx->allocator({}), tensors_[i].dtype(),
                                        shapes_[i]);
              TF_RETURN_IF_ERROR(batch_util::MaybeMoveSliceToElement(
                  &tensors_[i], &out_tensors->back(), current_index_));
            }
            ++current_index_;
            *end_of_sequence = false;
            return Status::OK();
          }
          current_index_ = 0;
          current_batch_size_ = 0;
          tensors_.clear();
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &tensors_, end_of_sequence));
          if (!*end_of_sequence) {
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
                    tensors_[0].dim_size(0), " but component ", i,
                    " had size, ", tensors_[i].dim_size(0), ".");
              }
              shapes_[i] = tensors_[i].shape();
              shapes_[i].RemoveDim(0);
            }
            current_batch_size_ = tensors_[0].dim_size(0);
          }
        }
        input_impl_.reset();
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        // Unbatch assumes that all input components have the same leading
        // dimension. If it is statically known for any component, we model the
        // transformation using `KnownRatio`. Otherwise, we use `UnknownRatio`.
        for (auto& shape : dataset()->input_->output_shapes()) {
          if (shape.dims() > 0 && shape.dim_size(0) > 0) {
            return model::MakeKnownRatioNode(
                std::move(args), 1.0 / static_cast<double>(shape.dim_size(0)));
          }
        }
        return model::MakeUnknownRatioNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_14(mht_14_v, 407, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        }
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("current_index"), current_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("n"), current_batch_size_));
        if (current_index_ < current_batch_size_) {
          for (size_t i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("tensors[", i, "]")), tensors_[i]));
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSunbatch_dataset_opDTcc mht_15(mht_15_v, 432, "", "./tensorflow/core/kernels/data/experimental/unbatch_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        if (!reader->Contains(full_name("input_impl_empty"))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        } else {
          input_impl_.reset();
        }
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("current_index"), &current_index_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("n"), &current_batch_size_));
        tensors_.clear();
        tensors_.resize(dataset()->output_dtypes().size());
        if (current_index_ < current_batch_size_) {
          for (size_t i = 0; i < tensors_.size(); ++i) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(strings::StrCat("tensors[", i, "]")),
                &tensors_[i]));
            shapes_[i] = tensors_[i].shape();
            shapes_[i].RemoveDim(0);
          }
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      int64_t current_index_ TF_GUARDED_BY(mu_);
      int64_t current_batch_size_ TF_GUARDED_BY(mu_);
      std::vector<Tensor> tensors_ TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::vector<TensorShape> shapes_ TF_GUARDED_BY(mu_);
    };

    const DatasetBase* const input_;
    std::vector<PartialTensorShape> shapes_;
    // batch_size_ may or may not be known, with -1 as unknown
    int64_t batch_size_;
  };
};

REGISTER_KERNEL_BUILDER(Name("UnbatchDataset").Device(DEVICE_CPU),
                        UnbatchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalUnbatchDataset").Device(DEVICE_CPU),
                        UnbatchDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
