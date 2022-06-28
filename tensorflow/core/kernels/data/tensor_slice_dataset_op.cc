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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/tensor_slice_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const TensorSliceDatasetOp::kDatasetType;
/* static */ constexpr const char* const TensorSliceDatasetOp::kComponents;
/* static */ constexpr const char* const TensorSliceDatasetOp::kToutputTypes;
/* static */ constexpr const char* const TensorSliceDatasetOp::kOutputShapes;
/* static */ constexpr const char* const TensorSliceDatasetOp::kIsFiles;

class TensorSliceDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<Tensor> tensors,
                   bool is_files)
      : DatasetBase(DatasetContext(ctx)),
        tensors_(std::move(tensors)),
        is_files_(is_files) {
    for (const Tensor& t : tensors_) {
      dtypes_.push_back(t.dtype());
      gtl::InlinedVector<int64_t, 4> element_dim_sizes;
      // Handle scalar here. Check that everyone matches here? Or fail
      // at runtime?
      for (int i = 1; i < t.dims(); ++i) {
        element_dim_sizes.push_back(t.dim_size(i));
      }
      partial_shapes_.emplace_back(element_dim_sizes);
      shapes_.emplace_back(std::move(element_dim_sizes));
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "MakeSplitProviders");

    split_providers->push_back(
        absl::make_unique<IndexSplitProvider>(tensors_[0].dim_size(0)));
    return Status::OK();
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "output_dtypes");
 return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "output_shapes");

    return partial_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_3(mht_3_v, 258, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_4(mht_4_v, 265, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "CardinalityInternal");

    return tensors_[0].dim_size(0);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_5(mht_5_v, 272, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "CardinalityInternal");

    return tensors_[0].dim_size(0);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_7(mht_7_v, 286, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_8(mht_8_v, 292, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    out_tensors->clear();
    out_tensors->reserve(tensors_.size());
    for (int i = 0; i < tensors_.size(); ++i) {
      out_tensors->push_back(MaybeCopySubSlice(tensors_[i], index));
    }
    return Status::OK();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_9(mht_9_v, 308, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "AsGraphDefInternal");

    std::vector<Node*> components;
    components.reserve(tensors_.size());
    for (const Tensor& t : tensors_) {
      Node* node;
      if (!ctx->is_graph_rewrite()) {
        TF_RETURN_IF_ERROR(b->AddDatasetOrTensor(ctx, t, &node));
        if (is_files_) {
          Node* file_node;
          TF_RETURN_IF_ERROR(
              b->AddIdentity(ctx, "FileIdentity", &node, &file_node));
        }
      } else {
        TF_RETURN_IF_ERROR(b->AddPlaceholder(t, &node));
        DCHECK_NE(ctx->input_list(), nullptr);
        ctx->input_list()->emplace_back(node->name(), t);
      }
      components.emplace_back(node);
    }
    AttrValue dtypes;
    b->BuildAttrValue(dtypes_, &dtypes);
    AttrValue is_files;
    b->BuildAttrValue(is_files_, &is_files);
    TF_RETURN_IF_ERROR(b->AddDataset(this, {}, {{0, components}},
                                     {{kToutputTypes, dtypes}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_10(mht_10_v, 343, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_11(mht_11_v, 348, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "Initialize");

      if (ctx->split_providers().empty()) {
        split_provider_ = std::make_shared<IndexSplitProvider>(
            dataset()->tensors_[0].dim_size(0));
      } else {
        TF_ASSIGN_OR_RETURN(split_provider_,
                            GetSingleSplitProvider(ctx, dataset()));
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_12(mht_12_v, 364, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "GetNextInternal");

      Tensor split;
      TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, end_of_sequence));
      if (*end_of_sequence) {
        return Status::OK();
      }
      int64_t index = split.scalar<int64_t>()();
      out_tensors->reserve(dataset()->tensors_.size());
      for (size_t i = 0; i < dataset()->tensors_.size(); ++i) {
        out_tensors->push_back(
            MaybeCopySubSlice(dataset()->tensors_[i], index));
      }
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_13(mht_13_v, 390, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "SaveInternal");

      return split_provider_->Save(
          [this](const std::string& key) { return full_name(key); }, writer);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_14(mht_14_v, 399, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "RestoreInternal");

      return split_provider_->Restore(
          [this](const std::string& key) { return full_name(key); }, reader);
    }

   private:
    std::shared_ptr<SplitProvider> split_provider_;
  };

  const std::vector<Tensor> tensors_;
  DataTypeVector dtypes_;
  std::vector<TensorShape> shapes_;
  std::vector<PartialTensorShape> partial_shapes_;
  bool is_files_;
};

TensorSliceDatasetOp::TensorSliceDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_15(mht_15_v, 419, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "TensorSliceDatasetOp::TensorSliceDatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kToutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  if (ctx->HasAttr(kIsFiles)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kIsFiles, &is_files_));
  }
}

void TensorSliceDatasetOp::MakeDataset(OpKernelContext* ctx,
                                       DatasetBase** output) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStensor_slice_dataset_opDTcc mht_16(mht_16_v, 431, "", "./tensorflow/core/kernels/data/tensor_slice_dataset_op.cc", "TensorSliceDatasetOp::MakeDataset");

  OpInputList inputs;
  OP_REQUIRES_OK(ctx, ctx->input_list(kComponents, &inputs));
  std::vector<Tensor> components;
  components.reserve(inputs.size());
  OP_REQUIRES(
      ctx, inputs[0].dims() > 0,
      errors::InvalidArgument("All components must be at least 1-dimensional"));
  const int64_t num_slices = inputs[0].dim_size(0);
  for (const Tensor& t : inputs) {
    components.push_back(t);
    OP_REQUIRES(ctx, t.dims() > 0,
                errors::InvalidArgument(
                    "All components must be at least 1-dimensional"));
    OP_REQUIRES(
        ctx, t.dim_size(0) == num_slices,
        errors::InvalidArgument(
            "All components must have the same size in the 0th dimension"));
  }
  *output = new Dataset(ctx, std::move(components), is_files_);
  OP_REQUIRES_OK(ctx,
                 VerifyTypesMatch((*output)->output_dtypes(), output_types_));
  OP_REQUIRES_OK(
      ctx, VerifyShapesCompatible((*output)->output_shapes(), output_shapes_));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("TensorSliceDataset").Device(DEVICE_CPU),
                        TensorSliceDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
