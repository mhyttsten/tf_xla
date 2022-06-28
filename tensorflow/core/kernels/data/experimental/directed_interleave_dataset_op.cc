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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kDatasetType;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kSelectorInputDataset;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kDataInputDatasets;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kStopOnEmptyDataset;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kOutputTypes;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kOutputShapes;
/* static */ constexpr const char* const
    DirectedInterleaveDatasetOp::kNumInputDatasets;

class DirectedInterleaveDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* selector_input,
          std::vector<DatasetBase*> data_inputs, bool stop_on_empty_dataset)
      : DatasetBase(DatasetContext(ctx)),
        selector_input_(selector_input),
        data_inputs_(std::move(data_inputs)),
        stop_on_empty_dataset_(stop_on_empty_dataset) {
    selector_input_->Ref();

    output_shapes_ = data_inputs_[0]->output_shapes();
    data_inputs_[0]->Ref();
    for (size_t i = 1; i < data_inputs_.size(); ++i) {
      const DatasetBase* data_input = data_inputs_[i];
      data_input->Ref();
      for (size_t j = 0; j < output_shapes_.size(); ++j) {
        output_shapes_[j] = MostSpecificCompatibleShape(
            output_shapes_[j], data_input->output_shapes()[j]);
      }
    }
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_0(mht_0_v, 236, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "~Dataset");

    selector_input_->Unref();
    for (DatasetBase* data_input : data_inputs_) {
      data_input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_1(mht_1_v, 253, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "MakeSplitProviders");

    TF_ASSIGN_OR_RETURN(*split_providers, GetSplitProviders(this));
    return Status::OK();
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "output_dtypes");

    return data_inputs_[0]->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_3(mht_3_v, 268, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "output_shapes");

    return output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_4(mht_4_v, 275, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "CardinalityInternal");

    // As long as one of input dataset has infinite cardinality, the output
    // cardinality is infinite.
    for (const auto& input : data_inputs_) {
      int64_t n = input->Cardinality();
      if (n == kInfiniteCardinality) {
        return n;
      }
    }
    return kUnknownCardinality;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "InputDatasets");

    inputs->push_back(selector_input_);
    for (const auto& data_input : data_inputs_) {
      inputs->push_back(data_input);
    }
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_7(mht_7_v, 308, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "CheckExternalState");

    for (const auto& input : data_inputs_) {
      TF_RETURN_IF_ERROR(input->CheckExternalState());
    }
    return selector_input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_8(mht_8_v, 321, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "AsGraphDefInternal");

    Node* selector_input_node;
    TF_RETURN_IF_ERROR(
        b->AddInputDataset(ctx, selector_input_, &selector_input_node));
    std::vector<Node*> data_input_nodes(data_inputs_.size());
    for (size_t i = 0; i < data_inputs_.size(); ++i) {
      TF_RETURN_IF_ERROR(
          b->AddInputDataset(ctx, data_inputs_[i], &data_input_nodes[i]));
    }

    // Attr: stop_on_empty_dataset
    AttrValue stop_on_empty_dataset_attr;
    b->BuildAttrValue(stop_on_empty_dataset_, &stop_on_empty_dataset_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        /*inputs=*/{{0, selector_input_node}},
        /*list_inputs=*/{{1, data_input_nodes}},
        /*attrs=*/
        {std::make_pair(kStopOnEmptyDataset, stop_on_empty_dataset_attr)},
        output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          num_active_inputs_(params.dataset->data_inputs_.size()) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_9(mht_9_v, 353, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_10(mht_10_v, 358, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "Initialize");

      mutex_lock l(mu_);
      TF_ASSIGN_OR_RETURN(input_contexts_,
                          CreateInputIteratorContexts(ctx, dataset()));
      TF_RETURN_IF_ERROR(dataset()->selector_input_->MakeIterator(
          &input_contexts_[0], this, prefix(), &selector_input_impl_));
      data_input_impls_.resize(dataset()->data_inputs_.size());
      for (size_t i = 0; i < data_input_impls_.size(); ++i) {
        const DatasetBase* data_input = dataset()->data_inputs_[i];
        TF_RETURN_IF_ERROR(data_input->MakeIterator(
            &input_contexts_[i + 1], this,
            strings::StrCat(prefix(), "[", i, "]"), &data_input_impls_[i]));
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_11(mht_11_v, 379, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);
      if (!selector_input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      while (true) {
        std::vector<Tensor> selector_result;
        *end_of_sequence = false;
        TF_RETURN_IF_ERROR(selector_input_impl_->GetNext(
            &input_contexts_[0], &selector_result, end_of_sequence));
        if (*end_of_sequence) {
          ResetInputs();
          return Status::OK();
        }

        int64_t selected_input = selector_result[0].scalar<int64_t>()();
        if (selected_input < 0 || selected_input >= data_input_impls_.size()) {
          return errors::InvalidArgument(
              "Selector index out of range: ", selected_input,
              " >= ", data_input_impls_.size());
        }

        if (data_input_impls_[selected_input]) {
          bool end_of_selected_input = false;
          TF_RETURN_IF_ERROR(data_input_impls_[selected_input]->GetNext(
              &input_contexts_[selected_input + 1], out_tensors,
              &end_of_selected_input));

          if (!end_of_selected_input) {
            return Status::OK();
          }

          if (dataset()->stop_on_empty_dataset_) {
            *end_of_sequence = true;
            ResetInputs();
            return Status::OK();
          }

          data_input_impls_[selected_input].reset();
          --num_active_inputs_;

          if (num_active_inputs_ == 0) {
            selector_input_impl_.reset();
            *end_of_sequence = true;
            return Status::OK();
          }
        }

        VLOG(2) << "DirectedInterleave selected an exhausted input: "
                << selected_input;
      }
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeInterleaveManyNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_12(mht_12_v, 444, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      if (selector_input_impl_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, selector_input_impl_));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("selector_input_impl_empty"), ""));
      }
      for (size_t i = 0; i < data_input_impls_.size(); ++i) {
        const auto& data_input_impl = data_input_impls_[i];
        if (data_input_impl) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, data_input_impl));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("data_input_impl_empty[", i, "]")),
              ""));
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_13(mht_13_v, 469, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      if (!reader->Contains(full_name("selector_input_impl_empty"))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, selector_input_impl_));
      } else {
        selector_input_impl_.reset();
      }
      for (size_t i = 0; i < data_input_impls_.size(); ++i) {
        if (!reader->Contains(
                full_name(strings::StrCat("data_input_impl_empty[", i, "]")))) {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, data_input_impls_[i]));
        } else {
          data_input_impls_[i].reset();
        }
      }
      return Status::OK();
    }

   private:
    void ResetInputs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_14(mht_14_v, 491, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "ResetInputs");

      selector_input_impl_.reset();
      for (auto& data_input_impl : data_input_impls_) {
        data_input_impl.reset();
      }
      num_active_inputs_ = 0;
    }

    mutex mu_;
    // Iterator contexts for inputs datasets. The first context is for the
    // selector input, and the remaning contexts are for the data inputs.
    std::vector<IteratorContext> input_contexts_;
    std::unique_ptr<IteratorBase> selector_input_impl_ TF_GUARDED_BY(mu_);
    std::vector<std::unique_ptr<IteratorBase>> data_input_impls_
        TF_GUARDED_BY(mu_);
    int64_t num_active_inputs_ TF_GUARDED_BY(mu_);
  };

  static PartialTensorShape MostSpecificCompatibleShape(
      const PartialTensorShape& ts1, const PartialTensorShape& ts2) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_15(mht_15_v, 513, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "MostSpecificCompatibleShape");

    PartialTensorShape output_tensorshape;
    if (ts1.dims() != ts2.dims() || ts1.unknown_rank() || ts2.unknown_rank())
      return output_tensorshape;
    auto dims1 = ts1.dim_sizes();
    auto dims2 = ts2.dim_sizes();
    for (int d = 0; d < ts1.dims(); ++d) {
      if (dims1[d] == dims2[d])
        output_tensorshape.Concatenate(dims1[d]);
      else
        output_tensorshape.Concatenate(-1);
    }
    return output_tensorshape;
  }

  const DatasetBase* const selector_input_;
  const std::vector<DatasetBase*> data_inputs_;
  std::vector<PartialTensorShape> output_shapes_;
  const bool stop_on_empty_dataset_;
};

DirectedInterleaveDatasetOp::DirectedInterleaveDatasetOp(
    OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_16(mht_16_v, 539, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "DirectedInterleaveDatasetOp::DirectedInterleaveDatasetOp");

  if (ctx->HasAttr(kStopOnEmptyDataset)) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr(kStopOnEmptyDataset, &stop_on_empty_dataset_));
  }
}

void DirectedInterleaveDatasetOp::MakeDataset(OpKernelContext* ctx,
                                              DatasetBase** output) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSdirected_interleave_dataset_opDTcc mht_17(mht_17_v, 550, "", "./tensorflow/core/kernels/data/experimental/directed_interleave_dataset_op.cc", "DirectedInterleaveDatasetOp::MakeDataset");

  DatasetBase* selector_input;
  OP_REQUIRES_OK(ctx,
                 GetDatasetFromVariantTensor(ctx->input(0), &selector_input));

  OP_REQUIRES(
      ctx,
      selector_input->output_dtypes().size() == 1 &&
          selector_input->output_dtypes()[0] == DT_INT64 &&
          selector_input->output_shapes().size() == 1 &&
          selector_input->output_shapes()[0].IsCompatibleWith(
              PartialTensorShape({})),
      errors::InvalidArgument(
          "The selector input must be a dataset of scalar int64 elements."));

  // The first input is the selector, followed by dataset inputs.
  std::vector<DatasetBase*> data_inputs;
  for (size_t i = 1; i < ctx->num_inputs(); ++i) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
    data_inputs.push_back(input);

    OP_REQUIRES(ctx, data_inputs[0]->output_dtypes() == input->output_dtypes(),
                errors::InvalidArgument(
                    "All inputs must have the same output_dtypes. First input "
                    "has types ",
                    DataTypeVectorString(data_inputs[0]->output_dtypes()),
                    ", and input ", i - 1, " has types ",
                    DataTypeVectorString(input->output_dtypes())));
  }

  *output = new Dataset(ctx, selector_input, std::move(data_inputs),
                        stop_on_empty_dataset_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("DirectedInterleaveDataset").Device(DEVICE_CPU),
                        DirectedInterleaveDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalDirectedInterleaveDataset").Device(DEVICE_CPU),
    DirectedInterleaveDatasetOp);
}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
