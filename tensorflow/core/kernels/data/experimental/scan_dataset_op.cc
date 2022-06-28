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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc() {
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
#include <iterator>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class ScanDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ScanDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "ScanDatasetOp");

    FunctionMetadata::Params params;
    if (ctx->HasAttr("use_default_device")) {
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("use_default_device", &use_default_device_));
      params.use_default_device = use_default_device_;
    }
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "f", params, &func_metadata_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tstate", &state_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("preserve_cardinality", &preserve_cardinality_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "MakeDataset");

    OpInputList initial_state_inputs;
    OP_REQUIRES_OK(ctx,
                   ctx->input_list("initial_state", &initial_state_inputs));
    std::vector<Tensor> initial_state(initial_state_inputs.begin(),
                                      initial_state_inputs.end());

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(ctx, func_metadata_, "other_arguments",
                                      &captured_func));

    *output =
        new Dataset(ctx, input, std::move(initial_state),
                    std::move(captured_func), state_types_, output_types_,
                    output_shapes_, preserve_cardinality_, use_default_device_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::vector<Tensor> initial_state,
            std::unique_ptr<CapturedFunction> captured_func,
            const DataTypeVector& state_types,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            bool preserve_cardinality, bool use_default_device)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          initial_state_(std::move(initial_state)),
          captured_func_(std::move(captured_func)),
          state_types_(state_types),
          output_types_(output_types),
          output_shapes_(output_shapes),
          preserve_cardinality_(preserve_cardinality),
          use_default_device_(use_default_device) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_2(mht_2_v, 263, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_3(mht_3_v, 270, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Scan")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "output_dtypes");

      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_5(mht_5_v, 287, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "DebugString");
 return "ScanDatasetOp::Dataset"; }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_7(mht_7_v, 299, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "CardinalityInternal");

      if (preserve_cardinality_) {
        return input_->Cardinality();
      } else {
        return kUnknownCardinality;
      }
    }

    int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_8(mht_8_v, 310, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "CardinalityInternal");

      if (preserve_cardinality_) {
        return input_->Cardinality(options);
      } else {
        return kUnknownCardinality;
      }
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_9(mht_9_v, 322, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_10(mht_10_v, 330, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "CheckExternalState");

      TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_11(mht_11_v, 341, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "AsGraphDefInternal");

      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));
      std::vector<Node*> initial_state_nodes;
      initial_state_nodes.reserve(initial_state_.size());
      for (const Tensor& t : initial_state_) {
        Node* node;
        TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
        initial_state_nodes.emplace_back(node);
      }
      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                    &other_arguments_types));
      AttrValue f;
      b->BuildAttrValue(captured_func_->func(), &f);
      AttrValue state_types;
      b->BuildAttrValue(state_types_, &state_types);
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);
      AttrValue preserve_cardinality_attr;
      b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);
      AttrValue use_default_device_attr;
      b->BuildAttrValue(use_default_device_, &use_default_device_attr);
      TF_RETURN_IF_ERROR(
          b->AddDataset(this, {{0, input_node}},
                        {{1, initial_state_nodes}, {2, other_arguments}},
                        {{"f", f},
                         {"Tstate", state_types},
                         {"Targuments", other_arguments_types_attr},
                         {"preserve_cardinality", preserve_cardinality_attr},
                         {"use_default_device", use_default_device_attr}},
                        output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            state_(params.dataset->initial_state_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_12(mht_12_v, 385, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_13(mht_13_v, 390, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "Initialize");

        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_14(mht_14_v, 402, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);

        std::vector<Tensor> next_element;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &next_element, end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }

        std::vector<Tensor> args;
        args.reserve(state_.size() + next_element.size());
        std::copy(state_.begin(), state_.end(), std::back_inserter(args));
        std::copy(next_element.begin(), next_element.end(),
                  std::back_inserter(args));

        std::vector<Tensor> state_and_output;
        state_and_output.reserve(dataset()->state_types_.size() +
                                 output_dtypes().size());

        Status s = instantiated_captured_func_->Run(
            ctx, std::move(args), &state_and_output, model_node());
        DCHECK(state_and_output.size() <=
               dataset()->state_types_.size() + output_dtypes().size());
        if (s.ok()) {
          state_.clear();
          size_t i = 0;
          for (; i < dataset()->state_types_.size(); ++i) {
            if (state_and_output[i].dtype() != dataset()->state_types_[i]) {
              return errors::InvalidArgument(
                  "Got wrong type for scan_func return value ", i,
                  " (expected ", DataTypeString(dataset()->state_types_[i]),
                  ", got ", DataTypeString(state_and_output[i].dtype()), ").");
            }
            state_.push_back(std::move(state_and_output[i]));
          }
          for (; i < state_and_output.size(); ++i) {
            const size_t output_index = i - dataset()->state_types_.size();
            if (state_and_output[i].dtype() != output_dtypes()[output_index]) {
              return errors::InvalidArgument(
                  "Got wrong type for scan_func return value ", i,
                  " (expected ",
                  DataTypeString(dataset()->state_types_[output_index]),
                  ", got ", DataTypeString(state_and_output[i].dtype()), ").");
            }
            if (!output_shapes()[output_index].IsCompatibleWith(
                    state_and_output[i].shape())) {
              return errors::InvalidArgument(
                  "Got wrong shape for scan_func return value ", i,
                  " (expected ", output_shapes()[output_index].DebugString(),
                  ", got ", state_and_output[i].shape().DebugString(), ").");
            }

            out_tensors->push_back(std::move(state_and_output[i]));
          }
        } else if (errors::IsOutOfRange(s)) {
          if (dataset()->preserve_cardinality_) {
            // To guarantee that the transformation preserves the cardinality of
            // the dataset, we convert `OutOfRange` to `InvalidArgument` as the
            // former may be interpreted by a caller as the end of sequence.
            return errors::InvalidArgument(
                "Function invocation produced OutOfRangeError: ",
                s.error_message());
          } else {
            // `f` may deliberately raise `errors::OutOfRange` to indicate
            // that we should terminate the iteration early.
            *end_of_sequence = true;
            return Status::OK();
          }
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
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_15(mht_15_v, 486, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "SaveInternal");

        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_func_->CheckExternalState()));
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        if (!state_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("state_size"), state_.size()));
          for (int idx = 0; idx < state_.size(); idx++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("state[", idx, "]")), state_[idx]));
          }
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSscan_dataset_opDTcc mht_16(mht_16_v, 506, "", "./tensorflow/core/kernels/data/experimental/scan_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        if (reader->Contains(full_name("state_size"))) {
          int64_t size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("state_size"), &size));
          state_.resize(size);
          for (int idx = 0; idx < size; idx++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), full_name(strings::StrCat("state[", idx, "]")),
                &state_[idx]));
          }
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::vector<Tensor> state_ TF_GUARDED_BY(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    };

    const DatasetBase* const input_;
    const std::vector<Tensor> initial_state_;
    const std::unique_ptr<CapturedFunction> captured_func_;
    const DataTypeVector state_types_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const bool preserve_cardinality_;
    const bool use_default_device_;
  };

  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector state_types_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool preserve_cardinality_;
  bool use_default_device_;
};

REGISTER_KERNEL_BUILDER(Name("ScanDataset").Device(DEVICE_CPU), ScanDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalScanDataset").Device(DEVICE_CPU),
                        ScanDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("ScanDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalScanDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
