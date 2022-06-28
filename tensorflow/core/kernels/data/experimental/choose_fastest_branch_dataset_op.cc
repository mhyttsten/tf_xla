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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc() {
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

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/histogram/histogram.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

static const double kPercentile = 90.0;

// Each instance of this class wraps an iterator. Whenever an iterator created
// for this dataset invokes the `GetNext` method, the call is delegated to the
// wrapped iterator's `GetNext` method.
class WrapperDataset : public DatasetBase {
 public:
  WrapperDataset(DatasetContext::Params params,
                 const DataTypeVector* output_dtypes,
                 const std::vector<PartialTensorShape>* output_shapes,
                 IteratorBase* iterator)
      : DatasetBase(DatasetContext(std::move(params))),
        output_dtypes_(output_dtypes),
        output_shapes_(output_shapes),
        real_iterator_(iterator) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "WrapperDataset");
}

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "output_dtypes");

    return *output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "output_shapes");

    return *output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "DebugString");
 return "WrapperDataset"; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_5(mht_5_v, 246, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_6(mht_6_v, 254, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "AsGraphDefInternal");

    return errors::Unimplemented(DebugString(), "::AsGraphDefInternal");
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    // MakeIterator should only be called once per WrapperDataset. However,
    // since this function expects an iterator return value, we raise the
    // error only at iterator initialization time.
    bool error = iterator_created_;
    iterator_created_ = true;
    return absl::make_unique<WrapperIterator>(
        WrapperIterator::Params{this, strings::StrCat(prefix, "::Wrapper")},
        error);
  }

 private:
  class WrapperIterator : public DatasetIterator<WrapperDataset> {
   public:
    explicit WrapperIterator(const Params& params, bool error)
        : DatasetIterator<WrapperDataset>(params), error_(error) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_7(mht_7_v, 277, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "WrapperIterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_8(mht_8_v, 282, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "Initialize");

      if (error_) {
        return errors::InvalidArgument(
            "Cannot create more than one WrapperIterator per WrapperDataset. "
            "Make sure the branches to ChooseFastestDataset do not expect the "
            "input to repeat.");
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_9(mht_9_v, 297, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "GetNextInternal");

      return dataset()->real_iterator_->GetNext(ctx, out_tensors,
                                                end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1.0);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_10(mht_10_v, 312, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "SaveInternal");

      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_11(mht_11_v, 320, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "RestoreInternal");

      return Status::OK();
    }

   private:
    const bool error_;
  };

  mutable bool iterator_created_ = false;
  const DataTypeVector* const output_dtypes_;
  const std::vector<PartialTensorShape>* const output_shapes_;
  IteratorBase* const real_iterator_;  // not owned.
};

// This Dataset picks between some dataset function branches. Each function is
// expected to input a dataset and output a dataset. The datasets in the
// branches are expected to be stateless. For each iterator that can be produced
// by a functions output, it is expected to call the input dataset's
// MakeIterator method at most once; otherwise, undefined behavior may occur.
class ChooseFastestBranchDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ChooseFastestBranchDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_12(mht_12_v, 345, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "ChooseFastestBranchDatasetOp");

    std::vector<NameAttrList> funcs;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("branches", &funcs));
    func_metadatas_.resize(funcs.size());
    for (int i = 0; i < funcs.size(); ++i) {
      OP_REQUIRES_OK(
          ctx, FunctionMetadata::Create(ctx, std::move(funcs[i]), /*params=*/{},
                                        &func_metadatas_[i]));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_elements_per_branch",
                                     &num_elements_per_branch_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("other_arguments_lengths",
                                     &other_arguments_lengths_));

    OP_REQUIRES(
        ctx, func_metadatas_.size() == other_arguments_lengths_.size(),
        errors::InvalidArgument(
            "branches and other_arguments_lengths must have the same length."));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_13(mht_13_v, 371, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "MakeDataset");

    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "ratio_numerator",
                                                     &ratio_numerator_));
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "ratio_denominator",
                                                     &ratio_denominator_));
    OP_REQUIRES(ctx, ratio_numerator_ > 0,
                errors::InvalidArgument(
                    "`ratio_numerator` must be greater than zero."));
    OP_REQUIRES(ctx, ratio_denominator_ > 0,
                errors::InvalidArgument(
                    "`ratio_denominator` must be greater than zero."));
    OP_REQUIRES(ctx, num_elements_per_branch_ % ratio_denominator_ == 0,
                errors::InvalidArgument("`num_elements_per_branch` must be "
                                        "divisible by `ratio_denominator`."));

    std::vector<std::unique_ptr<CapturedFunction>> captured_funcs(
        func_metadatas_.size());
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("other_arguments", &inputs));

    // Keeps track of starting index into other_arguments for a given function.
    int index = 0;
    for (int i = 0; i < func_metadatas_.size(); ++i) {
      std::vector<Tensor> captured_args;
      captured_args.reserve(other_arguments_lengths_[i]);
      int end_index = index + other_arguments_lengths_[i];
      for (; index < end_index; ++index) {
        captured_args.push_back(inputs[index]);
      }
      OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, func_metadatas_[i],
                                                   std::move(captured_args),
                                                   &captured_funcs[i]));
    }
    *output = new Dataset(ctx, input, std::move(captured_funcs), output_types_,
                          output_shapes_, num_elements_per_branch_,
                          ratio_numerator_, ratio_denominator_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, DatasetBase* input,
            std::vector<std::unique_ptr<CapturedFunction>> captured_funcs,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes,
            int64_t num_elements_per_branch, int64_t ratio_numerator,
            int64_t ratio_denominator)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_funcs_(std::move(captured_funcs)),
          output_types_(output_types),
          output_shapes_(output_shapes),
          num_elements_per_branch_(num_elements_per_branch),
          ratio_numerator_(ratio_numerator),
          ratio_denominator_(ratio_denominator) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_14(mht_14_v, 428, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_15(mht_15_v, 435, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<ChooseFastestIterator>(
          ChooseFastestIterator::Params{
              this, strings::StrCat(prefix, "::ChooseFastestBranch")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_16(mht_16_v, 447, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "output_dtypes");

      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_17(mht_17_v, 454, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_18(mht_18_v, 461, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "DebugString");

      return "ChooseFastestBranchDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_19(mht_19_v, 468, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "CardinalityInternal");

      int64_t n = input_->Cardinality();
      if (n == kInfiniteCardinality || n == kUnknownCardinality) {
        return n;
      }
      // TODO(rachelim): this might be wrong if the ratio is not fixed, for
      // example, from a BatchDataset with drop_remainder = False
      return static_cast<double>(n) * ratio_numerator_ / ratio_denominator_;
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_20(mht_20_v, 482, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_21(mht_21_v, 490, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "CheckExternalState");

      for (const auto& captured_func : captured_funcs_) {
        TF_RETURN_IF_ERROR(captured_func->CheckExternalState());
      }
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_22(mht_22_v, 503, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* ratio_numerator_node;
      TF_RETURN_IF_ERROR(b->AddScalar(ratio_numerator_, &ratio_numerator_node));
      Node* ratio_denominator_node;
      TF_RETURN_IF_ERROR(
          b->AddScalar(ratio_denominator_, &ratio_denominator_node));

      std::vector<int32> other_arguments_lengths;
      other_arguments_lengths.reserve(captured_funcs_.size());
      int num_captured_inputs = 0;
      for (const auto& func : captured_funcs_) {
        num_captured_inputs += func->captured_inputs().size();
        other_arguments_lengths.push_back(func->captured_inputs().size());
      }
      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      other_arguments_types.reserve(num_captured_inputs);
      other_arguments.reserve(num_captured_inputs);
      for (const auto& captured_func : captured_funcs_) {
        TF_RETURN_IF_ERROR(captured_func->AddToGraph(ctx, b, &other_arguments,
                                                     &other_arguments_types));
      }

      // Targuments
      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      // num_elements_per_branch
      AttrValue num_elements_per_branch_attr;
      b->BuildAttrValue(num_elements_per_branch_,
                        &num_elements_per_branch_attr);

      // branches
      AttrValue branches_attr;
      std::vector<NameAttrList> funcs;
      funcs.resize(captured_funcs_.size());
      for (int i = 0; i < captured_funcs_.size(); ++i) {
        funcs[i] = captured_funcs_[i]->func();
      }
      b->BuildAttrValue(funcs, &branches_attr);

      // other_arguments_lengths
      AttrValue other_arguments_lengths_attr;
      b->BuildAttrValue(other_arguments_lengths, &other_arguments_lengths_attr);

      return b->AddDataset(
          this,
          /*inputs=*/
          {std::make_pair(0, input_graph_node),
           std::make_pair(1, ratio_numerator_node),
           std::make_pair(2, ratio_denominator_node)},
          /*list_inputs=*/{std::make_pair(3, other_arguments)},
          /*attrs=*/
          {std::make_pair("Targuments", other_arguments_types_attr),
           std::make_pair("num_elements_per_branch",
                          num_elements_per_branch_attr),
           std::make_pair("branches", branches_attr),
           std::make_pair("other_arguments_lengths",
                          other_arguments_lengths_attr)},
          output);
    }

   private:
    // This iterator picks the fastest of dataset branches by running
    // experiments for the first dataset()->num_elements_per_branch_ *
    // num_branches iterations.
    class ChooseFastestIterator : public DatasetIterator<Dataset> {
     public:
      explicit ChooseFastestIterator(const Params& params)
          : DatasetIterator<Dataset>(params),
            instantiated_captured_funcs_(dataset()->captured_funcs_.size()),
            histograms_(dataset()->captured_funcs_.size()) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_23(mht_23_v, 580, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "ChooseFastestIterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_24(mht_24_v, 585, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "Initialize");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));

        for (int i = 0; i < dataset()->captured_funcs_.size(); ++i) {
          TF_RETURN_IF_ERROR(dataset()->captured_funcs_[i]->Instantiate(
              ctx, &instantiated_captured_funcs_[i]));
        }

        return Status::OK();
      }

      // The first num_elements_per_branch * num_branches iterations, we run
      // experiments on the branches, using (branch_index_, experiment_counter_)
      // to keep track of which experiment we're on.
      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_25(mht_25_v, 606, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "GetNextInternal");

        {  // Locking scope
          mutex_lock l(mu_);
          if (branch_index_ < dataset()->captured_funcs_.size()) {
            // Still running experiments
            if (!current_iterator_) {
              TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, branch_index_,
                                                     /*is_experiment=*/true,
                                                     /*is_get_next=*/true));
            }

            Status s = GetNextFromExperiment(ctx, out_tensors, end_of_sequence);
            experiment_counter_++;

            if (experiment_counter_ >= dataset()->num_elements_per_branch_) {
              // Done experimenting with this branch. Increment the branch index
              // so that on the next iteration, we will draw from the next
              // branch.
              experiment_counter_ = 0;
              branch_index_++;
              current_iterator_.reset();
            }
            return s;
          }
          if (!current_iterator_) {
            SelectFastestInputIndex();
            TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, fastest_index_,
                                                   /*is_experiment=*/false,
                                                   /*is_get_next=*/true));
          }
        }

        return current_iterator_->GetNext(ctx, out_tensors, end_of_sequence);
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeKnownRatioNode(
            std::move(args),
            /*ratio=*/static_cast<double>(dataset()->ratio_numerator_) /
                dataset()->ratio_denominator_);
      }

      // TODO(rachelim): Save and restore histogram state as well. Currently,
      // if an iterator is saved and restored, the histograms start recording
      // from scratch.
      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_26(mht_26_v, 657, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "SaveInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("experiment_counter"),
                                               experiment_counter_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("branch_index"), branch_index_));
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name("fastest_index"), fastest_index_));
        if (current_iterator_) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, current_iterator_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impl_empty"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_27(mht_27_v, 679, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("experiment_counter"),
                                              &experiment_counter_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("branch_index"), &branch_index_));
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("fastest_index"), &fastest_index_));

        // Restore state of `current_iterator_` if it exists.
        if (!reader->Contains(full_name("input_impl_empty"))) {
          if (branch_index_ < dataset()->captured_funcs_.size()) {
            TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, branch_index_,
                                                   /*is_experiment=*/true,
                                                   /*is_get_next=*/false));
          } else {
            TF_RETURN_IF_ERROR(MakeCurrentIterator(ctx, fastest_index_,
                                                   /*is_experiment=*/false,
                                                   /*is_get_next=*/false));
          }
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, current_iterator_));
        }
        return Status::OK();
      }

     private:
      Status GetNextFromExperiment(IteratorContext* ctx,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_28(mht_28_v, 712, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "GetNextFromExperiment");

        DCHECK_GE(branch_index_, 0);
        DCHECK_LT(branch_index_, histograms_.size());

        int64_t start = EnvTime::NowNanos();
        Status s =
            current_iterator_->GetNext(ctx, out_tensors, end_of_sequence);

        if (experiment_counter_ > 0) {
          // Ignore the first experiment when benchmarking. It may be an outlier
          // due to session set up time and other overheads.
          histograms_[branch_index_].Add(
              static_cast<double>(EnvTime::NowNanos() - start));
        }
        return s;
      }

      // Select the fastest input to use based on the histograms of timings
      // of the completed iterations. The input with the best 90th percentile
      // iteration time is selected.
      void SelectFastestInputIndex() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_29(mht_29_v, 735, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "SelectFastestInputIndex");

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
      }

      Status MakeCurrentIterator(IteratorContext* ctx, int64_t branch_index,
                                 bool is_experiment, bool is_get_next)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSchoose_fastest_branch_dataset_opDTcc mht_30(mht_30_v, 759, "", "./tensorflow/core/kernels/data/experimental/choose_fastest_branch_dataset_op.cc", "MakeCurrentIterator");

        DCHECK_GE(branch_index, 0);
        DCHECK_LT(branch_index, histograms_.size());

        // `StoreDatasetInVariantTensor` transfers ownership of the dataset
        // to the tensor, so the tensor must persist between iterations.
        wrapper_dataset_tensor_ =
            absl::make_unique<Tensor>(DT_VARIANT, TensorShape({}));

        DatasetContext::Params params;
        params.type_string = "ChooseFastestBranch_Wrapper";
        params.node_name = strings::StrCat(params.type_string, branch_index);
        DatasetBase* temp_dataset = new WrapperDataset(
            std::move(params), &input_impl_->output_dtypes(),
            &input_impl_->output_shapes(), input_impl_.get());

        if (is_experiment) {
          // When running experiment iterations, we add a TakeDataset in between
          // the input and the function datasets. This is so that function
          // datasets with prefetching behavior won't consume more input
          // elements than they actually use to produce output.
          DatasetContext::Params take_dataset_params;
          take_dataset_params.type_string = "ChooseFastestBranch_Take";
          take_dataset_params.node_name =
              strings::StrCat(take_dataset_params.type_string, branch_index);
          int64_t count = dataset()->num_elements_per_branch_ *
                          dataset()->ratio_numerator_ /
                          dataset()->ratio_denominator_;
          temp_dataset = new TakeDataset(std::move(take_dataset_params), count,
                                         temp_dataset);
        }

        TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(
            temp_dataset, wrapper_dataset_tensor_.get()));

        if (is_get_next) {
          TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
              ctx, this, {*wrapper_dataset_tensor_}, branch_index,
              *instantiated_captured_funcs_[branch_index], prefix(),
              &current_iterator_, model_node()));
        } else {
          // NOTE: We intentionally ignore resource modeling outside GetNext().
          TF_RETURN_IF_ERROR(MakeIteratorFromInputElement(
              ctx, this, {*wrapper_dataset_tensor_}, branch_index,
              *instantiated_captured_funcs_[branch_index], prefix(),
              &current_iterator_, /*node=*/nullptr));
        }

        return Status::OK();
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::vector<std::unique_ptr<InstantiatedCapturedFunction>>
          instantiated_captured_funcs_ TF_GUARDED_BY(mu_);

      // For tracking the time taken for each input's iterations.
      std::vector<histogram::Histogram> histograms_ TF_GUARDED_BY(mu_);
      int64_t fastest_index_ = -1;
      std::unique_ptr<Tensor> wrapper_dataset_tensor_;
      std::unique_ptr<IteratorBase> current_iterator_;

      // Keeps track of which (branch, experiment) the next iteration is on.
      int64_t branch_index_ TF_GUARDED_BY(mu_) = 0;
      int64_t experiment_counter_ TF_GUARDED_BY(mu_) = 0;
    };  // class Iterator

    const DatasetBase* const input_;
    const std::vector<std::unique_ptr<CapturedFunction>> captured_funcs_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
    const int64_t num_elements_per_branch_;
    const int64_t ratio_numerator_;
    const int64_t ratio_denominator_;
  };  // class Dataset

  int64_t ratio_numerator_;
  int64_t ratio_denominator_;
  int64_t num_elements_per_branch_;
  std::vector<std::shared_ptr<FunctionMetadata>> func_metadatas_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::vector<int32> other_arguments_lengths_;
};  // class ChooseFastestBranchDatasetOp

// Register the kernel implementation for ChooseFastestBranchDataset.
REGISTER_KERNEL_BUILDER(Name("ChooseFastestBranchDataset").Device(DEVICE_CPU),
                        ChooseFastestBranchDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
