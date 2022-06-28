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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/map_dataset_op.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const MapDatasetOp::kDatasetType;
/* static */ constexpr const char* const MapDatasetOp::kInputDataset;
/* static */ constexpr const char* const MapDatasetOp::kOtherArguments;
/* static */ constexpr const char* const MapDatasetOp::kFunc;
/* static */ constexpr const char* const MapDatasetOp::kTarguments;
/* static */ constexpr const char* const MapDatasetOp::kOutputTypes;
/* static */ constexpr const char* const MapDatasetOp::kOutputShapes;
/* static */ constexpr const char* const MapDatasetOp::kUseInterOpParallelism;
/* static */ constexpr const char* const MapDatasetOp::kPreserveCardinality;

class MapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes,
          bool preserve_cardinality)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        preserve_cardinality_(preserve_cardinality),
        captured_func_(std::move(captured_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "output_dtypes");
 return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "output_shapes");

    return output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_3(mht_3_v, 250, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_4(mht_4_v, 257, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "CardinalityInternal");

    if (preserve_cardinality_) {
      return input_->Cardinality();
    } else {
      return kUnknownCardinality;
    }
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_5(mht_5_v, 268, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "CardinalityInternal");

    if (preserve_cardinality_) {
      return input_->Cardinality(options);
    } else {
      return kUnknownCardinality;
    }
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_6(mht_6_v, 279, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_7(mht_7_v, 287, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "CheckExternalState");

    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_8(mht_8_v, 296, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    std::vector<Tensor> args;
    TF_RETURN_IF_ERROR(input_->Get(ctx, index, &args));
    if (!instantiated_captured_func_) {
      TF_RETURN_IF_ERROR(
          captured_func_->Instantiate(InstantiateCapturedFunctionParams(ctx),
                                      &instantiated_captured_func_));
    }
    return instantiated_captured_func_->RunInstantiated(args, out_tensors);
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_9(mht_9_v, 314, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));

    // Attr: f
    AttrValue f_attr;
    b->BuildAttrValue(captured_func_->func(), &f_attr);

    // Attr: Targuments
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    // Attr: use_inter_op_parallelism
    AttrValue use_inter_op_parallelism_attr;
    b->BuildAttrValue(captured_func_->use_inter_op_parallelism(),
                      &use_inter_op_parallelism_attr);

    // Attr: preserve_cardinality
    AttrValue preserve_cardinality_attr;
    b->BuildAttrValue(preserve_cardinality_, &preserve_cardinality_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {std::make_pair(0, input_graph_node)},  // Single tensor inputs.
        {std::make_pair(1, other_arguments)},         // Tensor list inputs.
        {std::make_pair(kFunc, f_attr),
         std::make_pair(kTarguments, other_arguments_types_attr),
         std::make_pair(kUseInterOpParallelism, use_inter_op_parallelism_attr),
         std::make_pair(kPreserveCardinality,
                        preserve_cardinality_attr)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_10(mht_10_v, 359, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_11(mht_11_v, 364, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "Initialize");

      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_12(mht_12_v, 376, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "GetNextInternal");

      // NOTE(mrry): This method is thread-safe as long as
      // `input_impl_` and `f` are thread-safe. However, if multiple
      // threads enter this method, outputs may be observed in a
      // non-deterministic order.

      std::vector<Tensor> args;
      TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &args, end_of_sequence));
      if (*end_of_sequence) {
        return Status::OK();
      }

      Status s = instantiated_captured_func_->Run(ctx, std::move(args),
                                                  out_tensors, model_node());
      if (errors::IsOutOfRange(s)) {
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
      } else {
        return s;
      }
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args), /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_13(mht_13_v, 419, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "SaveInternal");

      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_14(mht_14_v, 430, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "RestoreInternal");

      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return Status::OK();
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const bool preserve_cardinality_;
  const std::unique_ptr<CapturedFunction> captured_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  // This is used for random access provided by Get().
  mutable std::unique_ptr<InstantiatedCapturedFunction>
      instantiated_captured_func_;
};

MapDatasetOp::MapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_15(mht_15_v, 454, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "MapDatasetOp::MapDatasetOp");

  FunctionMetadata::Params params;
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kUseInterOpParallelism,
                                   &params.use_inter_op_parallelism));
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFunc, params, &func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kPreserveCardinality, &preserve_cardinality_));
}

void MapDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                               DatasetBase** output) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmap_dataset_opDTcc mht_16(mht_16_v, 470, "", "./tensorflow/core/kernels/data/map_dataset_op.cc", "MapDatasetOp::MakeDataset");

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  *output = new Dataset(ctx, input, std::move(captured_func), output_types_,
                        output_shapes_, preserve_cardinality_);
}

namespace {

REGISTER_KERNEL_BUILDER(Name("MapDataset").Device(DEVICE_CPU), MapDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalMapDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        MapDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("MapDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
