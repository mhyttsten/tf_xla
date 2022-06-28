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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.h"

#include <map>
#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/rewrite_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace data {
namespace experimental {

/* static */ constexpr char AssertPrevDatasetOp::kInputDataset[];
/* static */ constexpr char AssertPrevDatasetOp::kDatasetType[];
/* static */ constexpr char AssertPrevDatasetOp::kTransformations[];
/* static */ constexpr char AssertPrevDatasetOp::kOutputTypes[];
/* static */ constexpr char AssertPrevDatasetOp::kOutputShapes[];

namespace {

// Returns a `NameAttrList` of an op name and attrs, parsed from
// `transformation`.
StatusOr<NameAttrList> GetAssertions(const tstring& transformation) {
  NameAttrList assertions;
  if (!std::is_base_of<protobuf::Message, NameAttrList>()) {
    return errors::InvalidArgument(
        "Portable proto implementations are not supported.");
  }
  if (!protobuf::TextFormat::ParseFromString(
          transformation, reinterpret_cast<protobuf::Message*>(&assertions))) {
    return errors::InvalidArgument("Couldn't parse transformation '",
                                   transformation, "'.");
  }
  return assertions;
}

// Returns `dataset`'s input dataset.
StatusOr<const DatasetBase*> GetPreviousDataset(const DatasetBase& dataset) {
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(dataset.InputDatasets(&inputs));
  if (inputs.empty()) {
    return errors::InvalidArgument("No previous transformation found.");
  }
  return inputs.back();
}

// Checks `dataset`'s op name against that in `assertions`.
Status CheckOpName(const DatasetBase& dataset, const NameAttrList& assertions) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_0(mht_0_v, 240, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "CheckOpName");

  if (!MatchesAnyVersion(assertions.name(), dataset.type_string())) {
    return errors::InvalidArgument("Asserted transformation matching '",
                                   assertions.name(), "', but found '",
                                   dataset.type_string(), "'.");
  }
  return Status::OK();
}

// Returns a NodeDef representation of `dataset`.
StatusOr<NodeDef> GetDatasetNode(const DatasetBase& dataset,
                                 absl::string_view op_name) {
  SerializationContext serialization_ctx((SerializationContext::Params()));
  GraphDefBuilder b;
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
      AsGraphDef(&dataset, std::move(serialization_ctx), &graph_def));
  TF_ASSIGN_OR_RETURN(NodeDef node, GetDatasetNodeDef(graph_def));
  return node;
}

// Checks `dataset`'s attrs against those in `assertions`.
Status CheckAttributes(const DatasetBase& dataset,
                       const NameAttrList& assertions) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_1(mht_1_v, 266, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "CheckAttributes");

  if (assertions.attr().empty()) return Status::OK();
  TF_ASSIGN_OR_RETURN(NodeDef node, GetDatasetNode(dataset, assertions.name()));
  std::vector<std::string> attrs_not_found;
  for (const auto& attr : assertions.attr()) {
    auto it = node.attr().find(attr.first);
    if (it != node.attr().end()) {
      if (!std::is_base_of<protobuf::Message, AttrValue>()) {
        return errors::InvalidArgument(
            "Portable proto implementations are not supported.");
      }
      if (!protobuf::util::MessageDifferencer::Equivalent(
              *reinterpret_cast<const protobuf::Message*>(&it->second),
              *reinterpret_cast<const protobuf::Message*>(&attr.second))) {
        return errors::InvalidArgument(
            "Asserted attribute '", attr.first, "' having a value of '",
            attr.second.DebugString(), "', but found value of '",
            it->second.DebugString(), "'.");
      }
    } else {
      return errors::InvalidArgument(
          "Asserted attribute '", attr.first, "' having a value of '",
          attr.second.DebugString(), "', but found no such attribute defined.");
    }
  }
  return Status::OK();
}

// Checks `dataset`'s op name and attrs against those in `transformation`.
Status CheckTransformation(const DatasetBase& dataset,
                           const tstring& transformation) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("transformation: \"" + (std::string)transformation + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_2(mht_2_v, 300, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "CheckTransformation");

  TF_ASSIGN_OR_RETURN(NameAttrList assertions, GetAssertions(transformation));
  TF_RETURN_IF_ERROR(CheckOpName(dataset, assertions));
  TF_RETURN_IF_ERROR(CheckAttributes(dataset, assertions));
  return Status::OK();
}

}  // namespace

class AssertPrevDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          const std::vector<tstring>& transformations,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        transformations_(transformations),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_3(mht_3_v, 326, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_4(mht_4_v, 337, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "output_dtypes");
 return output_types_; }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_5(mht_5_v, 341, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "output_shapes");

    return output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_6(mht_6_v, 348, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_7(mht_7_v, 355, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "CardinalityInternal");
 return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_8(mht_8_v, 360, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_9(mht_9_v, 368, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_10(mht_10_v, 378, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* transformations_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(transformations_, &transformations_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, transformations_node}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_11(mht_11_v, 395, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_12(mht_12_v, 400, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "Initialize");

      const DatasetBase* current_dataset = dataset();
      for (int i = 0; i < dataset()->transformations_.size(); ++i) {
        StatusOr<const DatasetBase*> previous_dataset =
            GetPreviousDataset(*current_dataset);
        if (!previous_dataset.ok()) {
          return errors::InvalidArgument(
              "Asserted previous ", dataset()->transformations_.size(),
              " transformations but encountered only ", i, ".");
        }

        Status s = CheckTransformation(**previous_dataset,
                                       dataset()->transformations_[i]);
        if (!s.ok()) {
          return errors::InvalidArgument(
              "Failure checking transformations at offset ", i, ": ",
              s.error_message());
        }

        current_dataset = *previous_dataset;
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_13(mht_13_v, 429, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "GetNextInternal");

      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_14(mht_14_v, 444, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "SaveInternal");

      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_15(mht_15_v, 453, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "RestoreInternal");

      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      return Status::OK();
    }

   private:
    std::unique_ptr<IteratorBase> input_impl_;
  };

  const DatasetBase* input_;
  const std::vector<tstring> transformations_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

AssertPrevDatasetOp::AssertPrevDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_16(mht_16_v, 472, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "AssertPrevDatasetOp::AssertPrevDatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void AssertPrevDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                      DatasetBase** output) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSassert_prev_dataset_opDTcc mht_17(mht_17_v, 481, "", "./tensorflow/core/kernels/data/experimental/assert_prev_dataset_op.cc", "AssertPrevDatasetOp::MakeDataset");

  std::vector<tstring> transformations;
  OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kTransformations,
                                                   &transformations));
  *output =
      new Dataset(ctx, input, transformations, output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("AssertPrevDataset").Device(DEVICE_CPU),
                        AssertPrevDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
