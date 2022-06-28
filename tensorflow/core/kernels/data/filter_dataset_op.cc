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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/filter_dataset_op.h"

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/stats_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const FilterDatasetOp::kDatasetType;
/* static */ constexpr const char* const FilterDatasetOp::kInputDataset;
/* static */ constexpr const char* const FilterDatasetOp::kOtherArguments;
/* static */ constexpr const char* const FilterDatasetOp::kPredicate;
/* static */ constexpr const char* const FilterDatasetOp::kTarguments;
/* static */ constexpr const char* const FilterDatasetOp::kOutputTypes;
/* static */ constexpr const char* const FilterDatasetOp::kOutputShapes;

constexpr char kInputImplsEmpty[] = "input_impls_empty";
constexpr char kFilteredElements[] = "filtered_elements";
constexpr char kDroppedElements[] = "dropped_elements";

class FilterDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_func)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_func_(std::move(captured_func)) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_0(mht_0_v, 227, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_3(mht_3_v, 252, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_4(mht_4_v, 259, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_5(mht_5_v, 267, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "CheckExternalState");

    TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_6(mht_6_v, 278, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    std::vector<Node*> other_arguments;
    DataTypeVector other_arguments_types;
    TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                  &other_arguments_types));
    AttrValue f;
    b->BuildAttrValue(captured_func_->func(), &f);
    AttrValue other_arguments_types_attr;
    b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {{0, input_graph_node}}, {{1, other_arguments}},
        {{kPredicate, f}, {kTarguments, other_arguments_types_attr}}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          filtered_elements_(0),
          dropped_elements_(0) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_8(mht_8_v, 310, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "Initialize");

      TF_RETURN_IF_ERROR(
          dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      return dataset()->captured_func_->Instantiate(
          ctx, &instantiated_captured_func_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_9(mht_9_v, 322, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "GetNextInternal");

      // NOTE(mrry): This method is thread-safe as long as
      // `input_impl_` and `f` are thread-safe. However, if multiple
      // threads enter this method, outputs may be observed in a
      // non-deterministic order.
      auto stats_aggregator = ctx->stats_aggregator();
      bool matched;
      do {
        {
          tf_shared_lock l(mu_);
          if (!input_impl_) {
            *end_of_sequence = true;
            return Status::OK();
          }
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        }
        if (*end_of_sequence) {
          mutex_lock l(mu_);
          input_impl_.reset();
          return Status::OK();
        }

        std::vector<Tensor> result;
        TF_RETURN_IF_ERROR(instantiated_captured_func_->RunWithBorrowedArgs(
            ctx, *out_tensors, &result, model_node()));

        if (result.size() != 1 || result[0].dtype() != DT_BOOL ||
            result[0].NumElements() != 1) {
          // Clear the output tensor list since there were errors with Filter
          // prediction result.
          out_tensors->clear();
          return errors::InvalidArgument(
              "Filter predicate `f` must return a scalar bool.");
        }
        matched = result[0].scalar<bool>()();

        if (!matched) {
          // Clear the output tensor list since it didn't match.
          out_tensors->clear();
          if (stats_aggregator) {
            mutex_lock l(mu_);
            dropped_elements_++;
            stats_aggregator->AddScalar(
                stats_utils::DroppedElementsScalarName(dataset()->node_name()),
                static_cast<float>(dropped_elements_), num_elements());

            stats_aggregator->IncrementCounter(dataset()->node_name(),
                                               stats_utils::kDroppedElements,
                                               static_cast<float>(1));
          }
        }
      } while (!matched);
      // TODO(shivaniagrawal): add ratio of dropped_elements and
      // filtered_elements as a histogram.
      if (stats_aggregator) {
        mutex_lock l(mu_);
        filtered_elements_++;
        stats_aggregator->AddScalar(
            stats_utils::FilterdElementsScalarName(dataset()->node_name()),
            static_cast<float>(filtered_elements_), num_elements());

        stats_aggregator->IncrementCounter(dataset()->node_name(),
                                           stats_utils::kFilteredElements,
                                           static_cast<float>(1));
      }
      *end_of_sequence = false;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_10(mht_10_v, 402, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "SaveInternal");

      TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
          dataset()->captured_func_->CheckExternalState()));
      mutex_lock l(mu_);
      if (input_impl_)
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      else
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kInputImplsEmpty), ""));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kFilteredElements),
                                             filtered_elements_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kDroppedElements), dropped_elements_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_11(mht_11_v, 422, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      if (reader->Contains(full_name(kInputImplsEmpty)))
        input_impl_.reset();
      else
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kFilteredElements),
                                            &filtered_elements_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kDroppedElements), &dropped_elements_));
      return Status::OK();
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    int64_t filtered_elements_ TF_GUARDED_BY(mu_);
    int64_t dropped_elements_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
  };

  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_func_;
};

FilterDatasetOp::FilterDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_12(mht_12_v, 451, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "FilterDatasetOp::FilterDatasetOp");

  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kPredicate, /*params=*/{},
                                               &func_metadata_));
  OP_REQUIRES(ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
              errors::InvalidArgument(
                  "predicate function has more than one return value."));
}

void FilterDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfilter_dataset_opDTcc mht_13(mht_13_v, 463, "", "./tensorflow/core/kernels/data/filter_dataset_op.cc", "FilterDatasetOp::MakeDataset");

  std::unique_ptr<CapturedFunction> captured_func;
  OP_REQUIRES_OK(ctx,
                 CapturedFunction::Create(ctx, func_metadata_, kOtherArguments,
                                          &captured_func));

  *output = new Dataset(ctx, input, std::move(captured_func));
}

namespace {

REGISTER_KERNEL_BUILDER(Name("FilterDataset").Device(DEVICE_CPU),
                        FilterDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("FilterDataset");

}  // namespace
}  // namespace data
}  // namespace tensorflow
