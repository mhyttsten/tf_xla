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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc() {
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
#include <iterator>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class TakeWhileDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit TakeWhileDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "TakeWhileDatasetOp");

    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(
                            ctx, "predicate", /*params=*/{}, &func_metadata_));
    OP_REQUIRES(ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
                errors::InvalidArgument(
                    "predicate function has more than one return value."));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_1(mht_1_v, 216, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "MakeDataset");

    std::unique_ptr<CapturedFunction> captured_func;
    OP_REQUIRES_OK(
        ctx, CapturedFunction::Create(ctx, func_metadata_, "other_arguments",
                                      &captured_func));
    *output = new Dataset(ctx, input, std::move(captured_func));
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_func)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_func_(std::move(captured_func)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_3(mht_3_v, 241, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return MakeUnique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::TakeWhile")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_4(mht_4_v, 252, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "output_dtypes");

      return input_->output_dtypes();
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_5(mht_5_v, 259, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "output_shapes");

      return input_->output_shapes();
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_6(mht_6_v, 266, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "DebugString");

      return "TakeWhileDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_7(mht_7_v, 273, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "CardinalityInternal");
 return kUnknownCardinality; }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_8(mht_8_v, 279, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_9(mht_9_v, 287, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "CheckExternalState");

      TF_RETURN_IF_ERROR(captured_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_10(mht_10_v, 298, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "AsGraphDefInternal");

      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_node));

      std::vector<Node*> other_arguments;
      DataTypeVector other_arguments_types;
      TF_RETURN_IF_ERROR(captured_func_->AddToGraph(ctx, b, &other_arguments,
                                                    &other_arguments_types));
      AttrValue f_attr;
      b->BuildAttrValue(captured_func_->func(), &f_attr);

      AttrValue other_arguments_types_attr;
      b->BuildAttrValue(other_arguments_types, &other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {std::make_pair(0, input_node)},
          {std::make_pair(1, other_arguments)},
          {std::make_pair("predicate", f_attr),
           std::make_pair("Targuments", other_arguments_types_attr)},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_11(mht_11_v, 328, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_12(mht_12_v, 333, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "Initialize");

        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        return dataset()->captured_func_->Instantiate(
            ctx, &instantiated_captured_func_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_13(mht_13_v, 345, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "GetNextInternal");

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
          return errors::InvalidArgument(
              "`predicate` must returns a scalar bool tensor.");
        }
        *end_of_sequence = !result[0].scalar<bool>()();
        if (*end_of_sequence) {
          mutex_lock l(mu_);
          input_impl_.reset();
          out_tensors->clear();
        }
        return Status::OK();
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_14(mht_14_v, 389, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "SaveInternal");

        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_func_->CheckExternalState()));
        mutex_lock l(mu_);
        if (input_impl_) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
        } else {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("input_impls_empty"), ""));
        }
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPStake_while_dataset_opDTcc mht_15(mht_15_v, 406, "", "./tensorflow/core/kernels/data/experimental/take_while_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        if (reader->Contains(full_name("input_impls_empty"))) {
          input_impl_.reset();
        } else {
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        }
        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_captured_func_;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_func_;
  };

  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
};

REGISTER_KERNEL_BUILDER(Name("TakeWhileDataset").Device(DEVICE_CPU),
                        TakeWhileDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalTakeWhileDataset").Device(DEVICE_CPU),
                        TakeWhileDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("TakeWhileDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalTakeWhileDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
