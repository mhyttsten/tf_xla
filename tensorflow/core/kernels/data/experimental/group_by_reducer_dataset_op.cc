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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc() {
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
#include <map>

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

class GroupByReducerDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GroupByReducerDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "GroupByReducerDatasetOp");

    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, "key_func", /*params=*/{},
                                                 &key_func_metadata_));
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "init_func", /*params=*/{},
                                            &init_func_metadata_));
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "reduce_func", /*params=*/{},
                                            &reduce_func_metadata_));
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "finalize_func", /*params=*/{},
                                            &finalize_func_metadata_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "MakeDataset");

    std::unique_ptr<CapturedFunction> captured_key_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, key_func_metadata_,
                                                 "key_func_other_arguments",
                                                 &captured_key_func));
    std::unique_ptr<CapturedFunction> captured_init_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, init_func_metadata_,
                                                 "init_func_other_arguments",
                                                 &captured_init_func));
    std::unique_ptr<CapturedFunction> captured_reduce_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, reduce_func_metadata_,
                                                 "reduce_func_other_arguments",
                                                 &captured_reduce_func));
    std::unique_ptr<CapturedFunction> captured_finalize_func;
    OP_REQUIRES_OK(ctx,
                   CapturedFunction::Create(ctx, finalize_func_metadata_,
                                            "finalize_func_other_arguments",
                                            &captured_finalize_func));

    *output = new Dataset(
        ctx, input, std::move(captured_key_func), std::move(captured_init_func),
        std::move(captured_reduce_func), std::move(captured_finalize_func),
        output_types_, output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_init_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            std::unique_ptr<CapturedFunction> captured_finalize_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_key_func_(std::move(captured_key_func)),
          captured_init_func_(std::move(captured_init_func)),
          captured_reduce_func_(std::move(captured_reduce_func)),
          captured_finalize_func_(std::move(captured_finalize_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_2(mht_2_v, 268, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::GroupByReducer")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_4(mht_4_v, 286, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "output_dtypes");

      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_5(mht_5_v, 292, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_6(mht_6_v, 299, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "DebugString");

      return "GroupByReducerDatasetOp::Dataset";
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_7(mht_7_v, 306, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "CheckExternalState");

      TF_RETURN_IF_ERROR(captured_key_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_init_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_reduce_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_finalize_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_8(mht_8_v, 320, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      std::vector<Node*> key_func_other_arguments_node;
      DataTypeVector key_func_other_arguments_types;
      TF_RETURN_IF_ERROR(
          captured_key_func_->AddToGraph(ctx, b, &key_func_other_arguments_node,
                                         &key_func_other_arguments_types));

      std::vector<Node*> init_func_other_arguments_node;
      DataTypeVector init_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_init_func_->AddToGraph(
          ctx, b, &init_func_other_arguments_node,
          &init_func_other_arguments_types));

      std::vector<Node*> reduce_func_other_arguments_node;
      DataTypeVector reduce_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_reduce_func_->AddToGraph(
          ctx, b, &reduce_func_other_arguments_node,
          &reduce_func_other_arguments_types));

      std::vector<Node*> finalize_func_other_arguments_node;
      DataTypeVector finalize_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_finalize_func_->AddToGraph(
          ctx, b, &finalize_func_other_arguments_node,
          &finalize_func_other_arguments_types));

      AttrValue key_func;
      b->BuildAttrValue(captured_key_func_->func(), &key_func);
      AttrValue init_func;
      b->BuildAttrValue(captured_init_func_->func(), &init_func);
      AttrValue reduce_func;
      b->BuildAttrValue(captured_reduce_func_->func(), &reduce_func);
      AttrValue finalize_func;
      b->BuildAttrValue(captured_finalize_func_->func(), &finalize_func);

      AttrValue key_func_other_arguments_types_attr;
      b->BuildAttrValue(key_func_other_arguments_types,
                        &key_func_other_arguments_types_attr);
      AttrValue init_func_other_arguments_types_attr;
      b->BuildAttrValue(init_func_other_arguments_types,
                        &init_func_other_arguments_types_attr);
      AttrValue reduce_func_other_arguments_types_attr;
      b->BuildAttrValue(reduce_func_other_arguments_types,
                        &reduce_func_other_arguments_types_attr);
      AttrValue finalize_func_other_arguments_types_attr;
      b->BuildAttrValue(finalize_func_other_arguments_types,
                        &finalize_func_other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {{0, input_graph_node}},
          {{1, key_func_other_arguments_node},
           {2, init_func_other_arguments_node},
           {3, reduce_func_other_arguments_node},
           {4, finalize_func_other_arguments_node}},
          {{"key_func", key_func},
           {"init_func", init_func},
           {"reduce_func", reduce_func},
           {"finalize_func", finalize_func},
           {"Tkey_func_other_arguments", key_func_other_arguments_types_attr},
           {"Tinit_func_other_arguments", init_func_other_arguments_types_attr},
           {"Treduce_func_other_arguments",
            reduce_func_other_arguments_types_attr},
           {"Tfinalize_func_other_arguments",
            finalize_func_other_arguments_types_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_9(mht_9_v, 397, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_10(mht_10_v, 402, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "Initialize");

        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(dataset()->captured_key_func_->Instantiate(
            ctx, &instantiated_key_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_init_func_->Instantiate(
            ctx, &instantiated_init_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Instantiate(
            ctx, &instantiated_reduce_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_finalize_func_->Instantiate(
            ctx, &instantiated_finalize_func_));
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_11(mht_11_v, 421, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);

        // Iterate through the input dataset, keying input elements to reducers.
        while (!end_of_input_) {
          std::vector<Tensor> next_input_element;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &next_input_element, &end_of_input_));

          if (!end_of_input_) {
            // Run the key function on the input element.
            std::vector<Tensor> key_func_output;
            TF_RETURN_IF_ERROR(instantiated_key_func_->RunWithBorrowedArgs(
                ctx, next_input_element, &key_func_output, model_node()));

            if (key_func_output.size() != 1 ||
                key_func_output[0].dtype() != DT_INT64 ||
                key_func_output[0].NumElements() != 1) {
              // TODO(b/78665031): Support non-int64 keys.
              return errors::InvalidArgument(
                  "`key_func` must return a scalar int64.");
            }
            const int64_t key = key_func_output[0].scalar<int64_t>()();

            if (states_.find(key) == states_.end()) {
              // Run the init function to create the initial state.
              std::vector<Tensor> init_func_output;
              TF_RETURN_IF_ERROR(instantiated_init_func_->Run(
                  ctx, std::move(key_func_output), &init_func_output,
                  model_node()));
              states_[key] = init_func_output;
            }

            // Run the reduce function to update the current state.
            std::vector<Tensor> args;
            args.reserve(states_[key].size() + next_input_element.size());
            std::copy(states_[key].begin(), states_[key].end(),
                      std::back_inserter(args));
            std::copy(next_input_element.begin(), next_input_element.end(),
                      std::back_inserter(args));

            std::vector<Tensor> reduce_func_output;
            TF_RETURN_IF_ERROR(instantiated_reduce_func_->Run(
                ctx, std::move(args), &reduce_func_output, model_node()));
            states_[key] = reduce_func_output;
          } else {
            keys_.resize(states_.size());
            int idx = 0;
            for (auto it = states_.begin(); it != states_.end(); ++idx, ++it) {
              keys_[idx] = it->first;
            }
          }
        }

        if (keys_index_ == keys_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }
        TF_RETURN_IF_ERROR(instantiated_finalize_func_->RunWithBorrowedArgs(
            ctx, states_[keys_[keys_index_++]], out_tensors, model_node()));
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
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_12(mht_12_v, 495, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "SaveInternal");

        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_key_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_init_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_reduce_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_finalize_func_->CheckExternalState()));
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));

        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }

        // Saving states_.
        if (!states_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("states_size"), states_.size()));
          int idx = 0;
          for (auto it = states_.begin(); it != states_.end(); ++idx, ++it) {
            int64_t key = it->first;
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("states[", idx, "]->key")), key));
            if (!it->second.empty()) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  full_name(strings::StrCat("states[", idx, "]->state_size")),
                  it->second.size()));
              for (int j = 0; j < it->second.size(); ++j) {
                TF_RETURN_IF_ERROR(writer->WriteTensor(
                    full_name(
                        strings::StrCat("states[", idx, "]->state[", j, "]")),
                    it->second[j]));
              }
            }
          }
        }

        // Saving keys_index_ and keys_.
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("keys_index"), keys_index_));
          if (!keys_.empty()) {
            TF_RETURN_IF_ERROR(
                writer->WriteScalar(full_name("keys_size"), keys_.size()));
            for (int idx = 0; idx < keys_.size(); ++idx) {
              TF_RETURN_IF_ERROR(writer->WriteScalar(
                  full_name(strings::StrCat("keys[", idx, "]")), keys_[idx]));
            }
          }
        }

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_reducer_dataset_opDTcc mht_13(mht_13_v, 556, "", "./tensorflow/core/kernels/data/experimental/group_by_reducer_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;

        // Restoring states_.
        if (reader->Contains(full_name("states_size"))) {
          int64_t size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("states_size"), &size));
          for (int idx = 0; idx < size; ++idx) {
            int64_t key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("states[", idx, "]->key")), &key));
            std::vector<Tensor> state;
            if (reader->Contains(full_name(
                    strings::StrCat("states[", idx, "]->state_size")))) {
              int64_t state_size;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat("states[", idx, "]->state_size")),
                  &state_size));
              state.resize(state_size);
              for (int j = 0; j < state_size; ++j) {
                TF_RETURN_IF_ERROR(reader->ReadTensor(
                    ctx->flr(),
                    full_name(
                        strings::StrCat("states[", idx, "]->state[", j, "]")),
                    &state[j]));
              }
            }
            states_[key] = state;
          }
        }

        // Restoring keys_index_ and keys_.
        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("keys_index"), &keys_index_));
          if (reader->Contains(full_name("keys_size"))) {
            int64_t size;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(full_name("keys_size"), &size));
            keys_.resize(size);
            for (int idx = 0; idx < size; ++idx) {
              int64_t key;
              TF_RETURN_IF_ERROR(reader->ReadScalar(
                  full_name(strings::StrCat("keys[", idx, "]")), &key));
              keys_[idx] = key;
            }
          }
        }

        return Status::OK();
      }

     private:
      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      bool end_of_input_ TF_GUARDED_BY(mu_) = false;
      std::map<int64_t, std::vector<Tensor>> states_ TF_GUARDED_BY(mu_);
      std::vector<int64_t> keys_ TF_GUARDED_BY(mu_);
      int64_t keys_index_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_key_func_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_init_func_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_reduce_func_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_finalize_func_;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_init_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const std::unique_ptr<CapturedFunction> captured_finalize_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  std::shared_ptr<FunctionMetadata> key_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> init_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> reduce_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> finalize_func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("GroupByReducerDataset").Device(DEVICE_CPU),
                        GroupByReducerDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalGroupByReducerDataset").Device(DEVICE_CPU),
    GroupByReducerDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("GroupByReducerDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalGroupByReducerDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
