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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc() {
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
#include <map>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/window_dataset.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class GroupByWindowDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit GroupByWindowDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "GroupByWindowDatasetOp");

    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, "key_func", /*params=*/{},
                                                 &key_func_metadata_));
    OP_REQUIRES_OK(ctx,
                   FunctionMetadata::Create(ctx, "reduce_func", /*params=*/{},
                                            &reduce_func_metadata_));
    OP_REQUIRES_OK(
        ctx, FunctionMetadata::Create(ctx, "window_size_func", /*params=*/{},
                                      &window_size_func_metadata_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "MakeDataset");

    std::unique_ptr<CapturedFunction> captured_key_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, key_func_metadata_,
                                                 "key_func_other_arguments",
                                                 &captured_key_func));

    std::unique_ptr<CapturedFunction> captured_reduce_func;
    OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, reduce_func_metadata_,
                                                 "reduce_func_other_arguments",
                                                 &captured_reduce_func));

    std::unique_ptr<CapturedFunction> captured_window_size_func;
    OP_REQUIRES_OK(ctx,
                   CapturedFunction::Create(ctx, window_size_func_metadata_,
                                            "window_size_func_other_arguments",
                                            &captured_window_size_func));

    *output = new Dataset(ctx, input, std::move(captured_key_func),
                          std::move(captured_reduce_func),
                          std::move(captured_window_size_func), output_types_,
                          output_shapes_);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            std::unique_ptr<CapturedFunction> captured_key_func,
            std::unique_ptr<CapturedFunction> captured_reduce_func,
            std::unique_ptr<CapturedFunction> captured_window_size_func,
            const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          captured_key_func_(std::move(captured_key_func)),
          captured_reduce_func_(std::move(captured_reduce_func)),
          captured_window_size_func_(std::move(captured_window_size_func)),
          output_types_(output_types),
          output_shapes_(output_shapes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_2(mht_2_v, 262, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_3(mht_3_v, 269, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::GroupByWindow")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "output_dtypes");

      return output_types_;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_5(mht_5_v, 286, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "output_shapes");

      return output_shapes_;
    }

    string DebugString() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "DebugString");

      return "GroupByWindowDatasetOp::Dataset";
    }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_7(mht_7_v, 300, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "CardinalityInternal");

      int64_t n = input_->Cardinality();
      if (n == kInfiniteCardinality) {
        return n;
      }
      return kUnknownCardinality;
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_8(mht_8_v, 312, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_9(mht_9_v, 320, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "CheckExternalState");

      TF_RETURN_IF_ERROR(captured_key_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_reduce_func_->CheckExternalState());
      TF_RETURN_IF_ERROR(captured_window_size_func_->CheckExternalState());
      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_10(mht_10_v, 333, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      std::vector<Node*> key_func_other_arguments_node;
      DataTypeVector key_func_other_arguments_types;
      TF_RETURN_IF_ERROR(
          captured_key_func_->AddToGraph(ctx, b, &key_func_other_arguments_node,
                                         &key_func_other_arguments_types));

      std::vector<Node*> reduce_func_other_arguments_node;
      DataTypeVector reduce_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_reduce_func_->AddToGraph(
          ctx, b, &reduce_func_other_arguments_node,
          &reduce_func_other_arguments_types));

      std::vector<Node*> window_size_func_other_arguments_node;
      DataTypeVector window_size_func_other_arguments_types;
      TF_RETURN_IF_ERROR(captured_window_size_func_->AddToGraph(
          ctx, b, &window_size_func_other_arguments_node,
          &window_size_func_other_arguments_types));

      AttrValue key_func;
      b->BuildAttrValue(captured_key_func_->func(), &key_func);
      AttrValue reduce_func;
      b->BuildAttrValue(captured_reduce_func_->func(), &reduce_func);
      AttrValue window_size_func;
      b->BuildAttrValue(captured_window_size_func_->func(), &window_size_func);

      AttrValue key_func_other_arguments_types_attr;
      b->BuildAttrValue(key_func_other_arguments_types,
                        &key_func_other_arguments_types_attr);
      AttrValue reduce_func_other_arguments_types_attr;
      b->BuildAttrValue(reduce_func_other_arguments_types,
                        &reduce_func_other_arguments_types_attr);
      AttrValue window_size_func_other_arguments_types_attr;
      b->BuildAttrValue(window_size_func_other_arguments_types,
                        &window_size_func_other_arguments_types_attr);

      TF_RETURN_IF_ERROR(b->AddDataset(
          this, {{0, input_graph_node}},
          {{1, key_func_other_arguments_node},
           {2, reduce_func_other_arguments_node},
           {3, window_size_func_other_arguments_node}},
          {{"key_func", key_func},
           {"reduce_func", reduce_func},
           {"window_size_func", window_size_func},
           {"Tkey_func_other_arguments", key_func_other_arguments_types_attr},
           {"Treduce_func_other_arguments",
            reduce_func_other_arguments_types_attr},
           {"Twindow_size_func_other_arguments",
            window_size_func_other_arguments_types_attr}},
          output));
      return Status::OK();
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_11(mht_11_v, 396, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "Iterator");
}

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_12(mht_12_v, 401, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "Initialize");

        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(dataset()->captured_key_func_->Instantiate(
            ctx, &instantiated_key_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_reduce_func_->Instantiate(
            ctx, &instantiated_reduce_func_));
        TF_RETURN_IF_ERROR(dataset()->captured_window_size_func_->Instantiate(
            ctx, &instantiated_window_size_func_));
        return Status::OK();
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_13(mht_13_v, 418, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        do {
          if (current_group_iterator_) {
            // We are currently processing a group, so try to get the
            // next element.
            bool end_of_group;
            TF_RETURN_IF_ERROR(current_group_iterator_->GetNext(
                MakeNestedIteratorContext(ctx), out_tensors, &end_of_group));
            if (!end_of_group) {
              // Produce the subelement as output.
              *end_of_sequence = false;
              return Status::OK();
            }
            // We have reached the end of the current group, so maybe move on
            // to the next group.
            current_group_iterator_.reset();
            groups_.erase(current_key_);
          }

          // Iterate through the input dataset until we get a full
          // group, or reach the end.
          while (!end_of_input_) {
            std::vector<Tensor> next_input_element;
            TF_RETURN_IF_ERROR(
                input_impl_->GetNext(MakeNestedIteratorContext(ctx),
                                     &next_input_element, &end_of_input_));

            if (!end_of_input_) {
              // Run the key function on the input element to identify its
              // group.
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

              if (window_sizes_.find(key) == window_sizes_.end()) {
                // Run the window size function on the key to identify its
                // window size.
                std::vector<Tensor> window_size_func_output;
                TF_RETURN_IF_ERROR(instantiated_window_size_func_->Run(
                    ctx, std::move(key_func_output), &window_size_func_output,
                    model_node()));

                if (window_size_func_output.size() != 1 ||
                    window_size_func_output[0].dtype() != DT_INT64 ||
                    window_size_func_output[0].NumElements() != 1) {
                  // TODO(mrry): Support non-int64 window sizes.
                  return errors::InvalidArgument(
                      "`window_size_func` must return a scalar int64.");
                }
                const int64_t window_size =
                    window_size_func_output[0].scalar<int64_t>()();
                if (window_size <= 0) {
                  return errors::InvalidArgument(
                      "Window size must be greater than zero, but got ",
                      window_size, ".");
                }
                window_sizes_[key] = window_size;
              }

              const int64_t window_size = window_sizes_[key];

              std::vector<std::vector<Tensor>>& group = groups_[key];
              group.push_back(std::move(next_input_element));

              if (group.size() == window_size) {
                current_key_ = key;
                TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, key));
                break;
              }
            }
          }

          if (end_of_input_) {
            if (!groups_.empty()) {
              // We have consumed all of the input, so flush an
              // arbitrarily chosen group.
              current_key_ = groups_.begin()->first;
              TF_RETURN_IF_ERROR(
                  StartFlushingGroup(ctx, groups_.begin()->first));
            }
          }
        } while (current_group_iterator_ || !end_of_input_);

        *end_of_sequence = true;
        return Status::OK();
      }

     protected:
      std::shared_ptr<model::Node> CreateNode(
          IteratorContext* ctx, model::Node::Args args) const override {
        return model::MakeUnknownRatioNode(std::move(args));
      }

      Status SaveInternal(SerializationContext* ctx,
                          IteratorStateWriter* writer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_14(mht_14_v, 525, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "SaveInternal");

        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_key_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_reduce_func_->CheckExternalState()));
        TF_RETURN_IF_ERROR(ctx->HandleCheckExternalStateStatus(
            dataset()->captured_window_size_func_->CheckExternalState()));
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));

        if (end_of_input_) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("end_of_input"), ""));
        }

        // Saving groups_
        if (!groups_.empty()) {
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("groups_size"), groups_.size()));
          int idx = 0;
          for (auto it = groups_.begin(); it != groups_.end(); it++) {
            int64_t key = it->first;
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("groups_[", idx, "]->key")), key));
            TF_RETURN_IF_ERROR(SaveGroup(
                writer, full_name(strings::StrCat("groups_[", idx, "]")),
                it->second));
            idx++;
          }
        }

        // Saving window_sizes_
        if (!window_sizes_.empty()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("window_sizes_size"),
                                                 window_sizes_.size()));
          int idx = 0;
          for (auto it = window_sizes_.begin(); it != window_sizes_.end();
               it++) {
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->key")),
                it->first));
            TF_RETURN_IF_ERROR(writer->WriteScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->value")),
                it->second));
            idx++;
          }
        }

        if (current_group_iterator_) {
          TF_RETURN_IF_ERROR(SaveInput(ctx, writer, current_group_iterator_));

          // Saving current_key_
          TF_RETURN_IF_ERROR(
              writer->WriteScalar(full_name("current_key"), current_key_));
        } else {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name("current_iterator_not_initialized"), ""));
        }
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("group_counter"),
                                               group_counter_ - 1));
        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_15(mht_15_v, 592, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "RestoreInternal");

        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

        if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;

        // Restoring groups_
        if (reader->Contains(full_name("groups_size"))) {
          int64_t size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("groups_size"), &size));
          for (int idx = 0; idx < size; idx++) {
            int64_t key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("groups_[", idx, "]->key")), &key));
            std::vector<std::vector<Tensor>> group;
            TF_RETURN_IF_ERROR(RestoreGroup(
                ctx, reader, full_name(strings::StrCat("groups_[", idx, "]")),
                &group));
            groups_[key] = group;
          }
        }

        // Restoring window_sizes_
        if (reader->Contains(full_name("window_sizes_size"))) {
          int64_t size;
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("window_sizes_size"), &size));
          for (int idx = 0; idx < size; idx++) {
            int64_t key;
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->key")),
                &key));
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("window_sizes_[", idx, "]->value")),
                &window_sizes_[key]));
          }
        }

        // Group counter needs to be restored before current group iterator.
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(full_name("group_counter"), &group_counter_));

        if (reader->Contains(full_name("current_iterator_not_initialized"))) {
          current_group_iterator_.reset();
        } else {
          // Restore current_key_
          TF_RETURN_IF_ERROR(
              reader->ReadScalar(full_name("current_key"), &current_key_));

          // Initialize current_group_iterator_
          TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, current_key_));
          // Restore current_group_iterator_ state
          TF_RETURN_IF_ERROR(
              RestoreInput(ctx, reader, current_group_iterator_));
        }
        return Status::OK();
      }

     private:
      Status SaveGroup(IteratorStateWriter* writer, const string& name,
                       const std::vector<std::vector<Tensor>>& group)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_16(mht_16_v, 658, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "SaveGroup");

        TF_RETURN_IF_ERROR(
            writer->WriteScalar(strings::StrCat(name, "_size"), group.size()));
        for (int i = 0; i < group.size(); i++) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              strings::StrCat(name, "[", i, "]_size"), group[i].size()));
          for (int j = 0; j < group[i].size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                strings::StrCat(name, "[", i, "][", j, "]"), group[i][j]));
          }
        }
        return Status::OK();
      }

      Status RestoreGroup(IteratorContext* ctx, IteratorStateReader* reader,
                          const string& name,
                          std::vector<std::vector<Tensor>>* group)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_17(mht_17_v, 679, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "RestoreGroup");

        int64_t group_size;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(strings::StrCat(name, "_size"), &group_size));
        group->resize(group_size);
        for (int i = 0; i < group_size; i++) {
          int64_t vector_size;
          TF_RETURN_IF_ERROR(reader->ReadScalar(
              strings::StrCat(name, "[", i, "]_size"), &vector_size));
          group->at(i).resize(vector_size);
          for (int j = 0; j < vector_size; j++) {
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                ctx->flr(), strings::StrCat(name, "[", i, "][", j, "]"),
                &group->at(i)[j]));
          }
        }
        return Status::OK();
      }

      Status StartFlushingGroup(IteratorContext* ctx, int64_t key)
          TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSgroup_by_window_dataset_opDTcc mht_18(mht_18_v, 702, "", "./tensorflow/core/kernels/data/experimental/group_by_window_dataset_op.cc", "StartFlushingGroup");

        DatasetBase* group_dataset;
        TF_RETURN_IF_ERROR(
            NewWindow(groups_[key], dataset()->input_->output_dtypes(),
                      dataset()->input_->output_shapes(), &group_dataset));

        Tensor key_arg(DT_INT64, TensorShape({}));
        key_arg.scalar<int64_t>()() = key;

        Tensor group_dataset_arg(DT_VARIANT, TensorShape({}));
        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(group_dataset, &group_dataset_arg));

        std::vector<Tensor> args(
            {std::move(key_arg), std::move(group_dataset_arg)});
        std::vector<Tensor> return_values;
        // If not restoring, pass the model node of this iterator in order to
        // exclude captured function run time from being added to the processing
        // time of the node. If restoring, pass nullptr to not record processing
        // time because iterator modeling is only used to model Iterator's
        // GetNext() resource usage.
        TF_RETURN_IF_ERROR(instantiated_reduce_func_->Run(
            ctx, std::move(args), &return_values,
            ctx->is_restoring() ? nullptr : model_node()));

        if (!(return_values.size() == 1 &&
              return_values[0].dtype() == DT_VARIANT &&
              TensorShapeUtils::IsScalar(return_values[0].shape()))) {
          return errors::InvalidArgument(
              "`reduce_func` must return a single scalar of dtype "
              "DT_VARIANT.");
        }

        // Retrieve the dataset that was created in `f`.
        // `returned_dataset` is borrowed from the `return_values[0]`.
        DatasetBase* returned_dataset;
        TF_RETURN_IF_ERROR(
            GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

        // Create an iterator for the dataset that was returned by `f`.
        return returned_dataset->MakeIterator(
            MakeNestedIteratorContext(ctx), this,
            strings::StrCat(prefix(), "[", group_counter_++, "]"),
            &current_group_iterator_);
      }

      mutex mu_;
      int64_t group_counter_ TF_GUARDED_BY(mu_) = 0;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      // TODO(mrry): Optimize for dense key space if appropriate.
      bool end_of_input_ TF_GUARDED_BY(mu_) = false;
      int64_t current_key_ TF_GUARDED_BY(mu_);
      std::map<int64_t, std::vector<std::vector<Tensor>>> groups_
          TF_GUARDED_BY(mu_);
      std::unique_ptr<IteratorBase> current_group_iterator_ TF_GUARDED_BY(mu_);
      std::map<int64_t, int64_t> window_sizes_ TF_GUARDED_BY(mu_);
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_key_func_;
      std::unique_ptr<InstantiatedCapturedFunction> instantiated_reduce_func_;
      std::unique_ptr<InstantiatedCapturedFunction>
          instantiated_window_size_func_;
    };

    const DatasetBase* const input_;
    const std::unique_ptr<CapturedFunction> captured_key_func_;
    const std::unique_ptr<CapturedFunction> captured_reduce_func_;
    const std::unique_ptr<CapturedFunction> captured_window_size_func_;
    const DataTypeVector output_types_;
    const std::vector<PartialTensorShape> output_shapes_;
  };

  std::shared_ptr<FunctionMetadata> key_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> reduce_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> window_size_func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name("GroupByWindowDataset").Device(DEVICE_CPU),
                        GroupByWindowDatasetOp);
REGISTER_KERNEL_BUILDER(
    Name("ExperimentalGroupByWindowDataset").Device(DEVICE_CPU),
    GroupByWindowDatasetOp);

REGISTER_INPUT_COLOCATION_EXEMPTION("GroupByWindowDataset");
REGISTER_INPUT_COLOCATION_EXEMPTION("ExperimentalGroupByWindowDataset");

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
