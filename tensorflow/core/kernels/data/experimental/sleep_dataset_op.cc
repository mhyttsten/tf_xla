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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc() {
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
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

class SleepDatasetOp : public UnaryDatasetOpKernel {
 public:
  using UnaryDatasetOpKernel::UnaryDatasetOpKernel;

  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "MakeDataset");

    int64_t sleep_microseconds;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, "sleep_microseconds",
                                                     &sleep_microseconds));

    OP_REQUIRES(ctx, sleep_microseconds >= 0,
                errors::InvalidArgument("`sleep_microseconds` must be >= 0"));

    *output = new Dataset(ctx, input, sleep_microseconds);
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const DatasetBase* input,
            int64_t sleep_microseconds)
        : DatasetBase(DatasetContext(ctx)),
          input_(input),
          sleep_microseconds_(sleep_microseconds) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_1(mht_1_v, 220, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "Dataset");

      input_->Ref();
    }

    ~Dataset() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_2(mht_2_v, 227, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "~Dataset");
 input_->Unref(); }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return absl::make_unique<Iterator>(
          Iterator::Params{this, strings::StrCat(prefix, "::Sleep")});
    }

    const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_3(mht_3_v, 238, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "output_dtypes");

      return input_->output_dtypes();
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_4(mht_4_v, 244, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "output_shapes");

      return input_->output_shapes();
    }

    string DebugString() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_5(mht_5_v, 251, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "DebugString");
 return "SleepDatasetOp::Dataset"; }

    int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_6(mht_6_v, 256, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "CardinalityInternal");

      return input_->Cardinality();
    }

    Status InputDatasets(
        std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_7(mht_7_v, 264, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "InputDatasets");

      inputs->push_back(input_);
      return Status::OK();
    }

    Status CheckExternalState() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_8(mht_8_v, 272, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "CheckExternalState");

      return input_->CheckExternalState();
    }

   protected:
    Status AsGraphDefInternal(SerializationContext* ctx,
                              DatasetGraphDefBuilder* b,
                              Node** output) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_9(mht_9_v, 282, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "AsGraphDefInternal");

      Node* input_graph_node = nullptr;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

      Node* sleep_microseconds = nullptr;
      TF_RETURN_IF_ERROR(
          b->AddScalar(sleep_microseconds_, &sleep_microseconds));

      return b->AddDataset(this,
                           {{0, input_graph_node},
                            {1, sleep_microseconds}},  // Single tensor inputs.
                           {},                         // Tensor list inputs.
                           {},                         // Attrs
                           output);
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_10(mht_10_v, 305, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "Iterator");
}

      ~Iterator() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_11(mht_11_v, 310, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "~Iterator");

        {
          mutex_lock l(mu_);
          cancelled_ = true;
        }
        if (deregister_fn_) {
          deregister_fn_();
        }
      }

      Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_12(mht_12_v, 323, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "Initialize");

        TF_RETURN_IF_ERROR(RegisterCancellationCallback(
            ctx->cancellation_manager(),
            [this]() {
              mutex_lock l(mu_);
              cancelled_ = true;
            },
            &deregister_fn_));
        return dataset()->input_->MakeIterator(ctx, this, prefix(),
                                               &input_impl_);
      }

      Status GetNextInternal(IteratorContext* ctx,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_13(mht_13_v, 340, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "GetNextInternal");

        mutex_lock l(mu_);
        RecordStop(ctx);
        bool cancelled = mu_.AwaitWithDeadline(
            Condition(&cancelled_),
            EnvTime::NowNanos() +
                dataset()->sleep_microseconds_ * EnvTime::kMicrosToNanos);
        RecordStart(ctx);
        if (cancelled) {
          return errors::Cancelled("Operation was cancelled");
        }
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_14(mht_14_v, 365, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "SaveInternal");

        return SaveInput(ctx, writer, input_impl_);
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSexperimentalPSsleep_dataset_opDTcc mht_15(mht_15_v, 373, "", "./tensorflow/core/kernels/data/experimental/sleep_dataset_op.cc", "RestoreInternal");

        return RestoreInput(ctx, reader, input_impl_);
      }

      mutex mu_;
      std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
      bool cancelled_ TF_GUARDED_BY(mu_) = false;
      std::function<void()> deregister_fn_;
    };

    const DatasetBase* const input_;
    // TODO(b/117612213): Investigate autotuning for this value.
    const int64_t sleep_microseconds_;
  };
};

REGISTER_KERNEL_BUILDER(Name("SleepDataset").Device(DEVICE_CPU),
                        SleepDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalSleepDataset").Device(DEVICE_CPU),
                        SleepDatasetOp);

REGISTER_KERNEL_BUILDER(Name("SleepDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("sleep_microseconds")
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        SleepDatasetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalSleepDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("sleep_microseconds")
                            .HostMemory("input_dataset")
                            .HostMemory("handle"),
                        SleepDatasetOp);

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
