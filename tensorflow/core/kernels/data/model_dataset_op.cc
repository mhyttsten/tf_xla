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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/model_dataset_op.h"

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/cancellation.h"

// On mobile we do not provide model dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

// Default share of available RAM that can be used by model's internal buffers.
constexpr double kRamBudgetShare = 0.5;

}  // namespace

/* static */ constexpr const char* const ModelDatasetOp::kDatasetType;
/* static */ constexpr const char* const ModelDatasetOp::kDatasetOp;
/* static */ constexpr const char* const ModelDatasetOp::kAlgorithm;
/* static */ constexpr const char* const ModelDatasetOp::kCpuBudget;
/* static */ constexpr const char* const ModelDatasetOp::kRamBudget;

class ModelDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          model::AutotuneAlgorithm algorithm, int64_t cpu_budget,
          int64_t ram_budget)
      : Dataset(DatasetContext(ctx), input, algorithm, cpu_budget, ram_budget) {
  }

  Dataset(DatasetContext&& ctx, const DatasetBase* input,
          model::AutotuneAlgorithm algorithm, int64_t cpu_budget,
          int64_t ram_budget)
      : DatasetBase(std::move(ctx)),
        input_(input),
        algorithm_(algorithm),
        cpu_budget_(cpu_budget),
        ram_budget_(ram_budget),
        traceme_metadata_(
            {{"algorithm", model::AutotuneAlgorithm_Name(algorithm)},
             {"cpu_budget",
              strings::Printf("%lld", static_cast<long long>(cpu_budget))},
             {"ram_budget",
              strings::Printf("%lldB", static_cast<long long>(ram_budget))}}) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_0(mht_0_v, 243, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::Model")});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_1(mht_1_v, 254, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_2(mht_2_v, 260, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "DebugString");
 return "ModelDatasetOp::Dataset"; }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "CardinalityInternal");
 return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_5(mht_5_v, 277, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_7(mht_7_v, 295, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    AttrValue algorithm_attr;
    b->BuildAttrValue(static_cast<int64_t>(algorithm_), &algorithm_attr);
    AttrValue cpu_budget_attr;
    b->BuildAttrValue(cpu_budget_, &cpu_budget_attr);
    AttrValue ram_budget_attr;
    b->BuildAttrValue(ram_budget_, &ram_budget_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node},
                      {std::make_pair(kAlgorithm, algorithm_attr),
                       std::make_pair(kCpuBudget, cpu_budget_attr),
                       std::make_pair(kRamBudget, ram_budget_attr)},
                      output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          cpu_budget_(dataset()->cpu_budget_ == 0 ? GetCpuBudget()
                                                  : dataset()->cpu_budget_),
          ram_budget_(dataset()->ram_budget_ == 0
                          ? kRamBudgetShare * port::AvailableRam()
                          : dataset()->ram_budget_) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_8(mht_8_v, 326, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "Iterator");

      cancellation_manager_ = absl::make_unique<CancellationManager>();
      model_ = std::make_shared<model::Model>();
    }

    ~Iterator() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_9(mht_9_v, 334, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "~Iterator");
 cancellation_manager_->StartCancel(); }

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_10(mht_10_v, 339, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "Initialize");

      return dataset()->input_->MakeIterator(IteratorContext(CreateParams(ctx)),
                                             this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_11(mht_11_v, 349, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "GetNextInternal");

      if (!ctx->model()) {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsureOptimizationLoopThreadStarted(ctx));
      }
      return input_impl_->GetNext(IteratorContext(CreateParams(ctx)),
                                  out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_12(mht_12_v, 369, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "SaveInternal");

      return SaveInput(ctx, writer, input_impl_);
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_13(mht_13_v, 377, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "RestoreInternal");

      return RestoreInput(IteratorContext(CreateParams(ctx)), reader,
                          input_impl_);
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_14(mht_14_v, 385, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "GetTraceMeMetadata");

      return dataset()->traceme_metadata_;
    }

   private:
    IteratorContext::Params CreateParams(IteratorContext* ctx) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_15(mht_15_v, 393, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "CreateParams");

      IteratorContext::Params params(ctx);
      if (!ctx->model()) {
        params.model = model_;
      }
      return params;
    }

    Status EnsureOptimizationLoopThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_16(mht_16_v, 405, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "EnsureOptimizationLoopThreadStarted");

      if (!model_thread_) {
        model_thread_ = ctx->StartThread("tf_data_model", [this]() {
          Status status =
              model_->OptimizeLoop(dataset()->algorithm_, cpu_budget_,
                                   ram_budget_, cancellation_manager_.get());
          if (!status.ok()) {
            LOG(WARNING) << "Optimization loop failed: " << status.ToString();
          }
        });
      }
      return Status::OK();
    }

    mutex mu_;
    std::shared_ptr<model::Model> model_;
    // Controls cancellation of `model_thread_`. Must be ordered before
    // `model_thread_` so that `model_thread_` is destroyed first.
    std::unique_ptr<CancellationManager> cancellation_manager_;
    std::unique_ptr<Thread> model_thread_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_;
    const int64_t cpu_budget_;
    const int64_t ram_budget_;
  };

  const DatasetBase* input_;
  const model::AutotuneAlgorithm algorithm_;
  const int64_t cpu_budget_;
  const int64_t ram_budget_;
  const TraceMeMetadata traceme_metadata_;
};

// static
void ModelDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            model::AutotuneAlgorithm algorithm,
                                            int64_t cpu_budget,
                                            int64_t ram_budget,
                                            DatasetBase** output) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_17(mht_17_v, 446, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "ModelDatasetOp::MakeDatasetFromOptions");

  *output = new ModelDatasetOp::Dataset(
      DatasetContext(DatasetContext::Params(
          {ModelDatasetOp::kDatasetType, ModelDatasetOp::kDatasetOp})),
      input, algorithm, cpu_budget, ram_budget);
}

ModelDatasetOp::ModelDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_18(mht_18_v, 457, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "ModelDatasetOp::ModelDatasetOp");

  if (ctx->HasAttr(kAlgorithm)) {
    int64_t algorithm;
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kAlgorithm, &algorithm));
    algorithm_ = model::AutotuneAlgorithm(algorithm);
  } else {
    algorithm_ = model::AutotuneAlgorithm::HILL_CLIMB;
  }
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCpuBudget, &cpu_budget_));
  OP_REQUIRES(ctx, cpu_budget_ >= 0,
              errors::InvalidArgument("CPU budget must be positive but is ",
                                      cpu_budget_, "."));
  if (ctx->HasAttr(kRamBudget)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kRamBudget, &ram_budget_));
  } else {
    ram_budget_ = 0;
  }
  OP_REQUIRES(ctx, ram_budget_ >= 0,
              errors::InvalidArgument("RAM budget must be positive but is ",
                                      ram_budget_, "."));
}

void ModelDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_19(mht_19_v, 483, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "ModelDatasetOp::MakeDataset");

  *output = new ModelDatasetOp::Dataset(ctx, input, algorithm_, cpu_budget_,
                                        ram_budget_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#else   // !IS_MOBILE_PLATFORM
namespace tensorflow {
namespace data {
// static
void ModelDatasetOp::MakeDatasetFromOptions(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            model::AutotuneAlgorithm algorithm,
                                            bool cpu_budget, bool ram_budget,
                                            DatasetBase** output) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_20(mht_20_v, 505, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "ModelDatasetOp::MakeDatasetFromOptions");

  input->Ref();
  *output = input;
}

ModelDatasetOp::ModelDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_21(mht_21_v, 514, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "ModelDatasetOp::ModelDatasetOp");
}

void ModelDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                 DatasetBase** output) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmodel_dataset_opDTcc mht_22(mht_22_v, 520, "", "./tensorflow/core/kernels/data/model_dataset_op.cc", "ModelDatasetOp::MakeDataset");

  input->Ref();
  *output = input;
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ModelDataset").Device(DEVICE_CPU),
                        ModelDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM
