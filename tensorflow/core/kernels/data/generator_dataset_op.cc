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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/generator_dataset_op.h"

#include <iterator>
#include <vector>

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

/* static */ constexpr const char* const GeneratorDatasetOp::kDatasetType;
/* static */ constexpr const char* const GeneratorDatasetOp::kInitFuncOtherArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kNextFuncOtherArgs;
/* static */ constexpr const char* const
    GeneratorDatasetOp::kFinalizeFuncOtherArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kInitFunc;
/* static */ constexpr const char* const GeneratorDatasetOp::kNextFunc;
/* static */ constexpr const char* const GeneratorDatasetOp::kFinalizeFunc;
/* static */ constexpr const char* const GeneratorDatasetOp::kTinitFuncArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kTnextFuncArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kTfinalizeFuncArgs;
/* static */ constexpr const char* const GeneratorDatasetOp::kOutputTypes;
/* static */ constexpr const char* const GeneratorDatasetOp::kOutputShapes;

class GeneratorDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, std::unique_ptr<CapturedFunction> init_func,
          std::unique_ptr<CapturedFunction> next_func,
          std::unique_ptr<CapturedFunction> finalize_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        init_func_(std::move(init_func)),
        next_func_(std::move(next_func)),
        finalize_func_(std::move(finalize_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_0(mht_0_v, 237, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "output_dtypes");
 return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_1(mht_1_v, 242, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "output_shapes");

    return output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_4(mht_4_v, 263, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "CheckExternalState");

    TF_RETURN_IF_ERROR(init_func_->CheckExternalState());
    TF_RETURN_IF_ERROR(next_func_->CheckExternalState());
    return finalize_func_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "AsGraphDefInternal");

    return errors::Unimplemented(DebugString(),
                                 " does not support serialization");
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_6(mht_6_v, 287, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "Iterator");
}

    ~Iterator() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_7(mht_7_v, 292, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "~Iterator");

      if (!finalized_ && initialized_) {
        std::vector<Tensor> ignored;
        Status s =
            instantiated_finalize_func_->RunInstantiated(state_, &ignored);
        if (!s.ok()) {
          LOG(WARNING)
              << "Error occurred when finalizing GeneratorDataset iterator: "
              << s;
        }
      }
    }

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_8(mht_8_v, 308, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "Initialize");

      TF_RETURN_IF_ERROR(
          dataset()->init_func_->Instantiate(ctx, &instantiated_init_func_));
      TF_RETURN_IF_ERROR(
          dataset()->next_func_->Instantiate(ctx, &instantiated_next_func_));
      TF_RETURN_IF_ERROR(dataset()->finalize_func_->Instantiate(
          ctx, &instantiated_finalize_func_));
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_9(mht_9_v, 323, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);

      if (!initialized_) {
        TF_RETURN_IF_ERROR(instantiated_init_func_->RunWithBorrowedArgs(
            ctx, {}, &state_, model_node()));
        initialized_ = true;
      }

      if (finalized_) {
        *end_of_sequence = true;
        return Status::OK();
      }

      Status s = instantiated_next_func_->RunWithBorrowedArgs(
          ctx, state_, out_tensors, model_node());
      if (s.ok()) {
        *end_of_sequence = false;
      } else if (errors::IsOutOfRange(s)) {
        // `next_func` may deliberately raise `errors::OutOfRange`
        // to indicate that we should terminate the iteration.
        s = Status::OK();
        *end_of_sequence = true;

        // NOTE(mrry): We ignore any tensors returned by the finalize function.
        std::vector<Tensor> ignored;
        TF_RETURN_IF_ERROR(instantiated_finalize_func_->RunWithBorrowedArgs(
            ctx, state_, &ignored, model_node()));
        finalized_ = true;
      }
      return s;
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_10(mht_10_v, 366, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "SaveInternal");

      return errors::Unimplemented(
          "GeneratorDataset does not support checkpointing.");
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_11(mht_11_v, 375, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "RestoreInternal");

      return errors::Unimplemented(
          "GeneratorDataset does not support checkpointing.");
    }

   private:
    mutex mu_;
    bool initialized_ TF_GUARDED_BY(mu_) = false;
    bool finalized_ TF_GUARDED_BY(mu_) = false;
    std::vector<Tensor> state_ TF_GUARDED_BY(mu_);
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_init_func_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_next_func_;
    std::unique_ptr<InstantiatedCapturedFunction> instantiated_finalize_func_;
  };

  const std::unique_ptr<CapturedFunction> init_func_;
  const std::unique_ptr<CapturedFunction> next_func_;
  const std::unique_ptr<CapturedFunction> finalize_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

GeneratorDatasetOp::GeneratorDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_12(mht_12_v, 401, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "GeneratorDatasetOp::GeneratorDatasetOp");

  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kInitFunc, /*params=*/{},
                                               &init_func_metadata_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kNextFunc, /*params=*/{},
                                               &next_func_metadata_));
  OP_REQUIRES_OK(ctx,
                 FunctionMetadata::Create(ctx, kFinalizeFunc, /*params=*/{},
                                          &finalize_func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void GeneratorDatasetOp::MakeDataset(OpKernelContext* ctx,
                                     DatasetBase** output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSgenerator_dataset_opDTcc mht_13(mht_13_v, 417, "", "./tensorflow/core/kernels/data/generator_dataset_op.cc", "GeneratorDatasetOp::MakeDataset");

  std::unique_ptr<CapturedFunction> init_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, init_func_metadata_,
                                               kInitFuncOtherArgs, &init_func));

  std::unique_ptr<CapturedFunction> next_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, next_func_metadata_,
                                               kNextFuncOtherArgs, &next_func));

  std::unique_ptr<CapturedFunction> finalize_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(ctx, finalize_func_metadata_,
                                    kFinalizeFuncOtherArgs, &finalize_func));

  *output =
      new Dataset(ctx, std::move(init_func), std::move(next_func),
                  std::move(finalize_func), output_types_, output_shapes_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("GeneratorDataset").Device(DEVICE_CPU).Priority(2),
                        GeneratorDatasetOp);
REGISTER_KERNEL_BUILDER(Name("GeneratorDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("handle")
                            .Priority(1),
                        GeneratorDatasetOp);
REGISTER_INPUT_COLOCATION_EXEMPTION("GeneratorDataset");
}  // namespace

}  // namespace data
}  // namespace tensorflow
