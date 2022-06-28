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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/repeat_dataset_op.h"

#include <utility>

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const RepeatDatasetOp::kDatasetType;
/* static */ constexpr const char* const RepeatDatasetOp::kInputDataset;
/* static */ constexpr const char* const RepeatDatasetOp::kCount;
/* static */ constexpr const char* const RepeatDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RepeatDatasetOp::kOutputShapes;

constexpr char kForeverRepeat[] = "ForeverRepeat";
constexpr char kEmptyRepeat[] = "EmptyRepeat";
constexpr char kFiniteRepeat[] = "FiniteRepeat";
constexpr char kCurIteration[] = "i";
constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kUninitialized[] = "uninitialized";
constexpr int64_t kKnownRatio = 1;

class RepeatDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t count, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)), count_(count), input_(input) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    if (count_ < 0) {
      return absl::make_unique<ForeverIterator>(ForeverIterator::Params{
          this, name_utils::IteratorPrefix(kForeverRepeat, prefix)});
    } else if (count_ == 0) {
      return absl::make_unique<EmptyIterator>(EmptyIterator::Params{
          this, name_utils::IteratorPrefix(kEmptyRepeat, prefix)});
    } else {
      return absl::make_unique<FiniteIterator>(FiniteIterator::Params{
          this, name_utils::IteratorPrefix(kFiniteRepeat, prefix)});
    }
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }
  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_2(mht_2_v, 244, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(RepeatDatasetOp::kDatasetType);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_4(mht_4_v, 258, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "CardinalityInternal");

    int64_t n = input_->Cardinality();
    if (count_ < 0) {
      if (n == 0) {
        return 0;
      }
      return kInfiniteCardinality;
    }
    if (count_ == 0) {
      return 0;
    }
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return count_ * n;
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "CardinalityInternal");

    int64_t n = input_->Cardinality(options);
    if (count_ < 0) {
      if (n == 0) {
        return 0;
      }
      return kInfiniteCardinality;
    }
    if (count_ == 0) {
      return 0;
    }
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    return count_ * n;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_6(mht_6_v, 298, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_7(mht_7_v, 306, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_8(mht_8_v, 314, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    return input_->Get(ctx, index % input_->Cardinality(), out_tensors);
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_9(mht_9_v, 325, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* count = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph_node, count}, output));
    return Status::OK();
  }

 private:
  class EmptyIterator : public DatasetIterator<Dataset> {
   public:
    explicit EmptyIterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_10(mht_10_v, 341, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "EmptyIterator");
}
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_11(mht_11_v, 347, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "GetNextInternal");

      *end_of_sequence = true;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/kKnownRatio);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_12(mht_12_v, 363, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "SaveInternal");

      return Status::OK();
    }
    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_13(mht_13_v, 370, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "RestoreInternal");

      return Status::OK();
    }
  };

  class FiniteIterator : public DatasetIterator<Dataset> {
   public:
    explicit FiniteIterator(const Params& params)
        : DatasetIterator<Dataset>(params), i_(0) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_14(mht_14_v, 381, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "FiniteIterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_15(mht_15_v, 386, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "Initialize");

      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_16(mht_16_v, 395, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
      if (!input_impl_) {
        *end_of_sequence = true;
        return Status::OK();
      }
      while (i_ < dataset()->count_) {
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        if (!*end_of_sequence) {
          return Status::OK();
        }
        ++i_;
        for (const auto& provider : ctx->split_providers()) {
          TF_RETURN_IF_ERROR(provider->Reset());
        }
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
      }
      *end_of_sequence = true;
      input_impl_.reset();
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/kKnownRatio);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_17(mht_17_v, 430, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurIteration), i_));
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_18(mht_18_v, 445, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurIteration), &i_));
      if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      return Status::OK();
    }

   private:
    mutex mu_;
    int64_t i_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  class ForeverIterator : public DatasetIterator<Dataset> {
   public:
    explicit ForeverIterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          input_impl_(nullptr),
          first_call_(true) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_19(mht_19_v, 470, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "ForeverIterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_20(mht_20_v, 475, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "Initialize");

      mutex_lock l(mu_);
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_21(mht_21_v, 485, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);  // TODO(mrry): Make locking less conservative.
      do {
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(
              ctx, this, prefix(), &input_impl_));
        }
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, out_tensors, end_of_sequence));
        DCHECK(!*end_of_sequence || out_tensors->empty());
        if (first_call_ && *end_of_sequence && ctx->split_providers().empty()) {
          // If the first call to GetNext() fails because the end of sequence
          // has been reached, we terminate the iteration immediately.
          // Otherwise, this iterator would loop infinitely and never produce a
          // value.
          input_impl_.reset();
          return Status::OK();
        }
        first_call_ = false;
        if (!*end_of_sequence) {
          return Status::OK();
        }
        for (const auto& provider : ctx->split_providers()) {
          TF_RETURN_IF_ERROR(provider->Reset());
        }
        input_impl_.reset();
        first_call_ = true;
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/kKnownRatio);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_22(mht_22_v, 526, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      if (!first_call_)
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      else
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kUninitialized), ""));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_23(mht_23_v, 539, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      if (reader->Contains(full_name(kUninitialized))) {
        input_impl_.reset();
        first_call_ = true;
      } else {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
        first_call_ = false;
      }
      return Status::OK();
    }

   private:
    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    bool first_call_ TF_GUARDED_BY(mu_);
  };

  const int64_t count_;
  const DatasetBase* const input_;
};

RepeatDatasetOp::RepeatDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_24(mht_24_v, 567, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "RepeatDatasetOp::RepeatDatasetOp");
}

void RepeatDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrepeat_dataset_opDTcc mht_25(mht_25_v, 573, "", "./tensorflow/core/kernels/data/repeat_dataset_op.cc", "RepeatDatasetOp::MakeDataset");

  // Create a new RepeatDatasetOp::Dataset, insert it in the step-local
  // container, and return it as the output.
  int64_t count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kCount, &count));
  *output = new Dataset(ctx, count, input);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RepeatDataset").Device(DEVICE_CPU),
                        RepeatDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
