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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/range_dataset_op.h"

#include <functional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/split_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const RangeDatasetOp::kDatasetType;
/* static */ constexpr const char* const RangeDatasetOp::kStart;
/* static */ constexpr const char* const RangeDatasetOp::kStop;
/* static */ constexpr const char* const RangeDatasetOp::kStep;
/* static */ constexpr const char* const RangeDatasetOp::kOutputTypes;
/* static */ constexpr const char* const RangeDatasetOp::kOutputShapes;

namespace {
constexpr char kNext[] = "next";
constexpr char kHasSplitProvider[] = "has_split_provider";
constexpr char kSlash[] = "/";
constexpr char kSplitProvider[] = "split_provider";

Status ConvertOutputTypes(const tensorflow::DataTypeVector& output_dtypes,
                          std::vector<Tensor>* out_tensors, int64 value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_0(mht_0_v, 218, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "ConvertOutputTypes");

  switch (output_dtypes[0]) {
#define HANDLE_TYPE(type)                                \
  case DataTypeToEnum<type>::value: {                    \
    out_tensors->emplace_back(static_cast<type>(value)); \
    break;                                               \
  }
    TF_CALL_NUMBER_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(output_dtypes[0]));
  }
  return Status::OK();
}

int64_t sgn(int64_t val) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_1(mht_1_v, 237, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "sgn");
 return (0 < val) - (val < 0); }

// Class which produces the elements of `range(start, stop, step)`. Threadsafe.
class RangeCounter {
 public:
  RangeCounter(int64_t start, int64_t stop, int64_t step)
      : start_(start), stop_(stop), step_(step), next_(start) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "RangeCounter");
}

  // Returns the next value for the counter. Sets `*end_of_counter` to indicate
  // whether the end of the counter was reached.
  int64_t GetNext(bool* end_of_counter) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_3(mht_3_v, 253, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "GetNext");

    mutex_lock l(mu_);
    if ((step_ > 0 && next_ >= stop_) || (step_ < 0 && next_ <= stop_)) {
      *end_of_counter = true;
      return -1;
    }
    *end_of_counter = false;
    int result = next_;
    next_ += step_;
    return result;
  }

  int64_t Peek() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Peek");

    mutex_lock l(mu_);
    return next_;
  }

  void Reset() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_5(mht_5_v, 276, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Reset");

    mutex_lock l(mu_);
    next_ = start_;
  }

  void SetNext(int64_t value) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_6(mht_6_v, 284, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "SetNext");

    mutex_lock l(mu_);
    next_ = value;
  }

 private:
  const int64_t start_;
  const int64_t stop_;
  const int64_t step_;
  mutable mutex mu_;
  int64_t next_ TF_GUARDED_BY(mu_);
};
}  // namespace

// Split provider where splits are individual outputs from RangeDataset.
// For example, the "splits" of range(0, 10, 2) will be {0, 2, 4, 6, 8}.
// The split tensors are scalars of type DT_INT64.
class RangeDatasetOp::RangeSplitProvider : public SplitProvider {
 public:
  RangeSplitProvider(int64_t start, int64_t stop, int64_t step)
      : counter_(start, stop, step) {}

  Status GetNext(Tensor* split, bool* end_of_splits) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_7(mht_7_v, 309, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "GetNext");

    int64_t next = counter_.GetNext(end_of_splits);
    if (*end_of_splits) {
      return Status::OK();
    }
    *split = Tensor(DT_INT64, TensorShape{});
    split->scalar<int64_t>()() = next;
    return Status::OK();
  }

  Status Reset() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_8(mht_8_v, 322, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Reset");

    counter_.Reset();
    return Status::OK();
  }

  Status Save(std::function<std::string(std::string)> key_name_fn,
              IteratorStateWriter* writer) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_9(mht_9_v, 331, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Save");

    TF_RETURN_IF_ERROR(
        writer->WriteScalar(key_name_fn(kNext), counter_.Peek()));
    return Status::OK();
  }

  Status Restore(std::function<std::string(std::string)> key_name_fn,
                 IteratorStateReader* reader) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_10(mht_10_v, 341, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Restore");

    int64_t next;
    TF_RETURN_IF_ERROR(reader->ReadScalar(key_name_fn(kNext), &next));
    counter_.SetNext(next);
    return Status::OK();
  }

 private:
  RangeCounter counter_;
};

class RangeDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t start, int64_t stop, int64_t step,
          DataTypeVector output_dtypes)
      : DatasetBase(DatasetContext(ctx)),
        start_(start),
        stop_(stop),
        step_(step),
        output_dtypes_(output_dtypes) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_11(mht_11_v, 371, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "output_dtypes");

    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_12(mht_12_v, 378, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "output_shapes");

    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({PartialTensorShape({})});
    return *shapes;
  }

  string DebugString() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_13(mht_13_v, 387, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.set_args(start_, stop_, step_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_14(mht_14_v, 396, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "CardinalityInternal");

    // If start_ == stop_ or if the sign of stop_ - start_ and step do not agree
    // (or are zero), return zero.
    if (sgn(stop_ - start_) * sgn(step_) <= 0) {
      return 0;
    } else if (step_ > 0) {
      return std::max(int64_t{0}, (stop_ - start_ - 1) / step_ + 1);
    } else {
      return std::max(int64_t{0}, (start_ - stop_ - 1) / -step_ + 1);
    }
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_15(mht_15_v, 411, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "CardinalityInternal");

    // If start_ == stop_ or if the sign of stop_ - start_ and step do not agree
    // (or are zero), return zero.
    if (sgn(stop_ - start_) * sgn(step_) <= 0) {
      return 0;
    } else if (step_ > 0) {
      return std::max(int64_t{0}, (stop_ - start_ - 1) / step_ + 1);
    } else {
      return std::max(int64_t{0}, (start_ - stop_ - 1) / -step_ + 1);
    }
  }

  Status MakeSplitProviders(std::vector<std::unique_ptr<SplitProvider>>*
                                split_providers) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_16(mht_16_v, 427, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "MakeSplitProviders");

    split_providers->push_back(
        absl::make_unique<RangeSplitProvider>(start_, stop_, step_));
    return Status::OK();
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_17(mht_17_v, 436, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "InputDatasets");

    inputs->clear();
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_18(mht_18_v, 444, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_19(mht_19_v, 450, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    return ConvertOutputTypes(output_dtypes(), out_tensors,
                              start_ + (index * step_));
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_20(mht_20_v, 462, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "AsGraphDefInternal");

    Node* start = nullptr;
    Node* stop = nullptr;
    Node* step = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(start_, &start));
    TF_RETURN_IF_ERROR(b->AddScalar(stop_, &stop));
    TF_RETURN_IF_ERROR(b->AddScalar(step_, &step));
    TF_RETURN_IF_ERROR(b->AddDataset(this, {start, stop, step}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_21(mht_21_v, 480, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_22(mht_22_v, 485, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "Initialize");

      if (ctx->split_providers().empty()) {
        counter_ = absl::make_unique<RangeCounter>(
            dataset()->start_, dataset()->stop_, dataset()->step_);
      } else {
        TF_ASSIGN_OR_RETURN(split_provider_,
                            GetSingleSplitProvider(ctx, dataset()));
      }
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_23(mht_23_v, 501, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "GetNextInternal");

      int64_t value;
      if (split_provider_ != nullptr) {
        Tensor split;
        TF_RETURN_IF_ERROR(split_provider_->GetNext(&split, end_of_sequence));
        if (*end_of_sequence) {
          return Status::OK();
        }
        value = split.scalar<int64_t>()();
      } else {
        value = counter_->GetNext(end_of_sequence);
        if (*end_of_sequence) {
          return Status::OK();
        }
      }
      out_tensors->reserve(1);
      return ConvertOutputTypes(output_dtypes(), out_tensors, value);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_24(mht_24_v, 530, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "SaveInternal");

      if (split_provider_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kHasSplitProvider), true));
        TF_RETURN_IF_ERROR(split_provider_->Save(
            [this](const std::string& key) {
              return SplitProviderKeyNameFn(key);
            },
            writer));
      } else {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kNext), counter_->Peek()));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_25(mht_25_v, 550, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "RestoreInternal");

      if (reader->Contains(full_name(kHasSplitProvider))) {
        TF_RETURN_IF_ERROR(split_provider_->Restore(
            [this](const std::string& key) {
              return SplitProviderKeyNameFn(key);
            },
            reader));
      } else {
        int64_t next;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNext), &next));
        counter_->SetNext(next);
      }
      return Status::OK();
    }

    std::string SplitProviderKeyNameFn(const std::string& key) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_26(mht_26_v, 569, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "SplitProviderKeyNameFn");

      return full_name(absl::StrCat(kSplitProvider, kSlash, key));
    }

   private:
    std::unique_ptr<RangeCounter> counter_;
    std::shared_ptr<SplitProvider> split_provider_;
  };

  const int64_t start_;
  const int64_t stop_;
  const int64_t step_;
  const DataTypeVector output_dtypes_;
};

RangeDatasetOp::RangeDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_27(mht_27_v, 588, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "RangeDatasetOp::RangeDatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
}

void RangeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase** output) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSrange_dataset_opDTcc mht_28(mht_28_v, 595, "", "./tensorflow/core/kernels/data/range_dataset_op.cc", "RangeDatasetOp::MakeDataset");

  int64_t start;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kStart, &start));

  int64_t stop;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kStop, &stop));

  int64_t step;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kStep, &step));
  OP_REQUIRES(ctx, step != 0,
              errors::InvalidArgument("step must be a non-zero integer."));

  *output = new Dataset(ctx, start, stop, step, output_types_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("RangeDataset").Device(DEVICE_CPU),
                        RangeDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
