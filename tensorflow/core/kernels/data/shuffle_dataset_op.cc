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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/shuffle_dataset_op.h"

#include <cstdint>
#include <deque>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/random_seed_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const ShuffleDatasetOpBase::kInputDataset;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kBufferSize;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kSeed;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kSeed2;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kOutputTypes;
/* static */ constexpr const char* const ShuffleDatasetOpBase::kOutputShapes;
/* static */ constexpr const char* const
    ShuffleDatasetOpBase::kReshuffleEachIteration;

/* static */ constexpr const char* const ShuffleDatasetOp::kDatasetType;

/* static */ constexpr const char* const
    ShuffleAndRepeatDatasetOp::kDatasetType;
/* static */ constexpr const char* const ShuffleAndRepeatDatasetOp::kCount;

const int64_t kLogIntervalMicros = 10 * 1000000;  // 10 seconds.
const int64_t kMaxEpochsInBuffer = 3;

constexpr char kNumRandomSamples[] = "num_random_samples";
constexpr char kDataProduced[] = "data_produced";
constexpr char kEndOfInputSequence[] = "end_of_input_sequence";
constexpr char kEpoch[] = "epoch";
constexpr char kNumElements[] = "num_elements";
constexpr char kSlicesSize[] = "slices_size";
constexpr char kSlicesStart[] = "slices_start";
constexpr char kSlicesEnd[] = "slices_end";
constexpr char kSeedGenerator[] = "SeedGenerator";
constexpr char kEpochNumRandomSamples[] = "epoch_num_random_samples";
constexpr char kShuffleDatasetV1[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2[] = "ShuffleDatasetV2";
constexpr char kShuffleDatasetV3[] = "ShuffleDatasetV3";
constexpr char kShuffleAndRepeatDatasetV1[] = "ShuffleAndRepeatDataset";
constexpr char kShuffleAndRepeatDatasetV2[] = "ShuffleAndRepeatDatasetV2";

ShuffleDatasetOpBase::ShuffleDatasetOpBase(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_0(mht_0_v, 249, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ShuffleDatasetOpBase::ShuffleDatasetOpBase");
}

// Abstract base dataset that implements a shuffling iterator.
class ShuffleDatasetOpBase::ShuffleDatasetBase : public DatasetBase {
 public:
  ShuffleDatasetBase(OpKernelContext* ctx, const DatasetBase* input,
                     int64_t buffer_size,
                     std::shared_ptr<SeedGenerator> seed_generator,
                     int64_t count)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        seed_generator_(std::move(seed_generator)),
        count_(count),
        traceme_metadata_(
            {{"buffer_size",
              strings::Printf("%lld", static_cast<long long>(buffer_size))}}) {
    input_->Ref();
  }

  ~ShuffleDatasetBase() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_1(mht_1_v, 272, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "~ShuffleDatasetBase");
 input_->Unref(); }

  virtual string op_type() const = 0;

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_2(mht_2_v, 279, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "output_dtypes");

    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_3(mht_3_v, 286, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "output_shapes");

    return input_->output_shapes();
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_4(mht_4_v, 293, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "CardinalityInternal");

    if (count_ == -1 || input_->Cardinality() == kInfiniteCardinality) {
      return kInfiniteCardinality;
    } else if (input_->Cardinality() == kUnknownCardinality) {
      return kUnknownCardinality;
    } else {
      return input_->Cardinality() * count_;
    }
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_5(mht_5_v, 306, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "CardinalityInternal");

    if (count_ == -1 || input_->Cardinality(options) == kInfiniteCardinality) {
      return kInfiniteCardinality;
    } else if (input_->Cardinality(options) == kUnknownCardinality) {
      return kUnknownCardinality;
    } else {
      return input_->Cardinality(options) * count_;
    }
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_6(mht_6_v, 319, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_7(mht_7_v, 327, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_8(mht_8_v, 335, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "Get");

    TF_RETURN_IF_ERROR(CheckRandomAccessCompatible(index));
    {
      mutex_lock l(mu_);
      if (shuffled_indices_.empty()) {
        InitializeRandomAccessIndices();
      }
    }
    int64 shuffled_index;
    {
      tf_shared_lock l(mu_);
      shuffled_index = shuffled_indices_[index];
    }
    TF_RETURN_IF_ERROR(input_->Get(ctx, shuffled_index, out_tensors));
    return Status::OK();
  }

  string DebugString() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_9(mht_9_v, 355, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.set_args(buffer_size_, seed_generator_->seed(),
                    seed_generator_->seed2(), count_);
    return name_utils::DatasetDebugString(op_type(), params);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, name_utils::IteratorPrefix(op_type(), prefix)},
        seed_generator_.get());
  }

  void InitializeRandomAccessIndices() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const int64 cardinality = Cardinality();
    shuffled_indices_ = std::vector<std::int64_t>(cardinality);
    std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
    int64_t shuffled_index = 0;
    random::PhiloxRandom parent_generator =
        random::PhiloxRandom(seed_generator_->seed(), seed_generator_->seed2());
    random::SingleSampleAdapter<random::PhiloxRandom> generator =
        random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator);

    while (shuffled_index < cardinality) {
      int64_t offset = generator() % (cardinality - shuffled_index);
      std::swap(shuffled_indices_[shuffled_index + offset],
                shuffled_indices_[shuffled_index]);
      shuffled_index += 1;
    }
  }

 protected:
  class Iterator : public DatasetIterator<ShuffleDatasetBase> {
   public:
    explicit Iterator(const Params& params, SeedGenerator* seed_generator)
        : DatasetIterator<ShuffleDatasetBase>(params),
          seed_generator_(seed_generator),
          parent_generator_(seed_generator->seed(), seed_generator->seed2()),
          generator_(&parent_generator_) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_10(mht_10_v, 397, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "Iterator");

      buffer_ = absl::make_unique<std::vector<std::vector<Tensor>>>(
          params.dataset->buffer_size_);
    }

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_11(mht_11_v, 405, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "Initialize");

      mutex_lock l(mu_);
      seed_generator_->GenerateSeeds(&seed_, &seed2_);
      ResetRngs();
      return Status::OK();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_12(mht_12_v, 417, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(FillBuffer(ctx));
      if (num_elements_ == 0) {
        DCHECK(input_impl_ == nullptr);
        *end_of_sequence = true;
        return Status::OK();
      }

      *end_of_sequence = false;
      ClearEmptySlices();
      DCHECK(!slices_.empty());
      // Choose an element to produce uniformly at random from the first
      // slice, and then remove the element from the slice.
      int64_t offset =
          Random() % (slices_.front()->end - slices_.front()->start);
      int64_t index = (slices_.front()->start + offset) % buffer_->size();
      *out_tensors = std::move(buffer_->at(index));
      this->RecordBufferDequeue(ctx, *out_tensors);
      std::swap(buffer_->at(index),
                buffer_->at(slices_.front()->start % buffer_->size()));
      slices_.front()->start++;
      num_elements_--;
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    void ResetRngs() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_13(mht_13_v, 453, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ResetRngs");

      // Reset the generators based on the current iterator seeds.
      parent_generator_ = random::PhiloxRandom(seed_, seed2_);
      generator_ =
          random::SingleSampleAdapter<random::PhiloxRandom>(&parent_generator_);
      generator_.Skip(num_random_samples_);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_14(mht_14_v, 465, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      // Save state needed to restore the random number generators.
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kEpochNumRandomSamples),
                              seed_generator_->num_random_samples()));
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kNumRandomSamples),
                                             num_random_samples_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kSeed), seed_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kSeed2), seed2_));

      // Save input iterator if it hasn't been exhausted else write
      // "end_of_input_sequence".
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(kEndOfInputSequence), ""));
      } else {
        TF_RETURN_IF_ERROR(this->SaveInput(ctx, writer, input_impl_));
      }

      // Save the epoch counter, buffer, and buffer slices.
      TF_RETURN_IF_ERROR(writer->WriteScalar(this->full_name(kEpoch), epoch_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name(kNumElements), num_elements_));
      TF_RETURN_IF_ERROR(WriteElementsToCheckpoint(writer, prefix(), *buffer_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(this->full_name(kSlicesSize), slices_.size()));
      for (size_t i = 0; i < slices_.size(); ++i) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(absl::StrJoin(
                                    std::make_tuple(kSlicesStart, i), "_")),
                                slices_[i]->start));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            this->full_name(absl::StrJoin(std::make_tuple(kSlicesEnd, i), "_")),
            slices_[i]->end));
      }
      if (data_produced_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(this->full_name(kDataProduced), ""));
      }

      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_15(mht_15_v, 513, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      // Restore the random number generators.
      int64_t num_random_samples;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kEpochNumRandomSamples),
                                            &num_random_samples));
      seed_generator_->set_num_random_samples(num_random_samples);
      seed_generator_->Reset();
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kNumRandomSamples),
                                            &num_random_samples_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kSeed), &seed_));
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kSeed2), &seed2_));
      ResetRngs();

      // Restore the input iterator if it wasn't already exhausted.
      if (!reader->Contains(this->full_name(kEndOfInputSequence))) {
        TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
            ctx, this, this->prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(this->RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }

      // Restore the epoch counter, buffer, and buffer slices.
      TF_RETURN_IF_ERROR(reader->ReadScalar(this->full_name(kEpoch), &epoch_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(this->full_name(kNumElements), &num_elements_));
      size_t slices_size;
      {
        int64_t temp;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name(kSlicesSize), &temp));
        slices_size = static_cast<size_t>(temp);
      }
      buffer_ = absl::make_unique<std::vector<std::vector<Tensor>>>();
      TF_RETURN_IF_ERROR(
          ReadElementsFromCheckpoint(ctx, reader, prefix(), buffer_.get()));
      for (const auto& element : *buffer_) {
        RecordBufferEnqueue(ctx, element);
      }
      buffer_->resize(dataset()->buffer_size_);
      slices_.clear();
      for (size_t i = 0; i < slices_size; ++i) {
        int64_t start;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(this->full_name(absl::StrJoin(
                                   std::make_tuple(kSlicesStart, i), "_")),
                               &start));
        int64_t end;
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            this->full_name(absl::StrJoin(std::make_tuple(kSlicesEnd, i), "_")),
            &end));
        slices_.push_back(absl::make_unique<Slice>(start, end));
      }
      data_produced_ = reader->Contains(this->full_name(kDataProduced));

      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_16(mht_16_v, 575, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "GetTraceMeMetadata");

      return this->dataset()->traceme_metadata_;
    }

   private:
    // Used to represent slices of `buffer_` that belong to different epochs.
    // The invariant maintained by the implementation is: `start` <= `end`.
    // When using `start` and `end` to index into `buffer_`, their values
    // should be taken modulo the size of `buffer_` as their absolute value
    // can be greater than the range of `buffer_`.
    struct Slice {
      Slice(int64_t start, int64_t end) : start(start), end(end) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_17(mht_17_v, 589, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "Slice");
}

      int64_t start;
      int64_t end;
    };

    random::SingleSampleAdapter<random::PhiloxRandom>::ResultType Random()
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_18(mht_18_v, 599, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "Random");

      num_random_samples_++;
      auto out = generator_();
      return out;
    }

    // Fills the shuffle buffer, preparing the buffer for sampling.
    Status FillBuffer(IteratorContext* ctx) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_19(mht_19_v, 609, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "FillBuffer");

      int64_t start_micros = EnvTime::NowMicros();
      int64_t num_log_entries = 0;
      while (ShouldFillBuffer()) {
        if (EnvTime::NowMicros() >
            ((num_log_entries + 1) * kLogIntervalMicros) + start_micros) {
          num_log_entries++;
          LOG(INFO) << "Filling up shuffle buffer (this may take a while): "
                    << num_elements_ << " of " << BufferSizeString();
        }
        if (!input_impl_) {
          TF_RETURN_IF_ERROR(PrepareNextEpoch(ctx));
        }
        std::vector<Tensor> input_element;
        bool end_of_input_sequence = false;
        TF_RETURN_IF_ERROR(
            input_impl_->GetNext(ctx, &input_element, &end_of_input_sequence));
        if (!end_of_input_sequence) {
          AddToShuffleBuffer(ctx, std::move(input_element));
          continue;
        }
        input_impl_.reset();
        // Reached end of input_impl_.
        if (ctx->split_providers().empty() && !data_produced_ &&
            this->dataset()->count_ == -1) {
          // If we encounter the end of sequence without producing data, we
          // terminate the iteration immediately. (Otherwise, this iterator
          // would loop infinitely and never produce a value.)
          return Status::OK();
        }
      }
      if (num_log_entries > 0) {
        LOG(INFO) << "Shuffle buffer filled.";
      }
      return Status::OK();
    }

    bool ShouldFillBuffer() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_20(mht_20_v, 649, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ShouldFillBuffer");

      if (!input_impl_ && dataset()->count_ != -1 &&
          epoch_ >= dataset()->count_) {
        return false;
      }
      if (slices_.size() > kMaxEpochsInBuffer && num_elements_ > 0) {
        // When the elements stored in `buffer_` span more than
        // `kMaxEpochsInBuffer` epochs, we do not fill the buffer further to
        // conserve memory. This means that the upper bound on the size of
        // `buffer_` is `kMaxEpochsInBuffer * cardinality(input_dataset) +
        // 1`.
        return false;
      }
      return num_elements_ < buffer_->size();
    }

    Status PrepareNextEpoch(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_21(mht_21_v, 669, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "PrepareNextEpoch");

      if (epoch_ == 0) {
        slices_.push_back(absl::make_unique<Slice>(0, 0));
      } else {
        int64_t n = slices_.back()->end;
        slices_.push_back(absl::make_unique<Slice>(n, n));
        for (const auto& provider : ctx->split_providers()) {
          TF_RETURN_IF_ERROR(provider->Reset());
        }
      }
      TF_RETURN_IF_ERROR(this->dataset()->input_->MakeIterator(
          ctx, this, this->prefix(), &input_impl_));
      epoch_++;
      return Status::OK();
    }

    void AddToShuffleBuffer(IteratorContext* ctx, std::vector<Tensor>&& element)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_22(mht_22_v, 689, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "AddToShuffleBuffer");

      data_produced_ = true;
      if (num_elements_ == 0) {
        VLOG(1) << "Starting to fill up shuffle buffer of size: "
                << BufferSizeString();
      }
      this->RecordBufferEnqueue(ctx, element);
      size_t index = slices_.back()->end % buffer_->size();
      buffer_->at(index) = std::move(element);
      num_elements_++;
      slices_.back()->end++;
    }

    void ClearEmptySlices() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_23(mht_23_v, 705, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ClearEmptySlices");

      // Garbage collect all empty slices.
      while (slices_.front()->start == slices_.front()->end) {
        slices_.pop_front();
        // Reinitialize the RNG state for the next epoch.
        num_random_samples_ = 0;
        seed_generator_->GenerateSeeds(&seed_, &seed2_);
        ResetRngs();
      }
    }

    std::string BufferSizeString() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_24(mht_24_v, 719, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "BufferSizeString");

      return absl::StrCat(dataset()->buffer_size_);
    }

    mutex mu_;
    SeedGenerator* const seed_generator_ TF_GUARDED_BY(mu_);  // Not owned.
    std::unique_ptr<std::vector<std::vector<Tensor>>> buffer_
        TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_) = nullptr;
    int64_t epoch_ TF_GUARDED_BY(mu_) = 0;
    int64_t num_elements_ TF_GUARDED_BY(mu_) = 0;
    int64_t seed_ TF_GUARDED_BY(mu_) = 0;
    int64_t seed2_ TF_GUARDED_BY(mu_) = 0;
    // Indices into `buffer_` indicating which data belongs to which epoch.
    // The slice at the front of the deque references data from the earliest
    // buffered epoch. It is an invariant that all slices reference
    // non-overlapping sections of `buffer_`.
    std::deque<std::unique_ptr<Slice>> slices_ TF_GUARDED_BY(mu_);
    random::PhiloxRandom parent_generator_ TF_GUARDED_BY(mu_);
    random::SingleSampleAdapter<random::PhiloxRandom> generator_
        TF_GUARDED_BY(mu_);
    int64_t num_random_samples_ TF_GUARDED_BY(mu_) = 0;
    bool data_produced_ TF_GUARDED_BY(mu_) = false;
  };

  const DatasetBase* const input_;
  const int64_t buffer_size_;
  const std::shared_ptr<SeedGenerator> seed_generator_;
  // The number of epochs to run for. Normally this is just 1, but sometimes we
  // fuse shuffle and repeat together, and make the shuffle dataset op
  // responsible for repeating as well.
  const int64_t count_;
  const TraceMeMetadata traceme_metadata_;
  mutable mutex mu_;
  mutable std::vector<std::int64_t> shuffled_indices_ TF_GUARDED_BY(mu_);
};  // ShuffleDatasetBase

// This version of memory dataset has an exclusive ownership of the seed
// generator resource. It supports sharing of the seed generator across
// different iterations of the `repeat` transformation but not across different
// iterators.
class ShuffleDatasetOp::Dataset : public ShuffleDatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
          int64_t count, RandomSeeds&& seeds, SeedGeneratorManager* manager,
          ResourceHandle&& resource_handle)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~Dataset() override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_25(mht_25_v, 774, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "~Dataset");

    manager_->Unref();
    Status s = resource_mgr_->Delete<SeedGeneratorManager>(
        resource_handle_.container(), resource_handle_.name());
    if (!s.ok()) {
      LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
    }
  }

  string op_type() const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_26(mht_26_v, 786, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "op_type");
 return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_27(mht_27_v, 794, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    AttrValue reshuffle_each_iteration;

    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {input_graph_node, buffer_size_node, seed_node, seed2_node},  // Inputs
        {std::make_pair(kReshuffleEachIteration,
                        reshuffle_each_iteration)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned.
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

// This version of shuffle dataset has a shared ownership of the seed generator
// resource. It supports sharing of the generator state across different
// iterations of the `repeat` transformation and also across different
// iterators.
class ShuffleDatasetOp::DatasetV2 : public ShuffleDatasetBase {
 public:
  DatasetV2(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
            int64_t count, SeedGeneratorManager* manager,
            ResourceHandle&& resource_handle, bool owns_resource)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()) {}

  ~DatasetV2() override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_28(mht_28_v, 841, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "~DatasetV2");

    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
      }
    }
  }

  string op_type() const override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_29(mht_29_v, 855, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "op_type");
 return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_30(mht_30_v, 863, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this,
        {input_graph_node, buffer_size_node, resource_handle_node},  // Inputs
        {},                                                          // Attrs
        output));
    return Status::OK();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned.
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
};

// This version of shuffle dataset extends the functionality of DatasetV2 with
// the ability to preserve seed generator configuration (i.e. initial seeds and
// whether to reshuffle each iteration) across serialization of the dataset.
class ShuffleDatasetOp::DatasetV3 : public ShuffleDatasetBase {
 public:
  DatasetV3(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
            int64_t count, RandomSeeds&& seeds, SeedGeneratorManager* manager,
            ResourceHandle&& resource_handle, bool owns_resource)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~DatasetV3() override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_31(mht_31_v, 905, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "~DatasetV3");

    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
      }
    }
  }

  string op_type() const override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_32(mht_32_v, 919, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "op_type");
 return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_33(mht_33_v, 927, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {input_graph_node, buffer_size_node, seed_node,
                       seed2_node, resource_handle_node},  // Inputs
                      {std::make_pair(kReshuffleEachIteration,
                                      reshuffle_each_iteration)},  // Attrs
                      output));
    return Status::OK();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

ShuffleDatasetOp::ShuffleDatasetOp(OpKernelConstruction* ctx)
    : ShuffleDatasetOpBase(ctx) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_34(mht_34_v, 965, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ShuffleDatasetOp::ShuffleDatasetOp");

  auto& op_name = ctx->def().op();
  if (op_name == kShuffleDatasetV3) {
    op_version_ = 3;
  } else if (op_name == kShuffleDatasetV2) {
    op_version_ = 2;
  } else if (op_name == kShuffleDatasetV1) {
    op_version_ = 1;
  }
  if (ctx->HasAttr(kReshuffleEachIteration)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kReshuffleEachIteration, &reshuffle_each_iteration_));
  }
}

void ShuffleDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                   DatasetBase** output) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_35(mht_35_v, 984, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ShuffleDatasetOp::MakeDataset");

  int64_t buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("buffer_size must be greater than zero."));

  int64_t count = 1;
  static std::atomic<int64_t> resource_id_counter(0);
  const string& container = ctx->resource_manager()->default_container();
  auto name = strings::StrCat(ctx->op_kernel().name(), "/", kSeedGenerator, "_",
                              resource_id_counter.fetch_add(1));
  if (op_version_ == 3) {
    auto handle = HandleFromInput(ctx, 4);
    SeedGeneratorManager* manager = nullptr;
    Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    int64_t seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
    int64_t seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
    RandomSeeds seeds(seed, seed2);
    bool owns_resource = false;
    if (errors::IsNotFound(s)) {
      owns_resource = true;
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
              container, name, &manager,
              [reshuffle = reshuffle_each_iteration_,
               &seeds](SeedGeneratorManager** manager) {
                if (reshuffle) {
                  *manager =
                      new SeedGeneratorManager(new RandomSeedGenerator(seeds));
                } else {
                  *manager =
                      new SeedGeneratorManager(new FixedSeedGenerator(seeds));
                }
                return Status::OK();
              }));
      handle = MakeResourceHandle<SeedGenerator>(ctx, container, name);
    } else {
      OP_REQUIRES_OK(ctx, s);
    }

    // Ownership of manager is transferred onto `DatasetV3`.
    *output = new ShuffleDatasetOp::DatasetV3(ctx, input, buffer_size, count,
                                              std::move(seeds), manager,
                                              std::move(handle), owns_resource);
  } else if (op_version_ == 2) {
    auto handle = HandleFromInput(ctx, 2);
    SeedGeneratorManager* manager = nullptr;
    Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    bool owns_resource = false;
    if (errors::IsNotFound(s)) {
      owns_resource = true;
      LOG(WARNING) << "Failed to find seed generator resource. Falling back to "
                      "using a non-deterministically seeded generator and "
                      "reshuffling each iteration.";
      RandomSeeds seeds(0, 0);
      OP_REQUIRES_OK(
          ctx, ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
                   container, name, &manager,
                   [&seeds](SeedGeneratorManager** manager) {
                     *manager = new SeedGeneratorManager(
                         new RandomSeedGenerator(seeds));
                     return Status::OK();
                   }));
      handle = MakeResourceHandle<SeedGeneratorManager>(ctx, container, name);
    } else {
      OP_REQUIRES_OK(ctx, s);
    }

    // Ownership of manager is transferred onto `DatasetV2`.
    *output =
        new ShuffleDatasetOp::DatasetV2(ctx, input, buffer_size, count, manager,
                                        std::move(handle), owns_resource);
  } else {
    if (op_version_ != 1) {
      LOG(WARNING) << "Unsupported version of shuffle dataset op: "
                   << op_version_ << ". Defaulting to version 1.";
    }
    int64_t seed;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));
    int64_t seed2;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));
    RandomSeeds seeds(seed, seed2);
    SeedGeneratorManager* manager;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
            container, name, &manager,
            [reshuffle = reshuffle_each_iteration_,
             &seeds](SeedGeneratorManager** manager) {
              if (reshuffle) {
                *manager =
                    new SeedGeneratorManager(new RandomSeedGenerator(seeds));
              } else {
                *manager =
                    new SeedGeneratorManager(new FixedSeedGenerator(seeds));
              }
              return Status::OK();
            }));
    auto handle =
        MakeResourceHandle<SeedGeneratorManager>(ctx, container, name);

    // Ownership of manager is transferred onto `Dataset`.
    *output = new ShuffleDatasetOp::Dataset(ctx, input, buffer_size, count,
                                            std::move(seeds), manager,
                                            std::move(handle));
  }
}

class ShuffleAndRepeatDatasetOp::Dataset : public ShuffleDatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
          RandomSeeds&& seeds, SeedGeneratorManager* manager, int64_t count,
          ResourceHandle&& resource_handle)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~Dataset() override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_36(mht_36_v, 1113, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "~Dataset");

    manager_->Unref();
    Status s = resource_mgr_->Delete<SeedGeneratorManager>(
        resource_handle_.container(), resource_handle_.name());
    if (!s.ok()) {
      LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
    }
  }

  string op_type() const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_37(mht_37_v, 1125, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "op_type");
 return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_38(mht_38_v, 1133, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    Node* seed = nullptr;
    Node* seed2 = nullptr;
    Node* count = nullptr;

    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2));
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count));
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size, seed, seed2, count},  // Inputs
        {std::make_pair(kReshuffleEachIteration,
                        reshuffle_each_iteration)},  // Attrs
        output));
    return Status::OK();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned.
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

class ShuffleAndRepeatDatasetOp::DatasetV2 : public ShuffleDatasetBase {
 public:
  DatasetV2(OpKernelContext* ctx, const DatasetBase* input, int64_t buffer_size,
            int64_t count, RandomSeeds&& seeds, SeedGeneratorManager* manager,
            ResourceHandle&& resource_handle, bool owns_resource)
      : ShuffleDatasetBase(ctx, input, buffer_size, manager->get(), count),
        manager_(manager),
        owns_resource_(owns_resource),
        resource_handle_(std::move(resource_handle)),
        resource_mgr_(ctx->resource_manager()),
        seeds_(std::move(seeds)) {}

  ~DatasetV2() override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_39(mht_39_v, 1178, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "~DatasetV2");

    manager_->Unref();
    if (owns_resource_) {
      Status s = resource_mgr_->Delete<SeedGeneratorManager>(
          resource_handle_.container(), resource_handle_.name());
      if (!s.ok()) {
        LOG(WARNING) << "Failed to delete RNG resource: " << s.ToString();
      }
    }
  }

  string op_type() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_40(mht_40_v, 1192, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "op_type");
 return kDatasetType; }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_41(mht_41_v, 1200, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size_node = nullptr;
    Node* seed_node = nullptr;
    Node* seed2_node = nullptr;
    Node* count_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed(), &seed_node));
    TF_RETURN_IF_ERROR(b->AddScalar(seeds_.input_seed2(), &seed2_node));
    TF_RETURN_IF_ERROR(b->AddScalar(count_, &count_node));
    Node* resource_handle_node = nullptr;
    Tensor handle(DT_RESOURCE, TensorShape({}));
    handle.scalar<ResourceHandle>()() = resource_handle_;
    TF_RETURN_IF_ERROR(b->AddTensor(handle, &resource_handle_node));
    AttrValue reshuffle_each_iteration;
    b->BuildAttrValue(seed_generator_->reshuffle_each_iteration(),
                      &reshuffle_each_iteration);
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {input_graph_node, buffer_size_node, seed_node,
                       seed2_node, count_node, resource_handle_node},  // Inputs
                      {std::make_pair(kReshuffleEachIteration,
                                      reshuffle_each_iteration)},  // Attrs
                      output));
    return Status::OK();
  }

 private:
  SeedGeneratorManager* const manager_;  // Owned
  const bool owns_resource_;
  const ResourceHandle resource_handle_;
  ResourceMgr* const resource_mgr_;  // Not owned.
  const RandomSeeds seeds_;
};

ShuffleAndRepeatDatasetOp::ShuffleAndRepeatDatasetOp(OpKernelConstruction* ctx)
    : ShuffleDatasetOpBase(ctx) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_42(mht_42_v, 1240, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ShuffleAndRepeatDatasetOp::ShuffleAndRepeatDatasetOp");

  auto& op_name = ctx->def().op();
  if (op_name == kShuffleAndRepeatDatasetV2) {
    op_version_ = 2;
  } else if (op_name == kShuffleAndRepeatDatasetV1) {
    op_version_ = 1;
  }
  if (ctx->HasAttr(kReshuffleEachIteration)) {
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr(kReshuffleEachIteration, &reshuffle_each_iteration_));
  }
}

void ShuffleAndRepeatDatasetOp::MakeDataset(OpKernelContext* ctx,
                                            DatasetBase* input,
                                            DatasetBase** output) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSshuffle_dataset_opDTcc mht_43(mht_43_v, 1258, "", "./tensorflow/core/kernels/data/shuffle_dataset_op.cc", "ShuffleAndRepeatDatasetOp::MakeDataset");

  int64_t buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(
      ctx, buffer_size > 0,
      errors::InvalidArgument("buffer_size must be greater than zero."));

  int64_t seed;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed, &seed));

  int64_t seed2;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSeed2, &seed2));

  int64_t count;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kCount, &count));

  OP_REQUIRES(ctx, count > 0 || count == -1,
              errors::InvalidArgument(
                  "count must be greater than zero or equal to -1."));

  RandomSeeds seeds(seed, seed2);

  static std::atomic<int64_t> resource_id_counter(0);
  const string& container = ctx->resource_manager()->default_container();
  auto name = strings::StrCat(ctx->op_kernel().name(), "/", kSeedGenerator, "_",
                              resource_id_counter.fetch_add(1));
  if (op_version_ == 2) {
    auto handle = HandleFromInput(ctx, 5);
    SeedGeneratorManager* manager = nullptr;
    Status s = ctx->resource_manager()->Lookup<SeedGeneratorManager>(
        handle.container(), handle.name(), &manager);
    bool owns_resource = false;
    if (errors::IsNotFound(s)) {
      owns_resource = true;
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
              container, name, &manager,
              [reshuffle = reshuffle_each_iteration_,
               &seeds](SeedGeneratorManager** manager) {
                if (reshuffle) {
                  *manager =
                      new SeedGeneratorManager(new RandomSeedGenerator(seeds));
                } else {
                  *manager =
                      new SeedGeneratorManager(new FixedSeedGenerator(seeds));
                }
                return Status::OK();
              }));
      handle = MakeResourceHandle<SeedGenerator>(ctx, container, name);
    } else {
      OP_REQUIRES_OK(ctx, s);
    }

    // Ownership of manager is transferred onto `DatasetV2`.
    *output = new ShuffleAndRepeatDatasetOp::DatasetV2(
        ctx, input, buffer_size, count, std::move(seeds), manager,
        std::move(handle), owns_resource);
  } else {
    if (op_version_ != 1) {
      LOG(WARNING) << "Unsupported version of shuffle dataset op: "
                   << op_version_ << ". Defaulting to version 1.";
    }
    SeedGeneratorManager* manager;
    OP_REQUIRES_OK(
        ctx,
        ctx->resource_manager()->LookupOrCreate<SeedGeneratorManager>(
            container, name, &manager,
            [reshuffle = reshuffle_each_iteration_,
             &seeds](SeedGeneratorManager** manager) {
              if (reshuffle) {
                *manager =
                    new SeedGeneratorManager(new RandomSeedGenerator(seeds));
              } else {
                *manager =
                    new SeedGeneratorManager(new FixedSeedGenerator(seeds));
              }
              return Status::OK();
            }));
    auto handle =
        MakeResourceHandle<SeedGeneratorManager>(ctx, container, name);

    // Ownership of manager is transferred onto `Dataset`.
    *output = new Dataset(ctx, input, buffer_size, std::move(seeds), manager,
                          count, std::move(handle));
  }
}

namespace {
REGISTER_KERNEL_BUILDER(Name("ShuffleDataset").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleDatasetV2").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleDatasetV3").Device(DEVICE_CPU),
                        ShuffleDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleAndRepeatDataset").Device(DEVICE_CPU),
                        ShuffleAndRepeatDatasetOp);

REGISTER_KERNEL_BUILDER(Name("ShuffleAndRepeatDatasetV2").Device(DEVICE_CPU),
                        ShuffleAndRepeatDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
