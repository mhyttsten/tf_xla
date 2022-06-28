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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/window_dataset_op.h"

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/window_dataset.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const WindowDatasetOp::kDatasetType;
/* static */ constexpr const char* const WindowDatasetOp::kInputDataset;
/* static */ constexpr const char* const WindowDatasetOp::kSize;
/* static */ constexpr const char* const WindowDatasetOp::kShift;
/* static */ constexpr const char* const WindowDatasetOp::kStride;
/* static */ constexpr const char* const WindowDatasetOp::kDropRemainder;
/* static */ constexpr const char* const WindowDatasetOp::kOutputTypes;
/* static */ constexpr const char* const WindowDatasetOp::kOutputShapes;

constexpr char kInputImplEmpty[] = "input_impl_empty";
constexpr char kBufferSize[] = "buffer_size";
constexpr char kBuffer[] = "buffer";
constexpr char kSizeSuffix[] = ".size";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessage[] = ".error_message";

class WindowDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64_t window_size,
          int64_t window_shift, int64_t window_stride, bool drop_remainder)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        window_size_(window_size),
        window_shift_(window_shift),
        window_stride_(window_stride),
        drop_remainder_(drop_remainder),
        output_dtypes_(input_->output_dtypes().size(), {DT_VARIANT}),
        output_shapes_(input_->output_shapes().size(), TensorShape({})),
        traceme_metadata_(
            {{"window_size",
              strings::Printf("%lld", static_cast<long long>(window_size))},
             {"window_shift",
              strings::Printf("%lld", static_cast<long long>(window_shift))},
             {"window_stride", strings::Printf("%lld", static_cast<long long>(
                                                           window_stride))}}) {
    input_->Ref();
  }

  ~Dataset() override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_0(mht_0_v, 235, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "~Dataset");
 input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "output_dtypes");

    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_2(mht_2_v, 253, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "output_shapes");

    return output_shapes_;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.set_args(window_size_, window_shift_, window_stride_,
                    drop_remainder_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_4(mht_4_v, 270, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "CardinalityInternal");

    int64_t n = input_->Cardinality();
    if (n == kInfiniteCardinality || n == kUnknownCardinality) {
      return n;
    }
    int64_t cardinality = 0;
    if (drop_remainder_) {
      // Compute rest_elements, the number of elements after the last element
      // of the initial window. If it is negative, we know that the
      // cardinality is 0. Otherwise, it will be the number of valid shifts
      // over the rest_elements.
      int64_t rest_elements = n - ((window_size_ - 1) * window_stride_ + 1);
      cardinality = rest_elements < 0 ? 0 : rest_elements / window_shift_ + 1;
    } else {
      cardinality = n / window_shift_ + (n % window_shift_ == 0 ? 0 : 1);
    }
    return cardinality;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_5(mht_5_v, 292, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "InputDatasets");

    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_6(mht_6_v, 300, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "CheckExternalState");

    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_7(mht_7_v, 310, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "AsGraphDefInternal");

    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* window_size_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(window_size_, &window_size_node));
    Node* window_shift_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(window_shift_, &window_shift_node));
    Node* window_stride_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(window_stride_, &window_stride_node));
    Node* drop_remainder_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(drop_remainder_, &drop_remainder_node));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {input_graph_node, window_size_node, window_shift_node,
                       window_stride_node, drop_remainder_node},
                      output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_8(mht_8_v, 336, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_9(mht_9_v, 341, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "Initialize");

      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_10(mht_10_v, 350, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "GetNextInternal");

      const int64_t window_size = dataset()->window_size_;
      const int64_t window_shift = dataset()->window_shift_;
      const int64_t window_stride = dataset()->window_stride_;
      std::vector<std::vector<Tensor>> window_elements;
      Status status = Status::OK();
      {
        const size_t target_size = TargetBufferSize(window_size, window_stride);

        mutex_lock l(mu_);
        if (!input_impl_ &&
            (buffer_.empty() ||
             (dataset()->drop_remainder_ && buffer_.size() < target_size))) {
          *end_of_sequence = true;
          return Status::OK();
        }

        // Add elements to the buffer.
        if (input_impl_) {
          *end_of_sequence = false;
          for (size_t i = buffer_.size(); i < target_size && !*end_of_sequence;
               ++i) {
            std::vector<Tensor> element;
            Status status =
                input_impl_->GetNext(ctx, &element, end_of_sequence);
            if (!*end_of_sequence) {
              RecordBufferEnqueue(ctx, element);
              buffer_.emplace_back(std::move(element), status);
            } else {
              input_impl_.reset();
            }
          }
        }

        // If there are not enough elements and `drop_remainder` is set, we do
        // not wish to return a smaller window.
        if (buffer_.empty() ||
            (dataset()->drop_remainder_ && buffer_.size() < target_size)) {
          DCHECK(*end_of_sequence);
          return Status::OK();
        }

        int num_elements = 1 + (buffer_.size() - 1) / window_stride;
        window_elements.reserve(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
          status.Update(buffer_[window_stride * i].status);
          if (!status.ok()) {
            break;
          }
          window_elements.emplace_back(buffer_[window_stride * i].result);
        }

        // Shift the window, discarding elements if necessary.
        int buffer_size = buffer_.size();
        if (window_shift >= buffer_size) {
          for (size_t i = buffer_size; input_impl_ && i < window_shift; ++i) {
            bool end_of_input;
            std::vector<Tensor> element;
            // Ignore non-error status of discarded elements.
            input_impl_->GetNext(ctx, &element, &end_of_input).IgnoreError();
            if (end_of_input) {
              input_impl_.reset();
            }
          }
          for (size_t i = 0; i < buffer_.size(); ++i) {
            RecordBufferDequeue(ctx, buffer_.at(i).result);
          }
          buffer_.clear();
        } else {
          for (size_t i = 0; i < window_shift; ++i) {
            RecordBufferDequeue(ctx, buffer_.at(i).result);
          }
          buffer_.erase(buffer_.begin(), buffer_.begin() + window_shift);
        }
      }

      if (!status.ok()) {
        return status;
      }

      // Construct output tensors.
      const size_t num_tuple_components = window_elements[0].size();
      const int64_t num_window_elements = window_elements.size();
      *end_of_sequence = false;
      for (size_t idx = 0; idx < num_tuple_components; ++idx) {
        DatasetBase* window_dataset;
        std::vector<std::vector<Tensor>> window_component_elements;
        window_component_elements.reserve(num_window_elements);
        // Build the output tuple component by copying one slice
        // from each input element in the window.
        for (size_t i = 0; i < num_window_elements; ++i) {
          std::vector<Tensor> component_element;
          component_element.push_back(std::move(window_elements[i][idx]));
          window_component_elements.push_back(component_element);
        }
        DataTypeVector output_types({dataset()->input_->output_dtypes()[idx]});
        std::vector<PartialTensorShape> output_shapes(
            {dataset()->input_->output_shapes()[idx]});
        TF_RETURN_IF_ERROR(NewWindow(window_component_elements, output_types,
                                     output_shapes, &window_dataset));
        out_tensors->emplace_back(DT_VARIANT, TensorShape({}));
        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(window_dataset, &out_tensors->back()));
      }
      return Status::OK();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeKnownRatioNode(std::move(args),
                                       dataset()->window_shift_);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_11(mht_11_v, 468, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      if (!input_impl_) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kInputImplEmpty), ""));
      } else {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      }
      // Save buffer.
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kBufferSize), buffer_.size()));
      for (int64_t i = 0; i < buffer_.size(); i++) {
        TF_RETURN_IF_ERROR(WriteStatusLocked(writer, i, buffer_[i].status));
        TF_RETURN_IF_ERROR(writer->WriteScalar(
            full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
            buffer_[i].result.size()));
        for (int64_t j = 0; j < buffer_[i].result.size(); j++) {
          TF_RETURN_IF_ERROR(writer->WriteTensor(
              full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
              buffer_[i].result[j]));
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_12(mht_12_v, 496, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      if (!reader->Contains(full_name(kInputImplEmpty))) {
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      } else {
        input_impl_.reset();
      }
      // Restore buffer.
      int64_t buffer_size = 0;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kBufferSize), &buffer_size));
      buffer_.resize(buffer_size);
      for (int64_t i = 0; i < buffer_size; i++) {
        int64_t vector_size;
        TF_RETURN_IF_ERROR(ReadStatusLocked(reader, i, &buffer_[i].status));
        TF_RETURN_IF_ERROR(reader->ReadScalar(
            full_name(strings::StrCat(kBuffer, "[", i, "]", kSizeSuffix)),
            &vector_size));
        buffer_[i].result.resize(vector_size);
        for (int64_t j = 0; j < vector_size; j++) {
          TF_RETURN_IF_ERROR(reader->ReadTensor(
              ctx->flr(),
              full_name(strings::StrCat(kBuffer, "[", i, "][", j, "]")),
              &buffer_[i].result[j]));
        }
      }
      return Status::OK();
    }

    TraceMeMetadata GetTraceMeMetadata() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_13(mht_13_v, 528, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "GetTraceMeMetadata");

      return dataset()->traceme_metadata_;
    }

   private:
    struct InvocationResult {
      InvocationResult() = default;
      InvocationResult(std::vector<Tensor>&& result, const Status& status)
          : result(result), status(status) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_14(mht_14_v, 539, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "InvocationResult");
}

      std::vector<Tensor> result;
      Status status;
    };

    Status WriteStatusLocked(IteratorStateWriter* writer, size_t index,
                             const Status& status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_15(mht_15_v, 550, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "WriteStatusLocked");

      TF_RETURN_IF_ERROR(writer->WriteScalar(
          CodeKey(index), static_cast<int64_t>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                               status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatusLocked(IteratorStateReader* reader, size_t index,
                            Status* status) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_16(mht_16_v, 564, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "ReadStatusLocked");

      int64_t code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(ErrorMessageKey(index), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey(size_t index) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_17(mht_17_v, 583, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "CodeKey");

      return full_name(strings::StrCat(kBuffer, "[", index, "]", kCodeSuffix));
    }

    string ErrorMessageKey(size_t index) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_18(mht_18_v, 590, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "ErrorMessageKey");

      return full_name(
          strings::StrCat(kBuffer, "[", index, "]", kErrorMessage));
    }

    size_t TargetBufferSize(int64_t window_size, int64_t window_stride) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_19(mht_19_v, 598, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "TargetBufferSize");

      return (window_size - 1) * window_stride + 1;
    }

    mutex mu_;
    std::deque<InvocationResult> buffer_ TF_GUARDED_BY(mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
  };

  const DatasetBase* const input_;
  const int64_t window_size_;
  const int64_t window_shift_;
  const int64_t window_stride_;
  const bool drop_remainder_;
  const DataTypeVector output_dtypes_;
  const std::vector<PartialTensorShape> output_shapes_;
  const TraceMeMetadata traceme_metadata_;
};

WindowDatasetOp::WindowDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_20(mht_20_v, 621, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "WindowDatasetOp::WindowDatasetOp");
}

void WindowDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                  DatasetBase** output) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSwindow_dataset_opDTcc mht_21(mht_21_v, 627, "", "./tensorflow/core/kernels/data/window_dataset_op.cc", "WindowDatasetOp::MakeDataset");

  int64_t window_size = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kSize, &window_size));
  OP_REQUIRES(
      ctx, window_size > 0,
      errors::InvalidArgument("Window size must be greater than zero."));

  int64_t window_shift = 0;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kShift, &window_shift));
  OP_REQUIRES(
      ctx, window_shift > 0,
      errors::InvalidArgument("Window shift must be greater than zero."));

  int64_t window_stride = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kStride, &window_stride));
  OP_REQUIRES(
      ctx, window_stride > 0,
      errors::InvalidArgument("Window stride must be greater than zero."));

  bool drop_remainder;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<bool>(ctx, kDropRemainder, &drop_remainder));

  *output = new Dataset(ctx, input, window_size, window_shift, window_stride,
                        drop_remainder);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("WindowDataset").Device(DEVICE_CPU),
                        WindowDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
