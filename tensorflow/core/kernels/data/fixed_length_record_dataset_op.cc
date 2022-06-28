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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/fixed_length_record_dataset_op.h"

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace data {

/* static */ constexpr const char* const
    FixedLengthRecordDatasetOp::kDatasetType;
/* static */ constexpr const char* const FixedLengthRecordDatasetOp::kFileNames;
/* static */ constexpr const char* const
    FixedLengthRecordDatasetOp::kHeaderBytes;
/* static */ constexpr const char* const
    FixedLengthRecordDatasetOp::kRecordBytes;
/* static */ constexpr const char* const
    FixedLengthRecordDatasetOp::kFooterBytes;
/* static */ constexpr const char* const
    FixedLengthRecordDatasetOp::kBufferSize;
/* static */ constexpr const char* const
    FixedLengthRecordDatasetOp::kCompressionType;

constexpr char kFixedLengthRecordDataset[] = "FixedLengthRecordDataset";
constexpr char kCurrentFileIndex[] = "current_file_index";
constexpr char kCurrentPos[] = "current_pos";
constexpr char kZLIB[] = "ZLIB";
constexpr char kGZIP[] = "GZIP";

class FixedLengthRecordDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                   int64_t header_bytes, int64_t record_bytes,
                   int64_t footer_bytes, int64_t buffer_size,
                   const string& compression_type, int op_version)
      : DatasetBase(DatasetContext(ctx)),
        filenames_(std::move(filenames)),
        header_bytes_(header_bytes),
        record_bytes_(record_bytes),
        footer_bytes_(footer_bytes),
        buffer_size_(buffer_size),
        compression_type_(compression_type),
        op_version_(op_version) {}

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    params.op_version = op_version_;
    if (compression_type_.empty()) {
      return absl::make_unique<UncompressedIterator>(
          UncompressedIterator::Params{
              this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
    } else {
      return absl::make_unique<CompressedIterator>(CompressedIterator::Params{
          this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
    }
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_0(mht_0_v, 249, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "output_dtypes");

    static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_1(mht_1_v, 257, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "output_shapes");

    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_2(mht_2_v, 266, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "DebugString");

    name_utils::DatasetDebugStringParams params;
    params.op_version = op_version_;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_4(mht_4_v, 282, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_5(mht_5_v, 290, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "AsGraphDefInternal");

    Node* filenames = nullptr;
    Node* header_bytes = nullptr;
    Node* record_bytes = nullptr;
    Node* footer_bytes = nullptr;
    Node* buffer_size = nullptr;
    Node* compression_type = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    TF_RETURN_IF_ERROR(b->AddScalar(header_bytes_, &header_bytes));
    TF_RETURN_IF_ERROR(b->AddScalar(record_bytes_, &record_bytes));
    TF_RETURN_IF_ERROR(b->AddScalar(footer_bytes_, &footer_bytes));
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      {filenames, header_bytes, record_bytes, footer_bytes,
                       buffer_size, compression_type},
                      output));
    return Status::OK();
  }

 private:
  class UncompressedIterator : public DatasetIterator<Dataset> {
   public:
    explicit UncompressedIterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_6(mht_6_v, 318, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "UncompressedIterator");
}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_7(mht_7_v, 325, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "GetNextInternal");

      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to read the next record.
        if (input_buffer_) {
          const int64_t current_pos = input_buffer_->Tell();
          DCHECK_GE(file_pos_limit_, 0);
          if (current_pos < file_pos_limit_) {
            string record;
            TF_RETURN_IF_ERROR(
                input_buffer_->ReadNBytes(dataset()->record_bytes_, &record));
            static monitoring::CounterCell* bytes_counter =
                metrics::GetTFDataBytesReadCounter(kDatasetType);
            bytes_counter->IncrementBy(dataset()->record_bytes_);

            // Produce the record as output.
            Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
            record_tensor.scalar<tstring>()() = record;
            out_tensors->emplace_back(std::move(record_tensor));
            *end_of_sequence = false;
            return Status::OK();
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          input_buffer_.reset();
          file_.reset();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        // Actually move on to next file.
        uint64 file_size;
        const std::string& next_filename =
            dataset()->filenames_[current_file_index_];
        TF_RETURN_IF_ERROR(ctx->env()->GetFileSize(next_filename, &file_size));
        file_pos_limit_ = file_size - dataset()->footer_bytes_;

        uint64 body_size =
            file_size - (dataset()->header_bytes_ + dataset()->footer_bytes_);

        if (body_size % dataset()->record_bytes_ != 0) {
          return errors::InvalidArgument(
              "Excluding the header (", dataset()->header_bytes_,
              " bytes) and footer (", dataset()->footer_bytes_,
              " bytes), input file \"", next_filename, "\" has body length ",
              body_size,
              " bytes, which is not an exact multiple of the record length (",
              dataset()->record_bytes_, " bytes).");
        }
        TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
            TranslateFileName(next_filename), &file_));
        input_buffer_ = absl::make_unique<io::InputBuffer>(
            file_.get(), dataset()->buffer_size_);
        TF_RETURN_IF_ERROR(input_buffer_->SkipNBytes(dataset()->header_bytes_));
      } while (true);
    }

   protected:
    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_8(mht_8_v, 393, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentFileIndex),
                                             current_file_index_));

      // `input_buffer_` is empty if
      // 1. GetNext has not been called even once.
      // 2. All files have been read and iterator has been exhausted.
      int64_t current_pos = input_buffer_ ? input_buffer_->Tell() : -1;
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCurrentPos), current_pos));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_9(mht_9_v, 411, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      int64_t current_file_index;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentFileIndex),
                                            &current_file_index));
      current_file_index_ = size_t(current_file_index);
      int64_t current_pos;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCurrentPos), &current_pos));

      // Seek to current_pos.
      input_buffer_.reset();
      file_.reset();
      if (current_pos >= 0) {  // There was an active input_buffer_.
        uint64 file_size;
        const std::string& current_filename =
            dataset()->filenames_[current_file_index_];
        TF_RETURN_IF_ERROR(
            ctx->env()->GetFileSize(current_filename, &file_size));
        file_pos_limit_ = file_size - dataset()->footer_bytes_;
        TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
            TranslateFileName(current_filename), &file_));
        input_buffer_ = absl::make_unique<io::InputBuffer>(
            file_.get(), dataset()->buffer_size_);
        TF_RETURN_IF_ERROR(input_buffer_->Seek(current_pos));
      }

      return Status::OK();
    }

   private:
    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;
    std::unique_ptr<RandomAccessFile> file_
        TF_GUARDED_BY(mu_);  // must outlive input_buffer_
    std::unique_ptr<io::InputBuffer> input_buffer_ TF_GUARDED_BY(mu_);
    int64_t file_pos_limit_ TF_GUARDED_BY(mu_) = -1;
  };

  class CompressedIterator : public DatasetIterator<Dataset> {
   public:
    explicit CompressedIterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_10(mht_10_v, 456, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "CompressedIterator");
}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_11(mht_11_v, 463, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "GetNextInternal");

      static monitoring::CounterCell* bytes_counter =
          metrics::GetTFDataBytesReadCounter(kDatasetType);
      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to read the next record.
        if (buffered_input_stream_) {
          const int64_t current_pos = buffered_input_stream_->Tell();
          if (dataset()->compression_type_.empty()) {
            DCHECK_GE(file_pos_limit_, 0);
            if (current_pos < file_pos_limit_) {
              tstring record;
              TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(
                  dataset()->record_bytes_, &record));
              bytes_counter->IncrementBy(dataset()->record_bytes_);

              // Produce the record as output.
              Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
              record_tensor.scalar<tstring>()() = std::move(record);
              out_tensors->emplace_back(std::move(record_tensor));
              *end_of_sequence = false;
              return Status::OK();
            }
          } else {
            tstring record;
            Status s = buffered_input_stream_->ReadNBytes(
                dataset()->record_bytes_, &record);
            if (s.ok()) {
              bytes_counter->IncrementBy(dataset()->record_bytes_);
              lookahead_cache_.append(record);
              StringPiece lookahead_cache_view(lookahead_cache_);
              record = tstring(
                  lookahead_cache_view.substr(0, dataset()->record_bytes_));
              lookahead_cache_ = tstring(
                  lookahead_cache_view.substr(dataset()->record_bytes_));
              // Produce the record as output.
              Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
              record_tensor.scalar<tstring>()() = std::move(record);
              out_tensors->emplace_back(std::move(record_tensor));
              *end_of_sequence = false;
              return Status::OK();
            }
            if (errors::IsOutOfRange(s) && !record.empty()) {
              uint64 body_size =
                  current_pos + record.size() -
                  (dataset()->header_bytes_ + dataset()->footer_bytes_);
              return errors::DataLoss(
                  "Excluding the header (", dataset()->header_bytes_,
                  " bytes) and footer (", dataset()->footer_bytes_,
                  " bytes), input file \"",
                  dataset()->filenames_[current_file_index_],
                  "\" has body length ", body_size,
                  " bytes, which is not an exact multiple of the record "
                  "length (",
                  dataset()->record_bytes_, " bytes).");
            }
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          buffered_input_stream_.reset();
          file_.reset();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        // Actually move on to next file.
        if (dataset()->compression_type_.empty()) {
          uint64 file_size;
          TF_RETURN_IF_ERROR(ctx->env()->GetFileSize(
              dataset()->filenames_[current_file_index_], &file_size));
          file_pos_limit_ = file_size - dataset()->footer_bytes_;

          uint64 body_size =
              file_size - (dataset()->header_bytes_ + dataset()->footer_bytes_);

          if (body_size % dataset()->record_bytes_ != 0) {
            return errors::InvalidArgument(
                "Excluding the header (", dataset()->header_bytes_,
                " bytes) and footer (", dataset()->footer_bytes_,
                " bytes), input file \"",
                dataset()->filenames_[current_file_index_],
                "\" has body length ", body_size,
                " bytes, which is not an exact multiple of the record length "
                "(",
                dataset()->record_bytes_, " bytes).");
          }
        }
        TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
            TranslateFileName(dataset()->filenames_[current_file_index_]),
            &file_));
        if (!dataset()->compression_type_.empty()) {
          const io::ZlibCompressionOptions zlib_options =
              dataset()->compression_type_ == kZLIB
                  ? io::ZlibCompressionOptions::DEFAULT()
                  : io::ZlibCompressionOptions::GZIP();
          file_stream_ =
              absl::make_unique<io::RandomAccessInputStream>(file_.get());
          buffered_input_stream_ = absl::make_unique<io::ZlibInputStream>(
              file_stream_.get(), dataset()->buffer_size_,
              dataset()->buffer_size_, zlib_options);
        } else {
          buffered_input_stream_ = absl::make_unique<io::BufferedInputStream>(
              file_.get(), dataset()->buffer_size_);
        }
        TF_RETURN_IF_ERROR(
            buffered_input_stream_->SkipNBytes(dataset()->header_bytes_));
        lookahead_cache_.clear();
        if (!dataset()->compression_type_.empty()) {
          TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(
              dataset()->footer_bytes_, &lookahead_cache_));
        }
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_12(mht_12_v, 593, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentFileIndex),
                                             current_file_index_));

      // `buffered_input_stream_` is empty if
      // 1. GetNext has not been called even once.
      // 2. All files have been read and iterator has been exhausted.
      int64_t current_pos =
          buffered_input_stream_ ? buffered_input_stream_->Tell() : -1;
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kCurrentPos), current_pos));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_13(mht_13_v, 612, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      int64_t current_file_index;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentFileIndex),
                                            &current_file_index));
      current_file_index_ = size_t(current_file_index);
      int64_t current_pos;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kCurrentPos), &current_pos));

      // Seek to current_pos.
      buffered_input_stream_.reset();
      file_.reset();
      if (current_pos >= 0) {  // There was an active buffered_input_stream_.
        TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(
            TranslateFileName(dataset()->filenames_[current_file_index_]),
            &file_));
        const io::ZlibCompressionOptions zlib_options =
            dataset()->compression_type_ == kZLIB
                ? io::ZlibCompressionOptions::DEFAULT()
                : io::ZlibCompressionOptions::GZIP();
        file_stream_ =
            absl::make_unique<io::RandomAccessInputStream>(file_.get());
        buffered_input_stream_ = absl::make_unique<io::ZlibInputStream>(
            file_stream_.get(), dataset()->buffer_size_,
            dataset()->buffer_size_, zlib_options);
        lookahead_cache_.clear();
        TF_RETURN_IF_ERROR(buffered_input_stream_->SkipNBytes(
            current_pos - dataset()->footer_bytes_));
        TF_RETURN_IF_ERROR(buffered_input_stream_->ReadNBytes(
            dataset()->footer_bytes_, &lookahead_cache_));
      }

      return Status::OK();
    }

   private:
    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;
    std::unique_ptr<RandomAccessFile> file_
        TF_GUARDED_BY(mu_);  // must outlive buffered_input_stream_
    std::unique_ptr<io::RandomAccessInputStream>
        file_stream_;  // must outlive buffered_input_stream_
    std::unique_ptr<io::InputStreamInterface> buffered_input_stream_
        TF_GUARDED_BY(mu_);
    int64_t file_pos_limit_ TF_GUARDED_BY(mu_) = -1;
    tstring lookahead_cache_ TF_GUARDED_BY(mu_);
  };

  const std::vector<string> filenames_;
  const int64_t header_bytes_;
  const int64_t record_bytes_;
  const int64_t footer_bytes_;
  const int64_t buffer_size_;
  const tstring compression_type_;
  const int op_version_;
};

FixedLengthRecordDatasetOp::FixedLengthRecordDatasetOp(
    OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx),
      op_version_(ctx->def().op() == kFixedLengthRecordDataset ? 1 : 2) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_14(mht_14_v, 676, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "FixedLengthRecordDatasetOp::FixedLengthRecordDatasetOp");
}

void FixedLengthRecordDatasetOp::MakeDataset(OpKernelContext* ctx,
                                             DatasetBase** output) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSfixed_length_record_dataset_opDTcc mht_15(mht_15_v, 682, "", "./tensorflow/core/kernels/data/fixed_length_record_dataset_op.cc", "FixedLengthRecordDatasetOp::MakeDataset");

  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(kFileNames, &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  std::vector<string> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
    metrics::RecordTFDataFilename(kDatasetType, filenames[i]);
  }

  int64_t header_bytes = -1;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<int64_t>(ctx, kHeaderBytes, &header_bytes));
  OP_REQUIRES(ctx, header_bytes >= 0,
              errors::InvalidArgument("`header_bytes` must be >= 0"));

  int64_t record_bytes = -1;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<int64_t>(ctx, kRecordBytes, &record_bytes));
  OP_REQUIRES(ctx, record_bytes > 0,
              errors::InvalidArgument("`record_bytes` must be > 0"));

  int64_t footer_bytes = -1;
  OP_REQUIRES_OK(
      ctx, ParseScalarArgument<int64_t>(ctx, kFooterBytes, &footer_bytes));
  OP_REQUIRES(ctx, footer_bytes >= 0,
              errors::InvalidArgument("`footer_bytes` must be >= 0"));

  int64_t buffer_size = -1;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0,
              errors::InvalidArgument("`buffer_size` must be >= 0"));
  if (buffer_size == 0) {
    buffer_size = 256 << 10;  // 256 kB as default.
  }
  tstring compression_type;
  if (op_version_ > 1) {
    OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kCompressionType,
                                                     &compression_type));
    OP_REQUIRES(ctx,
                compression_type.empty() || compression_type == kZLIB ||
                    compression_type == kGZIP,
                errors::InvalidArgument("Unsupported compression_type."));
  }
  *output =
      new Dataset(ctx, std::move(filenames), header_bytes, record_bytes,
                  footer_bytes, buffer_size, compression_type, op_version_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordDataset").Device(DEVICE_CPU),
                        FixedLengthRecordDatasetOp);
REGISTER_KERNEL_BUILDER(Name("FixedLengthRecordDatasetV2").Device(DEVICE_CPU),
                        FixedLengthRecordDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
