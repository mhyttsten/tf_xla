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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc() {
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
#include "tensorflow/core/kernels/data/tf_record_dataset_op.h"

#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/utils.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following ops.

/* static */ constexpr const char* const TFRecordDatasetOp::kDatasetType;
/* static */ constexpr const char* const TFRecordDatasetOp::kFileNames;
/* static */ constexpr const char* const TFRecordDatasetOp::kCompressionType;
/* static */ constexpr const char* const TFRecordDatasetOp::kBufferSize;

constexpr char kCurrentFileIndex[] = "current_file_index";
constexpr char kOffset[] = "offset";
constexpr char kGcsFsPrefix[] = "gs://";
constexpr char kS3FsPrefix[] = "s3://";
constexpr int64_t kCloudTpuBlockSize = 127LL << 20;  // 127MB.
constexpr int64_t kS3BlockSize = kCloudTpuBlockSize;

bool is_cloud_tpu_gcs_fs() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_0(mht_0_v, 216, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "is_cloud_tpu_gcs_fs");

#if (defined(PLATFORM_CLOUD_TPU) && defined(TPU_GCS_FS)) || \
    defined(LIBTPU_ON_GCE)
  return true;
#endif
  return false;
}

class TFRecordDatasetOp::Dataset : public DatasetBase {
 public:
  explicit Dataset(OpKernelContext* ctx, std::vector<string> filenames,
                   const string& compression_type, int64_t buffer_size)
      : DatasetBase(DatasetContext(ctx)),
        filenames_(std::move(filenames)),
        compression_type_(compression_type),
        options_(io::RecordReaderOptions::CreateRecordReaderOptions(
            compression_type)) {
    if (buffer_size > 0) {
      options_.buffer_size = buffer_size;
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_1(mht_1_v, 247, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "output_dtypes");

    static DataTypeVector* dtypes = new DataTypeVector({DT_STRING});
    return *dtypes;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_2(mht_2_v, 255, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "output_shapes");

    static std::vector<PartialTensorShape>* shapes =
        new std::vector<PartialTensorShape>({{}});
    return *shapes;
  }

  string DebugString() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_3(mht_3_v, 264, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "DebugString");

    return name_utils::DatasetDebugString(kDatasetType);
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_4(mht_4_v, 271, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_6(mht_6_v, 286, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "AsGraphDefInternal");

    Node* filenames = nullptr;
    TF_RETURN_IF_ERROR(b->AddVector(filenames_, &filenames));
    Node* compression_type = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(compression_type_, &compression_type));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {filenames, compression_type, buffer_size}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_7(mht_7_v, 305, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "Iterator");
}

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_8(mht_8_v, 312, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "GetNextInternal");

      out_tensors->reserve(1);
      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to read the next record.
        if (reader_) {
          out_tensors->emplace_back(ctx->allocator({}), DT_STRING,
                                    TensorShape({}));
          Status s =
              reader_->ReadRecord(&out_tensors->back().scalar<tstring>()());
          if (s.ok()) {
            static monitoring::CounterCell* bytes_counter =
                metrics::GetTFDataBytesReadCounter(kDatasetType);
            bytes_counter->IncrementBy(
                out_tensors->back().scalar<tstring>()().size());
            *end_of_sequence = false;
            return Status::OK();
          }
          out_tensors->pop_back();
          if (!errors::IsOutOfRange(s)) {
            // In case of other errors e.g., DataLoss, we still move forward
            // the file index so that it works with ignore_errors.
            // Otherwise the same file will repeat.
            ResetStreamsLocked();
            ++current_file_index_;
            return s;
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          ResetStreamsLocked();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

    Status SkipInternal(IteratorContext* ctx, int num_to_skip,
                        bool* end_of_sequence, int* num_skipped) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_9(mht_9_v, 360, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "SkipInternal");

      *num_skipped = 0;
      mutex_lock l(mu_);
      do {
        // We are currently processing a file, so try to skip reading
        // the next (num_to_skip - *num_skipped) record.
        if (reader_) {
          int last_num_skipped;
          Status s = reader_->SkipRecords(num_to_skip - *num_skipped,
                                          &last_num_skipped);
          *num_skipped += last_num_skipped;
          if (s.ok()) {
            *end_of_sequence = false;
            return Status::OK();
          }
          if (!errors::IsOutOfRange(s)) {
            // In case of other errors e.g., DataLoss, we still move forward
            // the file index so that it works with ignore_errors.
            // Otherwise the same file will repeat.
            ResetStreamsLocked();
            ++current_file_index_;
            return s;
          }

          // We have reached the end of the current file, so maybe move on to
          // next file.
          ResetStreamsLocked();
          ++current_file_index_;
        }

        // Iteration ends when there are no more files to process.
        if (current_file_index_ == dataset()->filenames_.size()) {
          *end_of_sequence = true;
          return Status::OK();
        }

        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
      } while (true);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeSourceNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_10(mht_10_v, 410, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "SaveInternal");

      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentFileIndex),
                                             current_file_index_));

      if (reader_) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(full_name(kOffset), reader_->TellOffset()));
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_11(mht_11_v, 426, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "RestoreInternal");

      mutex_lock l(mu_);
      ResetStreamsLocked();
      int64_t current_file_index;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentFileIndex),
                                            &current_file_index));
      current_file_index_ = size_t(current_file_index);
      if (reader->Contains(full_name(kOffset))) {
        int64_t offset;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kOffset), &offset));
        TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        TF_RETURN_IF_ERROR(reader_->SeekOffset(offset));
      }
      return Status::OK();
    }

   private:
    // Sets up reader streams to read from the file at `current_file_index_`.
    Status SetupStreamsLocked(Env* env) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_12(mht_12_v, 447, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "SetupStreamsLocked");

      if (current_file_index_ >= dataset()->filenames_.size()) {
        return errors::InvalidArgument(
            "current_file_index_:", current_file_index_,
            " >= filenames_.size():", dataset()->filenames_.size());
      }

      // Actually move on to next file.
      TF_RETURN_IF_ERROR(env->NewRandomAccessFile(
          TranslateFileName(dataset()->filenames_[current_file_index_]),
          &file_));
      reader_ = absl::make_unique<io::SequentialRecordReader>(
          file_.get(), dataset()->options_);
      return Status::OK();
    }

    // Resets all reader streams.
    void ResetStreamsLocked() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_13(mht_13_v, 467, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "ResetStreamsLocked");

      reader_.reset();
      file_.reset();
    }

    mutex mu_;
    size_t current_file_index_ TF_GUARDED_BY(mu_) = 0;

    // `reader_` will borrow the object that `file_` points to, so
    // we must destroy `reader_` before `file_`.
    std::unique_ptr<RandomAccessFile> file_ TF_GUARDED_BY(mu_);
    std::unique_ptr<io::SequentialRecordReader> reader_ TF_GUARDED_BY(mu_);
  };

  const std::vector<string> filenames_;
  const tstring compression_type_;
  io::RecordReaderOptions options_;
};

TFRecordDatasetOp::TFRecordDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_14(mht_14_v, 490, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "TFRecordDatasetOp::TFRecordDatasetOp");
}

void TFRecordDatasetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPStf_record_dataset_opDTcc mht_15(mht_15_v, 496, "", "./tensorflow/core/kernels/data/tf_record_dataset_op.cc", "TFRecordDatasetOp::MakeDataset");

  const Tensor* filenames_tensor;
  OP_REQUIRES_OK(ctx, ctx->input(kFileNames, &filenames_tensor));
  OP_REQUIRES(
      ctx, filenames_tensor->dims() <= 1,
      errors::InvalidArgument("`filenames` must be a scalar or a vector."));

  bool is_gcs_fs = true;
  bool is_s3_fs = true;
  std::vector<string> filenames;
  filenames.reserve(filenames_tensor->NumElements());
  for (int i = 0; i < filenames_tensor->NumElements(); ++i) {
    VLOG(2) << "Reading file: " << filenames_tensor->flat<tstring>()(i);
    filenames.push_back(filenames_tensor->flat<tstring>()(i));
    is_gcs_fs &= absl::StartsWith(filenames[i], kGcsFsPrefix);
    is_s3_fs &= absl::StartsWith(filenames[i], kS3FsPrefix);
    metrics::RecordTFDataFilename(kDatasetType, filenames[i]);
  }

  tstring compression_type;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<tstring>(ctx, kCompressionType,
                                                   &compression_type));

  int64_t buffer_size = -1;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0,
              errors::InvalidArgument(
                  "`buffer_size` must be >= 0 (0 == no buffering)"));

  if (is_gcs_fs && is_cloud_tpu_gcs_fs() && buffer_size < kCloudTpuBlockSize) {
    VLOG(2) << "User buffer size is too small for reading Cloud TPU "
            << "TFRecords stored in GCS. Overriding " << buffer_size
            << " to the minimum recommended buffer_size = "
            << kCloudTpuBlockSize;
    buffer_size = kCloudTpuBlockSize;
  }

  if (is_s3_fs && buffer_size < kS3BlockSize) {
    VLOG(2) << "User buffer size is too small for reading "
            << "TFRecords stored in S3. Overriding " << buffer_size
            << " to the minimum recommended buffer_size = " << kS3BlockSize;
    buffer_size = kS3BlockSize;
  }

  *output =
      new Dataset(ctx, std::move(filenames), compression_type, buffer_size);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("TFRecordDataset").Device(DEVICE_CPU),
                        TFRecordDatasetOp);
}  // namespace
}  // namespace data
}  // namespace tensorflow
