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
class MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/snapshot_utils.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"

namespace tensorflow {
namespace data {
namespace snapshot_util {
namespace {

constexpr const char* const kOutputTypes = "output_types";
constexpr const char* const kOutputShapes = "output_shapes";
constexpr const char* const kCompression = "compression";
constexpr const char* const kVersion = "version";
constexpr const char* const kCurrentCheckpointID = "current_checkpoint_id";
constexpr const char* const kIndex = "index";
constexpr const char* const kStartIndex = "start_index";

}  // namespace

/* static */ constexpr const int64_t
    CustomReader::kSnappyReaderInputBufferSizeBytes;
/* static */ constexpr const int64_t
    CustomReader::kSnappyReaderOutputBufferSizeBytes;

std::string HashDirectory(const std::string& path, uint64 hash) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_0(mht_0_v, 238, "", "./tensorflow/core/data/snapshot_utils.cc", "HashDirectory");

  return io::JoinPath(
      path, strings::Printf("%llu", static_cast<unsigned long long>(hash)));
}

std::string RunDirectory(const std::string& hash_directory, uint64 run_id) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("hash_directory: \"" + hash_directory + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_1(mht_1_v, 247, "", "./tensorflow/core/data/snapshot_utils.cc", "RunDirectory");

  return RunDirectory(
      hash_directory,
      strings::Printf("%llu", static_cast<unsigned long long>(run_id)));
}

std::string RunDirectory(const std::string& hash_directory,
                         const std::string& run_id) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("hash_directory: \"" + hash_directory + "\"");
   mht_2_v.push_back("run_id: \"" + run_id + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/data/snapshot_utils.cc", "RunDirectory");

  return io::JoinPath(hash_directory, run_id);
}

std::string ShardDirectory(const std::string& run_directory, int64_t shard_id) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("run_directory: \"" + run_directory + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/data/snapshot_utils.cc", "ShardDirectory");

  return io::JoinPath(
      run_directory,
      strings::Printf("%08llu%s", static_cast<unsigned long long>(shard_id),
                      kShardDirectorySuffix));
}
std::string GetCheckpointFileName(const std::string& shard_directory,
                                  uint64 checkpoint_id) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("shard_directory: \"" + shard_directory + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_4(mht_4_v, 278, "", "./tensorflow/core/data/snapshot_utils.cc", "GetCheckpointFileName");

  return io::JoinPath(
      shard_directory,
      strings::Printf("%08llu.snapshot",
                      static_cast<unsigned long long>(checkpoint_id)));
}

Status Writer::Create(Env* env, const std::string& filename,
                      const std::string& compression_type, int version,
                      const DataTypeVector& dtypes,
                      std::unique_ptr<Writer>* out_writer) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("filename: \"" + filename + "\"");
   mht_5_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_5(mht_5_v, 293, "", "./tensorflow/core/data/snapshot_utils.cc", "Writer::Create");

  switch (version) {
    case 1:
      *out_writer =
          absl::make_unique<CustomWriter>(filename, compression_type, dtypes);
      break;
    case 2:
      *out_writer =
          absl::make_unique<TFRecordWriter>(filename, compression_type);
      break;
    default:
      return errors::InvalidArgument("Snapshot writer version: ", version,
                                     " is not supported.");
  }

  return (*out_writer)->Initialize(env);
}

TFRecordWriter::TFRecordWriter(const std::string& filename,
                               const std::string& compression_type)
    : filename_(filename), compression_type_(compression_type) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("filename: \"" + filename + "\"");
   mht_6_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_6(mht_6_v, 318, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordWriter::TFRecordWriter");
}

Status TFRecordWriter::Initialize(tensorflow::Env* env) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_7(mht_7_v, 323, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordWriter::Initialize");

  TF_RETURN_IF_ERROR(env->NewAppendableFile(filename_, &dest_));

  record_writer_ = absl::make_unique<io::RecordWriter>(
      dest_.get(), io::RecordWriterOptions::CreateRecordWriterOptions(
                       /*compression_type=*/compression_type_));
  return Status::OK();
}

Status TFRecordWriter::WriteTensors(const std::vector<Tensor>& tensors) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_8(mht_8_v, 335, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordWriter::WriteTensors");

  for (const auto& tensor : tensors) {
    TensorProto proto;
    tensor.AsProtoTensorContent(&proto);
#if defined(TF_CORD_SUPPORT)
    // Creating raw pointer here because std::move() in a releases in OSS TF
    // will result in a smart pointer being moved upon function creation, which
    // will result in proto_buffer == nullptr when WriteRecord happens.
    auto proto_buffer = new std::string();
    proto.SerializeToString(proto_buffer);
    absl::Cord proto_serialized = absl::MakeCordFromExternal(
        *proto_buffer,
        [proto_buffer](absl::string_view) { delete proto_buffer; });
    TF_RETURN_IF_ERROR(record_writer_->WriteRecord(proto_serialized));
#else   // TF_CORD_SUPPORT
    TF_RETURN_IF_ERROR(record_writer_->WriteRecord(proto.SerializeAsString()));
#endif  // TF_CORD_SUPPORT
  }
  return Status::OK();
}

Status TFRecordWriter::Sync() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_9(mht_9_v, 359, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordWriter::Sync");

  TF_RETURN_IF_ERROR(record_writer_->Flush());
  return dest_->Flush();
}

Status TFRecordWriter::Close() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_10(mht_10_v, 367, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordWriter::Close");

  if (record_writer_ != nullptr) {
    TF_RETURN_IF_ERROR(Sync());
    TF_RETURN_IF_ERROR(record_writer_->Close());
    TF_RETURN_IF_ERROR(dest_->Close());
    record_writer_ = nullptr;
    dest_ = nullptr;
  }
  return Status::OK();
}

TFRecordWriter::~TFRecordWriter() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_11(mht_11_v, 381, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordWriter::~TFRecordWriter");

  Status s = Close();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to close snapshot file " << filename_ << ": " << s;
  }
}

CustomWriter::CustomWriter(const std::string& filename,
                           const std::string& compression_type,
                           const DataTypeVector& dtypes)
    : filename_(filename),
      compression_type_(compression_type),
      dtypes_(dtypes) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("filename: \"" + filename + "\"");
   mht_12_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_12(mht_12_v, 398, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::CustomWriter");
}

Status CustomWriter::Initialize(tensorflow::Env* env) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_13(mht_13_v, 403, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::Initialize");

  TF_RETURN_IF_ERROR(env->NewAppendableFile(filename_, &dest_));
#if defined(IS_SLIM_BUILD)
  if (compression_type_ != io::compression::kNone) {
    LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
               << "off compression.";
  }
#else   // IS_SLIM_BUILD
  if (compression_type_ == io::compression::kGzip) {
    zlib_underlying_dest_.swap(dest_);
    io::ZlibCompressionOptions zlib_options;
    zlib_options = io::ZlibCompressionOptions::GZIP();

    io::ZlibOutputBuffer* zlib_output_buffer = new io::ZlibOutputBuffer(
        zlib_underlying_dest_.get(), zlib_options.input_buffer_size,
        zlib_options.output_buffer_size, zlib_options);
    TF_CHECK_OK(zlib_output_buffer->Init());
    dest_.reset(zlib_output_buffer);
  }
#endif  // IS_SLIM_BUILD
  simple_tensor_mask_.reserve(dtypes_.size());
  for (const auto& dtype : dtypes_) {
    if (DataTypeCanUseMemcpy(dtype)) {
      simple_tensor_mask_.push_back(true);
      num_simple_++;
    } else {
      simple_tensor_mask_.push_back(false);
      num_complex_++;
    }
  }

  return Status::OK();
}

Status CustomWriter::WriteTensors(const std::vector<Tensor>& tensors) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_14(mht_14_v, 440, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::WriteTensors");

  if (compression_type_ != io::compression::kSnappy) {
    experimental::SnapshotRecord record;
    for (const auto& tensor : tensors) {
      TensorProto* t = record.add_tensor();
      tensor.AsProtoTensorContent(t);
    }
#if defined(TF_CORD_SUPPORT)
    auto record_buffer = new std::string();
    record.SerializeToString(record_buffer);
    absl::Cord record_serialized = absl::MakeCordFromExternal(
        *record_buffer,
        [record_buffer](absl::string_view) { delete record_buffer; });
    return WriteRecord(record_serialized);
#else   // TF_CORD_SUPPORT
    return WriteRecord(record.SerializeAsString());
#endif  // TF_CORD_SUPPORT
  }

  std::vector<const TensorBuffer*> tensor_buffers;
  tensor_buffers.reserve(num_simple_);
  std::vector<TensorProto> tensor_protos;
  tensor_protos.reserve(num_complex_);
  experimental::SnapshotTensorMetadata metadata;
  int64_t total_size = 0;
  for (int i = 0, end = tensors.size(); i < end; ++i) {
    const Tensor& tensor = tensors[i];
    experimental::TensorMetadata* tensor_metadata =
        metadata.add_tensor_metadata();
    tensor.shape().AsProto(tensor_metadata->mutable_tensor_shape());
    int64_t size = 0;
    if (simple_tensor_mask_[i]) {
      auto tensor_buffer = DMAHelper::buffer(&tensor);
      tensor_buffers.push_back(tensor_buffer);
      size = tensor_buffer->size();
    } else {
      TensorProto proto;
      tensor.AsProtoTensorContent(&proto);
      size = proto.ByteSizeLong();
      tensor_protos.push_back(std::move(proto));
    }
    tensor_metadata->set_tensor_size_bytes(size);
    total_size += size;
  }

  std::vector<char> uncompressed(total_size);
  char* position = uncompressed.data();
  int buffer_index = 0;
  int proto_index = 0;
  for (int i = 0, end = tensors.size(); i < end; ++i) {
    const auto& tensor_metadata = metadata.tensor_metadata(i);
    if (simple_tensor_mask_[i]) {
      memcpy(position, tensor_buffers[buffer_index]->data(),
             tensor_metadata.tensor_size_bytes());
      buffer_index++;
    } else {
      tensor_protos[proto_index].SerializeToArray(
          position, tensor_metadata.tensor_size_bytes());
      proto_index++;
    }
    position += tensor_metadata.tensor_size_bytes();
  }
  DCHECK_EQ(position, uncompressed.data() + total_size);

  string output;
  if (!port::Snappy_Compress(uncompressed.data(), total_size, &output)) {
    return errors::Internal("Failed to compress using snappy.");
  }

#if defined(TF_CORD_SUPPORT)
  auto metadata_buffer = new std::string();
  metadata.SerializeToString(metadata_buffer);
  absl::Cord metadata_serialized = absl::MakeCordFromExternal(
      *metadata_buffer,
      [metadata_buffer](absl::string_view) { delete metadata_buffer; });
#else
  std::string metadata_serialized = metadata.SerializeAsString();
#endif  // TF_CORD_SUPPORT
  TF_RETURN_IF_ERROR(WriteRecord(metadata_serialized));
  TF_RETURN_IF_ERROR(WriteRecord(output));
  return Status::OK();
}

Status CustomWriter::Sync() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_15(mht_15_v, 526, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::Sync");
 return dest_->Sync(); }

Status CustomWriter::Close() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_16(mht_16_v, 531, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::Close");

  if (dest_ != nullptr) {
    TF_RETURN_IF_ERROR(dest_->Close());
    dest_ = nullptr;
  }
  if (zlib_underlying_dest_ != nullptr) {
    TF_RETURN_IF_ERROR(zlib_underlying_dest_->Close());
    zlib_underlying_dest_ = nullptr;
  }
  return Status::OK();
}

CustomWriter::~CustomWriter() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_17(mht_17_v, 546, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::~CustomWriter");

  Status s = Close();
  if (!s.ok()) {
    LOG(ERROR) << "Could not finish writing file: " << s;
  }
}

Status CustomWriter::WriteRecord(const StringPiece& data) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_18(mht_18_v, 556, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomWriter::WriteRecord");

  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}

#if defined(TF_CORD_SUPPORT)
Status CustomWriter::WriteRecord(const absl::Cord& data) {
  char header[kHeaderSize];
  core::EncodeFixed64(header, data.size());
  TF_RETURN_IF_ERROR(dest_->Append(StringPiece(header, sizeof(header))));
  return dest_->Append(data);
}
#endif  // TF_CORD_SUPPORT

Status Reader::Create(Env* env, const std::string& filename,
                      const string& compression_type, int version,
                      const DataTypeVector& dtypes,
                      std::unique_ptr<Reader>* out_reader) {
  switch (version) {
    // CustomReader is able to read a legacy snapshot file format (v0) though
    // custom writer doesn't have the ability to write it any more since it is
    // strictly worse than V1.
    case 0:
    case 1:
      *out_reader = absl::make_unique<CustomReader>(filename, compression_type,
                                                    version, dtypes);
      break;
    case 2:
      *out_reader =
          absl::make_unique<TFRecordReader>(filename, compression_type, dtypes);
      break;
    default:
      return errors::InvalidArgument("Snapshot reader version: ", version,
                                     " is not supported.");
  }

  return (*out_reader)->Initialize(env);
}

Status Reader::SkipRecords(int64_t num_records) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_19(mht_19_v, 600, "", "./tensorflow/core/data/snapshot_utils.cc", "Reader::SkipRecords");

  // TODO(frankchn): Optimize to not parse the entire Tensor and actually skip.
  for (int i = 0; i < num_records; ++i) {
    std::vector<Tensor> unused_tensors;
    TF_RETURN_IF_ERROR(ReadTensors(&unused_tensors));
  }
  return Status::OK();
}

class Reader::Dataset : public DatasetBase {
 public:
  Dataset(DatasetContext&& ctx, const std::string& shard_dir,
          const std::string& compression, const int64_t version,
          const DataTypeVector& dtypes,
          const std::vector<PartialTensorShape>& shapes,
          const int64_t start_index)
      : DatasetBase(std::move(ctx)),
        shard_dir_(shard_dir),
        compression_(compression),
        version_(version),
        dtypes_(dtypes),
        shapes_(shapes),
        start_index_(start_index) {}

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_20(mht_20_v, 627, "", "./tensorflow/core/data/snapshot_utils.cc", "output_dtypes");
 return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_21(mht_21_v, 632, "", "./tensorflow/core/data/snapshot_utils.cc", "output_shapes");

    return shapes_;
  }

  std::string DebugString() const override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_22(mht_22_v, 639, "", "./tensorflow/core/data/snapshot_utils.cc", "DebugString");
 return "SnapshotDatasetReader"; }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_23(mht_23_v, 644, "", "./tensorflow/core/data/snapshot_utils.cc", "InputDatasets");

    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_24(mht_24_v, 651, "", "./tensorflow/core/data/snapshot_utils.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_25(mht_25_v, 659, "", "./tensorflow/core/data/snapshot_utils.cc", "AsGraphDefInternal");

    Node* shard_dir = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(shard_dir_, &shard_dir));

    Node* start_index = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(start_index_, &start_index));

    AttrValue compression;
    b->BuildAttrValue(compression_, &compression);

    AttrValue version;
    b->BuildAttrValue(version_, &version);

    return b->AddDataset(
        this,
        /*inputs=*/
        {std::make_pair(0, shard_dir), std::make_pair(1, start_index)},
        /*list_inputs=*/{},
        /*attrs=*/
        {{kCompression, compression}, {kVersion, version}},
        /*use_dataset_name=*/true, node);
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          start_index_(dataset()->start_index_) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_26(mht_26_v, 696, "", "./tensorflow/core/data/snapshot_utils.cc", "Iterator");
}

    Status Initialize(IteratorContext* ctx) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_27(mht_27_v, 701, "", "./tensorflow/core/data/snapshot_utils.cc", "Initialize");

      // TODO(jsimsa): This only needs to happen when we are not restoring but
      // parallel_interleave op implementation caches IteratorContext (and thus
      // the is_restoring bit ends up being inaccurate).
      TF_RETURN_IF_ERROR(Reader::Create(
          ctx->env(), GetCurrentFilename(), dataset()->compression_,
          dataset()->version_, dataset()->dtypes_, &reader_));
      return AdvanceToStartIndex(ctx);
    }

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_28(mht_28_v, 717, "", "./tensorflow/core/data/snapshot_utils.cc", "GetNextInternal");

      *end_of_sequence = false;
      Status s = reader_->ReadTensors(out_tensors);
      if (!errors::IsOutOfRange(s)) {
        start_index_++;
        return s;
      }
      Status status = AdvanceToNextFile(ctx->env());
      if (errors::IsNotFound(status)) {
        *end_of_sequence = true;
        return Status::OK();
      }
      return status;
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_29(mht_29_v, 736, "", "./tensorflow/core/data/snapshot_utils.cc", "SaveInternal");

      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentCheckpointID),
                                             current_checkpoint_id_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kStartIndex), start_index_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_30(mht_30_v, 748, "", "./tensorflow/core/data/snapshot_utils.cc", "RestoreInternal");

      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentCheckpointID),
                                            &current_checkpoint_id_));
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kStartIndex), &start_index_));
      TF_RETURN_IF_ERROR(ctx->env()->FileExists(GetCurrentFilename()));
      TF_RETURN_IF_ERROR(Reader::Create(
          ctx->env(), GetCurrentFilename(), dataset()->compression_,
          dataset()->version_, dataset()->dtypes_, &reader_));
      return AdvanceToStartIndex(ctx);
    }

   private:
    Status AdvanceToNextFile(Env* env) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_31(mht_31_v, 764, "", "./tensorflow/core/data/snapshot_utils.cc", "AdvanceToNextFile");

      start_index_ = 0;
      current_checkpoint_id_++;
      TF_RETURN_IF_ERROR(env->FileExists(GetCurrentFilename()));
      return Reader::Create(env, GetCurrentFilename(), dataset()->compression_,
                            dataset()->version_, dataset()->dtypes_, &reader_);
    }

    std::string GetCurrentFilename() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_32(mht_32_v, 775, "", "./tensorflow/core/data/snapshot_utils.cc", "GetCurrentFilename");

      return GetCheckpointFileName(dataset()->shard_dir_,
                                   current_checkpoint_id_);
    }

    // TODO(frankchn): Optimize this to not parse every single element.
    Status AdvanceToStartIndex(IteratorContext* ctx) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_33(mht_33_v, 784, "", "./tensorflow/core/data/snapshot_utils.cc", "AdvanceToStartIndex");

      for (int64_t i = 0; i < start_index_; ++i) {
        std::vector<Tensor> unused;
        TF_RETURN_IF_ERROR(reader_->ReadTensors(&unused));
      }
      return Status::OK();
    }

    std::unique_ptr<Reader> reader_;

    // Stores the id current checkpoint file that we are in the process of
    // reading (e.g. if the file is currently 00000001.snapshot, then this will
    // be 1).
    int64_t current_checkpoint_id_ = 0;
    int64_t start_index_;
  };

  const tstring shard_dir_;
  const std::string compression_;
  const int64_t version_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
  const int64_t start_index_;
};

Reader::DatasetOp::DatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_34(mht_34_v, 812, "", "./tensorflow/core/data/snapshot_utils.cc", "Reader::DatasetOp::DatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kCompression, &compression_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kVersion, &version_));
}

void Reader::DatasetOp::MakeDataset(OpKernelContext* ctx,
                                    DatasetBase** output) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_35(mht_35_v, 823, "", "./tensorflow/core/data/snapshot_utils.cc", "Reader::DatasetOp::MakeDataset");

  tstring shard_dir;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "shard_dir", &shard_dir));

  int64_t start_index;
  OP_REQUIRES_OK(ctx, ParseScalarArgument(ctx, "start_index", &start_index));

  *output =
      new Reader::Dataset(DatasetContext(ctx), shard_dir, compression_,
                          version_, output_types_, output_shapes_, start_index);
}

class Reader::NestedDataset : public DatasetBase {
 public:
  explicit NestedDataset(DatasetContext&& ctx,
                         std::vector<DatasetBase*> datasets)
      : DatasetBase(std::move(ctx)), datasets_(datasets) {
    dtypes_.push_back(DT_VARIANT);
    gtl::InlinedVector<int64_t, 1> element_dim_sizes;
    element_dim_sizes.push_back(1);
    partial_shapes_.emplace_back(element_dim_sizes);
  }

  const DataTypeVector& output_dtypes() const override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_36(mht_36_v, 849, "", "./tensorflow/core/data/snapshot_utils.cc", "output_dtypes");
 return dtypes_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_37(mht_37_v, 854, "", "./tensorflow/core/data/snapshot_utils.cc", "output_shapes");

    return partial_shapes_;
  }

  std::string DebugString() const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_38(mht_38_v, 861, "", "./tensorflow/core/data/snapshot_utils.cc", "DebugString");

    return "SnapshotNestedDatasetReader";
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_39(mht_39_v, 868, "", "./tensorflow/core/data/snapshot_utils.cc", "InputDatasets");

    inputs->clear();
    return Status::OK();
  }

  Status CheckExternalState() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_40(mht_40_v, 876, "", "./tensorflow/core/data/snapshot_utils.cc", "CheckExternalState");
 return Status::OK(); }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** node) const override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_41(mht_41_v, 884, "", "./tensorflow/core/data/snapshot_utils.cc", "AsGraphDefInternal");

    std::vector<Node*> input_graph_nodes;
    input_graph_nodes.reserve(datasets_.size());
    for (const auto& dataset : datasets_) {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, dataset, &input_node));
      input_graph_nodes.emplace_back(input_node);
    }
    TF_RETURN_IF_ERROR(
        b->AddDataset(this, /*inputs=*/{},
                      /*list_inputs=*/{std::make_pair(0, input_graph_nodes)},
                      /*attrs=*/{}, node));
    return Status::OK();
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(node_name(), prefix)});
  }

 private:
  std::vector<DatasetBase*> datasets_;
  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> partial_shapes_;

  class Iterator : public DatasetIterator<NestedDataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<NestedDataset>(params) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_42(mht_42_v, 916, "", "./tensorflow/core/data/snapshot_utils.cc", "Iterator");
}

   protected:
    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_43(mht_43_v, 924, "", "./tensorflow/core/data/snapshot_utils.cc", "GetNextInternal");

      const int64_t num_datasets = dataset()->datasets_.size();
      *end_of_sequence = num_datasets == index_;
      if (!*end_of_sequence) {
        Tensor tensor(DT_VARIANT, TensorShape({}));

        TF_RETURN_IF_ERROR(
            StoreDatasetInVariantTensor(dataset()->datasets_[index_], &tensor));
        out_tensors->clear();
        out_tensors->push_back(std::move(tensor));

        index_++;
      }
      return Status::OK();
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_44(mht_44_v, 944, "", "./tensorflow/core/data/snapshot_utils.cc", "SaveInternal");

      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kIndex), index_));
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_45(mht_45_v, 953, "", "./tensorflow/core/data/snapshot_utils.cc", "RestoreInternal");

      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kIndex), &index_));
      return Status::OK();
    }

   private:
    int64_t index_ = 0;
  };
};

Reader::NestedDatasetOp::NestedDatasetOp(OpKernelConstruction* ctx)
    : DatasetOpKernel(ctx) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_46(mht_46_v, 967, "", "./tensorflow/core/data/snapshot_utils.cc", "Reader::NestedDatasetOp::NestedDatasetOp");

  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void Reader::NestedDatasetOp::MakeDataset(OpKernelContext* ctx,
                                          DatasetBase** output) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_47(mht_47_v, 976, "", "./tensorflow/core/data/snapshot_utils.cc", "Reader::NestedDatasetOp::MakeDataset");

  std::vector<DatasetBase*> inputs;
  for (size_t i = 0; i < ctx->num_inputs(); ++i) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
    inputs.push_back(input);
  }
  *output = new Reader::NestedDataset(DatasetContext(ctx), inputs);
  (*output)->Initialize(/*metadata=*/{});
}

Status Reader::MakeNestedDataset(Env* env,
                                 const std::vector<std::string>& shard_dirs,
                                 const string& compression_type, int version,
                                 const DataTypeVector& dtypes,
                                 const std::vector<PartialTensorShape>& shapes,
                                 const int64_t start_index,
                                 DatasetBase** output) {
   std::vector<std::string> mht_48_v;
   mht_48_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_48(mht_48_v, 997, "", "./tensorflow/core/data/snapshot_utils.cc", "Reader::MakeNestedDataset");

  std::vector<DatasetBase*> datasets;

  datasets.reserve(shard_dirs.size());
  for (int64_t i = 0; i < shard_dirs.size(); ++i) {
    // TODO(frankchn): The reading pattern could be controlled in a non-round
    // robin fashion, so we cannot assume a round-robin manner when restoring.
    int64_t dataset_start_index = start_index / shard_dirs.size();
    if (start_index % shard_dirs.size() > datasets.size()) {
      dataset_start_index++;
    }

    datasets.push_back(
        new Dataset(DatasetContext(DatasetContext::Params(
                        {"SnapshotDatasetReader",
                         strings::StrCat("SnapshotDatasetReader/_", i)})),
                    shard_dirs.at(i), compression_type, version, dtypes, shapes,
                    dataset_start_index));
    datasets.back()->Initialize(/*metadata=*/{});
  }

  // Rotate the vector such that the first dataset contains the next element
  // to be produced, but not if there are no shards at all (then we just
  // construct an empty dataset).
  if (!shard_dirs.empty()) {
    std::rotate(datasets.begin(),
                datasets.begin() + (start_index % shard_dirs.size()),
                datasets.end());
  }

  *output = new NestedDataset(
      DatasetContext(DatasetContext::Params(
          {"SnapshotNestedDatasetReader", "SnapshotNestedDatasetReader"})),
      datasets);
  (*output)->Initialize(/*metadata=*/{});
  return Status::OK();
}

TFRecordReader::TFRecordReader(const std::string& filename,
                               const string& compression_type,
                               const DataTypeVector& dtypes)
    : filename_(filename),
      offset_(0),
      compression_type_(compression_type),
      dtypes_(dtypes) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("filename: \"" + filename + "\"");
   mht_49_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_49(mht_49_v, 1046, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordReader::TFRecordReader");
}

Status TFRecordReader::Initialize(Env* env) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_50(mht_50_v, 1051, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordReader::Initialize");

  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename_, &file_));

  record_reader_ = absl::make_unique<io::RecordReader>(
      file_.get(), io::RecordReaderOptions::CreateRecordReaderOptions(
                       /*compression_type=*/compression_type_));
  return Status::OK();
}

Status TFRecordReader::ReadTensors(std::vector<Tensor>* read_tensors) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_51(mht_51_v, 1063, "", "./tensorflow/core/data/snapshot_utils.cc", "TFRecordReader::ReadTensors");

  read_tensors->reserve(dtypes_.size());
  for (int i = 0; i < dtypes_.size(); ++i) {
    tstring record;
    TF_RETURN_IF_ERROR(record_reader_->ReadRecord(&offset_, &record));

    TensorProto proto;
    proto.ParseFromArray(record.data(), record.size());

    Tensor tensor;
    if (!tensor.FromProto(proto)) {
      return errors::DataLoss("Unable to parse tensor from stored proto.");
    }

    read_tensors->push_back(std::move(tensor));
  }
  return Status::OK();
}

CustomReader::CustomReader(const std::string& filename,
                           const string& compression_type, const int version,
                           const DataTypeVector& dtypes)
    : filename_(filename),
      compression_type_(compression_type),
      version_(version),
      dtypes_(dtypes) {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("filename: \"" + filename + "\"");
   mht_52_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_52(mht_52_v, 1093, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomReader::CustomReader");
}

Status CustomReader::Initialize(Env* env) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_53(mht_53_v, 1098, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomReader::Initialize");

  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename_, &file_));
  input_stream_ = std::make_unique<io::RandomAccessInputStream>(file_.get());

#if defined(IS_SLIM_BUILD)
  if (compression_type_ != io::compression::kNone) {
    LOG(ERROR) << "Compression is unsupported on mobile platforms. Turning "
               << "off compression.";
  }
#else   // IS_SLIM_BUILD
  if (compression_type_ == io::compression::kGzip) {
    io::ZlibCompressionOptions zlib_options;
    zlib_options = io::ZlibCompressionOptions::GZIP();

    input_stream_ = absl::make_unique<io::ZlibInputStream>(
        input_stream_.release(), zlib_options.input_buffer_size,
        zlib_options.output_buffer_size, zlib_options, true);
  } else if (compression_type_ == io::compression::kSnappy) {
    if (version_ == 0) {
      input_stream_ = absl::make_unique<io::SnappyInputBuffer>(
          file_.get(), /*input_buffer_bytes=*/kSnappyReaderInputBufferSizeBytes,
          /*output_buffer_bytes=*/kSnappyReaderOutputBufferSizeBytes);
    } else {
      input_stream_ =
          absl::make_unique<io::BufferedInputStream>(file_.get(), 64 << 20);
    }
  }
#endif  // IS_SLIM_BUILD
  simple_tensor_mask_.reserve(dtypes_.size());
  for (const auto& dtype : dtypes_) {
    if (DataTypeCanUseMemcpy(dtype)) {
      simple_tensor_mask_.push_back(true);
      num_simple_++;
    } else {
      simple_tensor_mask_.push_back(false);
      num_complex_++;
    }
  }

  return Status::OK();
}

Status CustomReader::ReadTensors(std::vector<Tensor>* read_tensors) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_54(mht_54_v, 1143, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomReader::ReadTensors");

  profiler::TraceMe activity(
      [&]() { return absl::StrCat(kClassName, kSeparator, "ReadTensors"); },
      profiler::TraceMeLevel::kInfo);
  if (version_ == 0 || compression_type_ != io::compression::kSnappy) {
    return ReadTensorsV0(read_tensors);
  }
  if (version_ != 1) {
    return errors::InvalidArgument("Version: ", version_, " is not supported.");
  }
  if (compression_type_ != io::compression::kSnappy) {
    return errors::InvalidArgument("Compression ", compression_type_,
                                   " is not supported.");
  }

  experimental::SnapshotTensorMetadata metadata;
  tstring metadata_str;
  TF_RETURN_IF_ERROR(ReadRecord(&metadata_str));
  if (!metadata.ParseFromArray(metadata_str.data(), metadata_str.size())) {
    return errors::DataLoss("Could not parse SnapshotTensorMetadata");
  }
  read_tensors->reserve(metadata.tensor_metadata_size());

  std::vector<Tensor> simple_tensors;
  simple_tensors.reserve(num_simple_);
  std::vector<std::pair<std::unique_ptr<char[]>, size_t>> tensor_proto_strs;
  tensor_proto_strs.reserve(num_complex_);
  TF_RETURN_IF_ERROR(
      SnappyUncompress(&metadata, &simple_tensors, &tensor_proto_strs));

  int simple_index = 0;
  int complex_index = 0;
  for (int i = 0, end = simple_tensor_mask_.size(); i < end; ++i) {
    if (simple_tensor_mask_[i]) {
      read_tensors->push_back(std::move(simple_tensors[simple_index]));
      simple_index++;
    } else {
      auto tensor_proto_str = std::move(tensor_proto_strs[complex_index].first);
      size_t tensor_proto_size = tensor_proto_strs[complex_index].second;
      TensorProto tp;
      if (!tp.ParseFromArray(tensor_proto_str.get(), tensor_proto_size)) {
        return errors::Internal("Could not parse TensorProto");
      }
      Tensor t;
      if (!t.FromProto(tp)) {
        return errors::Internal("Could not parse Tensor");
      }
      read_tensors->push_back(std::move(t));
      complex_index++;
    }
  }
  return Status::OK();
}

Status CustomReader::ReadTensorsV0(std::vector<Tensor>* read_tensors) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_55(mht_55_v, 1200, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomReader::ReadTensorsV0");

  experimental::SnapshotRecord record;
#if defined(PLATFORM_GOOGLE)
  absl::Cord c;
  TF_RETURN_IF_ERROR(ReadRecord(&c));
  record.ParseFromCord(c);
#else   // PLATFORM_GOOGLE
  tstring record_bytes;
  TF_RETURN_IF_ERROR(ReadRecord(&record_bytes));
  record.ParseFromArray(record_bytes.data(), record_bytes.size());
#endif  // PLATFORM_GOOGLE
  read_tensors->reserve(record.tensor_size());
  for (int i = 0; i < record.tensor_size(); ++i) {
    read_tensors->emplace_back();
    if (!read_tensors->back().FromProto(record.tensor(i))) {
      return errors::DataLoss("Unable to parse tensor from proto.");
    }
  }
  return Status::OK();
}

Status CustomReader::SnappyUncompress(
    const experimental::SnapshotTensorMetadata* metadata,
    std::vector<Tensor>* simple_tensors,
    std::vector<std::pair<std::unique_ptr<char[]>, size_t>>*
        tensor_proto_strs) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_56(mht_56_v, 1228, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomReader::SnappyUncompress");

  tstring compressed;
  TF_RETURN_IF_ERROR(ReadRecord(&compressed));
  size_t size;
  if (!port::Snappy_GetUncompressedLength(compressed.data(), compressed.size(),
                                          &size)) {
    return errors::Internal("Could not get snappy uncompressed length");
  }

  int num_tensors = metadata->tensor_metadata_size();
  std::vector<struct iovec> iov(num_tensors);
  int index = 0;
  int64_t total_size = 0;
  for (int i = 0, end = simple_tensor_mask_.size(); i < end; ++i) {
    const auto& tensor_metadata = metadata->tensor_metadata(i);
    if (simple_tensor_mask_[i]) {
      TensorShape shape(tensor_metadata.tensor_shape());
      Tensor simple_tensor(dtypes_[i], shape);
      TensorBuffer* buffer = DMAHelper::buffer(&simple_tensor);
      iov[index].iov_base = buffer->data();
      iov[index].iov_len = buffer->size();
      simple_tensors->push_back(std::move(simple_tensor));
    } else {
      auto tensor_proto_str =
          absl::make_unique<char[]>(tensor_metadata.tensor_size_bytes());
      iov[index].iov_base = tensor_proto_str.get();
      iov[index].iov_len = tensor_metadata.tensor_size_bytes();
      tensor_proto_strs->push_back(std::make_pair(
          std::move(tensor_proto_str), tensor_metadata.tensor_size_bytes()));
    }
    total_size += iov[index].iov_len;
    index++;
  }
  const int64_t size_int = size;
  if (size_int != total_size) {
    return errors::Internal("Uncompressed size mismatch. Snappy expects ", size,
                            " whereas the tensor metadata suggests ",
                            total_size);
  }
  if (!port::Snappy_UncompressToIOVec(compressed.data(), compressed.size(),
                                      iov.data(), num_tensors)) {
    return errors::Internal("Failed to perform snappy decompression.");
  }
  return Status::OK();
}

Status CustomReader::ReadRecord(tstring* record) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_57(mht_57_v, 1277, "", "./tensorflow/core/data/snapshot_utils.cc", "CustomReader::ReadRecord");

  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  return input_stream_->ReadNBytes(length, record);
}

#if defined(TF_CORD_SUPPORT)
Status CustomReader::ReadRecord(absl::Cord* record) {
  tstring header;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(kHeaderSize, &header));
  uint64 length = core::DecodeFixed64(header.data());
  if (compression_type_ == io::compression::kNone) {
    return input_stream_->ReadNBytes(length, record);
  } else {
    auto tmp_str = new tstring();
    TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(length, tmp_str));
    absl::string_view tmp_str_view(*tmp_str);
    record->Append(absl::MakeCordFromExternal(
        tmp_str_view, [tmp_str](absl::string_view) { delete tmp_str; }));
    return Status::OK();
  }
}
#endif  // TF_CORD_SUPPORT

Status WriteMetadataFile(Env* env, const string& dir,
                         const experimental::SnapshotMetadataRecord* metadata) {
  string metadata_filename = io::JoinPath(dir, kMetadataFilename);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(dir));
  std::string tmp_filename =
      absl::StrCat(metadata_filename, "-tmp-", random::New64());
  TF_RETURN_IF_ERROR(WriteBinaryProto(env, tmp_filename, *metadata));
  return env->RenameFile(tmp_filename, metadata_filename);
}

Status ReadMetadataFile(Env* env, const string& dir,
                        experimental::SnapshotMetadataRecord* metadata,
                        bool* file_exists) {
   std::vector<std::string> mht_58_v;
   mht_58_v.push_back("dir: \"" + dir + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_58(mht_58_v, 1318, "", "./tensorflow/core/data/snapshot_utils.cc", "ReadMetadataFile");

  string metadata_filename = io::JoinPath(dir, kMetadataFilename);
  Status s = env->FileExists(metadata_filename);
  *file_exists = s.ok();

  if (*file_exists) {
    return ReadBinaryProto(env, metadata_filename, metadata);
  } else {
    return Status::OK();
  }
}

Status DumpDatasetGraph(Env* env, const std::string& path, uint64 hash,
                        const GraphDef* graph) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_59(mht_59_v, 1335, "", "./tensorflow/core/data/snapshot_utils.cc", "DumpDatasetGraph");

  std::string hash_hex =
      strings::StrCat(strings::Hex(hash, strings::kZeroPad16));
  std::string graph_file =
      io::JoinPath(path, absl::StrCat(hash_hex, "-graph.pbtxt"));

  LOG(INFO) << "Graph hash is " << hash_hex << ", writing to " << graph_file;
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(path));
  return WriteTextProto(env, graph_file, *graph);
}

Status DetermineOpState(const std::string& mode_string, bool file_exists,
                        const experimental::SnapshotMetadataRecord* metadata,
                        const uint64 pending_snapshot_expiry_seconds,
                        Mode* mode) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("mode_string: \"" + mode_string + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_60(mht_60_v, 1353, "", "./tensorflow/core/data/snapshot_utils.cc", "DetermineOpState");

  if (mode_string == kModeRead) {
    // In read mode, we should expect a metadata file is written.
    if (!file_exists) {
      return errors::NotFound("Metadata file does not exist.");
    }
    LOG(INFO) << "Overriding mode to reader.";
    *mode = READER;
    return Status::OK();
  }

  if (mode_string == kModeWrite) {
    LOG(INFO) << "Overriding mode to writer.";
    *mode = WRITER;
    return Status::OK();
  }

  if (mode_string == kModePassthrough) {
    LOG(INFO) << "Overriding mode to passthrough.";
    *mode = PASSTHROUGH;
    return Status::OK();
  }

  if (!file_exists) {
    *mode = WRITER;
    return Status::OK();
  }

  if (metadata->finalized()) {
    // File found, snapshot has been finalized.
    *mode = READER;
    return Status::OK();
  }

  int64_t expiration_timer = static_cast<int64_t>(EnvTime::NowMicros()) -
                             pending_snapshot_expiry_seconds * 1000000;

  if (metadata->creation_timestamp() >= expiration_timer) {
    // Someone else is already writing and time has not expired.
    *mode = PASSTHROUGH;
    return Status::OK();
  } else {
    // Time has expired, we write regardless.
    *mode = WRITER;
    return Status::OK();
  }
}

AsyncWriter::AsyncWriter(Env* env, int64_t file_index,
                         const std::string& shard_directory,
                         uint64 checkpoint_id, const std::string& compression,
                         int64_t version, const DataTypeVector& output_types,
                         std::function<void(Status)> done) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("shard_directory: \"" + shard_directory + "\"");
   mht_61_v.push_back("compression: \"" + compression + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_61(mht_61_v, 1410, "", "./tensorflow/core/data/snapshot_utils.cc", "AsyncWriter::AsyncWriter");

  thread_ = absl::WrapUnique(env->StartThread(
      ThreadOptions(), absl::StrCat("writer_thread_", file_index),
      [this, env, shard_directory, checkpoint_id, compression, version,
       &output_types, done = std::move(done)] {
        done(WriterThread(env, shard_directory, checkpoint_id, compression,
                          version, output_types));
      }));
}

void AsyncWriter::Write(const std::vector<Tensor>& tensors) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_62(mht_62_v, 1423, "", "./tensorflow/core/data/snapshot_utils.cc", "AsyncWriter::Write");

  mutex_lock l(mu_);
  ElementOrEOF element;
  element.value = tensors;
  deque_.push_back(std::move(element));
}

void AsyncWriter::SignalEOF() {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_63(mht_63_v, 1433, "", "./tensorflow/core/data/snapshot_utils.cc", "AsyncWriter::SignalEOF");

  mutex_lock l(mu_);
  ElementOrEOF be;
  be.end_of_sequence = true;
  deque_.push_back(std::move(be));
}

void AsyncWriter::Consume(ElementOrEOF* be) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_64(mht_64_v, 1443, "", "./tensorflow/core/data/snapshot_utils.cc", "AsyncWriter::Consume");

  mutex_lock l(mu_);
  mu_.Await(tensorflow::Condition(this, &AsyncWriter::ElementAvailable));
  *be = deque_.front();
  deque_.pop_front();
}

bool AsyncWriter::ElementAvailable() {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_65(mht_65_v, 1453, "", "./tensorflow/core/data/snapshot_utils.cc", "AsyncWriter::ElementAvailable");
 return !deque_.empty(); }

Status AsyncWriter::WriterThread(Env* env, const std::string& shard_directory,
                                 uint64 checkpoint_id,
                                 const std::string& compression,
                                 int64_t version, DataTypeVector output_types) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("shard_directory: \"" + shard_directory + "\"");
   mht_66_v.push_back("compression: \"" + compression + "\"");
   MHTracer_DTPStensorflowPScorePSdataPSsnapshot_utilsDTcc mht_66(mht_66_v, 1463, "", "./tensorflow/core/data/snapshot_utils.cc", "AsyncWriter::WriterThread");

  std::unique_ptr<snapshot_util::Writer> writer;
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(shard_directory));

  TF_RETURN_IF_ERROR(snapshot_util::Writer::Create(
      env, GetCheckpointFileName(shard_directory, checkpoint_id), compression,
      version, std::move(output_types), &writer));

  while (true) {
    ElementOrEOF be;
    Consume(&be);

    if (be.end_of_sequence) {
      TF_RETURN_IF_ERROR(writer->Close());
      break;
    }

    TF_RETURN_IF_ERROR(writer->WriteTensors(be.value));
  }
  return Status::OK();
}

namespace {

REGISTER_KERNEL_BUILDER(Name("SnapshotDatasetReader").Device(DEVICE_CPU),
                        Reader::DatasetOp);
REGISTER_KERNEL_BUILDER(Name("SnapshotNestedDatasetReader").Device(DEVICE_CPU),
                        Reader::NestedDatasetOp);

}  // namespace
}  // namespace snapshot_util
}  // namespace data
}  // namespace tensorflow
