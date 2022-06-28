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
class MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/record_reader.h"

#include <limits.h>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace io {

RecordReaderOptions RecordReaderOptions::CreateRecordReaderOptions(
    const string& compression_type) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("compression_type: \"" + compression_type + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReaderOptions::CreateRecordReaderOptions");

  RecordReaderOptions options;

#if defined(IS_SLIM_BUILD)
  if (compression_type != compression::kNone) {
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
  }
#else
  if (compression_type == compression::kZlib) {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
    options.zlib_options = io::ZlibCompressionOptions::GZIP();
  } else if (compression_type == compression::kSnappy) {
    options.compression_type = io::RecordReaderOptions::SNAPPY_COMPRESSION;
  } else if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No compression will be used.";
  }
#endif
  return options;
}

RecordReader::RecordReader(RandomAccessFile* file,
                           const RecordReaderOptions& options)
    : options_(options),
      input_stream_(new RandomAccessInputStream(file)),
      last_read_failed_(false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReader::RecordReader");

  if (options.buffer_size > 0) {
    input_stream_.reset(new BufferedInputStream(input_stream_.release(),
                                                options.buffer_size, true));
  }
#if defined(IS_SLIM_BUILD)
  if (options.compression_type != RecordReaderOptions::NONE) {
    LOG(FATAL) << "Compression is unsupported on mobile platforms.";
  }
#else
  if (options.compression_type == RecordReaderOptions::ZLIB_COMPRESSION) {
    input_stream_.reset(new ZlibInputStream(
        input_stream_.release(), options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options, true));
  } else if (options.compression_type ==
             RecordReaderOptions::SNAPPY_COMPRESSION) {
    input_stream_.reset(
        new SnappyInputStream(input_stream_.release(),
                              options.snappy_options.output_buffer_size, true));
  } else if (options.compression_type == RecordReaderOptions::NONE) {
    // Nothing to do.
  } else {
    LOG(FATAL) << "Unrecognized compression type :" << options.compression_type;
  }
#endif
}

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
//
// offset corresponds to the user-provided value to ReadRecord()
// and is used only in error messages.
Status RecordReader::ReadChecksummed(uint64 offset, size_t n, tstring* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_2(mht_2_v, 269, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReader::ReadChecksummed");

  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  const size_t expected = n + sizeof(uint32);
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(expected, result));

  if (result->size() != expected) {
    if (result->empty()) {
      return errors::OutOfRange("eof");
    } else {
      return errors::DataLoss("truncated record at ", offset);
    }
  }

  const uint32 masked_crc = core::DecodeFixed32(result->data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(result->data(), n)) {
    return errors::DataLoss("corrupted record at ", offset);
  }
  result->resize(n);
  return Status::OK();
}

Status RecordReader::GetMetadata(Metadata* md) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_3(mht_3_v, 296, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReader::GetMetadata");

  if (!md) {
    return errors::InvalidArgument(
        "Metadata object call to GetMetadata() was null");
  }

  // Compute the metadata of the TFRecord file if not cached.
  if (!cached_metadata_) {
    TF_RETURN_IF_ERROR(input_stream_->Reset());

    int64_t data_size = 0;
    int64_t entries = 0;

    // Within the loop, we always increment offset positively, so this
    // loop should be guaranteed to either return after reaching EOF
    // or encountering an error.
    uint64 offset = 0;
    tstring record;
    while (true) {
      // Read header, containing size of data.
      Status s = ReadChecksummed(offset, sizeof(uint64), &record);
      if (!s.ok()) {
        if (errors::IsOutOfRange(s)) {
          // We should reach out of range when the record file is complete.
          break;
        }
        return s;
      }

      // Read the length of the data.
      const uint64 length = core::DecodeFixed64(record.data());

      // Skip reading the actual data since we just want the number
      // of records and the size of the data.
      TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(length + kFooterSize));
      offset += kHeaderSize + length + kFooterSize;

      // Increment running stats.
      data_size += length;
      ++entries;
    }

    cached_metadata_.reset(new Metadata());
    cached_metadata_->stats.entries = entries;
    cached_metadata_->stats.data_size = data_size;
    cached_metadata_->stats.file_size =
        data_size + (kHeaderSize + kFooterSize) * entries;
  }

  md->stats = cached_metadata_->stats;
  return Status::OK();
}

Status RecordReader::PositionInputStream(uint64 offset) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_4(mht_4_v, 352, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReader::PositionInputStream");

  int64_t curr_pos = input_stream_->Tell();
  int64_t desired_pos = static_cast<int64_t>(offset);
  if (curr_pos > desired_pos || curr_pos < 0 /* EOF */ ||
      (curr_pos == desired_pos && last_read_failed_)) {
    last_read_failed_ = false;
    TF_RETURN_IF_ERROR(input_stream_->Reset());
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos));
  } else if (curr_pos < desired_pos) {
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos - curr_pos));
  }
  DCHECK_EQ(desired_pos, input_stream_->Tell());
  return Status::OK();
}

Status RecordReader::ReadRecord(uint64* offset, tstring* record) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_5(mht_5_v, 370, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReader::ReadRecord");

  TF_RETURN_IF_ERROR(PositionInputStream(*offset));

  // Read header data.
  Status s = ReadChecksummed(*offset, sizeof(uint64), record);
  if (!s.ok()) {
    last_read_failed_ = true;
    return s;
  }
  const uint64 length = core::DecodeFixed64(record->data());

  // Read data
  s = ReadChecksummed(*offset + kHeaderSize, length, record);
  if (!s.ok()) {
    last_read_failed_ = true;
    if (errors::IsOutOfRange(s)) {
      s = errors::DataLoss("truncated record at ", *offset, "' failed with ",
                           s.error_message());
    }
    return s;
  }

  *offset += kHeaderSize + length + kFooterSize;
  DCHECK_EQ(*offset, input_stream_->Tell());
  return Status::OK();
}

Status RecordReader::SkipRecords(uint64* offset, int num_to_skip,
                                 int* num_skipped) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_6(mht_6_v, 401, "", "./tensorflow/core/lib/io/record_reader.cc", "RecordReader::SkipRecords");

  TF_RETURN_IF_ERROR(PositionInputStream(*offset));

  Status s;
  tstring record;
  *num_skipped = 0;
  for (int i = 0; i < num_to_skip; ++i) {
    s = ReadChecksummed(*offset, sizeof(uint64), &record);
    if (!s.ok()) {
      last_read_failed_ = true;
      return s;
    }
    const uint64 length = core::DecodeFixed64(record.data());

    // Skip data
    s = input_stream_->SkipNBytes(length + kFooterSize);
    if (!s.ok()) {
      last_read_failed_ = true;
      if (errors::IsOutOfRange(s)) {
        s = errors::DataLoss("truncated record at ", *offset, "' failed with ",
                             s.error_message());
      }
      return s;
    }
    *offset += kHeaderSize + length + kFooterSize;
    DCHECK_EQ(*offset, input_stream_->Tell());
    (*num_skipped)++;
  }
  return Status::OK();
}

SequentialRecordReader::SequentialRecordReader(
    RandomAccessFile* file, const RecordReaderOptions& options)
    : underlying_(file, options), offset_(0) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTcc mht_7(mht_7_v, 437, "", "./tensorflow/core/lib/io/record_reader.cc", "SequentialRecordReader::SequentialRecordReader");
}

}  // namespace io
}  // namespace tensorflow
