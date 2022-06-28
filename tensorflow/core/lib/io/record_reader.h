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

#ifndef TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
#define TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh() {
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


#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/snappy/snappy_compression_options.h"
#include "tensorflow/core/lib/io/snappy/snappy_inputstream.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_inputstream.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

struct RecordReaderOptions {
  enum CompressionType {
    NONE = 0,
    ZLIB_COMPRESSION = 1,
    SNAPPY_COMPRESSION = 2
  };
  CompressionType compression_type = NONE;

  // If buffer_size is non-zero, then all reads must be sequential, and no
  // skipping around is permitted. (Note: this is the same behavior as reading
  // compressed files.) Consider using SequentialRecordReader.
  int64_t buffer_size = 0;

  static RecordReaderOptions CreateRecordReaderOptions(
      const string& compression_type);

#if !defined(IS_SLIM_BUILD)
  // Options specific to compression.
  ZlibCompressionOptions zlib_options;
  SnappyCompressionOptions snappy_options;
#endif  // IS_SLIM_BUILD
};

// Low-level interface to read TFRecord files.
//
// If using compression or buffering, consider using SequentialRecordReader.
//
// Note: this class is not thread safe; external synchronization required.
class RecordReader {
 public:
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  static constexpr size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static constexpr size_t kFooterSize = sizeof(uint32);

  // Statistics (sizes are in units of bytes)
  struct Stats {
    int64_t file_size = -1;
    int64_t data_size = -1;
    int64_t entries = -1;  // Number of values
  };

  // Metadata for the TFRecord file.
  struct Metadata {
    Stats stats;
  };

  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit RecordReader(
      RandomAccessFile* file,
      const RecordReaderOptions& options = RecordReaderOptions());

  virtual ~RecordReader() = default;

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(uint64* offset, tstring* record);

  // Skip num_to_skip record starting at "*offset" and update *offset
  // to point to the offset of the next num_to_skip + 1 record.
  // Return OK on success, OUT_OF_RANGE for end of file, or something
  // else for an error. "*num_skipped" records the number of records that
  // are actually skipped. It should be equal to num_to_skip on success.
  Status SkipRecords(uint64* offset, int num_to_skip, int* num_skipped);

  // Return the metadata of the Record file.
  //
  // The current implementation scans the file to completion,
  // skipping over the data regions, to extract the metadata once
  // on the first call to GetStats().  An improved implementation
  // would change RecordWriter to write the metadata into TFRecord
  // so that GetMetadata() could be a const method.
  //
  // 'metadata' must not be nullptr.
  Status GetMetadata(Metadata* md);

 private:
  Status ReadChecksummed(uint64 offset, size_t n, tstring* result);
  Status PositionInputStream(uint64 offset);

  RecordReaderOptions options_;
  std::unique_ptr<InputStreamInterface> input_stream_;
  bool last_read_failed_;

  std::unique_ptr<Metadata> cached_metadata_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecordReader);
};

// High-level interface to read TFRecord files.
//
// Note: this class is not thread safe; external synchronization required.
class SequentialRecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit SequentialRecordReader(
      RandomAccessFile* file,
      const RecordReaderOptions& options = RecordReaderOptions());

  virtual ~SequentialRecordReader() = default;

  // Read the next record in the file into *record. Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(tstring* record) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh mht_0(mht_0_v, 315, "", "./tensorflow/core/lib/io/record_reader.h", "ReadRecord");

    return underlying_.ReadRecord(&offset_, record);
  }

  // Skip the next num_to_skip record in the file. Return OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  // "*num_skipped" records the number of records that are actually skipped.
  // It should be equal to num_to_skip on success.
  Status SkipRecords(int num_to_skip, int* num_skipped) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh mht_1(mht_1_v, 326, "", "./tensorflow/core/lib/io/record_reader.h", "SkipRecords");

    return underlying_.SkipRecords(&offset_, num_to_skip, num_skipped);
  }

  // Return the current offset in the file.
  uint64 TellOffset() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh mht_2(mht_2_v, 334, "", "./tensorflow/core/lib/io/record_reader.h", "TellOffset");
 return offset_; }

  // Seek to this offset within the file and set this offset as the current
  // offset. Trying to seek backward will throw error.
  Status SeekOffset(uint64 offset) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_readerDTh mht_3(mht_3_v, 341, "", "./tensorflow/core/lib/io/record_reader.h", "SeekOffset");

    if (offset < offset_)
      return errors::InvalidArgument(
          "Trying to seek offset: ", offset,
          " which is less than the current offset: ", offset_);
    offset_ = offset;
    return Status::OK();
  }

 private:
  RecordReader underlying_;
  uint64 offset_ = 0;
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_RECORD_READER_H_
