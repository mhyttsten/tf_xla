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

#ifndef TENSORFLOW_CORE_LIB_IO_RECORD_WRITER_H_
#define TENSORFLOW_CORE_LIB_IO_RECORD_WRITER_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh() {
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


#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#if !defined(IS_SLIM_BUILD)
#include "tensorflow/core/lib/io/snappy/snappy_compression_options.h"
#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/lib/io/zlib_outputbuffer.h"
#endif  // IS_SLIM_BUILD
#include "tensorflow/core/platform/cord.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class WritableFile;

namespace io {

struct RecordWriterOptions {
 public:
  enum CompressionType {
    NONE = 0,
    ZLIB_COMPRESSION = 1,
    SNAPPY_COMPRESSION = 2
  };
  CompressionType compression_type = NONE;

  static RecordWriterOptions CreateRecordWriterOptions(
      const string& compression_type);

#if !defined(IS_SLIM_BUILD)
  // Options specific to compression.
  tensorflow::io::ZlibCompressionOptions zlib_options;
  tensorflow::io::SnappyCompressionOptions snappy_options;
#endif  // IS_SLIM_BUILD
};

class RecordWriter {
 public:
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  static constexpr size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static constexpr size_t kFooterSize = sizeof(uint32);

  // Create a writer that will append data to "*dest".
  // "*dest" must be initially empty.
  // "*dest" must remain live while this Writer is in use.
  explicit RecordWriter(WritableFile* dest, const RecordWriterOptions& options =
                                                RecordWriterOptions());

  // Calls Close() and logs if an error occurs.
  //
  // TODO(jhseu): Require that callers explicitly call Close() and remove the
  // implicit Close() call in the destructor.
  ~RecordWriter();

  Status WriteRecord(StringPiece data);

#if defined(TF_CORD_SUPPORT)
  Status WriteRecord(const absl::Cord& data);
#endif

  // Flushes any buffered data held by underlying containers of the
  // RecordWriter to the WritableFile. Does *not* flush the
  // WritableFile.
  Status Flush();

  // Writes all output to the file. Does *not* close the WritableFile.
  //
  // After calling Close(), any further calls to `WriteRecord()` or `Flush()`
  // are invalid.
  Status Close();

  // Utility method to populate TFRecord headers.  Populates record-header in
  // "header[0,kHeaderSize-1]".  The record-header is based on data[0, n-1].
  inline static void PopulateHeader(char* header, const char* data, size_t n);

#if defined(TF_CORD_SUPPORT)
  inline static void PopulateHeader(char* header, const absl::Cord& data);
#endif

  // Utility method to populate TFRecord footers.  Populates record-footer in
  // "footer[0,kFooterSize-1]".  The record-footer is based on data[0, n-1].
  inline static void PopulateFooter(char* footer, const char* data, size_t n);

#if defined(TF_CORD_SUPPORT)
  inline static void PopulateFooter(char* footer, const absl::Cord& data);
#endif

 private:
  WritableFile* dest_;
  RecordWriterOptions options_;

  inline static uint32 MaskedCrc(const char* data, size_t n) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh mht_0(mht_0_v, 287, "", "./tensorflow/core/lib/io/record_writer.h", "MaskedCrc");

    return crc32c::Mask(crc32c::Value(data, n));
  }

#if defined(TF_CORD_SUPPORT)
  inline static uint32 MaskedCrc(const absl::Cord& data) {
    return crc32c::Mask(crc32c::Value(data));
  }
#endif

  TF_DISALLOW_COPY_AND_ASSIGN(RecordWriter);
};

void RecordWriter::PopulateHeader(char* header, const char* data, size_t n) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("header: \"" + (header == nullptr ? std::string("nullptr") : std::string((char*)header)) + "\"");
   mht_1_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh mht_1(mht_1_v, 305, "", "./tensorflow/core/lib/io/record_writer.h", "RecordWriter::PopulateHeader");

  core::EncodeFixed64(header + 0, n);
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
}

void RecordWriter::PopulateFooter(char* footer, const char* data, size_t n) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("footer: \"" + (footer == nullptr ? std::string("nullptr") : std::string((char*)footer)) + "\"");
   mht_2_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh mht_2(mht_2_v, 316, "", "./tensorflow/core/lib/io/record_writer.h", "RecordWriter::PopulateFooter");

  core::EncodeFixed32(footer, MaskedCrc(data, n));
}

#if defined(TF_CORD_SUPPORT)
void RecordWriter::PopulateHeader(char* header, const absl::Cord& data) {
  core::EncodeFixed64(header + 0, data.size());
  core::EncodeFixed32(header + sizeof(uint64),
                      MaskedCrc(header, sizeof(uint64)));
}

void RecordWriter::PopulateFooter(char* footer, const absl::Cord& data) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("footer: \"" + (footer == nullptr ? std::string("nullptr") : std::string((char*)footer)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSrecord_writerDTh mht_3(mht_3_v, 331, "", "./tensorflow/core/lib/io/record_writer.h", "RecordWriter::PopulateFooter");

  core::EncodeFixed32(footer, MaskedCrc(data));
}
#endif

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_RECORD_WRITER_H_
