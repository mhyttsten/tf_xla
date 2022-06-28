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
class MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc() {
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

#include "tensorflow/core/lib/io/snappy/snappy_inputstream.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/snappy.h"

namespace tensorflow {
namespace io {

SnappyInputStream::SnappyInputStream(InputStreamInterface* input_stream,
                                     size_t output_buffer_bytes,
                                     bool owns_input_stream)
    : input_stream_(input_stream),
      output_buffer_bytes_(output_buffer_bytes),
      owns_input_stream_(owns_input_stream),
      bytes_read_(0),
      output_buffer_(new char[output_buffer_bytes]),
      next_out_(nullptr),
      avail_out_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::SnappyInputStream");
}

SnappyInputStream::SnappyInputStream(InputStreamInterface* input_stream,
                                     size_t output_buffer_bytes)
    : SnappyInputStream(input_stream, output_buffer_bytes, false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::SnappyInputStream");
}

SnappyInputStream::~SnappyInputStream() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_2(mht_2_v, 215, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::~SnappyInputStream");

  if (owns_input_stream_) {
    delete input_stream_;
  }
}

Status SnappyInputStream::ReadNBytes(int64_t bytes_to_read, tstring* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_3(mht_3_v, 224, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::ReadNBytes");

  result->clear();
  result->resize_uninitialized(bytes_to_read);

  char* result_ptr = result->mdata();

  // Read as many bytes as possible from the cache.
  size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
  bytes_to_read -= bytes_read;
  result_ptr += bytes_read;

  while (bytes_to_read > 0) {
    DCHECK_EQ(avail_out_, 0);

    // Fill the cache with more data.
    TF_RETURN_IF_ERROR(Inflate());

    size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
    bytes_to_read -= bytes_read;
    result_ptr += bytes_read;
  }

  return Status::OK();
}

#if defined(TF_CORD_SUPPORT)
Status SnappyInputStream::ReadNBytes(int64_t bytes_to_read,
                                     absl::Cord* result) {
  // TODO(frankchn): Optimize this instead of bouncing through the buffer.
  tstring buf;
  TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &buf));
  result->Clear();
  result->Append(buf.data());
  return Status::OK();
}
#endif

Status SnappyInputStream::Inflate() {
  tstring compressed_block_length_ts;
  uint32 compressed_block_length;

  TF_RETURN_IF_ERROR(
      input_stream_->ReadNBytes(sizeof(uint32), &compressed_block_length_ts));
  for (int i = 0; i < sizeof(uint32); ++i) {
    compressed_block_length =
        (compressed_block_length << 8) |
        static_cast<unsigned char>(compressed_block_length_ts.data()[i]);
  }

  tstring compressed_block;
  compressed_block.resize_uninitialized(compressed_block_length);

  Status s =
      input_stream_->ReadNBytes(compressed_block_length, &compressed_block);
  if (errors::IsOutOfRange(s)) {
    return errors::DataLoss("Failed to read ", compressed_block_length,
                            " bytes from file. Possible data corruption.");
  }
  TF_RETURN_IF_ERROR(s);

  size_t uncompressed_length;
  if (!port::Snappy_GetUncompressedLength(compressed_block.data(),
                                          compressed_block_length,
                                          &uncompressed_length)) {
    return errors::DataLoss("Parsing error in Snappy_GetUncompressedLength");
  }

  DCHECK_EQ(avail_out_, 0);
  if (output_buffer_bytes_ < uncompressed_length) {
    return errors::ResourceExhausted(
        "Output buffer(size: ", output_buffer_bytes_,
        " bytes"
        ") too small. Should be larger than ",
        uncompressed_length, " bytes.");
  }

  next_out_ = output_buffer_.get();
  if (!port::Snappy_Uncompress(compressed_block.data(), compressed_block_length,
                               output_buffer_.get())) {
    return errors::DataLoss("Snappy_Uncompress failed.");
  }
  avail_out_ += uncompressed_length;

  return Status::OK();
}

size_t SnappyInputStream::ReadBytesFromCache(size_t bytes_to_read,
                                             char* result) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("result: \"" + (result == nullptr ? std::string("nullptr") : std::string((char*)result)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_4(mht_4_v, 315, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::ReadBytesFromCache");

  size_t can_read_bytes = std::min(bytes_to_read, avail_out_);
  if (can_read_bytes) {
    memcpy(result, next_out_, can_read_bytes);
    next_out_ += can_read_bytes;
    avail_out_ -= can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

int64_t SnappyInputStream::Tell() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_5(mht_5_v, 329, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::Tell");
 return bytes_read_; }

Status SnappyInputStream::Reset() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputstreamDTcc mht_6(mht_6_v, 334, "", "./tensorflow/core/lib/io/snappy/snappy_inputstream.cc", "SnappyInputStream::Reset");

  TF_RETURN_IF_ERROR(input_stream_->Reset());
  avail_out_ = 0;
  bytes_read_ = 0;
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
