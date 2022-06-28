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
class MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/zlib_inputstream.h"

#include <zlib.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace io {

struct ZStreamDef {
  ZStreamDef(size_t input_buffer_capacity, size_t output_buffer_capacity)
      : input(new Bytef[input_buffer_capacity]),
        output(new Bytef[output_buffer_capacity]),
        stream(new z_stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZStreamDef");
}

  // Buffer for storing contents read from compressed stream.
  // TODO(srbs): Consider using circular buffers. That would greatly simplify
  // the implementation.
  std::unique_ptr<Bytef[]> input;

  // Buffer for storing inflated contents of `input_stream_`.
  std::unique_ptr<Bytef[]> output;

  // Configuration passed to `inflate`.
  //
  // z_stream_def_->stream->next_in:
  //   Next byte to de-compress. Points to some byte in
  //   z_stream_def_->streamdef_.input buffer.
  // z_stream_def_->stream->avail_in:
  //   Number of bytes available to be decompressed at this time.
  // z_stream_def_->stream->next_out:
  //   Next byte to write de-compressed data to. Points to some byte in
  //   z_stream_def_->streamdef_.output buffer.
  // z_stream_def_->stream->avail_out:
  //   Number of free bytes available at write location.
  std::unique_ptr<z_stream> stream;
};

ZlibInputStream::ZlibInputStream(
    InputStreamInterface* input_stream,
    size_t input_buffer_bytes,   // size of z_stream.next_in buffer
    size_t output_buffer_bytes,  // size of z_stream.next_out buffer
    const ZlibCompressionOptions& zlib_options, bool owns_input_stream)
    : owns_input_stream_(owns_input_stream),
      input_stream_(input_stream),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      zlib_options_(zlib_options),
      z_stream_def_(
          new ZStreamDef(input_buffer_capacity_, output_buffer_capacity_)),
      bytes_read_(0) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::ZlibInputStream");

  InitZlibBuffer();
}

ZlibInputStream::ZlibInputStream(InputStreamInterface* input_stream,
                                 size_t input_buffer_bytes,
                                 size_t output_buffer_bytes,
                                 const ZlibCompressionOptions& zlib_options)
    : ZlibInputStream(input_stream, input_buffer_bytes, output_buffer_bytes,
                      zlib_options, false) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_2(mht_2_v, 251, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::ZlibInputStream");
}

ZlibInputStream::~ZlibInputStream() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::~ZlibInputStream");

  if (z_stream_def_->stream && !init_error_) {
    inflateEnd(z_stream_def_->stream.get());
  }
  if (owns_input_stream_) {
    delete input_stream_;
  }
}

Status ZlibInputStream::Reset() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::Reset");

  if (init_error_) {
    return errors::DataLoss("unable to reset stream, cannot decompress.");
  }
  TF_RETURN_IF_ERROR(input_stream_->Reset());
  inflateEnd(z_stream_def_->stream.get());
  InitZlibBuffer();
  bytes_read_ = 0;
  return Status::OK();
}

void ZlibInputStream::InitZlibBuffer() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_5(mht_5_v, 282, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::InitZlibBuffer");

  memset(z_stream_def_->stream.get(), 0, sizeof(z_stream));

  z_stream_def_->stream->zalloc = Z_NULL;
  z_stream_def_->stream->zfree = Z_NULL;
  z_stream_def_->stream->opaque = Z_NULL;
  z_stream_def_->stream->next_in = Z_NULL;
  z_stream_def_->stream->avail_in = 0;

  int status =
      inflateInit2(z_stream_def_->stream.get(), zlib_options_.window_bits);

  if (zlib_options_.soft_fail_on_error && status != Z_OK) {
    init_error_ = true;
    return;
  }
  CHECK_EQ(status, Z_OK) << "inflateInit failed with status " << status;

  z_stream_def_->stream->next_in = z_stream_def_->input.get();
  z_stream_def_->stream->next_out = z_stream_def_->output.get();
  next_unread_byte_ = reinterpret_cast<char*>(z_stream_def_->output.get());
  z_stream_def_->stream->avail_in = 0;
  z_stream_def_->stream->avail_out = output_buffer_capacity_;
}

Status ZlibInputStream::ReadFromStream() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_6(mht_6_v, 310, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::ReadFromStream");

  int bytes_to_read = input_buffer_capacity_;
  char* read_location = reinterpret_cast<char*>(z_stream_def_->input.get());

  // If there are unread bytes in the input stream we move them to the head
  // of the stream to maximize the space available to read new data into.
  if (z_stream_def_->stream->avail_in > 0) {
    uLong read_bytes =
        z_stream_def_->stream->next_in - z_stream_def_->input.get();
    // Remove `read_bytes` from the head of the input stream.
    // Move unread bytes to the head of the input stream.
    if (read_bytes > 0) {
      memmove(z_stream_def_->input.get(), z_stream_def_->stream->next_in,
              z_stream_def_->stream->avail_in);
    }

    bytes_to_read -= z_stream_def_->stream->avail_in;
    read_location += z_stream_def_->stream->avail_in;
  }
  tstring data;
  // Try to read enough data to fill up z_stream_def_->input.
  // TODO(rohanj): Add a char* version of ReadNBytes to InputStreamInterface
  // and use that instead to make this more efficient.
  Status s = input_stream_->ReadNBytes(bytes_to_read, &data);
  memcpy(read_location, data.data(), data.size());

  // Since we moved unread data to the head of the input stream we can point
  // next_in to the head of the input stream.
  z_stream_def_->stream->next_in = z_stream_def_->input.get();

  // Note: data.size() could be different from bytes_to_read.
  z_stream_def_->stream->avail_in += data.size();

  if (!s.ok() && !errors::IsOutOfRange(s)) {
    return s;
  }

  // We throw OutOfRange error iff no new data has been read from stream.
  // Since we never check how much data is remaining in the stream, it is
  // possible that on the last read there isn't enough data in the stream to
  // fill up the buffer in which case input_stream_->ReadNBytes would return an
  // OutOfRange error.
  if (data.empty()) {
    return errors::OutOfRange("EOF reached");
  }
  if (errors::IsOutOfRange(s)) {
    return Status::OK();
  }

  return s;
}

size_t ZlibInputStream::ReadBytesFromCache(size_t bytes_to_read,
                                           tstring* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_7(mht_7_v, 366, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::ReadBytesFromCache");

  size_t unread_bytes =
      reinterpret_cast<char*>(z_stream_def_->stream->next_out) -
      next_unread_byte_;
  size_t can_read_bytes = std::min(bytes_to_read, unread_bytes);
  if (can_read_bytes > 0) {
    result->append(next_unread_byte_, can_read_bytes);
    next_unread_byte_ += can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

size_t ZlibInputStream::NumUnreadBytes() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_8(mht_8_v, 382, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::NumUnreadBytes");

  size_t read_bytes =
      next_unread_byte_ - reinterpret_cast<char*>(z_stream_def_->output.get());
  return output_buffer_capacity_ - z_stream_def_->stream->avail_out -
         read_bytes;
}

Status ZlibInputStream::ReadNBytes(int64_t bytes_to_read, tstring* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_9(mht_9_v, 392, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::ReadNBytes");

  if (init_error_) {
    return errors::DataLoss("Unable to decompress Zlib file.");
  }

  result->clear();
  // Read as many bytes as possible from cache.
  bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);

  while (bytes_to_read > 0) {
    // At this point we can be sure that cache has been emptied.
    DCHECK_EQ(NumUnreadBytes(), 0);

    // Now that the cache is empty we need to inflate more data.

    // Step 1. Setup output stream.
    z_stream_def_->stream->next_out = z_stream_def_->output.get();
    next_unread_byte_ = reinterpret_cast<char*>(z_stream_def_->output.get());
    z_stream_def_->stream->avail_out = output_buffer_capacity_;

    // Step 2. Try to inflate some input data.
    TF_RETURN_IF_ERROR(Inflate());

    // Step 3. Read any data produced by inflate. If no progress was made by
    // inflate, read more compressed data from the input stream.
    if (NumUnreadBytes() == 0) {
      TF_RETURN_IF_ERROR(ReadFromStream());
    } else {
      bytes_to_read -= ReadBytesFromCache(bytes_to_read, result);
    }
  }

  return Status::OK();
}

#if defined(TF_CORD_SUPPORT)
Status ZlibInputStream::ReadNBytes(int64_t bytes_to_read, absl::Cord* result) {
  // TODO(frankchn): Optimize this instead of bouncing through the buffer.
  tstring buf;
  TF_RETURN_IF_ERROR(ReadNBytes(bytes_to_read, &buf));
  result->Clear();
  result->Append(buf.data());
  return Status::OK();
}
#endif

int64_t ZlibInputStream::Tell() const { return bytes_read_; }

Status ZlibInputStream::Inflate() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_inputstreamDTcc mht_10(mht_10_v, 443, "", "./tensorflow/core/lib/io/zlib_inputstream.cc", "ZlibInputStream::Inflate");

  int error = inflate(z_stream_def_->stream.get(), zlib_options_.flush_mode);
  // Source: http://zlib.net/manual.html
  // Z_BUF_ERROR: `inflate` returns Z_BUF_ERROR if no progress was made. This is
  // not fatal and `inflate` can be called again with more input and output
  // space to continue inflating.
  if (error != Z_OK && error != Z_STREAM_END && error != Z_BUF_ERROR) {
    string error_string =
        strings::StrCat("inflate() failed with error ", error);
    if (z_stream_def_->stream->msg != nullptr) {
      strings::StrAppend(&error_string, ": ", z_stream_def_->stream->msg);
    }
    return errors::DataLoss(error_string);
  }
  if (error == Z_STREAM_END && zlib_options_.window_bits == MAX_WBITS + 16) {
    inflateReset(z_stream_def_->stream.get());
  }
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
