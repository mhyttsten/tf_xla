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
class MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc() {
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

#include "tensorflow/core/lib/io/zlib_outputbuffer.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace io {

ZlibOutputBuffer::ZlibOutputBuffer(
    WritableFile* file,
    int32_t input_buffer_bytes,  // size of z_stream.next_in buffer
    int32_t output_buffer_bytes,
    const ZlibCompressionOptions&
        zlib_options)  // size of z_stream.next_out buffer
    : file_(file),
      init_status_(),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      z_stream_input_(new Bytef[input_buffer_bytes]),
      z_stream_output_(new Bytef[output_buffer_bytes]),
      zlib_options_(zlib_options),
      z_stream_(new z_stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::ZlibOutputBuffer");
}

ZlibOutputBuffer::~ZlibOutputBuffer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::~ZlibOutputBuffer");

  if (z_stream_) {
    LOG(WARNING) << "ZlibOutputBuffer::Close() not called. Possible data loss";
  }
}

Status ZlibOutputBuffer::Init() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Init");

  // Output buffer size should be greater than 1 because deflation needs at
  // least one byte for book keeping etc.
  if (output_buffer_capacity_ <= 1) {
    return errors::InvalidArgument(
        "output_buffer_bytes should be greater than "
        "1");
  }
  memset(z_stream_.get(), 0, sizeof(z_stream));
  z_stream_->zalloc = Z_NULL;
  z_stream_->zfree = Z_NULL;
  z_stream_->opaque = Z_NULL;
  int status =
      deflateInit2(z_stream_.get(), zlib_options_.compression_level,
                   zlib_options_.compression_method, zlib_options_.window_bits,
                   zlib_options_.mem_level, zlib_options_.compression_strategy);
  if (status != Z_OK) {
    z_stream_.reset(nullptr);
    return errors::InvalidArgument("deflateInit failed with status", status);
  }
  z_stream_->next_in = z_stream_input_.get();
  z_stream_->next_out = z_stream_output_.get();
  z_stream_->avail_in = 0;
  z_stream_->avail_out = output_buffer_capacity_;
  return Status::OK();
}

int32 ZlibOutputBuffer::AvailableInputSpace() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_3(mht_3_v, 249, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::AvailableInputSpace");

  return input_buffer_capacity_ - z_stream_->avail_in;
}

void ZlibOutputBuffer::AddToInputBuffer(StringPiece data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_4(mht_4_v, 256, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::AddToInputBuffer");

  size_t bytes_to_write = data.size();
  CHECK_LE(bytes_to_write, AvailableInputSpace());

  // Input stream ->
  // [....................input_buffer_capacity_...............]
  // [<...read_bytes...><...avail_in...>......empty space......]
  //  ^                 ^
  //  |                 |
  //  z_stream_input_   next_in
  //
  // Data in the input stream is sharded as show above. z_stream_->next_in could
  // be pointing to some byte in the buffer with avail_in number of bytes
  // available to be read.
  //
  // In order to avoid shifting the avail_in bytes at next_in to the head of
  // the buffer we try to fit `data` in the empty space at the tail of the
  // input stream.
  // TODO(srbs): This could be avoided if we had a circular buffer.
  // If it doesn't fit we free the space at the head of the stream and then
  // append `data` at the end of existing data.

  int32_t read_bytes = z_stream_->next_in - z_stream_input_.get();
  int32_t unread_bytes = z_stream_->avail_in;
  int32_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  if (static_cast<int32>(bytes_to_write) > free_tail_bytes) {
    memmove(z_stream_input_.get(), z_stream_->next_in, z_stream_->avail_in);
    z_stream_->next_in = z_stream_input_.get();
  }
  memcpy(z_stream_->next_in + z_stream_->avail_in, data.data(), bytes_to_write);
  z_stream_->avail_in += bytes_to_write;
}

Status ZlibOutputBuffer::DeflateBuffered(int flush_mode) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_5(mht_5_v, 294, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::DeflateBuffered");

  do {
    // From zlib manual (http://www.zlib.net/manual.html):
    //
    // "In the case of a Z_FULL_FLUSH or Z_SYNC_FLUSH, make sure that
    // avail_out is greater than six to avoid repeated flush markers due
    // to avail_out == 0 on return."
    //
    // If above condition is met or if output buffer is full we flush contents
    // to file.
    if (z_stream_->avail_out == 0 ||
        (IsSyncOrFullFlush(flush_mode) && z_stream_->avail_out < 6)) {
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
    TF_RETURN_IF_ERROR(Deflate(flush_mode));
  } while (z_stream_->avail_out == 0);

  DCHECK(z_stream_->avail_in == 0);
  z_stream_->next_in = z_stream_input_.get();
  return Status::OK();
}

Status ZlibOutputBuffer::FlushOutputBufferToFile() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_6(mht_6_v, 319, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::FlushOutputBufferToFile");

  uint32 bytes_to_write = output_buffer_capacity_ - z_stream_->avail_out;
  if (bytes_to_write > 0) {
    Status s = file_->Append(StringPiece(
        reinterpret_cast<char*>(z_stream_output_.get()), bytes_to_write));
    if (s.ok()) {
      z_stream_->next_out = z_stream_output_.get();
      z_stream_->avail_out = output_buffer_capacity_;
    }
    return s;
  }
  return Status::OK();
}

Status ZlibOutputBuffer::Append(StringPiece data) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_7(mht_7_v, 336, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Append");

  // If there is sufficient free space in z_stream_input_ to fit data we
  // add it there and return.
  // If there isn't enough space we deflate the existing contents of
  // z_input_stream_. If data now fits in z_input_stream_ we add it there
  // else we directly deflate it.
  //
  // The deflated output is accumulated in z_stream_output_ and gets written to
  // file as and when needed.

  size_t bytes_to_write = data.size();

  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(DeflateBuffered(zlib_options_.flush_mode));

  // At this point input stream should be empty.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  // `data` is too large to fit in input buffer so we deflate it directly.
  // Note that at this point we have already deflated all existing input so
  // we do not need to backup next_in and avail_in.
  z_stream_->next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
  z_stream_->avail_in = bytes_to_write;

  do {
    if (z_stream_->avail_out == 0) {
      // No available output space.
      // Write output buffer to file.
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
    TF_RETURN_IF_ERROR(Deflate(zlib_options_.flush_mode));
  } while (z_stream_->avail_out == 0);

  DCHECK(z_stream_->avail_in == 0);  // All input will be used up.

  // Restore z_stream input pointers.
  z_stream_->next_in = z_stream_input_.get();

  return Status::OK();
}

#if defined(TF_CORD_SUPPORT)
Status ZlibOutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return Status::OK();
}
#endif

Status ZlibOutputBuffer::Flush() {
  TF_RETURN_IF_ERROR(DeflateBuffered(Z_PARTIAL_FLUSH));
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return file_->Flush();
}

Status ZlibOutputBuffer::Name(StringPiece* result) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_8(mht_8_v, 402, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Name");

  return file_->Name(result);
}

Status ZlibOutputBuffer::Sync() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_9(mht_9_v, 409, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Sync");

  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

Status ZlibOutputBuffer::Close() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_10(mht_10_v, 417, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Close");

  if (z_stream_) {
    TF_RETURN_IF_ERROR(DeflateBuffered(Z_FINISH));
    TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    deflateEnd(z_stream_.get());
    z_stream_.reset(nullptr);
  }
  return Status::OK();
}

Status ZlibOutputBuffer::Deflate(int flush) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_11(mht_11_v, 430, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Deflate");

  int error = deflate(z_stream_.get(), flush);
  if (error == Z_OK || error == Z_BUF_ERROR ||
      (error == Z_STREAM_END && flush == Z_FINISH)) {
    return Status::OK();
  }
  string error_string = strings::StrCat("deflate() failed with error ", error);
  if (z_stream_->msg != nullptr) {
    strings::StrAppend(&error_string, ": ", z_stream_->msg);
  }
  return errors::DataLoss(error_string);
}

Status ZlibOutputBuffer::Tell(int64_t* position) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSzlib_outputbufferDTcc mht_12(mht_12_v, 446, "", "./tensorflow/core/lib/io/zlib_outputbuffer.cc", "ZlibOutputBuffer::Tell");

  return file_->Tell(position);
}

}  // namespace io
}  // namespace tensorflow
