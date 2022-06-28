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
class MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc() {
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

#include "tensorflow/core/lib/io/snappy/snappy_outputbuffer.h"

namespace tensorflow {
namespace io {

SnappyOutputBuffer::SnappyOutputBuffer(WritableFile* file,
                                       int32_t input_buffer_bytes,
                                       int32_t output_buffer_bytes)
    : file_(file),
      input_buffer_(new char[input_buffer_bytes]),
      input_buffer_capacity_(input_buffer_bytes),
      next_in_(input_buffer_.get()),
      output_buffer_(new char[output_buffer_bytes]),
      output_buffer_capacity_(output_buffer_bytes),
      next_out_(output_buffer_.get()),
      avail_out_(output_buffer_bytes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::SnappyOutputBuffer");
}

SnappyOutputBuffer::~SnappyOutputBuffer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::~SnappyOutputBuffer");

  size_t bytes_to_write = output_buffer_capacity_ - avail_out_;
  if (bytes_to_write > 0) {
    LOG(WARNING) << "There is still data in the output buffer. "
                 << "Possible data loss has occurred.";
  }
}

Status SnappyOutputBuffer::Append(StringPiece data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_2(mht_2_v, 216, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Append");
 return Write(data); }

#if defined(TF_CORD_SUPPORT)
Status SnappyOutputBuffer::Append(const absl::Cord& cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    TF_RETURN_IF_ERROR(Append(fragment));
  }
  return Status::OK();
}
#endif

Status SnappyOutputBuffer::Close() {
  // Given that we do not own `file`, we don't close it.
  return Flush();
}

Status SnappyOutputBuffer::Name(StringPiece* result) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Name");

  return file_->Name(result);
}

Status SnappyOutputBuffer::Sync() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_4(mht_4_v, 242, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Sync");

  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

Status SnappyOutputBuffer::Tell(int64_t* position) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_5(mht_5_v, 250, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Tell");

  return file_->Tell(position);
}

Status SnappyOutputBuffer::Write(StringPiece data) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_6(mht_6_v, 257, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Write");

  //
  // The deflated output is accumulated in output_buffer_ and gets written to
  // file as and when needed.

  size_t bytes_to_write = data.size();

  // If there is sufficient free space in input_buffer_ to fit data we
  // add it there and return.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  // If there isn't enough available space in the input_buffer_ we empty it
  // by uncompressing its contents. If data now fits in input_buffer_
  // we add it there else we directly deflate it.
  TF_RETURN_IF_ERROR(DeflateBuffered());

  // input_buffer_ should be empty at this point.
  if (static_cast<int32>(bytes_to_write) <= AvailableInputSpace()) {
    AddToInputBuffer(data);
    return Status::OK();
  }

  // `data` is too large to fit in input buffer so we deflate it directly.
  // Note that at this point we have already deflated all existing input so
  // we do not need to backup next_in and avail_in.
  next_in_ = const_cast<char*>(data.data());
  avail_in_ = bytes_to_write;

  TF_RETURN_IF_ERROR(Deflate());

  DCHECK_EQ(avail_in_, 0);  // All input will be used up.

  next_in_ = input_buffer_.get();

  return Status::OK();
}

Status SnappyOutputBuffer::Flush() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_7(mht_7_v, 300, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Flush");

  TF_RETURN_IF_ERROR(DeflateBuffered());
  TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
  return Status::OK();
}

int32 SnappyOutputBuffer::AvailableInputSpace() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_8(mht_8_v, 309, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::AvailableInputSpace");

  return input_buffer_capacity_ - avail_in_;
}

void SnappyOutputBuffer::AddToInputBuffer(StringPiece data) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_9(mht_9_v, 316, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::AddToInputBuffer");

  size_t bytes_to_write = data.size();
  DCHECK_LE(bytes_to_write, AvailableInputSpace());

  // Input stream ->
  // [....................input_buffer_capacity_...............]
  // [<...read_bytes...><...avail_in...>......empty space......]
  //  ^                 ^
  //  |                 |
  //  input_buffer_   next_in
  //
  // Data in the input stream is sharded as shown above. next_in_ could
  // be pointing to some byte in the buffer with avail_in number of bytes
  // available to be read.
  //
  // In order to avoid shifting the avail_in bytes at next_in to the head of
  // the buffer we try to fit `data` in the empty space at the tail of the
  // input stream.
  // TODO(srbs): This could be avoided if we had a circular buffer.
  // If it doesn't fit we free the space at the head of the stream and then
  // append `data` at the end of existing data.

  const int32_t read_bytes = next_in_ - input_buffer_.get();
  const int32_t unread_bytes = avail_in_;
  const int32_t free_tail_bytes =
      input_buffer_capacity_ - (read_bytes + unread_bytes);

  if (static_cast<int32>(bytes_to_write) > free_tail_bytes) {
    memmove(input_buffer_.get(), next_in_, avail_in_);
    next_in_ = input_buffer_.get();
  }
  memcpy(next_in_ + avail_in_, data.data(), bytes_to_write);
  avail_in_ += bytes_to_write;
}

Status SnappyOutputBuffer::AddToOutputBuffer(const char* data, size_t length) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_10(mht_10_v, 355, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::AddToOutputBuffer");

  while (length > 0) {
    size_t bytes_to_copy = std::min(length, avail_out_);
    memcpy(next_out_, data, bytes_to_copy);
    data += bytes_to_copy;
    next_out_ += bytes_to_copy;
    avail_out_ -= bytes_to_copy;
    length -= bytes_to_copy;
    if (avail_out_ == 0) {
      TF_RETURN_IF_ERROR(FlushOutputBufferToFile());
    }
  }
  return Status::OK();
}

Status SnappyOutputBuffer::DeflateBuffered() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_11(mht_11_v, 373, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::DeflateBuffered");

  TF_RETURN_IF_ERROR(Deflate());
  DCHECK_EQ(avail_in_, 0);
  next_in_ = input_buffer_.get();
  return Status::OK();
}

Status SnappyOutputBuffer::FlushOutputBufferToFile() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_12(mht_12_v, 383, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::FlushOutputBufferToFile");

  size_t bytes_to_write = output_buffer_capacity_ - avail_out_;
  if (bytes_to_write > 0) {
    Status s = file_->Append(StringPiece(
        reinterpret_cast<char*>(output_buffer_.get()), bytes_to_write));
    if (s.ok()) {
      next_out_ = output_buffer_.get();
      avail_out_ = output_buffer_capacity_;
    }
    return s;
  }
  return Status::OK();
}

Status SnappyOutputBuffer::Deflate() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_outputbufferDTcc mht_13(mht_13_v, 400, "", "./tensorflow/core/lib/io/snappy/snappy_outputbuffer.cc", "SnappyOutputBuffer::Deflate");

  if (avail_in_ == 0) {
    return Status::OK();
  }
  string output;
  if (!port::Snappy_Compress(next_in_, avail_in_, &output)) {
    return errors::DataLoss("Snappy_Compress failed");
  }

  // Write length of compressed block to output buffer.
  char compressed_length_array[4];
  std::fill(compressed_length_array, compressed_length_array + 4, 0);
  for (int i = 0; i < 4; i++) {
    // Little endian.
    compressed_length_array[i] = output.size() >> (8 * (3 - i));
  }
  TF_RETURN_IF_ERROR(AddToOutputBuffer(compressed_length_array, 4));

  // Write compressed output to buffer.
  TF_RETURN_IF_ERROR(AddToOutputBuffer(output.data(), output.size()));
  next_in_ += avail_in_;
  avail_in_ = 0;

  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow
