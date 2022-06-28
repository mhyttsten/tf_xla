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
class MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc() {
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

#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"

namespace tensorflow {
namespace io {
SnappyInputBuffer::SnappyInputBuffer(
    RandomAccessFile* file,
    size_t input_buffer_bytes,  // size of input_buffer_
    size_t output_buffer_bytes  // size of output_buffer_
    )
    : file_(file),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      input_buffer_(new char[input_buffer_capacity_]),
      output_buffer_(new char[output_buffer_capacity_]),
      next_in_(input_buffer_.get()),
      bytes_read_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_0(mht_0_v, 200, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::SnappyInputBuffer");
}

Status SnappyInputBuffer::ReadNBytes(int64_t bytes_to_read, tstring* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::ReadNBytes");

  result->clear();
  result->resize_uninitialized(bytes_to_read);

  char* result_ptr = result->mdata();

  // Read as many bytes as possible from cache.
  size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
  bytes_to_read -= bytes_read;
  result_ptr += bytes_read;

  while (bytes_to_read > 0) {
    // At this point we can be sure that cache has been emptied.
    DCHECK_EQ(avail_out_, 0);

    // Now that the cache is empty we need to inflate more data.
    TF_RETURN_IF_ERROR(Inflate());

    bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
    bytes_to_read -= bytes_read;
    result_ptr += bytes_read;
  }

  return Status::OK();
}

int64_t SnappyInputBuffer::Tell() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::Tell");
 return bytes_read_; }

Status SnappyInputBuffer::Reset() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::Reset");

  file_pos_ = 0;
  avail_in_ = 0;
  avail_out_ = 0;
  next_in_ = input_buffer_.get();
  bytes_read_ = 0;
  return Status::OK();
}

size_t SnappyInputBuffer::ReadBytesFromCache(size_t bytes_to_read,
                                             char* result_ptr) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("result_ptr: \"" + (result_ptr == nullptr ? std::string("nullptr") : std::string((char*)result_ptr)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_4(mht_4_v, 253, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::ReadBytesFromCache");

  size_t can_read_bytes = std::min(bytes_to_read, avail_out_);
  if (can_read_bytes > 0) {
    memcpy(result_ptr, next_out_, can_read_bytes);
    next_out_ += can_read_bytes;
    avail_out_ -= can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

Status SnappyInputBuffer::Inflate() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_5(mht_5_v, 267, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::Inflate");

  // Read length of compressed block.
  uint32 compressed_block_length;
  TF_RETURN_IF_ERROR(ReadCompressedBlockLength(&compressed_block_length));

  // If the entire block is not in cache do a read from file.
  if (avail_in_ < compressed_block_length) {
    TF_RETURN_IF_ERROR(ReadFromFile());
    if (avail_in_ < compressed_block_length) {
      if (compressed_block_length > input_buffer_capacity_) {
        return errors::ResourceExhausted(
            "Input buffer(size: ", input_buffer_capacity_,
            " bytes) too small. Should be larger ", "than ",
            compressed_block_length, " bytes.");
      } else {
        return errors::DataLoss(
            strings::StrCat("Failed to read ", compressed_block_length,
                            " bytes from file. Possible data corruption."));
      }
    }
  }

  size_t uncompressed_length;
  if (!port::Snappy_GetUncompressedLength(next_in_, compressed_block_length,
                                          &uncompressed_length)) {
    return errors::DataLoss("Parsing error in Snappy_GetUncompressedLength");
  }

  // Output buffer must have been cleared before uncompressing more input.
  DCHECK_EQ(avail_out_, 0);

  // Output buffer must be large enough to fit the uncompressed block.
  DCHECK_GE(output_buffer_capacity_, uncompressed_length);
  next_out_ = output_buffer_.get();

  bool status = port::Snappy_Uncompress(next_in_, compressed_block_length,
                                        output_buffer_.get());
  if (!status) {
    return errors::DataLoss("Snappy_Uncompress failed");
  }
  next_in_ += compressed_block_length;
  avail_in_ -= compressed_block_length;
  avail_out_ += uncompressed_length;
  return Status::OK();
}

Status SnappyInputBuffer::ReadCompressedBlockLength(uint32* length) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_6(mht_6_v, 316, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::ReadCompressedBlockLength");

  *length = 0;
  size_t bytes_to_read = 4;
  while (bytes_to_read > 0) {
    if (avail_in_ == 0) {
      TF_RETURN_IF_ERROR(ReadFromFile());
    }
    size_t readable = std::min(bytes_to_read, avail_in_);

    for (size_t i = 0; i < readable; i++) {
      // The "unsigned char" type cast is intentional to avoid implicit type
      // casting of the signed char to unsigned int during bitwise OR which
      // causes weird overflow errors.
      *length = (*length << 8) | static_cast<unsigned char>(next_in_[0]);
      bytes_to_read--;
      next_in_++;
      avail_in_--;
    }
  }
  return Status::OK();
}

Status SnappyInputBuffer::ReadFromFile() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSsnappyPSsnappy_inputbufferDTcc mht_7(mht_7_v, 341, "", "./tensorflow/core/lib/io/snappy/snappy_inputbuffer.cc", "SnappyInputBuffer::ReadFromFile");

  int bytes_to_read = input_buffer_capacity_;
  char* read_location = reinterpret_cast<char*>(input_buffer_.get());

  // If there are unread bytes in the input stream we move them to the head
  // of the stream to maximize the space available to read new data into.
  // TODO(srbs): A circular buffer would be useful here.
  if (avail_in_ > 0) {
    size_t read_bytes = next_in_ - input_buffer_.get();
    // Remove `read_bytes` from the head of the input stream.
    // Move unread bytes to the head of the input stream.
    if (read_bytes > 0) {
      memmove(input_buffer_.get(), next_in_, avail_in_);
    }

    bytes_to_read -= avail_in_;
    read_location += avail_in_;
  }
  StringPiece data;
  // Try to read enough data to fill up input_buffer_.
  Status s = file_->Read(file_pos_, bytes_to_read, &data, read_location);
  if (data.data() != read_location) {
    memmove(read_location, data.data(), data.size());
  }

  // Since we moved unread data to the head of the input stream we can point
  // next_in to the head of the input stream.
  next_in_ = input_buffer_.get();

  // Note: data.size() could be different from bytes_to_read.
  avail_in_ += data.size();
  file_pos_ += data.size();

  if (!s.ok() && !errors::IsOutOfRange(s)) {
    return s;
  }

  // We throw OutOfRange error iff no new data has been read from file.
  // Since we never check how much data is remaining in the file, it is
  // possible that on the last read there isn't enough data in the file to
  // fill up the buffer in which case file_->ReadNBytes would return an
  // OutOfRange error.
  if (data.empty()) {
    return errors::OutOfRange("EOF reached");
  }
  if (errors::IsOutOfRange(s)) {
    return Status::OK();
  }

  return s;
}

}  // namespace io
}  // namespace tensorflow
