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
class MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc() {
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

#include "tensorflow/core/lib/io/inputbuffer.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace io {

InputBuffer::InputBuffer(RandomAccessFile* file, size_t buffer_bytes)
    : file_(file),
      file_pos_(0),
      size_(buffer_bytes),
      buf_(new char[size_]),
      pos_(buf_),
      limit_(buf_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::InputBuffer");
}

InputBuffer::~InputBuffer() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::~InputBuffer");
 delete[] buf_; }

Status InputBuffer::FillBuffer() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::FillBuffer");

  StringPiece data;
  Status s = file_->Read(file_pos_, size_, &data, buf_);
  if (data.data() != buf_) {
    memmove(buf_, data.data(), data.size());
  }
  pos_ = buf_;
  limit_ = pos_ + data.size();
  file_pos_ += data.size();
  return s;
}

template <typename T>
Status InputBuffer::ReadLine(T* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_3(mht_3_v, 225, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::ReadLine");

  result->clear();
  Status s;
  do {
    size_t buf_remain = limit_ - pos_;
    char* newline = static_cast<char*>(memchr(pos_, '\n', buf_remain));
    if (newline != nullptr) {
      size_t result_len = newline - pos_;
      result->append(pos_, result_len);
      pos_ = newline + 1;
      if (!result->empty() && result->back() == '\r') {
        result->resize(result->size() - 1);
      }
      return Status::OK();
    }
    if (buf_remain > 0) result->append(pos_, buf_remain);
    // Get more data into buffer
    s = FillBuffer();
    DCHECK_EQ(pos_, buf_);
  } while (limit_ != buf_);
  if (!result->empty() && result->back() == '\r') {
    result->resize(result->size() - 1);
  }
  if (errors::IsOutOfRange(s) && !result->empty()) {
    return Status::OK();
  }
  return s;
}

template Status InputBuffer::ReadLine<std::string>(std::string* result);
template Status InputBuffer::ReadLine<tstring>(tstring* result);

Status InputBuffer::ReadNBytes(int64_t bytes_to_read, std::string* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_4(mht_4_v, 260, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::ReadNBytes");

  result->clear();
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }
  result->resize(bytes_to_read);
  size_t bytes_read = 0;
  Status status = ReadNBytes(bytes_to_read, &(*result)[0], &bytes_read);
  if (bytes_read < bytes_to_read) result->resize(bytes_read);
  return status;
}

Status InputBuffer::ReadNBytes(int64_t bytes_to_read, char* result,
                               size_t* bytes_read) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("result: \"" + (result == nullptr ? std::string("nullptr") : std::string((char*)result)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::ReadNBytes");

  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }
  Status status;
  *bytes_read = 0;
  while (*bytes_read < static_cast<size_t>(bytes_to_read)) {
    if (pos_ == limit_) {
      // Get more data into buffer.
      status = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    // Do not go over the buffer boundary.
    const int64_t bytes_to_copy =
        std::min<int64_t>(limit_ - pos_, bytes_to_read - *bytes_read);
    // Copies buffered data into the destination.
    memcpy(result + *bytes_read, pos_, bytes_to_copy);
    pos_ += bytes_to_copy;
    *bytes_read += bytes_to_copy;
  }
  if (errors::IsOutOfRange(status) &&
      (*bytes_read == static_cast<size_t>(bytes_to_read))) {
    return Status::OK();
  }
  return status;
}

Status InputBuffer::ReadVarint32Fallback(uint32* result) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_6(mht_6_v, 311, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::ReadVarint32Fallback");

  Status s = ReadVarintFallback(result, core::kMaxVarint32Bytes);
  if (errors::IsDataLoss(s)) {
    return errors::DataLoss("Stored data is too large to be a varint32.");
  }
  return s;
}

Status InputBuffer::ReadVarint64Fallback(uint64* result) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_7(mht_7_v, 322, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::ReadVarint64Fallback");

  Status s = ReadVarintFallback(result, core::kMaxVarint64Bytes);
  if (errors::IsDataLoss(s)) {
    return errors::DataLoss("Stored data is too large to be a varint64.");
  }
  return s;
}

template <typename T>
Status InputBuffer::ReadVarintFallback(T* result, int max_bytes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_8(mht_8_v, 334, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::ReadVarintFallback");

  uint8 scratch = 0;
  auto* p = reinterpret_cast<char*>(&scratch);
  size_t unused_bytes_read = 0;

  *result = 0;
  for (int index = 0; index < max_bytes; index++) {
    int shift = 7 * index;
    TF_RETURN_IF_ERROR(ReadNBytes(1, p, &unused_bytes_read));
    *result |= (static_cast<T>(scratch) & 127) << shift;
    if (!(scratch & 128)) return Status::OK();
  }
  return errors::DataLoss("Stored data longer than ", max_bytes, " bytes.");
}

Status InputBuffer::SkipNBytes(int64_t bytes_to_skip) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_9(mht_9_v, 352, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::SkipNBytes");

  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can only skip forward, not ",
                                   bytes_to_skip);
  }
  int64_t bytes_skipped = 0;
  Status s;
  while (bytes_skipped < bytes_to_skip) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    const int64_t bytes_to_advance =
        std::min<int64_t>(limit_ - pos_, bytes_to_skip - bytes_skipped);
    bytes_skipped += bytes_to_advance;
    pos_ += bytes_to_advance;
  }
  if (errors::IsOutOfRange(s) && bytes_skipped == bytes_to_skip) {
    return Status::OK();
  }
  return s;
}

Status InputBuffer::Seek(int64_t position) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_10(mht_10_v, 381, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::Seek");

  if (position < 0) {
    return errors::InvalidArgument("Seeking to a negative position: ",
                                   position);
  }
  // Position of the buffer within file.
  const int64_t bufpos = file_pos_ - static_cast<int64_t>(limit_ - buf_);
  if (position >= bufpos && position < file_pos_) {
    // Seeks to somewhere inside the buffer.
    pos_ = buf_ + (position - bufpos);
    DCHECK(pos_ >= buf_ && pos_ < limit_);
  } else {
    // Seeks to somewhere outside.  Discards the buffered data.
    pos_ = limit_ = buf_;
    file_pos_ = position;
  }
  return Status::OK();
}

Status InputBuffer::Hint(int64_t bytes_to_read) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSinputbufferDTcc mht_11(mht_11_v, 403, "", "./tensorflow/core/lib/io/inputbuffer.cc", "InputBuffer::Hint");

  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }

  // The internal buffer is too small. Do nothing.
  if (bytes_to_read > size_) {
    return Status::OK();
  }

  const int64_t bytes_remain_in_buf = static_cast<int64_t>(limit_ - pos_);

  // There are enough data in the buffer. Do nothing.
  if (bytes_to_read <= bytes_remain_in_buf) {
    return Status::OK();
  }

  // Additional read from file is necessary. Make some room.
  memmove(buf_, pos_, bytes_remain_in_buf);
  pos_ = buf_;
  limit_ = buf_ + bytes_remain_in_buf;
  bytes_to_read -= bytes_remain_in_buf;

  // Read the remaining bytes from file.
  StringPiece data;
  Status s = file_->Read(file_pos_, bytes_to_read, &data, limit_);
  if (data.data() != limit_) {
    memmove(limit_, data.data(), data.size());
  }
  limit_ += data.size();
  file_pos_ += data.size();

  if (errors::IsOutOfRange(s) && data.size() == bytes_to_read) {
    return Status::OK();
  } else {
    return s;
  }
}

}  // namespace io
}  // namespace tensorflow
