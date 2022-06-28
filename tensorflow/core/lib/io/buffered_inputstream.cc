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
class MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc() {
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

#include "tensorflow/core/lib/io/buffered_inputstream.h"

#include "tensorflow/core/lib/io/random_inputstream.h"

namespace tensorflow {
namespace io {

BufferedInputStream::BufferedInputStream(InputStreamInterface* input_stream,
                                         size_t buffer_bytes,
                                         bool owns_input_stream)
    : input_stream_(input_stream),
      size_(buffer_bytes),
      owns_input_stream_(owns_input_stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::BufferedInputStream");

  buf_.reserve(size_);
}

BufferedInputStream::BufferedInputStream(RandomAccessFile* file,
                                         size_t buffer_bytes)
    : BufferedInputStream(new RandomAccessInputStream(file), buffer_bytes,
                          true) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::BufferedInputStream");
}

BufferedInputStream::~BufferedInputStream() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_2(mht_2_v, 212, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::~BufferedInputStream");

  if (owns_input_stream_) {
    delete input_stream_;
  }
}

Status BufferedInputStream::FillBuffer() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_3(mht_3_v, 221, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::FillBuffer");

  if (!file_status_.ok()) {
    pos_ = 0;
    limit_ = 0;
    return file_status_;
  }
  Status s = input_stream_->ReadNBytes(size_, &buf_);
  pos_ = 0;
  limit_ = buf_.size();
  if (!s.ok()) {
    file_status_ = s;
  }
  return s;
}

template <typename StringType>
Status BufferedInputStream::ReadLineHelper(StringType* result,
                                           bool include_eol) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_4(mht_4_v, 241, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::ReadLineHelper");

  result->clear();
  Status s;
  size_t start_pos = pos_;
  while (true) {
    if (pos_ == limit_) {
      result->append(buf_.data() + start_pos, pos_ - start_pos);
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == 0) {
        break;
      }
      start_pos = pos_;
    }
    char c = buf_[pos_];
    if (c == '\n') {
      result->append(buf_.data() + start_pos, pos_ - start_pos);
      if (include_eol) {
        result->append(1, c);
      }
      pos_++;
      return Status::OK();
    }
    // We don't append '\r' to *result
    if (c == '\r') {
      result->append(buf_.data() + start_pos, pos_ - start_pos);
      start_pos = pos_ + 1;
    }
    pos_++;
  }
  if (errors::IsOutOfRange(s) && !result->empty()) {
    return Status::OK();
  }
  return s;
}

Status BufferedInputStream::ReadNBytes(int64_t bytes_to_read, tstring* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_5(mht_5_v, 280, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::ReadNBytes");

  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }
  result->clear();
  if (pos_ == limit_ && !file_status_.ok() && bytes_to_read > 0) {
    return file_status_;
  }
  result->reserve(bytes_to_read);

  Status s;
  while (result->size() < static_cast<size_t>(bytes_to_read)) {
    // Check whether the buffer is fully read or not.
    if (pos_ == limit_) {
      s = FillBuffer();
      // If we didn't read any bytes, we're at the end of the file; break out.
      if (limit_ == 0) {
        DCHECK(!s.ok());
        file_status_ = s;
        break;
      }
    }
    const int64_t bytes_to_copy =
        std::min<int64_t>(limit_ - pos_, bytes_to_read - result->size());
    result->insert(result->size(), buf_, pos_, bytes_to_copy);
    pos_ += bytes_to_copy;
  }
  // Filling the buffer might lead to a situation when we go past the end of
  // the file leading to an OutOfRange() status return. But we might have
  // obtained enough data to satisfy the function call. Returning OK then.
  if (errors::IsOutOfRange(s) &&
      (result->size() == static_cast<size_t>(bytes_to_read))) {
    return Status::OK();
  }
  return s;
}

Status BufferedInputStream::SkipNBytes(int64_t bytes_to_skip) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_6(mht_6_v, 321, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::SkipNBytes");

  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can only skip forward, not ",
                                   bytes_to_skip);
  }
  if (pos_ + bytes_to_skip < limit_) {
    // If we aren't skipping too much, then we can just move pos_;
    pos_ += bytes_to_skip;
  } else {
    // Otherwise, we already have read limit_ - pos_, so skip the rest. At this
    // point we need to get fresh data into the buffer, so reset pos_ and
    // limit_.
    Status s = input_stream_->SkipNBytes(bytes_to_skip - (limit_ - pos_));
    pos_ = 0;
    limit_ = 0;
    if (errors::IsOutOfRange(s)) {
      file_status_ = s;
    }
    return s;
  }
  return Status::OK();
}

int64_t BufferedInputStream::Tell() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_7(mht_7_v, 347, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::Tell");

  return input_stream_->Tell() - (limit_ - pos_);
}

Status BufferedInputStream::Seek(int64_t position) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_8(mht_8_v, 354, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::Seek");

  if (position < 0) {
    return errors::InvalidArgument("Seeking to a negative position: ",
                                   position);
  }

  // Position of the buffer's lower limit within file.
  const int64_t buf_lower_limit = input_stream_->Tell() - limit_;
  if (position < buf_lower_limit) {
    // Seek before buffer, reset input stream and skip 'position' bytes.
    TF_RETURN_IF_ERROR(Reset());
    return SkipNBytes(position);
  }

  if (position < Tell()) {
    // Seek within buffer before 'pos_'
    pos_ -= Tell() - position;
    return Status::OK();
  }

  // Seek after 'pos_'
  return SkipNBytes(position - Tell());
}

template <typename T>
Status BufferedInputStream::ReadAll(T* result) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_9(mht_9_v, 382, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::ReadAll");

  result->clear();
  Status status;
  while (status.ok()) {
    status = FillBuffer();
    if (limit_ == 0) {
      break;
    }
    result->append(buf_);
    pos_ = limit_;
  }

  if (errors::IsOutOfRange(status)) {
    file_status_ = status;
    return Status::OK();
  }
  return status;
}

template Status BufferedInputStream::ReadAll<std::string>(std::string* result);
template Status BufferedInputStream::ReadAll<tstring>(tstring* result);

Status BufferedInputStream::Reset() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_10(mht_10_v, 407, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::Reset");

  TF_RETURN_IF_ERROR(input_stream_->Reset());
  pos_ = 0;
  limit_ = 0;
  file_status_ = Status::OK();
  return Status::OK();
}

Status BufferedInputStream::ReadLine(std::string* result) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_11(mht_11_v, 418, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::ReadLine");

  return ReadLineHelper(result, false);
}

Status BufferedInputStream::ReadLine(tstring* result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_12(mht_12_v, 425, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::ReadLine");

  return ReadLineHelper(result, false);
}

std::string BufferedInputStream::ReadLineAsString() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_13(mht_13_v, 432, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::ReadLineAsString");

  std::string result;
  ReadLineHelper(&result, true).IgnoreError();
  return result;
}

Status BufferedInputStream::SkipLine() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSbuffered_inputstreamDTcc mht_14(mht_14_v, 441, "", "./tensorflow/core/lib/io/buffered_inputstream.cc", "BufferedInputStream::SkipLine");

  Status s;
  bool skipped = false;
  while (true) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == 0) {
        break;
      }
    }
    char c = buf_[pos_++];
    skipped = true;
    if (c == '\n') {
      return Status::OK();
    }
  }
  if (errors::IsOutOfRange(s) && skipped) {
    return Status::OK();
  }
  return s;
}

}  // namespace io
}  // namespace tensorflow
