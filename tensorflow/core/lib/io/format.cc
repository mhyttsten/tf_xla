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
class MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc() {
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

#include <limits>

#include "tensorflow/core/lib/io/format.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/block.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/snappy.h"

namespace tensorflow {
namespace table {

void BlockHandle::EncodeTo(string* dst) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/lib/io/format.cc", "BlockHandle::EncodeTo");

  // Sanity check that all fields have been set
  assert(offset_ != ~static_cast<uint64>(0));
  assert(size_ != ~static_cast<uint64>(0));
  core::PutVarint64(dst, offset_);
  core::PutVarint64(dst, size_);
}

Status BlockHandle::DecodeFrom(StringPiece* input) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/lib/io/format.cc", "BlockHandle::DecodeFrom");

  if (core::GetVarint64(input, &offset_) && core::GetVarint64(input, &size_)) {
    return Status::OK();
  } else {
    return errors::DataLoss("bad block handle");
  }
}

void Footer::EncodeTo(string* dst) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/lib/io/format.cc", "Footer::EncodeTo");

#ifndef NDEBUG
  const size_t original_size = dst->size();
#endif
  metaindex_handle_.EncodeTo(dst);
  index_handle_.EncodeTo(dst);
  dst->resize(2 * BlockHandle::kMaxEncodedLength);  // Padding
  core::PutFixed32(dst, static_cast<uint32>(kTableMagicNumber & 0xffffffffu));
  core::PutFixed32(dst, static_cast<uint32>(kTableMagicNumber >> 32));
  assert(dst->size() == original_size + kEncodedLength);
}

Status Footer::DecodeFrom(StringPiece* input) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/lib/io/format.cc", "Footer::DecodeFrom");

  const char* magic_ptr = input->data() + kEncodedLength - 8;
  const uint32 magic_lo = core::DecodeFixed32(magic_ptr);
  const uint32 magic_hi = core::DecodeFixed32(magic_ptr + 4);
  const uint64 magic =
      ((static_cast<uint64>(magic_hi) << 32) | (static_cast<uint64>(magic_lo)));
  if (magic != kTableMagicNumber) {
    return errors::DataLoss("not an sstable (bad magic number)");
  }

  Status result = metaindex_handle_.DecodeFrom(input);
  if (result.ok()) {
    result = index_handle_.DecodeFrom(input);
  }
  if (result.ok()) {
    // We skip over any leftover data (just padding for now) in "input"
    const char* end = magic_ptr + 8;
    *input = StringPiece(end, input->data() + input->size() - end);
  }
  return result;
}

Status ReadBlock(RandomAccessFile* file, const BlockHandle& handle,
                 BlockContents* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTcc mht_4(mht_4_v, 262, "", "./tensorflow/core/lib/io/format.cc", "ReadBlock");

  result->data = StringPiece();
  result->cacheable = false;
  result->heap_allocated = false;

  // Read the block contents as well as the type/crc footer.
  // See table_builder.cc for the code that built this structure.
  size_t n = static_cast<size_t>(handle.size());

  if (kBlockTrailerSize > std::numeric_limits<size_t>::max() - n) {
    return errors::DataLoss("handle.size() too big");
  }

  char* buf = new char[n + kBlockTrailerSize];
  StringPiece contents;
  Status s = file->Read(handle.offset(), n + kBlockTrailerSize, &contents, buf);
  if (!s.ok()) {
    delete[] buf;
    return s;
  }
  if (contents.size() != n + kBlockTrailerSize) {
    delete[] buf;
    return errors::DataLoss("truncated block read");
  }

  // Check the crc of the type and the block contents
  const char* data = contents.data();  // Pointer to where Read put the data
  // This checksum verification is optional.  We leave it on for now
  const bool verify_checksum = true;
  if (verify_checksum) {
    const uint32 crc = crc32c::Unmask(core::DecodeFixed32(data + n + 1));
    const uint32 actual = crc32c::Value(data, n + 1);
    if (actual != crc) {
      delete[] buf;
      s = errors::DataLoss("block checksum mismatch");
      return s;
    }
  }

  switch (data[n]) {
    case kNoCompression:
      if (data != buf) {
        // File implementation gave us pointer to some other data.
        // Use it directly under the assumption that it will be live
        // while the file is open.
        delete[] buf;
        result->data = StringPiece(data, n);
        result->heap_allocated = false;
        result->cacheable = false;  // Do not double-cache
      } else {
        result->data = StringPiece(buf, n);
        result->heap_allocated = true;
        result->cacheable = true;
      }

      // Ok
      break;
    case kSnappyCompression: {
      size_t ulength = 0;
      if (!port::Snappy_GetUncompressedLength(data, n, &ulength)) {
        delete[] buf;
        return errors::DataLoss("corrupted compressed block contents");
      }
      char* ubuf = new char[ulength];
      if (!port::Snappy_Uncompress(data, n, ubuf)) {
        delete[] buf;
        delete[] ubuf;
        return errors::DataLoss("corrupted compressed block contents");
      }
      delete[] buf;
      result->data = StringPiece(ubuf, ulength);
      result->heap_allocated = true;
      result->cacheable = true;
      break;
    }
    default:
      delete[] buf;
      return errors::DataLoss("bad block type");
  }

  return Status::OK();
}

}  // namespace table
}  // namespace tensorflow
