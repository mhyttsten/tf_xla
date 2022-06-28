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

#ifndef TENSORFLOW_LIB_IO_FORMAT_H_
#define TENSORFLOW_LIB_IO_FORMAT_H_
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
class MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh() {
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


#include <stdint.h>
#include <string>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/table_builder.h"

namespace tensorflow {
class RandomAccessFile;
namespace table {

class Block;

// BlockHandle is a pointer to the extent of a file that stores a data
// block or a meta block.
class BlockHandle {
 public:
  BlockHandle();

  // The offset of the block in the file.
  uint64 offset() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_0(mht_0_v, 207, "", "./tensorflow/core/lib/io/format.h", "offset");
 return offset_; }
  void set_offset(uint64 offset) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_1(mht_1_v, 211, "", "./tensorflow/core/lib/io/format.h", "set_offset");
 offset_ = offset; }

  // The size of the stored block
  uint64 size() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_2(mht_2_v, 217, "", "./tensorflow/core/lib/io/format.h", "size");
 return size_; }
  void set_size(uint64 size) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_3(mht_3_v, 221, "", "./tensorflow/core/lib/io/format.h", "set_size");
 size_ = size; }

  void EncodeTo(string* dst) const;
  Status DecodeFrom(StringPiece* input);

  // Maximum encoding length of a BlockHandle
  enum { kMaxEncodedLength = 10 + 10 };

 private:
  uint64 offset_;
  uint64 size_;
};

// Footer encapsulates the fixed information stored at the tail
// end of every table file.
class Footer {
 public:
  Footer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_4(mht_4_v, 241, "", "./tensorflow/core/lib/io/format.h", "Footer");
}

  // The block handle for the metaindex block of the table
  const BlockHandle& metaindex_handle() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_5(mht_5_v, 247, "", "./tensorflow/core/lib/io/format.h", "metaindex_handle");
 return metaindex_handle_; }
  void set_metaindex_handle(const BlockHandle& h) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_6(mht_6_v, 251, "", "./tensorflow/core/lib/io/format.h", "set_metaindex_handle");
 metaindex_handle_ = h; }

  // The block handle for the index block of the table
  const BlockHandle& index_handle() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_7(mht_7_v, 257, "", "./tensorflow/core/lib/io/format.h", "index_handle");
 return index_handle_; }
  void set_index_handle(const BlockHandle& h) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_8(mht_8_v, 261, "", "./tensorflow/core/lib/io/format.h", "set_index_handle");
 index_handle_ = h; }

  void EncodeTo(string* dst) const;
  Status DecodeFrom(StringPiece* input);

  // Encoded length of a Footer.  Note that the serialization of a
  // Footer will always occupy exactly this many bytes.  It consists
  // of two block handles and a magic number.
  enum { kEncodedLength = 2 * BlockHandle::kMaxEncodedLength + 8 };

 private:
  BlockHandle metaindex_handle_;
  BlockHandle index_handle_;
};

// kTableMagicNumber was picked by running
//    echo http://code.google.com/p/leveldb/ | sha1sum
// and taking the leading 64 bits.
static const uint64 kTableMagicNumber = 0xdb4775248b80fb57ull;

// 1-byte type + 32-bit crc
static const size_t kBlockTrailerSize = 5;

struct BlockContents {
  StringPiece data;     // Actual contents of data
  bool cacheable;       // True iff data can be cached
  bool heap_allocated;  // True iff caller should delete[] data.data()
};

// Read the block identified by "handle" from "file".  On failure
// return non-OK.  On success fill *result and return OK.
extern Status ReadBlock(RandomAccessFile* file, const BlockHandle& handle,
                        BlockContents* result);

// Implementation details follow.  Clients should ignore,

inline BlockHandle::BlockHandle()
    : offset_(~static_cast<uint64>(0)), size_(~static_cast<uint64>(0)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSformatDTh mht_9(mht_9_v, 301, "", "./tensorflow/core/lib/io/format.h", "BlockHandle::BlockHandle");
}

}  // namespace table
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_FORMAT_H_
