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
class MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc() {
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

// Decodes the blocks generated by block_builder.cc.

#include "tensorflow/core/lib/io/block.h"

#include <algorithm>
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace table {

inline uint32 Block::NumRestarts() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/lib/io/block.cc", "Block::NumRestarts");

  assert(size_ >= sizeof(uint32));
  return core::DecodeFixed32(data_ + size_ - sizeof(uint32));
}

Block::Block(const BlockContents& contents)
    : data_(contents.data.data()),
      size_(contents.data.size()),
      owned_(contents.heap_allocated) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/lib/io/block.cc", "Block::Block");

  if (size_ < sizeof(uint32)) {
    size_ = 0;  // Error marker
  } else {
    size_t max_restarts_allowed = (size_ - sizeof(uint32)) / sizeof(uint32);
    if (NumRestarts() > max_restarts_allowed) {
      // The size is too small for NumRestarts()
      size_ = 0;
    } else {
      restart_offset_ = size_ - (1 + NumRestarts()) * sizeof(uint32);
    }
  }
}

Block::~Block() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_2(mht_2_v, 226, "", "./tensorflow/core/lib/io/block.cc", "Block::~Block");

  if (owned_) {
    delete[] data_;
  }
}

// Helper routine: decode the next block entry starting at "p",
// storing the number of shared key bytes, non_shared key bytes,
// and the length of the value in "*shared", "*non_shared", and
// "*value_length", respectively.  Will not dereference past "limit".
//
// If any errors are detected, returns NULL.  Otherwise, returns a
// pointer to the key delta (just past the three decoded values).
static inline const char* DecodeEntry(const char* p, const char* limit,
                                      uint32* shared, uint32* non_shared,
                                      uint32* value_length) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("p: \"" + (p == nullptr ? std::string("nullptr") : std::string((char*)p)) + "\"");
   mht_3_v.push_back("limit: \"" + (limit == nullptr ? std::string("nullptr") : std::string((char*)limit)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_3(mht_3_v, 246, "", "./tensorflow/core/lib/io/block.cc", "DecodeEntry");

  if (limit - p < 3) return nullptr;
  *shared = reinterpret_cast<const unsigned char*>(p)[0];
  *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
  *value_length = reinterpret_cast<const unsigned char*>(p)[2];
  if ((*shared | *non_shared | *value_length) < 128) {
    // Fast path: all three values are encoded in one byte each
    p += 3;
  } else {
    if ((p = core::GetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
    if ((p = core::GetVarint32Ptr(p, limit, non_shared)) == nullptr)
      return nullptr;
    if ((p = core::GetVarint32Ptr(p, limit, value_length)) == nullptr)
      return nullptr;
  }

  if (static_cast<uint32>(limit - p) < (*non_shared + *value_length)) {
    return nullptr;
  }
  return p;
}

class Block::Iter : public Iterator {
 private:
  const char* const data_;     // underlying block contents
  uint32 const restarts_;      // Offset of restart array (list of fixed32)
  uint32 const num_restarts_;  // Number of uint32 entries in restart array

  // current_ is offset in data_ of current entry.  >= restarts_ if !Valid
  uint32 current_;
  uint32 restart_index_;  // Index of restart block in which current_ falls
  string key_;
  StringPiece value_;
  Status status_;

  inline int Compare(const StringPiece& a, const StringPiece& b) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_4(mht_4_v, 284, "", "./tensorflow/core/lib/io/block.cc", "Compare");

    return a.compare(b);
  }

  // Return the offset in data_ just past the end of the current entry.
  inline uint32 NextEntryOffset() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_5(mht_5_v, 292, "", "./tensorflow/core/lib/io/block.cc", "NextEntryOffset");

    return (value_.data() + value_.size()) - data_;
  }

  uint32 GetRestartPoint(uint32 index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_6(mht_6_v, 299, "", "./tensorflow/core/lib/io/block.cc", "GetRestartPoint");

    assert(index < num_restarts_);
    return core::DecodeFixed32(data_ + restarts_ + index * sizeof(uint32));
  }

  void SeekToRestartPoint(uint32 index) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/lib/io/block.cc", "SeekToRestartPoint");

    key_.clear();
    restart_index_ = index;
    // current_ will be fixed by ParseNextKey();

    // ParseNextKey() starts at the end of value_, so set value_ accordingly
    uint32 offset = GetRestartPoint(index);
    value_ = StringPiece(data_ + offset, 0);
  }

 public:
  Iter(const char* data, uint32 restarts, uint32 num_restarts)
      : data_(data),
        restarts_(restarts),
        num_restarts_(num_restarts),
        current_(restarts_),
        restart_index_(num_restarts_) {
    assert(num_restarts_ > 0);
  }

  bool Valid() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_8(mht_8_v, 330, "", "./tensorflow/core/lib/io/block.cc", "Valid");
 return current_ < restarts_; }
  Status status() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_9(mht_9_v, 334, "", "./tensorflow/core/lib/io/block.cc", "status");
 return status_; }
  StringPiece key() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_10(mht_10_v, 338, "", "./tensorflow/core/lib/io/block.cc", "key");

    assert(Valid());
    return key_;
  }
  StringPiece value() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_11(mht_11_v, 345, "", "./tensorflow/core/lib/io/block.cc", "value");

    assert(Valid());
    return value_;
  }

  void Next() override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_12(mht_12_v, 353, "", "./tensorflow/core/lib/io/block.cc", "Next");

    assert(Valid());
    ParseNextKey();
  }

  void Seek(const StringPiece& target) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_13(mht_13_v, 361, "", "./tensorflow/core/lib/io/block.cc", "Seek");

    // Binary search in restart array to find the last restart point
    // with a key < target
    uint32 left = 0;
    uint32 right = num_restarts_ - 1;
    while (left < right) {
      uint32 mid = left + (right - left + 1) / 2;
      uint32 region_offset = GetRestartPoint(mid);
      uint32 shared, non_shared, value_length;
      const char* key_ptr =
          DecodeEntry(data_ + region_offset, data_ + restarts_, &shared,
                      &non_shared, &value_length);
      if (key_ptr == nullptr || (shared != 0)) {
        CorruptionError();
        return;
      }
      StringPiece mid_key(key_ptr, non_shared);
      if (Compare(mid_key, target) < 0) {
        // Key at "mid" is smaller than "target".  Therefore all
        // blocks before "mid" are uninteresting.
        left = mid;
      } else {
        // Key at "mid" is >= "target".  Therefore all blocks at or
        // after "mid" are uninteresting.
        right = mid - 1;
      }
    }

    // Linear search (within restart block) for first key >= target
    SeekToRestartPoint(left);
    while (true) {
      if (!ParseNextKey()) {
        return;
      }
      if (Compare(key_, target) >= 0) {
        return;
      }
    }
  }

  void SeekToFirst() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_14(mht_14_v, 404, "", "./tensorflow/core/lib/io/block.cc", "SeekToFirst");

    SeekToRestartPoint(0);
    ParseNextKey();
  }

 private:
  void CorruptionError() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_15(mht_15_v, 413, "", "./tensorflow/core/lib/io/block.cc", "CorruptionError");

    current_ = restarts_;
    restart_index_ = num_restarts_;
    status_ = errors::DataLoss("bad entry in block");
    key_.clear();
    value_ = StringPiece();
  }

  bool ParseNextKey() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_16(mht_16_v, 424, "", "./tensorflow/core/lib/io/block.cc", "ParseNextKey");

    current_ = NextEntryOffset();
    const char* p = data_ + current_;
    const char* limit = data_ + restarts_;  // Restarts come right after data
    if (p >= limit) {
      // No more entries to return.  Mark as invalid.
      current_ = restarts_;
      restart_index_ = num_restarts_;
      return false;
    }

    // Decode next entry
    uint32 shared, non_shared, value_length;
    p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    if (p == nullptr || key_.size() < shared) {
      CorruptionError();
      return false;
    } else {
      key_.resize(shared);
      key_.append(p, non_shared);
      value_ = StringPiece(p + non_shared, value_length);
      while (restart_index_ + 1 < num_restarts_ &&
             GetRestartPoint(restart_index_ + 1) < current_) {
        ++restart_index_;
      }
      return true;
    }
  }
};

Iterator* Block::NewIterator() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPSblockDTcc mht_17(mht_17_v, 457, "", "./tensorflow/core/lib/io/block.cc", "Block::NewIterator");

  if (size_ < sizeof(uint32)) {
    return NewErrorIterator(errors::DataLoss("bad block contents"));
  }
  const uint32 num_restarts = NumRestarts();
  if (num_restarts == 0) {
    return NewEmptyIterator();
  } else {
    return new Iter(data_, restart_offset_, num_restarts);
  }
}

}  // namespace table
}  // namespace tensorflow
