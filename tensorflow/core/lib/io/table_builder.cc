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
class MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc() {
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

#include "tensorflow/core/lib/io/table_builder.h"

#include <assert.h>
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/block_builder.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/lib/io/table_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/snappy.h"

namespace tensorflow {
namespace table {

namespace {

void FindShortestSeparator(string* start, const StringPiece& limit) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/lib/io/table_builder.cc", "FindShortestSeparator");

  // Find length of common prefix
  size_t min_length = std::min(start->size(), limit.size());
  size_t diff_index = 0;
  while ((diff_index < min_length) &&
         ((*start)[diff_index] == limit[diff_index])) {
    diff_index++;
  }

  if (diff_index >= min_length) {
    // Do not shorten if one string is a prefix of the other
  } else {
    uint8 diff_byte = static_cast<uint8>((*start)[diff_index]);
    if (diff_byte < static_cast<uint8>(0xff) &&
        diff_byte + 1 < static_cast<uint8>(limit[diff_index])) {
      (*start)[diff_index]++;
      start->resize(diff_index + 1);
      assert(StringPiece(*start).compare(limit) < 0);
    }
  }
}

void FindShortSuccessor(string* key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/lib/io/table_builder.cc", "FindShortSuccessor");

  // Find first character that can be incremented
  size_t n = key->size();
  for (size_t i = 0; i < n; i++) {
    const uint8 byte = (*key)[i];
    if (byte != static_cast<uint8>(0xff)) {
      (*key)[i] = byte + 1;
      key->resize(i + 1);
      return;
    }
  }
  // *key is a run of 0xffs.  Leave it alone.
}
}  // namespace

struct TableBuilder::Rep {
  Options options;
  Options index_block_options;
  WritableFile* file;
  uint64 offset;
  Status status;
  BlockBuilder data_block;
  BlockBuilder index_block;
  string last_key;
  int64_t num_entries;
  bool closed;  // Either Finish() or Abandon() has been called.

  // We do not emit the index entry for a block until we have seen the
  // first key for the next data block.  This allows us to use shorter
  // keys in the index block.  For example, consider a block boundary
  // between the keys "the quick brown fox" and "the who".  We can use
  // "the r" as the key for the index block entry since it is >= all
  // entries in the first block and < all entries in subsequent
  // blocks.
  //
  // Invariant: r->pending_index_entry is true only if data_block is empty.
  bool pending_index_entry;
  BlockHandle pending_handle;  // Handle to add to index block

  string compressed_output;

  Rep(const Options& opt, WritableFile* f)
      : options(opt),
        index_block_options(opt),
        file(f),
        offset(0),
        data_block(&options),
        index_block(&index_block_options),
        num_entries(0),
        closed(false),
        pending_index_entry(false) {
    index_block_options.block_restart_interval = 1;
  }
};

TableBuilder::TableBuilder(const Options& options, WritableFile* file)
    : rep_(new Rep(options, file)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_2(mht_2_v, 286, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::TableBuilder");
}

TableBuilder::~TableBuilder() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_3(mht_3_v, 291, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::~TableBuilder");

  assert(rep_->closed);  // Catch errors where caller forgot to call Finish()
  delete rep_;
}

void TableBuilder::Add(const StringPiece& key, const StringPiece& value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_4(mht_4_v, 299, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::Add");

  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->num_entries > 0) {
    assert(key.compare(StringPiece(r->last_key)) > 0);
    // See if this key+value would make our current block overly large.  If
    // so, emit the current block before adding this key/value
    const int kOverlyLargeBlockRatio = 2;
    const size_t this_entry_bytes = key.size() + value.size();
    if (this_entry_bytes >= kOverlyLargeBlockRatio * r->options.block_size) {
      Flush();
    }
  }

  if (r->pending_index_entry) {
    assert(r->data_block.empty());
    FindShortestSeparator(&r->last_key, key);
    string handle_encoding;
    r->pending_handle.EncodeTo(&handle_encoding);
    r->index_block.Add(r->last_key, StringPiece(handle_encoding));
    r->pending_index_entry = false;
  }

  r->last_key.assign(key.data(), key.size());
  r->num_entries++;
  r->data_block.Add(key, value);

  const size_t estimated_block_size = r->data_block.CurrentSizeEstimate();
  if (estimated_block_size >= r->options.block_size) {
    Flush();
  }
}

void TableBuilder::Flush() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_5(mht_5_v, 336, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::Flush");

  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->data_block.empty()) return;
  assert(!r->pending_index_entry);
  WriteBlock(&r->data_block, &r->pending_handle);
  if (ok()) {
    r->pending_index_entry = true;
    // We don't flush the underlying file as that can be slow.
  }
}

void TableBuilder::WriteBlock(BlockBuilder* block, BlockHandle* handle) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_6(mht_6_v, 352, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::WriteBlock");

  // File format contains a sequence of blocks where each block has:
  //    block_data: uint8[n]
  //    type: uint8
  //    crc: uint32
  assert(ok());
  Rep* r = rep_;
  StringPiece raw = block->Finish();

  StringPiece block_contents;
  CompressionType type = r->options.compression;
  // TODO(postrelease): Support more compression options: zlib?
  switch (type) {
    case kNoCompression:
      block_contents = raw;
      break;

    case kSnappyCompression: {
      string* compressed = &r->compressed_output;
      if (port::Snappy_Compress(raw.data(), raw.size(), compressed) &&
          compressed->size() < raw.size() - (raw.size() / 8u)) {
        block_contents = *compressed;
      } else {
        // Snappy not supported, or compressed less than 12.5%, so just
        // store uncompressed form
        block_contents = raw;
        type = kNoCompression;
      }
      break;
    }
  }
  WriteRawBlock(block_contents, type, handle);
  r->compressed_output.clear();
  block->Reset();
}

void TableBuilder::WriteRawBlock(const StringPiece& block_contents,
                                 CompressionType type, BlockHandle* handle) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_7(mht_7_v, 392, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::WriteRawBlock");

  Rep* r = rep_;
  handle->set_offset(r->offset);
  handle->set_size(block_contents.size());
  r->status = r->file->Append(block_contents);
  if (r->status.ok()) {
    char trailer[kBlockTrailerSize];
    trailer[0] = type;
    uint32 crc = crc32c::Value(block_contents.data(), block_contents.size());
    crc = crc32c::Extend(crc, trailer, 1);  // Extend crc to cover block type
    core::EncodeFixed32(trailer + 1, crc32c::Mask(crc));
    r->status = r->file->Append(StringPiece(trailer, kBlockTrailerSize));
    if (r->status.ok()) {
      r->offset += block_contents.size() + kBlockTrailerSize;
    }
  }
}

Status TableBuilder::status() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_8(mht_8_v, 413, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::status");
 return rep_->status; }

Status TableBuilder::Finish() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_9(mht_9_v, 418, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::Finish");

  Rep* r = rep_;
  Flush();
  assert(!r->closed);
  r->closed = true;

  BlockHandle metaindex_block_handle, index_block_handle;

  // Write metaindex block
  if (ok()) {
    BlockBuilder meta_index_block(&r->options);
    // TODO(postrelease): Add stats and other meta blocks
    WriteBlock(&meta_index_block, &metaindex_block_handle);
  }

  // Write index block
  if (ok()) {
    if (r->pending_index_entry) {
      FindShortSuccessor(&r->last_key);
      string handle_encoding;
      r->pending_handle.EncodeTo(&handle_encoding);
      r->index_block.Add(r->last_key, StringPiece(handle_encoding));
      r->pending_index_entry = false;
    }
    WriteBlock(&r->index_block, &index_block_handle);
  }

  // Write footer
  if (ok()) {
    Footer footer;
    footer.set_metaindex_handle(metaindex_block_handle);
    footer.set_index_handle(index_block_handle);
    string footer_encoding;
    footer.EncodeTo(&footer_encoding);
    r->status = r->file->Append(footer_encoding);
    if (r->status.ok()) {
      r->offset += footer_encoding.size();
    }
  }
  return r->status;
}

void TableBuilder::Abandon() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_10(mht_10_v, 463, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::Abandon");

  Rep* r = rep_;
  assert(!r->closed);
  r->closed = true;
}

uint64 TableBuilder::NumEntries() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_11(mht_11_v, 472, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::NumEntries");
 return rep_->num_entries; }

uint64 TableBuilder::FileSize() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPStable_builderDTcc mht_12(mht_12_v, 477, "", "./tensorflow/core/lib/io/table_builder.cc", "TableBuilder::FileSize");
 return rep_->offset; }

}  // namespace table
}  // namespace tensorflow
