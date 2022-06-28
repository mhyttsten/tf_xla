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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/ram_file_block_cache.h"
#include <cstring>
#include <memory>
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

bool RamFileBlockCache::BlockNotStale(const std::shared_ptr<Block>& block) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::BlockNotStale");

  mutex_lock l(block->mu);
  if (block->state != FetchState::FINISHED) {
    return true;  // No need to check for staleness.
  }
  if (max_staleness_ == 0) return true;  // Not enforcing staleness.
  return env_->NowSeconds() - block->timestamp <= max_staleness_;
}

std::shared_ptr<RamFileBlockCache::Block> RamFileBlockCache::Lookup(
    const Key& key) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_1(mht_1_v, 206, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::Lookup");

  mutex_lock lock(mu_);
  auto entry = block_map_.find(key);
  if (entry != block_map_.end()) {
    if (BlockNotStale(entry->second)) {
      if (cache_stats_ != nullptr) {
        cache_stats_->RecordCacheHitBlockSize(entry->second->data.size());
      }
      return entry->second;
    } else {
      // Remove the stale block and continue.
      RemoveFile_Locked(key.first);
    }
  }

  // Insert a new empty block, setting the bookkeeping to sentinel values
  // in order to update them as appropriate.
  auto new_entry = std::make_shared<Block>();
  lru_list_.push_front(key);
  lra_list_.push_front(key);
  new_entry->lru_iterator = lru_list_.begin();
  new_entry->lra_iterator = lra_list_.begin();
  new_entry->timestamp = env_->NowSeconds();
  block_map_.emplace(std::make_pair(key, new_entry));
  return new_entry;
}

// Remove blocks from the cache until we do not exceed our maximum size.
void RamFileBlockCache::Trim() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::Trim");

  while (!lru_list_.empty() && cache_size_ > max_bytes_) {
    RemoveBlock(block_map_.find(lru_list_.back()));
  }
}

/// Move the block to the front of the LRU list if it isn't already there.
Status RamFileBlockCache::UpdateLRU(const Key& key,
                                    const std::shared_ptr<Block>& block) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_3(mht_3_v, 248, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::UpdateLRU");

  mutex_lock lock(mu_);
  if (block->timestamp == 0) {
    // The block was evicted from another thread. Allow it to remain evicted.
    return Status::OK();
  }
  if (block->lru_iterator != lru_list_.begin()) {
    lru_list_.erase(block->lru_iterator);
    lru_list_.push_front(key);
    block->lru_iterator = lru_list_.begin();
  }

  // Check for inconsistent state. If there is a block later in the same file
  // in the cache, and our current block is not block size, this likely means
  // we have inconsistent state within the cache. Note: it's possible some
  // incomplete reads may still go undetected.
  if (block->data.size() < block_size_) {
    Key fmax = std::make_pair(key.first, std::numeric_limits<size_t>::max());
    auto fcmp = block_map_.upper_bound(fmax);
    if (fcmp != block_map_.begin() && key < (--fcmp)->first) {
      return errors::Internal("Block cache contents are inconsistent.");
    }
  }

  Trim();

  return Status::OK();
}

Status RamFileBlockCache::MaybeFetch(const Key& key,
                                     const std::shared_ptr<Block>& block) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_4(mht_4_v, 281, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::MaybeFetch");

  bool downloaded_block = false;
  auto reconcile_state =
      gtl::MakeCleanup([this, &downloaded_block, &key, &block] {
        // Perform this action in a cleanup callback to avoid locking mu_ after
        // locking block->mu.
        if (downloaded_block) {
          mutex_lock l(mu_);
          // Do not update state if the block is already to be evicted.
          if (block->timestamp != 0) {
            // Use capacity() instead of size() to account for all  memory
            // used by the cache.
            cache_size_ += block->data.capacity();
            // Put to beginning of LRA list.
            lra_list_.erase(block->lra_iterator);
            lra_list_.push_front(key);
            block->lra_iterator = lra_list_.begin();
            block->timestamp = env_->NowSeconds();
          }
        }
      });
  // Loop until either block content is successfully fetched, or our request
  // encounters an error.
  mutex_lock l(block->mu);
  Status status = Status::OK();
  while (true) {
    switch (block->state) {
      case FetchState::ERROR:
        TF_FALLTHROUGH_INTENDED;
      case FetchState::CREATED:
        block->state = FetchState::FETCHING;
        block->mu.unlock();  // Release the lock while making the API call.
        block->data.clear();
        block->data.resize(block_size_, 0);
        size_t bytes_transferred;
        status.Update(block_fetcher_(key.first, key.second, block_size_,
                                     block->data.data(), &bytes_transferred));
        if (cache_stats_ != nullptr) {
          cache_stats_->RecordCacheMissBlockSize(bytes_transferred);
        }
        block->mu.lock();  // Reacquire the lock immediately afterwards
        if (status.ok()) {
          block->data.resize(bytes_transferred, 0);
          // Shrink the data capacity to the actual size used.
          // NOLINTNEXTLINE: shrink_to_fit() may not shrink the capacity.
          std::vector<char>(block->data).swap(block->data);
          downloaded_block = true;
          block->state = FetchState::FINISHED;
        } else {
          block->state = FetchState::ERROR;
        }
        block->cond_var.notify_all();
        return status;
      case FetchState::FETCHING:
        block->cond_var.wait_for(l, std::chrono::seconds(60));
        if (block->state == FetchState::FINISHED) {
          return Status::OK();
        }
        // Re-loop in case of errors.
        break;
      case FetchState::FINISHED:
        return Status::OK();
    }
  }
  return errors::Internal(
      "Control flow should never reach the end of RamFileBlockCache::Fetch.");
}

Status RamFileBlockCache::Read(const string& filename, size_t offset, size_t n,
                               char* buffer, size_t* bytes_transferred) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("filename: \"" + filename + "\"");
   mht_5_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_5(mht_5_v, 355, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::Read");

  *bytes_transferred = 0;
  if (n == 0) {
    return Status::OK();
  }
  if (!IsCacheEnabled() || (n > max_bytes_)) {
    // The cache is effectively disabled, so we pass the read through to the
    // fetcher without breaking it up into blocks.
    return block_fetcher_(filename, offset, n, buffer, bytes_transferred);
  }
  // Calculate the block-aligned start and end of the read.
  size_t start = block_size_ * (offset / block_size_);
  size_t finish = block_size_ * ((offset + n) / block_size_);
  if (finish < offset + n) {
    finish += block_size_;
  }
  size_t total_bytes_transferred = 0;
  // Now iterate through the blocks, reading them one at a time.
  for (size_t pos = start; pos < finish; pos += block_size_) {
    Key key = std::make_pair(filename, pos);
    // Look up the block, fetching and inserting it if necessary, and update the
    // LRU iterator for the key and block.
    std::shared_ptr<Block> block = Lookup(key);
    DCHECK(block) << "No block for key " << key.first << "@" << key.second;
    TF_RETURN_IF_ERROR(MaybeFetch(key, block));
    TF_RETURN_IF_ERROR(UpdateLRU(key, block));
    // Copy the relevant portion of the block into the result buffer.
    const auto& data = block->data;
    if (offset >= pos + data.size()) {
      // The requested offset is at or beyond the end of the file. This can
      // happen if `offset` is not block-aligned, and the read returns the last
      // block in the file, which does not extend all the way out to `offset`.
      *bytes_transferred = total_bytes_transferred;
      return errors::OutOfRange("EOF at offset ", offset, " in file ", filename,
                                " at position ", pos, "with data size ",
                                data.size());
    }
    auto begin = data.begin();
    if (offset > pos) {
      // The block begins before the slice we're reading.
      begin += offset - pos;
    }
    auto end = data.end();
    if (pos + data.size() > offset + n) {
      // The block extends past the end of the slice we're reading.
      end -= (pos + data.size()) - (offset + n);
    }
    if (begin < end) {
      size_t bytes_to_copy = end - begin;
      memcpy(&buffer[total_bytes_transferred], &*begin, bytes_to_copy);
      total_bytes_transferred += bytes_to_copy;
    }
    if (data.size() < block_size_) {
      // The block was a partial block and thus signals EOF at its upper bound.
      break;
    }
  }
  *bytes_transferred = total_bytes_transferred;
  return Status::OK();
}

bool RamFileBlockCache::ValidateAndUpdateFileSignature(const string& filename,
                                                       int64_t file_signature) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_6(mht_6_v, 421, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::ValidateAndUpdateFileSignature");

  mutex_lock lock(mu_);
  auto it = file_signature_map_.find(filename);
  if (it != file_signature_map_.end()) {
    if (it->second == file_signature) {
      return true;
    }
    // Remove the file from cache if the signatures don't match.
    RemoveFile_Locked(filename);
    it->second = file_signature;
    return false;
  }
  file_signature_map_[filename] = file_signature;
  return true;
}

size_t RamFileBlockCache::CacheSize() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_7(mht_7_v, 440, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::CacheSize");

  mutex_lock lock(mu_);
  return cache_size_;
}

void RamFileBlockCache::Prune() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_8(mht_8_v, 448, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::Prune");

  while (!WaitForNotificationWithTimeout(&stop_pruning_thread_, 1000000)) {
    mutex_lock lock(mu_);
    uint64 now = env_->NowSeconds();
    while (!lra_list_.empty()) {
      auto it = block_map_.find(lra_list_.back());
      if (now - it->second->timestamp <= max_staleness_) {
        // The oldest block is not yet expired. Come back later.
        break;
      }
      // We need to make a copy of the filename here, since it could otherwise
      // be used within RemoveFile_Locked after `it` is deleted.
      RemoveFile_Locked(std::string(it->first.first));
    }
  }
}

void RamFileBlockCache::Flush() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_9(mht_9_v, 468, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::Flush");

  mutex_lock lock(mu_);
  block_map_.clear();
  lru_list_.clear();
  lra_list_.clear();
  cache_size_ = 0;
}

void RamFileBlockCache::RemoveFile(const string& filename) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_10(mht_10_v, 480, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::RemoveFile");

  mutex_lock lock(mu_);
  RemoveFile_Locked(filename);
}

void RamFileBlockCache::RemoveFile_Locked(const string& filename) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_11(mht_11_v, 489, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::RemoveFile_Locked");

  Key begin = std::make_pair(filename, 0);
  auto it = block_map_.lower_bound(begin);
  while (it != block_map_.end() && it->first.first == filename) {
    auto next = std::next(it);
    RemoveBlock(it);
    it = next;
  }
}

void RamFileBlockCache::RemoveBlock(BlockMap::iterator entry) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTcc mht_12(mht_12_v, 502, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.cc", "RamFileBlockCache::RemoveBlock");

  // This signals that the block is removed, and should not be inadvertently
  // reinserted into the cache in UpdateLRU.
  entry->second->timestamp = 0;
  lru_list_.erase(entry->second->lru_iterator);
  lra_list_.erase(entry->second->lra_iterator);
  cache_size_ -= entry->second->data.capacity();
  block_map_.erase(entry);
}

}  // namespace tensorflow
