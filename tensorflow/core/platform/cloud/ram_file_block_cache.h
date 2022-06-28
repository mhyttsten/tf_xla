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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_RAM_FILE_BLOCK_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_RAM_FILE_BLOCK_CACHE_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh() {
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


#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An LRU block cache of file contents, keyed by {filename, offset}.
///
/// This class should be shared by read-only random access files on a remote
/// filesystem (e.g. GCS).
class RamFileBlockCache : public FileBlockCache {
 public:
  /// The callback executed when a block is not found in the cache, and needs to
  /// be fetched from the backing filesystem. This callback is provided when the
  /// cache is constructed. The returned Status should be OK as long as the
  /// read from the remote filesystem succeeded (similar to the semantics of the
  /// read(2) system call).
  typedef std::function<Status(const string& filename, size_t offset,
                               size_t buffer_size, char* buffer,
                               size_t* bytes_transferred)>
      BlockFetcher;

  RamFileBlockCache(size_t block_size, size_t max_bytes, uint64 max_staleness,
                    BlockFetcher block_fetcher, Env* env = Env::Default())
      : block_size_(block_size),
        max_bytes_(max_bytes),
        max_staleness_(max_staleness),
        block_fetcher_(block_fetcher),
        env_(env) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh mht_0(mht_0_v, 228, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.h", "RamFileBlockCache");

    if (max_staleness_ > 0) {
      pruning_thread_.reset(env_->StartThread(ThreadOptions(), "TF_prune_FBC",
                                              [this] { Prune(); }));
    }
    VLOG(1) << "GCS file block cache is "
            << (IsCacheEnabled() ? "enabled" : "disabled");
  }

  ~RamFileBlockCache() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.h", "~RamFileBlockCache");

    if (pruning_thread_) {
      stop_pruning_thread_.Notify();
      // Destroying pruning_thread_ will block until Prune() receives the above
      // notification and returns.
      pruning_thread_.reset();
    }
  }

  /// Read `n` bytes from `filename` starting at `offset` into `out`. This
  /// method will return:
  ///
  /// 1) The error from the remote filesystem, if the read from the remote
  ///    filesystem failed.
  /// 2) PRECONDITION_FAILED if the read from the remote filesystem succeeded,
  ///    but the read returned a partial block, and the LRU cache contained a
  ///    block at a higher offset (indicating that the partial block should have
  ///    been a full block).
  /// 3) OUT_OF_RANGE if the read from the remote filesystem succeeded, but
  ///    the file contents do not extend past `offset` and thus nothing was
  ///    placed in `out`.
  /// 4) OK otherwise (i.e. the read succeeded, and at least one byte was placed
  ///    in `out`).
  Status Read(const string& filename, size_t offset, size_t n, char* buffer,
              size_t* bytes_transferred) override;

  // Validate the given file signature with the existing file signature in the
  // cache. Returns true if the signature doesn't change or the file doesn't
  // exist before. If the signature changes, update the existing signature with
  // the new one and remove the file from cache.
  bool ValidateAndUpdateFileSignature(const string& filename,
                                      int64_t file_signature) override
      TF_LOCKS_EXCLUDED(mu_);

  /// Remove all cached blocks for `filename`.
  void RemoveFile(const string& filename) override TF_LOCKS_EXCLUDED(mu_);

  /// Remove all cached data.
  void Flush() override TF_LOCKS_EXCLUDED(mu_);

  /// Accessors for cache parameters.
  size_t block_size() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh mht_2(mht_2_v, 284, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.h", "block_size");
 return block_size_; }
  size_t max_bytes() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh mht_3(mht_3_v, 288, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.h", "max_bytes");
 return max_bytes_; }
  uint64 max_staleness() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh mht_4(mht_4_v, 292, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.h", "max_staleness");
 return max_staleness_; }

  /// The current size (in bytes) of the cache.
  size_t CacheSize() const override TF_LOCKS_EXCLUDED(mu_);

  // Returns true if the cache is enabled. If false, the BlockFetcher callback
  // is always executed during Read.
  bool IsCacheEnabled() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSram_file_block_cacheDTh mht_5(mht_5_v, 302, "", "./tensorflow/core/platform/cloud/ram_file_block_cache.h", "IsCacheEnabled");

    return block_size_ > 0 && max_bytes_ > 0;
  }

 private:
  /// The size of the blocks stored in the LRU cache, as well as the size of the
  /// reads from the underlying filesystem.
  const size_t block_size_;
  /// The maximum number of bytes (sum of block sizes) allowed in the LRU cache.
  const size_t max_bytes_;
  /// The maximum staleness of any block in the LRU cache, in seconds.
  const uint64 max_staleness_;
  /// The callback to read a block from the underlying filesystem.
  const BlockFetcher block_fetcher_;
  /// The Env from which we read timestamps.
  Env* const env_;  // not owned

  /// \brief The key type for the file block cache.
  ///
  /// The file block cache key is a {filename, offset} pair.
  typedef std::pair<string, size_t> Key;

  /// \brief The state of a block.
  ///
  /// A block begins in the CREATED stage. The first thread will attempt to read
  /// the block from the filesystem, transitioning the state of the block to
  /// FETCHING. After completing, if the read was successful the state should
  /// be FINISHED. Otherwise the state should be ERROR. A subsequent read can
  /// re-fetch the block if the state is ERROR.
  enum class FetchState {
    CREATED,
    FETCHING,
    FINISHED,
    ERROR,
  };

  /// \brief A block of a file.
  ///
  /// A file block consists of the block data, the block's current position in
  /// the LRU cache, the timestamp (seconds since epoch) at which the block
  /// was cached, a coordination lock, and state & condition variables.
  ///
  /// Thread safety:
  /// The iterator and timestamp fields should only be accessed while holding
  /// the block-cache-wide mu_ instance variable. The state variable should only
  /// be accessed while holding the Block's mu lock. The data vector should only
  /// be accessed after state == FINISHED, and it should never be modified.
  ///
  /// In order to prevent deadlocks, never grab the block-cache-wide mu_ lock
  /// AFTER grabbing any block's mu lock. It is safe to grab mu without locking
  /// mu_.
  struct Block {
    /// The block data.
    std::vector<char> data;
    /// A list iterator pointing to the block's position in the LRU list.
    std::list<Key>::iterator lru_iterator;
    /// A list iterator pointing to the block's position in the LRA list.
    std::list<Key>::iterator lra_iterator;
    /// The timestamp (seconds since epoch) at which the block was cached.
    uint64 timestamp;
    /// Mutex to guard state variable
    mutex mu;
    /// The state of the block.
    FetchState state TF_GUARDED_BY(mu) = FetchState::CREATED;
    /// Wait on cond_var if state is FETCHING.
    condition_variable cond_var;
  };

  /// \brief The block map type for the file block cache.
  ///
  /// The block map is an ordered map from Key to Block.
  typedef std::map<Key, std::shared_ptr<Block>> BlockMap;

  /// Prune the cache by removing files with expired blocks.
  void Prune() TF_LOCKS_EXCLUDED(mu_);

  bool BlockNotStale(const std::shared_ptr<Block>& block)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Look up a Key in the block cache.
  std::shared_ptr<Block> Lookup(const Key& key) TF_LOCKS_EXCLUDED(mu_);

  Status MaybeFetch(const Key& key, const std::shared_ptr<Block>& block)
      TF_LOCKS_EXCLUDED(mu_);

  /// Trim the block cache to make room for another entry.
  void Trim() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Update the LRU iterator for the block at `key`.
  Status UpdateLRU(const Key& key, const std::shared_ptr<Block>& block)
      TF_LOCKS_EXCLUDED(mu_);

  /// Remove all blocks of a file, with mu_ already held.
  void RemoveFile_Locked(const string& filename)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// Remove the block `entry` from the block map and LRU list, and update the
  /// cache size accordingly.
  void RemoveBlock(BlockMap::iterator entry) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  /// The cache pruning thread that removes files with expired blocks.
  std::unique_ptr<Thread> pruning_thread_;

  /// Notification for stopping the cache pruning thread.
  Notification stop_pruning_thread_;

  /// Guards access to the block map, LRU list, and cached byte count.
  mutable mutex mu_;

  /// The block map (map from Key to Block).
  BlockMap block_map_ TF_GUARDED_BY(mu_);

  /// The LRU list of block keys. The front of the list identifies the most
  /// recently accessed block.
  std::list<Key> lru_list_ TF_GUARDED_BY(mu_);

  /// The LRA (least recently added) list of block keys. The front of the list
  /// identifies the most recently added block.
  ///
  /// Note: blocks are added to lra_list_ only after they have successfully been
  /// fetched from the underlying block store.
  std::list<Key> lra_list_ TF_GUARDED_BY(mu_);

  /// The combined number of bytes in all of the cached blocks.
  size_t cache_size_ TF_GUARDED_BY(mu_) = 0;

  // A filename->file_signature map.
  std::map<string, int64_t> file_signature_map_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_RAM_FILE_BLOCK_CACHE_H_
