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

#ifndef TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
#define TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
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
class MHTracer_DTPStensorflowPScorePSplatformPScloudPSfile_block_cacheDTh {
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
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSfile_block_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPScloudPSfile_block_cacheDTh() {
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

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FileBlockCache;

/// FileBlockCacheStatsInterface allows for instrumentation of the block cache.
///
/// FileBlockCacheStatsInterface and its subclasses must be safe to use from
/// multiple threads concurrently.
///
/// WARNING! This is an experimental interface that may change or go away at any
/// time.
class FileBlockCacheStatsInterface {
 public:
  /// Configure is called to provide instrumentation hooks.
  ///
  /// Note: Configure can be called multiple times (e.g. if the block cache is
  /// re-initialized).
  virtual void Configure(const FileBlockCache* block_cache) = 0;

  /// RecordBlockLoadRequest is called to record the size of a hit block.
  virtual void RecordCacheHitBlockSize(size_t bytes_transferred) = 0;

  /// RecordBlockLoadRequest is called to record the size of a missed block.
  virtual void RecordCacheMissBlockSize(size_t bytes_transferred) = 0;

  virtual ~FileBlockCacheStatsInterface() = default;
};

/// \brief A block cache of file contents, keyed by {filename, offset}.
///
/// This class should be shared by read-only random access files on a remote
/// filesystem (e.g. GCS).
class FileBlockCache {
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

  virtual ~FileBlockCache() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSfile_block_cacheDTh mht_0(mht_0_v, 247, "", "./tensorflow/core/platform/cloud/file_block_cache.h", "~FileBlockCache");
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
  virtual Status Read(const string& filename, size_t offset, size_t n,
                      char* buffer, size_t* bytes_transferred) = 0;

  // Validate the given file signature with the existing file signature in the
  // cache. Returns true if the signature doesn't change or the file did not
  // exist before. If the signature changes, update the existing signature with
  // the new one and remove the file from cache.
  virtual bool ValidateAndUpdateFileSignature(const string& filename,
                                              int64_t file_signature) = 0;

  /// Remove all cached blocks for `filename`.
  virtual void RemoveFile(const string& filename) = 0;

  /// Remove all cached data.
  virtual void Flush() = 0;

  /// Accessors for cache parameters.
  virtual size_t block_size() const = 0;
  virtual size_t max_bytes() const = 0;
  virtual uint64 max_staleness() const = 0;

  /// The current size (in bytes) of the cache.
  virtual size_t CacheSize() const = 0;

  // Returns true if the cache is enabled. If false, the BlockFetcher callback
  // is always executed during Read.
  virtual bool IsCacheEnabled() const = 0;

  void SetStats(FileBlockCacheStatsInterface* stats) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPScloudPSfile_block_cacheDTh mht_1(mht_1_v, 294, "", "./tensorflow/core/platform/cloud/file_block_cache.h", "SetStats");

    if (stats == nullptr) {
      LOG(ERROR)
          << "Attempted to monitor a NULL stats object. This may prevent the "
             "corresponding monitoring data from being exported";
      return;
    }
    cache_stats_ = stats;
    cache_stats_->Configure(this);
  }

 protected:
  FileBlockCacheStatsInterface* cache_stats_ = nullptr;  // Not owned.
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_CLOUD_FILE_BLOCK_CACHE_H_
