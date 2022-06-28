/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
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
class MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh() {
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

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/tf_status.h"

namespace tf_gcs_filesystem {

/// \brief An LRU cache of string keys and arbitrary values, with configurable
/// max item age (in seconds) and max entries.
///
/// This class is thread safe.
template <typename T>
class ExpiringLRUCache {
 public:
  /// A `max_age` of 0 means that nothing is cached. A `max_entries` of 0 means
  /// that there is no limit on the number of entries in the cache (however, if
  /// `max_age` is also 0, the cache will not be populated).
  ExpiringLRUCache(uint64_t max_age, size_t max_entries,
                   std::function<uint64_t()> timer_seconds = TF_NowSeconds)
      : max_age_(max_age),
        max_entries_(max_entries),
        timer_seconds_(timer_seconds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_0(mht_0_v, 215, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "ExpiringLRUCache");
}

  /// Insert `value` with key `key`. This will replace any previous entry with
  /// the same key.
  void Insert(const std::string& key, const T& value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_1(mht_1_v, 223, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "Insert");

    if (max_age_ == 0) {
      return;
    }
    absl::MutexLock lock(&mu_);
    InsertLocked(key, value);
  }

  // Delete the entry with key `key`. Return true if the entry was found for
  // `key`, false if the entry was not found. In both cases, there is no entry
  // with key `key` existed after the call.
  bool Delete(const std::string& key) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_2(mht_2_v, 238, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "Delete");

    absl::MutexLock lock(&mu_);
    return DeleteLocked(key);
  }

  /// Look up the entry with key `key` and copy it to `value` if found. Returns
  /// true if an entry was found for `key`, and its timestamp is not more than
  /// max_age_ seconds in the past.
  bool Lookup(const std::string& key, T* value) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_3(mht_3_v, 250, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "Lookup");

    if (max_age_ == 0) {
      return false;
    }
    absl::MutexLock lock(&mu_);
    return LookupLocked(key, value);
  }

  typedef std::function<void(const std::string&, T*, TF_Status*)> ComputeFunc;

  /// Look up the entry with key `key` and copy it to `value` if found. If not
  /// found, call `compute_func`. If `compute_func` set `status` to `TF_OK`,
  /// store a copy of the output parameter in the cache, and another copy in
  /// `value`.
  void LookupOrCompute(const std::string& key, T* value,
                       const ComputeFunc& compute_func, TF_Status* status) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_4(mht_4_v, 269, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "LookupOrCompute");

    if (max_age_ == 0) {
      return compute_func(key, value, status);
    }

    // Note: we hold onto mu_ for the rest of this function. In practice, this
    // is okay, as stat requests are typically fast, and concurrent requests are
    // often for the same file. Future work can split this up into one lock per
    // key if this proves to be a significant performance bottleneck.
    absl::MutexLock lock(&mu_);
    if (LookupLocked(key, value)) {
      return TF_SetStatus(status, TF_OK, "");
    }
    compute_func(key, value, status);
    if (TF_GetCode(status) == TF_OK) {
      InsertLocked(key, *value);
    }
  }

  /// Clear the cache.
  void Clear() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_5(mht_5_v, 292, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "Clear");

    absl::MutexLock lock(&mu_);
    cache_.clear();
    lru_list_.clear();
  }

  /// Accessors for cache parameters.
  uint64_t max_age() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_6(mht_6_v, 302, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "max_age");
 return max_age_; }
  size_t max_entries() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSfilesystemPSpluginsPSgcsPSexpiring_lru_cacheDTh mht_7(mht_7_v, 306, "", "./tensorflow/c/experimental/filesystem/plugins/gcs/expiring_lru_cache.h", "max_entries");
 return max_entries_; }

 private:
  struct Entry {
    /// The timestamp (seconds) at which the entry was added to the cache.
    uint64_t timestamp;

    /// The entry's value.
    T value;

    /// A list iterator pointing to the entry's position in the LRU list.
    std::list<std::string>::iterator lru_iterator;
  };

  bool LookupLocked(const std::string& key, T* value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    if (timer_seconds_() - it->second.timestamp > max_age_) {
      cache_.erase(it);
      return false;
    }
    *value = it->second.value;
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return true;
  }

  void InsertLocked(const std::string& key, const T& value)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    lru_list_.push_front(key);
    Entry entry{timer_seconds_(), value, lru_list_.begin()};
    auto insert = cache_.insert(std::make_pair(key, entry));
    if (!insert.second) {
      lru_list_.erase(insert.first->second.lru_iterator);
      insert.first->second = entry;
    } else if (max_entries_ > 0 && cache_.size() > max_entries_) {
      cache_.erase(lru_list_.back());
      lru_list_.pop_back();
    }
  }

  bool DeleteLocked(const std::string& key) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return false;
    }
    lru_list_.erase(it->second.lru_iterator);
    cache_.erase(it);
    return true;
  }

  /// The maximum age of entries in the cache, in seconds. A value of 0 means
  /// that no entry is ever placed in the cache.
  const uint64_t max_age_;

  /// The maximum number of entries in the cache. A value of 0 means there is no
  /// limit on entry count.
  const size_t max_entries_;

  /// The callback to read timestamps.
  std::function<uint64_t()> timer_seconds_;

  /// Guards access to the cache and the LRU list.
  absl::Mutex mu_;

  /// The cache (a map from string key to Entry).
  std::map<std::string, Entry> cache_ ABSL_GUARDED_BY(mu_);

  /// The LRU list of entries. The front of the list identifies the most
  /// recently accessed entry.
  std::list<std::string> lru_list_ ABSL_GUARDED_BY(mu_);
};

}  // namespace tf_gcs_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_GCS_EXPIRING_LRU_CACHE_H_
