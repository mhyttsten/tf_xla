/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_LRU_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_LRU_CACHE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh() {
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


#include "absl/container/node_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// A simple LRU cache. Not thread-safe.
// Value must be copyable and moveable. The intent is that Value is typically
// a smart-pointer type.
template <typename Key, typename Value,
          typename Hash = typename absl::node_hash_map<Key, Value>::hasher,
          typename Eq = typename absl::node_hash_map<Key, Value>::key_equal>
class LRUCache {
 private:
  struct LRUListEntry {
    LRUListEntry* next;
    LRUListEntry* prev;
  };

 public:
  // Multiple LRUCaches can share a LRU list, meaning that the capacity and
  // eviction policy is shared. The user provides an LRU list
  // to the cache constructor, and must ensure that it remains alive as long
  // as the cache does.
  class LRUList {
   public:
    explicit LRUList(int capacity) : capacity_(capacity) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_0(mht_0_v, 214, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "LRUList");

      head_.next = &head_;
      head_.prev = &head_;
    }
    ~LRUList() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_1(mht_1_v, 221, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "~LRUList");

      CHECK(head_.next == &head_);
      CHECK(head_.prev == &head_);
    }

    LRUList(const LRUList&) = delete;
    LRUList(LRUList&&) = delete;
    LRUList& operator=(const LRUList&) = delete;
    LRUList& operator=(LRUList&&) = delete;

    int Capacity() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "Capacity");
 return capacity_; }
    int Size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_3(mht_3_v, 238, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "Size");
 return size_; }

    void Clear();

   private:
    friend class LRUCache;
    int capacity_;
    int size_ = 0;

    // Root of a circular doubly-linked list of entries, in order from least
    // recently used to most recently used. An "empty" cache always contains
    // this element in the LRU list.
    LRUListEntry head_;
  };

  explicit LRUCache(LRUList* lru_list) : lru_list_(lru_list) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_4(mht_4_v, 256, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "LRUCache");
}
  ~LRUCache();

  LRUCache(const LRUCache&) = delete;
  LRUCache(LRUCache&&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;
  LRUCache& operator=(LRUCache&&) = delete;

  // Returns the `value` associated with `key`. Creates a value with `factory`
  // and inserts it if absent.
  Value GetOrCreateIfAbsent(const Key& key,
                            const std::function<Value(const Key&)>& factory);

  // Removes all entries from the cache.
  void Clear();

  int Size() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_5(mht_5_v, 275, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "Size");
 return entries_.size(); }
  int Capacity() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlru_cacheDTh mht_6(mht_6_v, 279, "", "./tensorflow/compiler/xla/pjrt/lru_cache.h", "Capacity");
 return lru_list_->Capacity(); }

 private:
  LRUList* lru_list_;

  struct Entry : public LRUListEntry {
    Entry() = default;

    // Pointer to the key in `entries_`. absl::node_hash_map<> promises
    // pointer stability for keys.
    const Key* key;
    LRUCache* container;
    absl::optional<Value> value;
  };

  // We use `node_hash_map` because we want to guarantee pointer stability for
  // keys and values.
  absl::node_hash_map<Key, Entry, Hash, Eq> entries_;
};

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::LRUList::Clear() {
  while (head_.next != &head_) {
    static_cast<Entry*>(head_.next)->container->Clear();
  }
  size_ = 0;
}

template <typename Key, typename Value, typename Hash, typename Eq>
void LRUCache<Key, Value, Hash, Eq>::Clear() {
  for (auto& e : entries_) {
    LRUListEntry* l = &e.second;
    l->next->prev = l->prev;
    l->prev->next = l->next;
    --lru_list_->size_;
  }
  entries_.clear();
}

template <typename Key, typename Value, typename Hash, typename Eq>
LRUCache<Key, Value, Hash, Eq>::~LRUCache() {
  Clear();
}

template <typename Key, typename Value, typename Hash, typename Eq>
Value LRUCache<Key, Value, Hash, Eq>::GetOrCreateIfAbsent(
    const Key& key, const std::function<Value(const Key&)>& factory) {
  typename absl::node_hash_map<Key, Entry, Hash, Eq>::iterator it;
  bool inserted;
  std::tie(it, inserted) = entries_.try_emplace(key);
  Entry& entry = it->second;
  if (inserted) {
    entry.key = &it->first;
    entry.value = factory(*entry.key);
    ++lru_list_->size_;
  } else {
    // Removes the entry from the LRU list, in preparation for adding it
    // to the back of the list.
    entry.prev->next = entry.next;
    entry.next->prev = entry.prev;
  }
  // (Re-)adds entry to the back of the LRU list. Since it is now the
  // most recently used element, it goes at the back.
  LRUListEntry& lru_head = lru_list_->head_;
  entry.container = this;
  entry.prev = lru_head.prev;
  entry.next = &lru_head;
  lru_head.prev->next = &entry;
  lru_head.prev = &entry;

  Value v = *entry.value;

  // Evict an LRU entry if we are over capacity.
  if (lru_list_->size_ > lru_list_->capacity_) {
    Entry* to_remove = static_cast<Entry*>(lru_head.next);
    to_remove->next->prev = &lru_head;
    lru_head.next = to_remove->next;
    to_remove->container->entries_.erase(*to_remove->key);
    --lru_list_->size_;
  }
  return v;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_LRU_CACHE_H_
