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
class MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc() {
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

#include "tensorflow/core/lib/io/cache.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace table {

Cache::~Cache() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/lib/io/cache.cc", "Cache::~Cache");
}

namespace {

// LRU cache implementation
//
// Cache entries have an "in_cache" boolean indicating whether the cache has a
// reference on the entry.  The only ways that this can become false without the
// entry being passed to its "deleter" are via Erase(), via Insert() when
// an element with a duplicate key is inserted, or on destruction of the cache.
//
// The cache keeps two linked lists of items in the cache.  All items in the
// cache are in one list or the other, and never both.  Items still referenced
// by clients but erased from the cache are in neither list.  The lists are:
// - in-use:  contains the items currently referenced by clients, in no
//   particular order.  (This list is used for invariant checking.  If we
//   removed the check, elements that would otherwise be on this list could be
//   left as disconnected singleton lists.)
// - LRU:  contains the items not currently referenced by clients, in LRU order
// Elements are moved between these lists by the Ref() and Unref() methods,
// when they detect an element in the cache acquiring or losing its only
// external reference.

// An entry is a variable length heap-allocated structure.  Entries
// are kept in a circular doubly linked list ordered by access time.
struct LRUHandle {
  void* value;
  void (*deleter)(const Slice&, void* value);
  LRUHandle* next_hash;
  LRUHandle* next;
  LRUHandle* prev;
  size_t charge;  // TODO(opt): Only allow uint32_t?
  size_t key_length;
  bool in_cache;     // Whether entry is in the cache.
  uint32_t refs;     // References, including cache reference, if present.
  uint32_t hash;     // Hash of key(); used for fast sharding and comparisons
  char key_data[1];  // Beginning of key

  Slice key() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_1(mht_1_v, 240, "", "./tensorflow/core/lib/io/cache.cc", "key");

    // next_ is only equal to this if the LRU handle is the list head of an
    // empty list. List heads never have meaningful keys.
    assert(next != this);

    return Slice(key_data, key_length);
  }
};

// We provide our own simple hash table since it removes a whole bunch
// of porting hacks and is also faster than some of the built-in hash
// table implementations in some of the compiler/runtime combinations
// we have tested.  E.g., readrandom speeds up by ~5% over the g++
// 4.4.3's builtin hashtable.
class HandleTable {
 public:
  HandleTable() : length_(0), elems_(0), list_(nullptr) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/lib/io/cache.cc", "HandleTable");
 Resize(); }
  ~HandleTable() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_3(mht_3_v, 263, "", "./tensorflow/core/lib/io/cache.cc", "~HandleTable");
 delete[] list_; }

  LRUHandle* Lookup(const Slice& key, uint32_t hash) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_4(mht_4_v, 268, "", "./tensorflow/core/lib/io/cache.cc", "Lookup");

    return *FindPointer(key, hash);
  }

  LRUHandle* Insert(LRUHandle* h) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_5(mht_5_v, 275, "", "./tensorflow/core/lib/io/cache.cc", "Insert");

    LRUHandle** ptr = FindPointer(h->key(), h->hash);
    LRUHandle* old = *ptr;
    h->next_hash = (old == nullptr ? nullptr : old->next_hash);
    *ptr = h;
    if (old == nullptr) {
      ++elems_;
      if (elems_ > length_) {
        // Since each cache entry is fairly large, we aim for a small
        // average linked list length (<= 1).
        Resize();
      }
    }
    return old;
  }

  LRUHandle* Remove(const Slice& key, uint32_t hash) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_6(mht_6_v, 294, "", "./tensorflow/core/lib/io/cache.cc", "Remove");

    LRUHandle** ptr = FindPointer(key, hash);
    LRUHandle* result = *ptr;
    if (result != nullptr) {
      *ptr = result->next_hash;
      --elems_;
    }
    return result;
  }

 private:
  // The table consists of an array of buckets where each bucket is
  // a linked list of cache entries that hash into the bucket.
  uint32_t length_;
  uint32_t elems_;
  LRUHandle** list_;

  // Return a pointer to slot that points to a cache entry that
  // matches key/hash.  If there is no such cache entry, return a
  // pointer to the trailing slot in the corresponding linked list.
  LRUHandle** FindPointer(const Slice& key, uint32_t hash) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_7(mht_7_v, 317, "", "./tensorflow/core/lib/io/cache.cc", "FindPointer");

    LRUHandle** ptr = &list_[hash & (length_ - 1)];
    while (*ptr != nullptr && ((*ptr)->hash != hash || key != (*ptr)->key())) {
      ptr = &(*ptr)->next_hash;
    }
    return ptr;
  }

  void Resize() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_8(mht_8_v, 328, "", "./tensorflow/core/lib/io/cache.cc", "Resize");

    uint32_t new_length = 4;
    while (new_length < elems_) {
      new_length *= 2;
    }
    LRUHandle** new_list = new LRUHandle*[new_length];
    memset(new_list, 0, sizeof(new_list[0]) * new_length);
    uint32_t count = 0;
    for (uint32_t i = 0; i < length_; i++) {
      LRUHandle* h = list_[i];
      while (h != nullptr) {
        LRUHandle* next = h->next_hash;
        uint32_t hash = h->hash;
        LRUHandle** ptr = &new_list[hash & (new_length - 1)];
        h->next_hash = *ptr;
        *ptr = h;
        h = next;
        count++;
      }
    }
    assert(elems_ == count);
    delete[] list_;
    list_ = new_list;
    length_ = new_length;
  }
};

// A single shard of sharded cache.
class LRUCache {
 public:
  LRUCache();
  ~LRUCache();

  // Separate from constructor so caller can easily make an array of LRUCache
  void SetCapacity(size_t capacity) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_9(mht_9_v, 365, "", "./tensorflow/core/lib/io/cache.cc", "SetCapacity");
 capacity_ = capacity; }

  // Like Cache methods, but with an extra "hash" parameter.
  Cache::Handle* Insert(const Slice& key, uint32_t hash, void* value,
                        size_t charge,
                        void (*deleter)(const Slice& key, void* value));
  Cache::Handle* Lookup(const Slice& key, uint32_t hash);
  void Release(Cache::Handle* handle);
  void Erase(const Slice& key, uint32_t hash);
  void Prune();
  size_t TotalCharge() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_10(mht_10_v, 378, "", "./tensorflow/core/lib/io/cache.cc", "TotalCharge");

    mutex_lock l(mutex_);
    return usage_;
  }

 private:
  void LRU_Remove(LRUHandle* e);
  void LRU_Append(LRUHandle* list, LRUHandle* e);
  void Ref(LRUHandle* e);
  void Unref(LRUHandle* e);
  bool FinishErase(LRUHandle* e) TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Initialized before use.
  size_t capacity_;

  // mutex_ protects the following state.
  mutable mutex mutex_;
  size_t usage_ TF_GUARDED_BY(mutex_);

  // Dummy head of LRU list.
  // lru.prev is newest entry, lru.next is oldest entry.
  // Entries have refs==1 and in_cache==true.
  LRUHandle lru_ TF_GUARDED_BY(mutex_);

  // Dummy head of in-use list.
  // Entries are in use by clients, and have refs >= 2 and in_cache==true.
  LRUHandle in_use_ TF_GUARDED_BY(mutex_);

  HandleTable table_ TF_GUARDED_BY(mutex_);
};

LRUCache::LRUCache() : capacity_(0), usage_(0) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_11(mht_11_v, 412, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::LRUCache");

  // Make empty circular linked lists.
  lru_.next = &lru_;
  lru_.prev = &lru_;
  in_use_.next = &in_use_;
  in_use_.prev = &in_use_;
}

LRUCache::~LRUCache() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_12(mht_12_v, 423, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::~LRUCache");

  assert(in_use_.next == &in_use_);  // Error if caller has an unreleased handle
  for (LRUHandle* e = lru_.next; e != &lru_;) {
    LRUHandle* next = e->next;
    assert(e->in_cache);
    e->in_cache = false;
    assert(e->refs == 1);  // Invariant of lru_ list.
    Unref(e);
    e = next;
  }
}

void LRUCache::Ref(LRUHandle* e) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_13(mht_13_v, 438, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Ref");

  if (e->refs == 1 && e->in_cache) {  // If on lru_ list, move to in_use_ list.
    LRU_Remove(e);
    LRU_Append(&in_use_, e);
  }
  e->refs++;
}

void LRUCache::Unref(LRUHandle* e) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_14(mht_14_v, 449, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Unref");

  assert(e->refs > 0);
  e->refs--;
  if (e->refs == 0) {  // Deallocate.
    assert(!e->in_cache);
    (*e->deleter)(e->key(), e->value);
    free(e);
  } else if (e->in_cache && e->refs == 1) {
    // No longer in use; move to lru_ list.
    LRU_Remove(e);
    LRU_Append(&lru_, e);
  }
}

void LRUCache::LRU_Remove(LRUHandle* e) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_15(mht_15_v, 466, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::LRU_Remove");

  e->next->prev = e->prev;
  e->prev->next = e->next;
}

void LRUCache::LRU_Append(LRUHandle* list, LRUHandle* e) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_16(mht_16_v, 474, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::LRU_Append");

  // Make "e" newest entry by inserting just before *list
  e->next = list;
  e->prev = list->prev;
  e->prev->next = e;
  e->next->prev = e;
}

Cache::Handle* LRUCache::Lookup(const Slice& key, uint32_t hash) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_17(mht_17_v, 485, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Lookup");

  mutex_lock l(mutex_);
  LRUHandle* e = table_.Lookup(key, hash);
  if (e != nullptr) {
    Ref(e);
  }
  return reinterpret_cast<Cache::Handle*>(e);
}

void LRUCache::Release(Cache::Handle* handle) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_18(mht_18_v, 497, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Release");

  mutex_lock l(mutex_);
  Unref(reinterpret_cast<LRUHandle*>(handle));
}

Cache::Handle* LRUCache::Insert(const Slice& key, uint32_t hash, void* value,
                                size_t charge,
                                void (*deleter)(const Slice& key,
                                                void* value)) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_19(mht_19_v, 508, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Insert");

  mutex_lock l(mutex_);

  LRUHandle* e =
      reinterpret_cast<LRUHandle*>(malloc(sizeof(LRUHandle) - 1 + key.size()));
  e->value = value;
  e->deleter = deleter;
  e->charge = charge;
  e->key_length = key.size();
  e->hash = hash;
  e->in_cache = false;
  e->refs = 1;  // for the returned handle.
  memcpy(e->key_data, key.data(), key.size());

  if (capacity_ > 0) {
    e->refs++;  // for the cache's reference.
    e->in_cache = true;
    LRU_Append(&in_use_, e);
    usage_ += charge;
    FinishErase(table_.Insert(e));
  } else {  // don't cache. (capacity_==0 is supported and turns off caching.)
    // next is read by key() in an assert, so it must be initialized
    e->next = nullptr;
  }
  while (usage_ > capacity_ && lru_.next != &lru_) {
    LRUHandle* old = lru_.next;
    assert(old->refs == 1);
    bool erased = FinishErase(table_.Remove(old->key(), old->hash));
    if (!erased) {  // to avoid unused variable when compiled NDEBUG
      assert(erased);
    }
  }

  return reinterpret_cast<Cache::Handle*>(e);
}

// If e != nullptr, finish removing *e from the cache; it has already been
// removed from the hash table.  Return whether e != nullptr.
bool LRUCache::FinishErase(LRUHandle* e) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_20(mht_20_v, 549, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::FinishErase");

  if (e != nullptr) {
    assert(e->in_cache);
    LRU_Remove(e);
    e->in_cache = false;
    usage_ -= e->charge;
    Unref(e);
  }
  return e != nullptr;
}

void LRUCache::Erase(const Slice& key, uint32_t hash) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_21(mht_21_v, 563, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Erase");

  mutex_lock l(mutex_);
  FinishErase(table_.Remove(key, hash));
}

void LRUCache::Prune() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_22(mht_22_v, 571, "", "./tensorflow/core/lib/io/cache.cc", "LRUCache::Prune");

  mutex_lock l(mutex_);
  while (lru_.next != &lru_) {
    LRUHandle* e = lru_.next;
    assert(e->refs == 1);
    bool erased = FinishErase(table_.Remove(e->key(), e->hash));
    if (!erased) {  // to avoid unused variable when compiled NDEBUG
      assert(erased);
    }
  }
}

static const int kNumShardBits = 4;
static const int kNumShards = 1 << kNumShardBits;

class ShardedLRUCache : public Cache {
 private:
  LRUCache shard_[kNumShards];
  mutex id_mutex_;
  uint64_t last_id_;

  static inline uint32_t HashSlice(const Slice& s) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_23(mht_23_v, 595, "", "./tensorflow/core/lib/io/cache.cc", "HashSlice");

    return Hash(s.data(), s.size(), 0);
  }

  static uint32_t Shard(uint32_t hash) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_24(mht_24_v, 602, "", "./tensorflow/core/lib/io/cache.cc", "Shard");
 return hash >> (32 - kNumShardBits); }

 public:
  explicit ShardedLRUCache(size_t capacity) : last_id_(0) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_25(mht_25_v, 608, "", "./tensorflow/core/lib/io/cache.cc", "ShardedLRUCache");

    const size_t per_shard = (capacity + (kNumShards - 1)) / kNumShards;
    for (int s = 0; s < kNumShards; s++) {
      shard_[s].SetCapacity(per_shard);
    }
  }
  ~ShardedLRUCache() override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_26(mht_26_v, 617, "", "./tensorflow/core/lib/io/cache.cc", "~ShardedLRUCache");
}
  Handle* Insert(const Slice& key, void* value, size_t charge,
                 void (*deleter)(const Slice& key, void* value)) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_27(mht_27_v, 622, "", "./tensorflow/core/lib/io/cache.cc", "Insert");

    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Insert(key, hash, value, charge, deleter);
  }
  Handle* Lookup(const Slice& key) override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_28(mht_28_v, 629, "", "./tensorflow/core/lib/io/cache.cc", "Lookup");

    const uint32_t hash = HashSlice(key);
    return shard_[Shard(hash)].Lookup(key, hash);
  }
  void Release(Handle* handle) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_29(mht_29_v, 636, "", "./tensorflow/core/lib/io/cache.cc", "Release");

    LRUHandle* h = reinterpret_cast<LRUHandle*>(handle);
    shard_[Shard(h->hash)].Release(handle);
  }
  void Erase(const Slice& key) override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_30(mht_30_v, 643, "", "./tensorflow/core/lib/io/cache.cc", "Erase");

    const uint32_t hash = HashSlice(key);
    shard_[Shard(hash)].Erase(key, hash);
  }
  void* Value(Handle* handle) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_31(mht_31_v, 650, "", "./tensorflow/core/lib/io/cache.cc", "Value");

    return reinterpret_cast<LRUHandle*>(handle)->value;
  }
  uint64_t NewId() override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_32(mht_32_v, 656, "", "./tensorflow/core/lib/io/cache.cc", "NewId");

    mutex_lock l(id_mutex_);
    return ++(last_id_);
  }
  void Prune() override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_33(mht_33_v, 663, "", "./tensorflow/core/lib/io/cache.cc", "Prune");

    for (int s = 0; s < kNumShards; s++) {
      shard_[s].Prune();
    }
  }
  size_t TotalCharge() const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_34(mht_34_v, 671, "", "./tensorflow/core/lib/io/cache.cc", "TotalCharge");

    size_t total = 0;
    for (int s = 0; s < kNumShards; s++) {
      total += shard_[s].TotalCharge();
    }
    return total;
  }

 private:
  // TODO(byronyi): Figure out why Hash32 fails EvictionPolicy test.
  static uint32_t Hash(const char* data, size_t n, uint32_t seed) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_35(mht_35_v, 685, "", "./tensorflow/core/lib/io/cache.cc", "Hash");

    // Similar to murmur hash
    const uint32_t m = 0xc6a4a793;
    const uint32_t r = 24;
    const char* limit = data + n;
    uint32_t h = seed ^ (n * m);

    // Pick up four bytes at a time
    while (data + 4 <= limit) {
      uint32_t w = core::DecodeFixed32(data);
      data += 4;
      h += w;
      h *= m;
      h ^= (h >> 16);
    }

    // Pick up remaining bytes
    switch (limit - data) {
      case 3:
        h += static_cast<uint8_t>(data[2]) << 16;
        ABSL_FALLTHROUGH_INTENDED;
      case 2:
        h += static_cast<uint8_t>(data[1]) << 8;
        ABSL_FALLTHROUGH_INTENDED;
      case 1:
        h += static_cast<uint8_t>(data[0]);
        h *= m;
        h ^= (h >> r);
        break;
    }
    return h;
  }
};

}  // end anonymous namespace

Cache* NewLRUCache(size_t capacity) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSlibPSioPScacheDTcc mht_36(mht_36_v, 724, "", "./tensorflow/core/lib/io/cache.cc", "NewLRUCache");
 return new ShardedLRUCache(capacity); }

}  // namespace table

}  // namespace tensorflow
