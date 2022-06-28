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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xrt/xrt_compilation_cache.h"

#include <stdlib.h>

#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

namespace {

int64_t get_uid() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "get_uid");

  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64_t>(unsigned_rand);
}

int64_t GetCompilationCacheSizeFromEnv() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "GetCompilationCacheSizeFromEnv");

  const char* env = getenv("TF_XRT_COMPILATION_CACHE_SIZE");
  return env == nullptr ? 1024 : std::stol(env);
}

}  // namespace

const char* kXRTCompilationCacheResourceName = "xrt_compilation_cache";

XRTCompilationCache::EntryRefImpl::EntryRefImpl(XRTCompilationCache* parent,
                                                CompiledSubgraph* entry)
    : parent_(parent), entry_(entry) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_2(mht_2_v, 222, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::EntryRefImpl::EntryRefImpl");

  entry_->Ref();
}

XRTCompilationCache::EntryRefImpl::~EntryRefImpl() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_3(mht_3_v, 229, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::EntryRefImpl::~EntryRefImpl");

  parent_->DiscardEntryRef(entry_);
}

XRTCompilationCacheEntry XRTCompilationCache::EntryRefImpl::get() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_4(mht_4_v, 236, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::EntryRefImpl::get");

  return XRTCompilationCacheEntry(entry_->program.get());
}

XRTCompilationCache::XRTCompilationCache(int max_number_of_entries)
    : max_cache_entries_(max_number_of_entries) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_5(mht_5_v, 244, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::XRTCompilationCache");

  CHECK_GE(max_cache_entries_, 0);
  VLOG(1) << "Created compilation cache max " << max_cache_entries_
          << " entries.";
}

XRTCompilationCache::~XRTCompilationCache() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_6(mht_6_v, 253, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::~XRTCompilationCache");

  VLOG(1) << "XRTCompilationCache::~XRTCompilationCache()";
  // A buggy client may be holding onto a reference, or a client might have
  // crashed while holding onto a reference. In either case, discard all
  // outstanding client references to avoid leaking storage.
  for (const auto& entry : entries_by_uid_) {
    while (!entry.second->RefCountIsOne()) {
      entry.second->Unref();
    }
  }
  while (!entries_by_last_use_.empty()) {
    MarkOldestEntryForEviction();
  }
  CHECK_EQ(cache_.size(), 0);
  CHECK_EQ(entries_by_uid_.size(), 0);
  CHECK_EQ(cache_entries_, 0);
  CHECK_EQ(marked_for_eviction_entries_, 0);
}

Status XRTCompilationCache::Release(int64_t uid) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_7(mht_7_v, 275, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::Release");

  absl::MutexLock lock(&mu_);
  auto iter = entries_by_uid_.find(uid);

  if (iter == entries_by_uid_.end()) {
    return errors::NotFound("No cache entry found for uid ", uid);
  }

  DiscardEntryRefLocked(iter->second);

  VLOG(1) << "After releasing entry " << uid << " refs cache is "
          << cache_.size() << " entries ("
          << cache_entries_ + marked_for_eviction_entries_
          << "), marked for eviction "
          << (cache_.size() - entries_by_last_use_.size()) << " entries ("
          << marked_for_eviction_entries_ << ").";

  return Status::OK();
}

void XRTCompilationCache::DiscardEntryRef(CompiledSubgraph* entry) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_8(mht_8_v, 298, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::DiscardEntryRef");

  absl::MutexLock lock(&mu_);
  DiscardEntryRefLocked(entry);
}

void XRTCompilationCache::DiscardEntryRefLocked(CompiledSubgraph* entry) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_9(mht_9_v, 306, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::DiscardEntryRefLocked");

  if (entry->RefCountIsOne()) {
    // The last reference to this entry is going away, so really delete it from
    // the cache in such a way that it can't be restored by being looked up
    // again.

    // Sanity-check that it has been marked for eviction.
    CHECK(entries_by_last_use_.find(entry->last_use) ==
          entries_by_last_use_.end());
    // Update the counter tracking how much space is taken up by entries that
    // are marked for eviction.
    --marked_for_eviction_entries_;

    // Remove the entry from the cache.
    auto erased = cache_.erase(entry->key);
    if (erased == 0) {
      LOG(FATAL) << "Tried to discard nonexistent cache entry";
    }
    erased = entries_by_uid_.erase(entry->uid);
    CHECK_EQ(erased, 1);
  }
  entry->Unref();
}

void XRTCompilationCache::MarkOldestEntryForEviction() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_10(mht_10_v, 333, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::MarkOldestEntryForEviction");

  CompiledSubgraph* entry_to_mark = entries_by_last_use_.begin()->second;
  VLOG(1) << "Marking " << entry_to_mark->key << " for eviction";
  entries_by_last_use_.erase(entry_to_mark->last_use);
  --cache_entries_;
  ++marked_for_eviction_entries_;
  // Discard the cache's reference to entry. If steps are holding onto
  // references to entry it won't be deleted until the last step holding it
  // completes. It stays in the cache in the meantime and can be resurrected
  // by a call to CompileIfKeyAbsent if that occurs before the last reference
  // expires.
  DiscardEntryRefLocked(entry_to_mark);
}

void XRTCompilationCache::LookupEntryMarkedForEviction(
    CompiledSubgraph* entry) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_11(mht_11_v, 351, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::LookupEntryMarkedForEviction");

  // The entry was previously marked for eviction (or is newly created) so
  // unmark it. Add a reference (owned by the cache), update the cache size, and
  // mark something old for eviction if necessary.
  entry->Ref();
  --marked_for_eviction_entries_;
  ++cache_entries_;

  // Mark the least-recently-used non-marked entry for eviction. Never mark the
  // most-recently used entry (i.e., do nothing if entries_by_last_use_ == 1
  // which means there's only one entry not already marked for eviction), so
  // that an entry persists in the cache even if it is larger than the allocated
  // cache size.
  while (entries_by_last_use_.size() > 1 &&
         cache_entries_ > max_cache_entries_) {
    MarkOldestEntryForEviction();
  }
}

XRTCompilationCache::CompiledSubgraph* XRTCompilationCache::InitializeEntry(
    const string& key,
    const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
        initialize_program) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_12(mht_12_v, 377, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::InitializeEntry");

  CompiledSubgraph* entry = new CompiledSubgraph();
  entry->parent = this;
  entry->key = key;
  entry->uid = get_uid();
  // Add the entry to the cache. Once the computation has been compiled,
  // UpdateEntryAfterCompilation will be called to potentially mark old entries
  // that don't fit any more for eviction.
  //
  // At this point there is one reference to entry, which is owned by the caller
  // who created the entry. A second reference, owned by the cache, will be
  // added below since we leave the entry in the 'marked for eviction' state
  // here.
  auto cache_inserted =
      cache_.insert(std::pair<string, CompiledSubgraph*>(key, entry));
  CHECK(cache_inserted.second);

  // Initialize the program outside the lock so that other cache operations
  // can proceed during the (potentially lengthy) initialization.
  Status s;
  std::unique_ptr<xla::LocalExecutable> program;
  {
    mu_.Unlock();
    { s = initialize_program(&program); }
    mu_.Lock();
  }

  // Add the entry to the uid index.
  auto uid_inserted = entries_by_uid_.insert(
      std::pair<int64_t, CompiledSubgraph*>(entry->uid, entry));
  CHECK(uid_inserted.second);

  entry->initialized = true;
  entry->initialization_status = s;
  if (s.ok()) {
    entry->program = std::move(program);
  }
  // Add the entry to marked_for_eviction_entries_ since it will be adjusted
  // down again when the newly-created entry gets unmarked.
  ++marked_for_eviction_entries_;
  return entry;
}

Status XRTCompilationCache::CompileIfKeyAbsent(
    const string& key, int64_t* uid,
    const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
        compile_function) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_13(mht_13_v, 427, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::CompileIfKeyAbsent");

  CompiledSubgraph* entry = nullptr;

  absl::MutexLock lock(&mu_);
  auto iter = cache_.find(key);

  if (iter == cache_.end()) {
    // The single ref on the newly-created entry is owned by the caller.
    VLOG(1) << "Before adding new entry for key " << key << " cache is "
            << cache_.size() << " entries ("
            << cache_entries_ + marked_for_eviction_entries_ << "), "
            << " marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_entries_ << ").";
    entry = InitializeEntry(key, compile_function);
  } else {
    VLOG(1) << "Before refreshing entry for key " << key << " cache is "
            << cache_.size() << " entries ("
            << cache_entries_ + marked_for_eviction_entries_ << "), "
            << " marked for eviction "
            << (cache_.size() - entries_by_last_use_.size()) << " entries ("
            << marked_for_eviction_entries_ << ").";
    entry = iter->second;
    // Make a new reference that is owned by the caller.
    entry->Ref();
    // Block if necessary until the subgraph has been initialized.
    mu_.Await(absl::Condition(
        +[](CompiledSubgraph* e) { return e->initialized; }, entry));
  }

  // Let the caller know the uid of the entry.
  *uid = entry->uid;

  // Remove the old LRU-table entry if it wasn't already marked for eviction.
  auto erased = entries_by_last_use_.erase(entry->last_use);
  // Update the LRU table indicating this entry is the most recently used.
  entry->last_use = use_counter_++;
  entries_by_last_use_[entry->last_use] = entry;
  if (erased == 0) {
    // The entry had been marked for eviction, or is newly created.
    LookupEntryMarkedForEviction(entry);
  }

  VLOG(1) << "After refreshing entry for key " << key << " cache is "
          << cache_.size() << " entries ("
          << cache_entries_ + marked_for_eviction_entries_ << "), "
          << " marked for eviction "
          << (cache_.size() - entries_by_last_use_.size()) << " entries ("
          << marked_for_eviction_entries_ << ").";

  return entry->initialization_status;
}

Status XRTCompilationCache::Lookup(
    int64_t uid, std::unique_ptr<XRTCompilationCacheEntryRef>* entry) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_14(mht_14_v, 484, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::Lookup");

  entry->reset();

  absl::MutexLock lock(&mu_);
  const auto iter = entries_by_uid_.find(uid);
  if (iter == entries_by_uid_.end()) {
    return errors::NotFound("No executable found for uid ", uid);
  }
  CompiledSubgraph* cache_entry = iter->second;
  *entry = std::unique_ptr<XRTCompilationCacheEntryRef>(
      new EntryRefImpl(this, cache_entry));
  return Status::OK();
}

string XRTCompilationCache::DebugString() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTcc mht_15(mht_15_v, 501, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.cc", "XRTCompilationCache::DebugString");

  return "XRTCompilationCache";
}

xla::StatusOr<RefPtr<XRTCompilationCache>> GetOrCreateCompilationCache(
    ResourceMgr* rm, int64_t max_number_of_entries) {
  if (max_number_of_entries == 0) {
    max_number_of_entries = GetCompilationCacheSizeFromEnv();
  }
  XRTCompilationCache* cache;
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<XRTCompilationCache>(
      rm->default_container(), kXRTCompilationCacheResourceName, &cache,
      [&](XRTCompilationCache** new_cache) {
        *new_cache = new XRTCompilationCache(max_number_of_entries);
        return Status::OK();
      }));
  return RefPtr<XRTCompilationCache>(cache);
}

}  // namespace tensorflow
