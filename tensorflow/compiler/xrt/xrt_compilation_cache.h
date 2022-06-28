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

#ifndef TENSORFLOW_COMPILER_XRT_XRT_COMPILATION_CACHE_H_
#define TENSORFLOW_COMPILER_XRT_XRT_COMPILATION_CACHE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTh() {
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


#include <memory>
#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xrt/xrt_refptr.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

extern const char* kXRTCompilationCacheResourceName;

struct XRTCompilationCacheEntry {
  explicit XRTCompilationCacheEntry(xla::LocalExecutable* executable)
      : executable(executable) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTh mht_0(mht_0_v, 204, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.h", "XRTCompilationCacheEntry");
}

  // Returns a non-owned pointer to an immutable executable.
  xla::LocalExecutable* get_executable() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxrtPSxrt_compilation_cacheDTh mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xrt/xrt_compilation_cache.h", "get_executable");
 return executable; }

 private:
  xla::LocalExecutable* executable;
};

// Base class for a reference to a cached executable. A unique_ptr to a
// XRTCompilationCacheEntryRef is returned by the cache Lookup methods below,
// and ensures the underlying executable is not garbage-collected until the
// client discards the ptr.
class XRTCompilationCacheEntryRef {
 public:
  virtual ~XRTCompilationCacheEntryRef() = default;

  // Returns a XRTCompilationCacheEntry that should not be used beyond the
  // lifetime of the XRTCompilationCacheEntryRef.
  virtual XRTCompilationCacheEntry get() = 0;
};

// Cache for compiled XLA executables.
// TODO(b/112646171) rationalize this with the other compilation caches.
//
// Each key identifies a unique XLA computation, and the value is executable
// generated by compiling the computation.
//
// When a computation is considered for compilation, the client calls
//
// auto key = <compute key for computation>;
// auto compile_function = <lambda to compile computation into executable>;
// int64 uid;
// CompileIfKeyAbsent(computation_key, &uid, compile_function);
//
// where computation_key is the key computed for the computation. On success,
// uid contains an identifier that can be used to look up the executable. If the
// compiled executable were not present in the cache, compile_function would be
// called to generate it.
//
// The caller is responsible for calling Release(uid) once for every
// call to CompileIfKeyAbsent(key, ...) to discard the reference to the
// compilation results, after the caller is sure it will not look up the
// compiled executables again.
//
// Subsequently the client can call
//
// std::unique_ptr<XRTCompilationCacheEntryRef> entry;
// Lookup(uid, &entry);
// auto proto = entry->get();
//
// to access a cached executable.
class XRTCompilationCache : public ResourceBase {
 public:
  // There is no way in general to discover the size taken by an XLA executable,
  // so the cache defaults to a specific number of entries to determine when to
  // start evicting programs. TODO(b/112592410) change this if the XLA API gets
  // a mechanism to query size.
  explicit XRTCompilationCache(int max_number_of_entries);
  ~XRTCompilationCache() override;

  // Ensures there is an entry for key present in the cache. By the time
  // CompileIfKeyAbsent returns there is guaranteed to be an entry in the cache
  // for key, and that entry will remain valid at least until Release is called
  // on the returned uid. The first call to CompileIfKeyAbsent with a key that
  // is not in the cache will evaluate compile_function to compute the value to
  // use in the entry. Subsequent calls with the same key will block until
  // compile_function completes. Other cache reads and inserts may proceed on
  // other threads while compile_function is executing. The caller is
  // responsible for calling Release(uid) to manually discard its reference to
  // the compiled program, once the caller will not look up the compiled program
  // again.
  //
  // compile_function should compile the computation represented by key and fill
  // the xla::LocalExecutable into its passed argument. It should return OK
  // if and only if compilation succeeds. The executable will be discarded on
  // non-OK status.
  Status CompileIfKeyAbsent(
      const string& key, int64_t* uid,
      const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
          compile_function);

  Status Release(int64_t uid);

  // Looks up an executable corresponding to uid. On success a pointer to an
  // EntryRef holding the program is returned in entry.
  Status Lookup(int64_t uid,
                std::unique_ptr<XRTCompilationCacheEntryRef>* entry);

  string DebugString() const override;

 private:
  // An entry in the compilation cache. The entry is deleted once it has been
  // marked for eviction from the cache _and_ all looked-up entries have been
  // released. When the entry is first created, it is uninitialized and a
  // client-supplied compilation function is run outside the cache's lock to
  // generate the program to be stored in the entry. Any other client that
  // requests the entry will block until it has been initialized. Each entry has
  // a last_use value that set from a monotonically-increasing counter in the
  // cache whenever the entry is referenced. When the cache becomes full,
  // entries are marked for eviction in LRU order.
  struct CompiledSubgraph : public core::RefCounted {
    ~CompiledSubgraph() override = default;

    XRTCompilationCache* parent = nullptr;  // Not owned.
    bool initialized = false;
    // The Status returned by the compilation function when the entry is
    // initialized. This status will be returned to any client that requests the
    // entry.
    Status initialization_status;
    // Counter to keep track of LRU entries for the eviction policy.
    int64_t last_use = -1;
    // The unique key describing this entry.
    string key;
    // The uid describing this entry.
    int64_t uid;
    // The compiled payload corresponding to the key.
    std::unique_ptr<xla::LocalExecutable> program;
  };

  // Wrapper for a cache entry that holds a reference to the entry until the
  // wrapper is deleted. This wrapper is the concrete type of
  // XRTCompilationCacheEntryRef returned by Lookup.
  class EntryRefImpl : public XRTCompilationCacheEntryRef {
   public:
    EntryRefImpl(XRTCompilationCache* parent, CompiledSubgraph* entry);
    ~EntryRefImpl() override;

    XRTCompilationCacheEntry get() override;

   private:
    XRTCompilationCache* parent_;  // Not owned.
    // A reference to entry_ is acquired in the contructor and released via
    // parent->DiscardEntryRef in the destructor.
    CompiledSubgraph* entry_;
  };

  // Releases one reference to entry. This is called by the cache when entry is
  // marked for eviction; or by an EntryRefImpl when it is destroyed. Before the
  // last reference to entry is released, entry is removed from cache_.
  void DiscardEntryRef(CompiledSubgraph* entry);
  void DiscardEntryRefLocked(CompiledSubgraph* entry)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Marks the oldest unmarked entry for eviction. Requires that there is at
  // least one such entry.
  void MarkOldestEntryForEviction() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Updates datastructures to indicate that entry, which had been marked for
  // eviction, has been looked up. This is called by CompileIfKeyAbsent when an
  // entry is newly created, or an entry that has been marked for eviction but
  // not yet evicted is looked up.
  //
  // First the entry is unmarked for eviction, i.e. the cache gains a reference
  // to entry, entry's last_use field is set to be the most recent value of
  // use_counter_ and entries_by_last_use_ is updated accordingly.
  //
  // Next, the size of the cache is examined to see if any other entries need to
  // be marked for eviction now that entry has been unmarked. While the total
  // number of unmarked cached entries is greater than max_cache_entries_,
  // entries are marked for eviction in LRU order. The most recently used entry
  // is never marked for eviction, so an entry larger than the max cache entries
  // will remain in the cache until it is replaced by something else.
  void LookupEntryMarkedForEviction(CompiledSubgraph* entry)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Creates a new entry by running initialize_program and places it in the
  // cache to be looked up by key. The new entry is in the 'marked for eviction'
  // state (not present in entries_by_last_use_) and the caller is expected to
  // call LookupEntryMarkedForEviction after InitializeEntry.
  //
  // **InitializeEntry releases mu_ during the call to initialize_program.**
  CompiledSubgraph* InitializeEntry(
      const string& key,
      const std::function<Status(std::unique_ptr<xla::LocalExecutable>*)>&
          initialize_program) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // The maximum number of entries that are stored in the cache before entries
  // are marked for eviction.
  const int max_cache_entries_;

  mutable absl::Mutex mu_;
  // The total number of entries that are stored and not marked for eviction.
  int cache_entries_ TF_GUARDED_BY(mu_) = 0;
  // The total number of entries that are marked for eviction.
  int marked_for_eviction_entries_ TF_GUARDED_BY(mu_) = 0;
  // The value to assign to the last_use field of the next entry that is looked
  // up.
  int64_t use_counter_ TF_GUARDED_BY(mu_) = 0;
  // All the executables that can be looked up in the cache index by key. An
  // entry is marked for eviction iff it is present in cache_ and not in
  // entries_by_last_use_.
  std::unordered_map<string, CompiledSubgraph*> cache_ TF_GUARDED_BY(mu_);
  // All the executable entries that can be looked up in the cache indexed by
  // uid.
  std::unordered_map<int64_t, CompiledSubgraph*> entries_by_uid_
      TF_GUARDED_BY(mu_);
  // Map from last_use to entry, used to mark entries for eviction in LRU
  // order. If an entry's last_use counter is not present as a key in
  // entries_by_last_use_ then the entry has been marked for eviction.
  std::map<int64_t, CompiledSubgraph*> entries_by_last_use_ TF_GUARDED_BY(mu_);
};

// Looks up or create an XRTCompilationCache object within the given resource
// manager, under the default container. The max_number_of_entries sets the
// maximum number of entries within the cache (which will be LRU-evicted).
// If max_number_of_entries is set to sero, the size of the cache will be
// configured using the TF_XRT_COMPILATION_CACHE_SIZE environment variable.
xla::StatusOr<RefPtr<XRTCompilationCache>> GetOrCreateCompilationCache(
    ResourceMgr* rm, int64_t max_number_of_entries);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_COMPILATION_CACHE_H_
