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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.h"

#include "grpcpp/security/credentials.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_support.h"

namespace tensorflow {
namespace tpu {
namespace {

#if defined(LIBTPU_ON_GCE)
using ResponseType = GetTpuProgramResponseExternal;
#else
using ResponseType = GetTpuProgramResponse;
#endif

static constexpr absl::Duration kProtoTimeout = absl::Minutes(15);
static gpr_timespec TimeToGprTimespec(absl::Time time) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TimeToGprTimespec");

  if (time == absl::InfiniteFuture()) {
    return gpr_inf_future(GPR_CLOCK_REALTIME);
  }
  if (time == absl::InfinitePast()) {
    return gpr_inf_past(GPR_CLOCK_REALTIME);
  }

  gpr_timespec spec;
  timespec t = absl::ToTimespec(time);
  spec.tv_sec = t.tv_sec;
  spec.tv_nsec = static_cast<int32_t>(t.tv_nsec);
  spec.clock_type = GPR_CLOCK_REALTIME;
  return spec;
}
}  // namespace
TpuCompilationCacheRpcLookup::TpuCompilationCacheRpcLookup(
    const std::string& server_address, int64_t max_cache_size)
    : max_cache_size_(max_cache_size) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("server_address: \"" + server_address + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_1(mht_1_v, 225, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TpuCompilationCacheRpcLookup::TpuCompilationCacheRpcLookup");

  // Ensure that large TPU program can get sent over the channel.
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int32>::max());
  auto channel =
      ::grpc::CreateCustomChannel(absl::StrCat("dns:///", server_address),
                                  CreateChannelCredentials(), args);
  stub_ = tpu::grpc::TpuCompilationCacheService::NewStub(channel);
  VLOG(1) << "Created RPC lookup cache size " << max_cache_size_ << " bytes.";
}

Status TpuCompilationCacheRpcLookup::Lookup(
    const std::string& proto_key,
    std::unique_ptr<CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("proto_key: \"" + proto_key + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TpuCompilationCacheRpcLookup::Lookup");

  profiler::TraceMe proto_lookup_traceme("Remote TPU proto cache lookup",
                                         /*level=*/2);
  entry->reset();
  std::shared_ptr<CacheEntry> cache_entry;
  // Keep a reference to CacheEntry objects evicted from the cache so that the
  // potential deletion happens outside the lock upon method exit.
  std::vector<std::shared_ptr<CacheEntry>> removed_entries;

  std::string local_proto_key = absl::StrCat(
      proto_key, "_", tpu::CompilationCacheFetchTarget_Name(fetch_target));

  {
    absl::MutexLock lock(&mu_);
    auto iter = cache_.find(local_proto_key);
    if (iter == cache_.end()) {
      tpu::GetTpuProgramRequest request;
      request.set_key(proto_key);
      request.set_fetch_target(fetch_target);
      TF_RETURN_IF_ERROR(
          RemoteLookupLocked(local_proto_key, request, &cache_entry));
    } else {
      VLOG(1) << "Found key " << local_proto_key << " in local proto cache.";
      cache_entry = iter->second;
      auto erased = entries_by_last_use_.erase(cache_entry->last_use);
      CHECK_EQ(erased, 1);
    }
    PostLookupLocked(&cache_entry, entry, &removed_entries);
  }
  return Status::OK();
}

Status TpuCompilationCacheRpcLookup::Lookup(
    int64_t uid, int proto_index,
    std::unique_ptr<CompilationCacheEntryRef>* entry,
    tpu::CompilationCacheFetchTarget fetch_target) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_3(mht_3_v, 281, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TpuCompilationCacheRpcLookup::Lookup");

  profiler::TraceMe proto_lookup_traceme("Remote TPU proto cache lookup by uid",
                                         /*level=*/2);
  entry->reset();
  std::shared_ptr<CacheEntry> cache_entry;
  // Keep a reference to CacheEntry objects evicted from the cache so that the
  // potential deletion happens outside the lock upon method exit.
  std::vector<std::shared_ptr<CacheEntry>> removed_entries;

  // Make a string key so that we can uniformly store cached entries under
  // string keys whether they are looked up by proto_key or uid+index. The
  // expectation is that any given executable will only ever be looked up
  // *either* by proto_key *or* by uid+index, so we are not concerned that the
  // same proto could be placed in the cache twice if it is looked up by both
  // methods.
  std::string local_proto_key =
      absl::StrCat(" _ ", uid, ":", proto_index, "_",
                   tpu::CompilationCacheFetchTarget_Name(fetch_target));
  {
    absl::MutexLock lock(&mu_);
    auto iter = cache_.find(local_proto_key);
    if (iter == cache_.end()) {
      tpu::GetTpuProgramRequest request;
      tpu::TpuCompilationUidAndIndex* uid_and_index =
          request.mutable_uid_and_index();
      uid_and_index->set_uid(uid);
      uid_and_index->set_proto_index(proto_index);
      request.set_fetch_target(fetch_target);
      TF_RETURN_IF_ERROR(
          RemoteLookupLocked(local_proto_key, request, &cache_entry));
    } else {
      VLOG(1) << "Found uid " << uid << " and index " << proto_index
              << " in local proto cache.";
      cache_entry = iter->second;
      auto erased = entries_by_last_use_.erase(cache_entry->last_use);
      CHECK_EQ(erased, 1);
    }
    PostLookupLocked(&cache_entry, entry, &removed_entries);
  }
  return Status::OK();
}

Status TpuCompilationCacheRpcLookup::RemoteLookupLocked(
    const std::string& local_proto_key,
    const tpu::GetTpuProgramRequest& request,
    std::shared_ptr<CacheEntry>* cache_entry) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("local_proto_key: \"" + local_proto_key + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_4(mht_4_v, 330, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TpuCompilationCacheRpcLookup::RemoteLookupLocked");

  profiler::TraceMe proto_lookup_traceme("Remote TPU proto cache fetch",
                                         /*level=*/2);
  // Perform the RPC while holding the lock unless it is demonstrated that
  // this causes a performance problem.
  ::grpc::ClientContext client_context;
  client_context.set_deadline(TimeToGprTimespec(::absl::Now() + kProtoTimeout));
  client_context.set_compression_algorithm(GRPC_COMPRESS_GZIP);

  ResponseType response;
  Status s =
      FromGrpcStatus(stub_->GetTpuProgram(&client_context, request, &response));
  VLOG(1) << "Looked up key " << local_proto_key
          << " in remote subgraph cache status " << s;
  TF_RETURN_IF_ERROR(s);

  TF_RETURN_IF_ERROR(DeserializeRpcResponseToCacheEntry(
      local_proto_key, &response, cache_entry));
  cache_.emplace(local_proto_key, (*cache_entry));
  cache_size_ += (*cache_entry)->size;

  return Status::OK();
}

void TpuCompilationCacheRpcLookup::PostLookupLocked(
    std::shared_ptr<CacheEntry>* cache_entry,
    std::unique_ptr<CompilationCacheEntryRef>* entry,
    std::vector<std::shared_ptr<CacheEntry>>* removed_entries) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_5(mht_5_v, 360, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TpuCompilationCacheRpcLookup::PostLookupLocked");

  (*cache_entry)->last_use = use_counter_++;
  entries_by_last_use_[(*cache_entry)->last_use] = cache_entry->get();
  *entry =
      std::unique_ptr<CompilationCacheEntryRef>(new CacheWrapper(*cache_entry));

  // Evict overflowing entries if necessary, but never evict the most recently
  // used entry.
  while (entries_by_last_use_.size() > 1 && cache_size_ > max_cache_size_) {
    auto entry_to_evict = entries_by_last_use_.begin()->second;
    entries_by_last_use_.erase(entry_to_evict->last_use);
    CHECK_GE(cache_size_, entry_to_evict->size);
    cache_size_ -= entry_to_evict->size;
    // Delete the cache's reference to the entry, though clients may still be
    // holding onto references. We use 'removed_entries' to delay the possible
    // CacheEntry destruction until the mu_ lock is released.
    auto entry_to_evict_it = cache_.find(entry_to_evict->key);
    CHECK(entry_to_evict_it != cache_.end())
        << "Missing entry key: " << entry_to_evict->key;
    removed_entries->push_back(entry_to_evict_it->second);
    cache_.erase(entry_to_evict_it);
  }
}

std::string TpuCompilationCacheRpcLookup::DebugString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_rpc_lookupDTcc mht_6(mht_6_v, 387, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_lookup.cc", "TpuCompilationCacheRpcLookup::DebugString");

  return "TpuCompilationCacheRpcLookup";
}
}  // namespace tpu
}  // namespace tensorflow
