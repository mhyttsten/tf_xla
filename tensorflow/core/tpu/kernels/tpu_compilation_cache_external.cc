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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_externalDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_externalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_externalDTcc() {
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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_external.h"

#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/tpu/kernels/compiled_subgraph.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_metrics.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/kernels/trace_util.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

namespace tensorflow {
namespace tpu {

namespace {

int64_t get_uid() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_externalDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_external.cc", "get_uid");

  uint64 unsigned_rand = random::New64() & INT64_MAX;
  return static_cast<int64_t>(unsigned_rand);
}

void PopulateEntry(const std::string& key, CompiledSubgraph* entry,
                   TpuProgramGroup tpu_program_group) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_externalDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_external.cc", "PopulateEntry");

  // Make the unique keys for each cached proto.
  for (int i = 0; i < tpu_program_group.program_count(); ++i) {
    entry->proto_key.push_back(ProtoKeyForComputation(key, i));
  }

  entry->tpu_program_group =
      absl::make_unique<TpuProgramGroup>(std::move(tpu_program_group));
  entry->initialized = true;

  if (entry->initialization_status.ok()) {
    // Compute the entries total size once all members are initialized.
    entry->total_size = entry->ComputeTotalSize();
  }
}

std::unique_ptr<CompiledSubgraph> CreateAndInitializeCompiledSubgraph(
    CompiledSubgraph* main_entry) {
  auto entry = absl::make_unique<CompiledSubgraph>();
  entry->main_entry = main_entry;
  entry->tpu_program_group = absl::make_unique<TpuProgramGroup>();
  return entry;
}
}  // namespace

CompiledSubgraph* TpuCompilationCacheExternal::InitializeEntry(
    const string& key,
    const std::function<Status(TpuProgramGroupInterface*)>& initialize_program,
    const TpuCompilationCacheKey& subgraph_key) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPStpu_compilation_cache_externalDTcc mht_2(mht_2_v, 249, "", "./tensorflow/core/tpu/kernels/tpu_compilation_cache_external.cc", "TpuCompilationCacheExternal::InitializeEntry");

  CompiledSubgraph* main_entry = new CompiledSubgraph();
  main_entry->parent = this;
  main_entry->subgraph_key = key;
  main_entry->uid = get_uid();
  main_entry->cache_entry_debug_string = subgraph_key.prefix;
  VLOG(1) << "Cache Initializing Entry Session Debug "
          << main_entry->cache_entry_debug_string;

  // Add the entry to the cache, with size zero since there are no compiled
  // programs in it. Once the subgraph has been compiled,
  // UpdateEntryAfterCompilation will be called to potentially mark old entries
  // that don't fit any more for eviction.
  //
  // At this point there is one reference to entry, which is owned by the caller
  // who created the entry. A second reference, owned by the cache, will be
  // added below since we leave the entry in the 'marked for eviction' state
  // here.
  InsertEntry(key, main_entry);

  // Initialize the programs outside the lock so that other cache operations
  // can proceed during the (potentially lengthy) initialization.
  Status initialization_status;

  TpuProgramGroup tpu_program_group;
  {
    mu_.Unlock();
    {
      profiler::TraceMe compile_programs_traceme(
          "TPU compilation cache compile",
          /*level=*/2);
      initialization_status = initialize_program(&tpu_program_group);
    }
    mu_.Lock();
  }

  main_entry->initialization_status = initialization_status;

  if (!initialization_status.ok()) {
    // Compilation failure might caused the subsequent tpu_program_group init
    // failed with assert error. Log the error here to make debugging easier.
    LOG(ERROR) << initialization_status.error_message();
  }

  // Add the entry to the uid index.
  auto uid_inserted = entries_by_uid_.insert(
      std::pair<int64_t, CompiledSubgraph*>(main_entry->uid, main_entry));
  CHECK(uid_inserted.second);

  if (tpu_program_group.has_sharding_program()) {
    main_entry->sharding_entry =
        CreateAndInitializeCompiledSubgraph(main_entry);
    TpuProgramGroup sharding_programs;
    sharding_programs.Initialize(
        tpu_program_group.tpu_programs(TpuProgramShardingType::kSharding));

    for (const auto& fingerprint : sharding_programs.fingerprints()) {
      main_entry->sharding_key.emplace_back(fingerprint);
    }

    PopulateEntry(key, main_entry->sharding_entry.get(),
                  std::move(sharding_programs));

    main_entry->unsharding_entry =
        CreateAndInitializeCompiledSubgraph(main_entry);
    TpuProgramGroup unsharding_programs;
    unsharding_programs.Initialize(
        tpu_program_group.tpu_programs(TpuProgramShardingType::kUnsharding));
    PopulateEntry(key, main_entry->unsharding_entry.get(),
                  std::move(unsharding_programs));
  }

  PopulateEntry(key, main_entry, std::move(tpu_program_group));

  for (int64_t i = 0; i < main_entry->proto_key.size(); ++i) {
    auto entry_inserted = entries_by_proto_key_.insert(
        std::pair<std::string, std::pair<CompiledSubgraph*, int>>(
            main_entry->proto_key[i], std::make_pair(main_entry, i)));
    CHECK(entry_inserted.second);
  }

  // Add the size to marked_for_eviction_size_ since it will be adjusted down
  // again when the newly-created entry gets unmarked.
  marked_for_eviction_size_ += main_entry->total_size;
  return main_entry;
}
}  // namespace tpu
}  // namespace tensorflow
