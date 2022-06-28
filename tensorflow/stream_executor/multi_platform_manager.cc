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
class MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/multi_platform_manager.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace stream_executor {
namespace {

class MultiPlatformManagerImpl {
 public:
  port::Status RegisterPlatform(std::unique_ptr<Platform> platform)
      TF_LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> PlatformWithName(absl::string_view target)
      TF_LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id)
      TF_LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> PlatformWithName(absl::string_view target,
                                             bool initialize_platform)
      TF_LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id,
                                           bool initialize_platform)
      TF_LOCKS_EXCLUDED(mu_);

  port::StatusOr<Platform*> InitializePlatformWithName(
      absl::string_view target,
      const std::map<std::string, std::string>& options) TF_LOCKS_EXCLUDED(mu_);
  port::StatusOr<Platform*> InitializePlatformWithId(
      const Platform::Id& id, const std::map<std::string, std::string>& options)
      TF_LOCKS_EXCLUDED(mu_);

  port::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
      const std::function<bool(const Platform*)>& filter,
      bool initialize_platform) TF_LOCKS_EXCLUDED(mu_);

  using Listener = MultiPlatformManager::Listener;
  port::Status RegisterListener(std::unique_ptr<Listener> listener)
      TF_LOCKS_EXCLUDED(mu_);

 private:
  // Looks up the platform object with the given name.  Assumes the Platforms
  // mutex is held.
  port::StatusOr<Platform*> LookupByNameLocked(absl::string_view target)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Looks up the platform object with the given id.  Assumes the Platforms
  // mutex is held.
  port::StatusOr<Platform*> LookupByIdLocked(const Platform::Id& id)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns the names of the initialied platforms satisfying the given filter.
  // By default, it will return all initialized platform names.
  std::vector<std::string> InitializedPlatformNamesWithFilter(
      const std::function<bool(const Platform*)>& filter = [](const Platform*) {
        return true;
      }) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  absl::Mutex mu_;
  std::vector<std::unique_ptr<Listener>> listeners_ TF_GUARDED_BY(mu_);
  absl::flat_hash_map<Platform::Id, Platform*> id_map_ TF_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, Platform*> name_map_ TF_GUARDED_BY(mu_);
};

port::Status MultiPlatformManagerImpl::RegisterPlatform(
    std::unique_ptr<Platform> platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_0(mht_0_v, 259, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::RegisterPlatform");

  CHECK(platform != nullptr);
  std::string key = absl::AsciiStrToLower(platform->Name());
  absl::MutexLock lock(&mu_);
  if (name_map_.find(key) != name_map_.end()) {
    return port::Status(port::error::INTERNAL,
                        "platform is already registered with name: \"" +
                            platform->Name() + "\"");
  }
  Platform* platform_ptr = platform.get();
  CHECK(id_map_.emplace(platform->id(), platform_ptr).second);
  // Release ownership/uniqueness to prevent destruction on program exit.
  // This avoids Platforms "cleaning up" on program exit, because otherwise,
  // there are _very_ tricky races between StreamExecutor and underlying
  // platforms (CUDA, OpenCL) during exit. Since these are fixed-size and 1x per
  // program, these are deemed acceptable.
  name_map_[key] = platform.release();
  for (const auto& listener : listeners_) {
    listener->PlatformRegistered(platform_ptr);
  }
  return port::Status::OK();
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::PlatformWithName(
    absl::string_view target) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_1(mht_1_v, 287, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::PlatformWithName");

  return PlatformWithName(target, /*initialize_platform=*/true);
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::PlatformWithId(
    const Platform::Id& id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_2(mht_2_v, 295, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::PlatformWithId");

  return PlatformWithId(id, /*initialize_platform=*/true);
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::PlatformWithName(
    absl::string_view target, bool initialize_platform) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_3(mht_3_v, 304, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::PlatformWithName");

  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByNameLocked(target));
  if (initialize_platform && !platform->Initialized()) {
    SE_RETURN_IF_ERROR(platform->Initialize({}));
  }

  return platform;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::PlatformWithId(
    const Platform::Id& id, bool initialize_platform) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_4(mht_4_v, 319, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::PlatformWithId");

  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByIdLocked(id));
  if (initialize_platform && !platform->Initialized()) {
    SE_RETURN_IF_ERROR(platform->Initialize({}));
  }

  return platform;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::InitializePlatformWithName(
    absl::string_view target,
    const std::map<std::string, std::string>& options) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_5(mht_5_v, 336, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::InitializePlatformWithName");

  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByNameLocked(target));
  if (platform->Initialized()) {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrCat("platform \"", target, "\" is already initialized"));
  }

  SE_RETURN_IF_ERROR(platform->Initialize(options));

  return platform;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::InitializePlatformWithId(
    const Platform::Id& id, const std::map<std::string, std::string>& options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_6(mht_6_v, 355, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::InitializePlatformWithId");

  absl::MutexLock lock(&mu_);

  SE_ASSIGN_OR_RETURN(Platform * platform, LookupByIdLocked(id));
  if (platform->Initialized()) {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrFormat("platform with id %p is already initialized", id));
  }

  SE_RETURN_IF_ERROR(platform->Initialize(options));

  return platform;
}

port::Status MultiPlatformManagerImpl::RegisterListener(
    std::unique_ptr<Listener> listener) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_7(mht_7_v, 374, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::RegisterListener");

  absl::MutexLock lock(&mu_);
  CHECK(id_map_.empty());
  CHECK(name_map_.empty());
  listeners_.push_back(std::move(listener));
  return port::Status::OK();
}

port::StatusOr<std::vector<Platform*>>
MultiPlatformManagerImpl::PlatformsWithFilter(
    const std::function<bool(const Platform*)>& filter,
    bool initialize_platform) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_8(mht_8_v, 388, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::PlatformsWithFilter");

  absl::MutexLock lock(&mu_);
  CHECK_EQ(id_map_.size(), name_map_.size());
  std::vector<Platform*> platforms;
  platforms.reserve(id_map_.size());
  for (const auto& entry : id_map_) {
    Platform* platform = entry.second;
    if (filter(platform)) {
      if (initialize_platform && !platform->Initialized()) {
        SE_RETURN_IF_ERROR(platform->Initialize({}));
      }
      platforms.push_back(platform);
    }
  }
  return platforms;
}

std::vector<std::string>
MultiPlatformManagerImpl::InitializedPlatformNamesWithFilter(
    const std::function<bool(const Platform*)>& filter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_9(mht_9_v, 410, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::InitializedPlatformNamesWithFilter");

  CHECK_EQ(id_map_.size(), name_map_.size());
  std::vector<std::string> initialized_platforms_names;
  initialized_platforms_names.reserve(id_map_.size());
  for (const auto& entry : id_map_) {
    Platform* platform = entry.second;
    if (filter(platform)) {
      if (platform->Initialized()) {
        initialized_platforms_names.push_back(platform->Name());
      }
    }
  }
  return initialized_platforms_names;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::LookupByNameLocked(
    absl::string_view target) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_10(mht_10_v, 430, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::LookupByNameLocked");

  auto it = name_map_.find(absl::AsciiStrToLower(target));
  if (it == name_map_.end()) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrCat("Could not find registered platform with name: \"", target,
                     "\". Available platform names are: ",
                     absl::StrJoin(InitializedPlatformNamesWithFilter(), " ")));
  }
  return it->second;
}

port::StatusOr<Platform*> MultiPlatformManagerImpl::LookupByIdLocked(
    const Platform::Id& id) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_11(mht_11_v, 446, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManagerImpl::LookupByIdLocked");

  auto it = id_map_.find(id);
  if (it == id_map_.end()) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrFormat("could not find registered platform with id: %p", id));
  }
  return it->second;
}

MultiPlatformManagerImpl& Impl() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_12(mht_12_v, 459, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "Impl");

  static MultiPlatformManagerImpl* impl = new MultiPlatformManagerImpl;
  return *impl;
}

}  // namespace

/*static*/ port::Status MultiPlatformManager::RegisterPlatform(
    std::unique_ptr<Platform> platform) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_13(mht_13_v, 470, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::RegisterPlatform");

  return Impl().RegisterPlatform(std::move(platform));
}

/*static*/ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithName(
    absl::string_view target) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_14(mht_14_v, 479, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::PlatformWithName");

  return Impl().PlatformWithName(target);
}

/*static*/ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithId(
    const Platform::Id& id) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_15(mht_15_v, 487, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::PlatformWithId");

  return Impl().PlatformWithId(id);
}

/*static*/ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithId(
    const Platform::Id& id, bool initialize_platform) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_16(mht_16_v, 495, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::PlatformWithId");

  return Impl().PlatformWithId(id, initialize_platform);
}

/*static*/ port::StatusOr<Platform*> MultiPlatformManager::PlatformWithName(
    absl::string_view target, bool initialize_platform) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_17(mht_17_v, 504, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::PlatformWithName");

  return Impl().PlatformWithName(target, initialize_platform);
}

/*static*/ port::StatusOr<Platform*>
MultiPlatformManager::InitializePlatformWithName(
    absl::string_view target,
    const std::map<std::string, std::string>& options) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("target: \"" + std::string(target.data(), target.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_18(mht_18_v, 515, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::InitializePlatformWithName");

  return Impl().InitializePlatformWithName(target, options);
}

/*static*/ port::StatusOr<Platform*>
MultiPlatformManager::InitializePlatformWithId(
    const Platform::Id& id, const std::map<std::string, std::string>& options) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_19(mht_19_v, 524, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::InitializePlatformWithId");

  return Impl().InitializePlatformWithId(id, options);
}

/*static*/ port::Status MultiPlatformManager::RegisterListener(
    std::unique_ptr<Listener> listener) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_20(mht_20_v, 532, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::RegisterListener");

  return Impl().RegisterListener(std::move(listener));
}

/*static*/ port::StatusOr<std::vector<Platform*>>
MultiPlatformManager::PlatformsWithFilter(
    const std::function<bool(const Platform*)>& filter) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_21(mht_21_v, 541, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::PlatformsWithFilter");

  return PlatformsWithFilter(filter, /*initialize_platform=*/true);
}

/*static*/ port::StatusOr<std::vector<Platform*>>
MultiPlatformManager::PlatformsWithFilter(
    const std::function<bool(const Platform*)>& filter,
    bool initialize_platform) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSstream_executorPSmulti_platform_managerDTcc mht_22(mht_22_v, 551, "", "./tensorflow/stream_executor/multi_platform_manager.cc", "MultiPlatformManager::PlatformsWithFilter");

  return Impl().PlatformsWithFilter(filter, initialize_platform);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    multi_platform_manager,
    {
        // Nothing -- this is just a module initializer
        // definition to reference for sequencing
        // purposes from Platform subclasses that register
        // themselves with the MultiPlatformManager.
    });

REGISTER_MODULE_INITIALIZER(
    multi_platform_manager_listener,
    {
        // Nothing -- this is just a module initializer definition to reference
        // for sequencing registration of listeners with the
        // MultiPlatformManager.
    });

// Listener registration should happen before platform registration.
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     multi_platform_manager);
