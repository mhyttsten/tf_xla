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
class MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc() {
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

#include "tensorflow/stream_executor/plugin_registry.h"

#include "absl/base/const_init.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace stream_executor {

const PluginId kNullPlugin = nullptr;

// Returns the string representation of the specified PluginKind.
std::string PluginKindString(PluginKind plugin_kind) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_0(mht_0_v, 199, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginKindString");

  switch (plugin_kind) {
    case PluginKind::kBlas:
      return "BLAS";
    case PluginKind::kDnn:
      return "DNN";
    case PluginKind::kFft:
      return "FFT";
    case PluginKind::kRng:
      return "RNG";
    case PluginKind::kInvalid:
    default:
      return "kInvalid";
  }
}

PluginRegistry::DefaultFactories::DefaultFactories() :
    blas(kNullPlugin), dnn(kNullPlugin), fft(kNullPlugin), rng(kNullPlugin) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_1(mht_1_v, 219, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::DefaultFactories::DefaultFactories");
 }

static absl::Mutex& GetPluginRegistryMutex() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_2(mht_2_v, 224, "", "./tensorflow/stream_executor/plugin_registry.cc", "GetPluginRegistryMutex");

  static absl::Mutex mu(absl::kConstInit);
  return mu;
}

/* static */ PluginRegistry* PluginRegistry::instance_ = nullptr;

PluginRegistry::PluginRegistry() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_3(mht_3_v, 234, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::PluginRegistry");
}

/* static */ PluginRegistry* PluginRegistry::Instance() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_4(mht_4_v, 239, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::Instance");

  absl::MutexLock lock{&GetPluginRegistryMutex()};
  if (instance_ == nullptr) {
    instance_ = new PluginRegistry();
  }
  return instance_;
}

void PluginRegistry::MapPlatformKindToId(PlatformKind platform_kind,
                                         Platform::Id platform_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_5(mht_5_v, 251, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::MapPlatformKindToId");

  platform_id_by_kind_[platform_kind] = platform_id;
}

template <typename FACTORY_TYPE>
port::Status PluginRegistry::RegisterFactoryInternal(
    PluginId plugin_id, const std::string& plugin_name, FACTORY_TYPE factory,
    std::map<PluginId, FACTORY_TYPE>* factories) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("plugin_name: \"" + plugin_name + "\"");
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_6(mht_6_v, 262, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::RegisterFactoryInternal");

  absl::MutexLock lock{&GetPluginRegistryMutex()};

  if (factories->find(plugin_id) != factories->end()) {
    return port::Status(
        port::error::ALREADY_EXISTS,
        absl::StrFormat("Attempting to register factory for plugin %s when "
                        "one has already been registered",
                        plugin_name));
  }

  (*factories)[plugin_id] = factory;
  plugin_names_[plugin_id] = plugin_name;
  return port::Status::OK();
}

template <typename FACTORY_TYPE>
port::StatusOr<FACTORY_TYPE> PluginRegistry::GetFactoryInternal(
    PluginId plugin_id, const std::map<PluginId, FACTORY_TYPE>& factories,
    const std::map<PluginId, FACTORY_TYPE>& generic_factories) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_7(mht_7_v, 284, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::GetFactoryInternal");

  auto iter = factories.find(plugin_id);
  if (iter == factories.end()) {
    iter = generic_factories.find(plugin_id);
    if (iter == generic_factories.end()) {
      return port::Status(
          port::error::NOT_FOUND,
          absl::StrFormat("Plugin ID %p not registered.", plugin_id));
    }
  }

  return iter->second;
}

bool PluginRegistry::SetDefaultFactory(Platform::Id platform_id,
                                       PluginKind plugin_kind,
                                       PluginId plugin_id) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_8(mht_8_v, 303, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::SetDefaultFactory");

  if (!HasFactory(platform_id, plugin_kind, plugin_id)) {
    port::StatusOr<Platform*> status =
        MultiPlatformManager::PlatformWithId(platform_id);
    std::string platform_name = "<unregistered platform>";
    if (status.ok()) {
      platform_name = status.ValueOrDie()->Name();
    }

    LOG(ERROR) << "A factory must be registered for a platform before being "
               << "set as default! "
               << "Platform name: " << platform_name
               << ", PluginKind: " << PluginKindString(plugin_kind)
               << ", PluginId: " << plugin_id;
    return false;
  }

  switch (plugin_kind) {
    case PluginKind::kBlas:
      default_factories_[platform_id].blas = plugin_id;
      break;
    case PluginKind::kDnn:
      default_factories_[platform_id].dnn = plugin_id;
      break;
    case PluginKind::kFft:
      default_factories_[platform_id].fft = plugin_id;
      break;
    case PluginKind::kRng:
      default_factories_[platform_id].rng = plugin_id;
      break;
    default:
      LOG(ERROR) << "Invalid plugin kind specified: "
                 << static_cast<int>(plugin_kind);
      return false;
  }

  return true;
}

bool PluginRegistry::HasFactory(const PluginFactories& factories,
                                PluginKind plugin_kind,
                                PluginId plugin_id) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_9(mht_9_v, 347, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::HasFactory");

  switch (plugin_kind) {
    case PluginKind::kBlas:
      return factories.blas.find(plugin_id) != factories.blas.end();
    case PluginKind::kDnn:
      return factories.dnn.find(plugin_id) != factories.dnn.end();
    case PluginKind::kFft:
      return factories.fft.find(plugin_id) != factories.fft.end();
    case PluginKind::kRng:
      return factories.rng.find(plugin_id) != factories.rng.end();
    default:
      LOG(ERROR) << "Invalid plugin kind specified: "
                 << PluginKindString(plugin_kind);
      return false;
  }
}

bool PluginRegistry::HasFactory(Platform::Id platform_id,
                                PluginKind plugin_kind,
                                PluginId plugin_id) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSplugin_registryDTcc mht_10(mht_10_v, 369, "", "./tensorflow/stream_executor/plugin_registry.cc", "PluginRegistry::HasFactory");

  auto iter = factories_.find(platform_id);
  if (iter != factories_.end()) {
    if (HasFactory(iter->second, plugin_kind, plugin_id)) {
      return true;
    }
  }

  return HasFactory(generic_factories_, plugin_kind, plugin_id);
}

// Explicit instantiations to support types exposed in user/public API.
#define EMIT_PLUGIN_SPECIALIZATIONS(FACTORY_TYPE, FACTORY_VAR, PLUGIN_STRING) \
  template port::StatusOr<PluginRegistry::FACTORY_TYPE>                       \
  PluginRegistry::GetFactoryInternal<PluginRegistry::FACTORY_TYPE>(           \
      PluginId plugin_id,                                                     \
      const std::map<PluginId, PluginRegistry::FACTORY_TYPE>& factories,      \
      const std::map<PluginId, PluginRegistry::FACTORY_TYPE>&                 \
          generic_factories) const;                                           \
                                                                              \
  template port::Status                                                       \
  PluginRegistry::RegisterFactoryInternal<PluginRegistry::FACTORY_TYPE>(      \
      PluginId plugin_id, const std::string& plugin_name,                     \
      PluginRegistry::FACTORY_TYPE factory,                                   \
      std::map<PluginId, PluginRegistry::FACTORY_TYPE>* factories);           \
                                                                              \
  template <>                                                                 \
  port::Status PluginRegistry::RegisterFactory<PluginRegistry::FACTORY_TYPE>( \
      Platform::Id platform_id, PluginId plugin_id, const std::string& name,  \
      PluginRegistry::FACTORY_TYPE factory) {                                 \
    return RegisterFactoryInternal(plugin_id, name, factory,                  \
                                   &factories_[platform_id].FACTORY_VAR);     \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  port::Status PluginRegistry::RegisterFactoryForAllPlatforms<                \
      PluginRegistry::FACTORY_TYPE>(PluginId plugin_id,                       \
                                    const std::string& name,                  \
                                    PluginRegistry::FACTORY_TYPE factory) {   \
    return RegisterFactoryInternal(plugin_id, name, factory,                  \
                                   &generic_factories_.FACTORY_VAR);          \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  port::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      Platform::Id platform_id, PluginId plugin_id) {                         \
    if (plugin_id == PluginConfig::kDefault) {                                \
      plugin_id = default_factories_[platform_id].FACTORY_VAR;                \
                                                                              \
      if (plugin_id == kNullPlugin) {                                         \
        return port::Status(                                                  \
            port::error::FAILED_PRECONDITION,                                 \
            "No suitable " PLUGIN_STRING                                      \
            " plugin registered. Have you linked in a " PLUGIN_STRING         \
            "-providing plugin?");                                            \
      } else {                                                                \
        VLOG(2) << "Selecting default " PLUGIN_STRING " plugin, "             \
                << plugin_names_[plugin_id];                                  \
      }                                                                       \
    }                                                                         \
    return GetFactoryInternal(plugin_id, factories_[platform_id].FACTORY_VAR, \
                              generic_factories_.FACTORY_VAR);                \
  }                                                                           \
                                                                              \
  /* TODO(b/22689637): Also temporary WRT MultiPlatformManager */             \
  template <>                                                                 \
  port::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      PlatformKind platform_kind, PluginId plugin_id) {                       \
    auto iter = platform_id_by_kind_.find(platform_kind);                     \
    if (iter == platform_id_by_kind_.end()) {                                 \
      return port::Status(port::error::FAILED_PRECONDITION,                   \
                          absl::StrFormat("Platform kind %d not registered.", \
                                          static_cast<int>(platform_kind)));  \
    }                                                                         \
    return GetFactory<PluginRegistry::FACTORY_TYPE>(iter->second, plugin_id); \
  }

EMIT_PLUGIN_SPECIALIZATIONS(BlasFactory, blas, "BLAS");
EMIT_PLUGIN_SPECIALIZATIONS(DnnFactory, dnn, "DNN");
EMIT_PLUGIN_SPECIALIZATIONS(FftFactory, fft, "FFT");
EMIT_PLUGIN_SPECIALIZATIONS(RngFactory, rng, "RNG");

}  // namespace stream_executor
