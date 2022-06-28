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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc() {
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

#include "tensorflow/compiler/xla/client/client_library.h"

#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

LocalClientOptions::LocalClientOptions(
    se::Platform* platform, int number_of_replicas,
    int intra_op_parallelism_threads,
    const absl::optional<std::set<int>>& allowed_devices)
    : platform_(platform),
      number_of_replicas_(number_of_replicas),
      intra_op_parallelism_threads_(intra_op_parallelism_threads),
      allowed_devices_(allowed_devices) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::LocalClientOptions");
}

LocalClientOptions& LocalClientOptions::set_platform(se::Platform* platform) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::set_platform");

  platform_ = platform;
  return *this;
}

se::Platform* LocalClientOptions::platform() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_2(mht_2_v, 218, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::platform");
 return platform_; }

LocalClientOptions& LocalClientOptions::set_number_of_replicas(
    int number_of_replicas) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_3(mht_3_v, 224, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::set_number_of_replicas");

  number_of_replicas_ = number_of_replicas;
  return *this;
}

int LocalClientOptions::number_of_replicas() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_4(mht_4_v, 232, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::number_of_replicas");

  return number_of_replicas_;
}

LocalClientOptions& LocalClientOptions::set_intra_op_parallelism_threads(
    int num_threads) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_5(mht_5_v, 240, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::set_intra_op_parallelism_threads");

  intra_op_parallelism_threads_ = num_threads;
  return *this;
}

int LocalClientOptions::intra_op_parallelism_threads() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_6(mht_6_v, 248, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::intra_op_parallelism_threads");

  return intra_op_parallelism_threads_;
}

LocalClientOptions& LocalClientOptions::set_allowed_devices(
    const absl::optional<std::set<int>>& allowed_devices) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_7(mht_7_v, 256, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::set_allowed_devices");

  allowed_devices_ = allowed_devices;
  return *this;
}

const absl::optional<std::set<int>>& LocalClientOptions::allowed_devices()
    const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_8(mht_8_v, 265, "", "./tensorflow/compiler/xla/client/client_library.cc", "LocalClientOptions::allowed_devices");

  return allowed_devices_;
}

/* static */ ClientLibrary& ClientLibrary::Singleton() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_9(mht_9_v, 272, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::Singleton");

  static ClientLibrary* c = new ClientLibrary;
  return *c;
}

ClientLibrary::ClientLibrary() = default;
ClientLibrary::~ClientLibrary() = default;

/* static */ StatusOr<LocalClient*> ClientLibrary::GetOrCreateLocalClient(
    se::Platform* platform, const absl::optional<std::set<int>>& device_set) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_10(mht_10_v, 284, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::GetOrCreateLocalClient");

  LocalClientOptions default_options;
  default_options.set_platform(platform);
  default_options.set_allowed_devices(device_set);
  return GetOrCreateLocalClient(default_options);
}

/* static */ StatusOr<LocalClient*> ClientLibrary::GetOrCreateLocalClient(
    const LocalClientOptions& options) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_11(mht_11_v, 295, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::GetOrCreateLocalClient");

  se::Platform* platform = options.platform();
  int replica_count = options.number_of_replicas();
  ClientLibrary& client_library = Singleton();
  absl::MutexLock lock(&client_library.service_mutex_);

  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  auto it = client_library.local_instances_.find(platform->id());
  if (it != client_library.local_instances_.end()) {
    return it->second->client.get();
  }

  ServiceOptions service_options;
  service_options.set_platform(platform);
  service_options.set_number_of_replicas(replica_count);
  service_options.set_intra_op_parallelism_threads(
      options.intra_op_parallelism_threads());
  service_options.set_allowed_devices(options.allowed_devices());
  auto instance = absl::make_unique<LocalInstance>();
  TF_ASSIGN_OR_RETURN(instance->service,
                      LocalService::NewService(service_options));
  instance->client = absl::make_unique<LocalClient>(instance->service.get());
  LocalClient* cl = instance->client.get();

  client_library.local_instances_.insert(
      std::make_pair(platform->id(), std::move(instance)));
  return cl;
}

/* static */ LocalClient* ClientLibrary::LocalClientOrDie() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_12(mht_12_v, 330, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::LocalClientOrDie");

  auto client_status = GetOrCreateLocalClient();
  TF_CHECK_OK(client_status.status());
  return client_status.ValueOrDie();
}

/* static */ LocalService* ClientLibrary::GetXlaService(
    se::Platform* platform) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_13(mht_13_v, 340, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::GetXlaService");

  ClientLibrary& client_library = Singleton();
  absl::MutexLock lock(&client_library.service_mutex_);
  auto it = client_library.local_instances_.find(platform->id());
  CHECK(it != client_library.local_instances_.end());
  return it->second->service.get();
}

/* static */ StatusOr<CompileOnlyClient*>
ClientLibrary::GetOrCreateCompileOnlyClient(se::Platform* platform) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_14(mht_14_v, 352, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::GetOrCreateCompileOnlyClient");

  ClientLibrary& client_library = Singleton();
  absl::MutexLock lock(&client_library.service_mutex_);

  if (platform == nullptr) {
    TF_ASSIGN_OR_RETURN(platform, PlatformUtil::GetDefaultPlatform());
  }

  auto it = client_library.compile_only_instances_.find(platform->id());
  if (it != client_library.compile_only_instances_.end()) {
    return it->second->client.get();
  }

  auto instance = absl::make_unique<CompileOnlyInstance>();
  TF_ASSIGN_OR_RETURN(instance->service,
                      CompileOnlyService::NewService(platform));
  instance->client =
      absl::make_unique<CompileOnlyClient>(instance->service.get());
  CompileOnlyClient* cl = instance->client.get();

  client_library.compile_only_instances_.insert(
      std::make_pair(platform->id(), std::move(instance)));
  return cl;
}

/* static */ void ClientLibrary::DestroyLocalInstances() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSclient_libraryDTcc mht_15(mht_15_v, 380, "", "./tensorflow/compiler/xla/client/client_library.cc", "ClientLibrary::DestroyLocalInstances");

  ClientLibrary& client_library = Singleton();
  absl::MutexLock lock(&client_library.service_mutex_);

  client_library.local_instances_.clear();
  client_library.compile_only_instances_.clear();
}

}  // namespace xla
