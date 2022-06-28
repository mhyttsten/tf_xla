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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"

#include "tensorflow/compiler/xla/service/hlo_proto_util.h"

namespace xla {

void XlaDebugInfoManager::RegisterModule(
    const ModuleIdentifier& module_id, std::shared_ptr<HloModule> hlo_module,
    std::shared_ptr<const BufferAssignmentProto> buffer_assignment) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager.cc", "XlaDebugInfoManager::RegisterModule");

  absl::MutexLock lock(&mutex_);
  if (active_modules_.find(module_id) != active_modules_.end()) {
    active_modules_[module_id].instances.emplace_back(hlo_module,
                                                      buffer_assignment);
  } else {
    XlaModuleEntry m;
    m.module_id = module_id;
    m.instances.emplace_back(hlo_module, buffer_assignment);
    active_modules_[module_id] = std::move(m);
  }
}

// Unregister an active module, when the last active module of the same
// module id is out of scope, we remove it from our database.
// However during tracing, we will defer the cleanup after serialization.
void XlaDebugInfoManager::UnregisterModule(
    const ModuleIdentifier& module_id, std::shared_ptr<HloModule> hlo_module,
    std::shared_ptr<const BufferAssignmentProto> buffer_assignment) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager.cc", "XlaDebugInfoManager::UnregisterModule");

  absl::MutexLock lock(&mutex_);
  CHECK(active_modules_.find(module_id) != active_modules_.end());
  XlaModuleEntry& active_module = active_modules_[module_id];
  auto instance_it =
      absl::c_find_if(active_module.instances, [&](XlaModuleInstance& e) {
        return e.hlo_module == hlo_module &&
               e.buffer_assignment == buffer_assignment;
      });

  CHECK(instance_it != active_module.instances.end());

  if (!tracing_active_) {
    active_module.instances.erase(instance_it);
    if (active_module.instances.empty()) {
      active_modules_.erase(module_id);
    }
  } else {
    instance_it->active = false;
  }
}

void XlaDebugInfoManager::StartTracing() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager.cc", "XlaDebugInfoManager::StartTracing");

  absl::MutexLock lock(&mutex_);
  tracing_active_ = true;
}

void XlaDebugInfoManager::StopTracing(
    std::vector<XlaModuleDebugInfo>* module_debug_info) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_managerDTcc mht_3(mht_3_v, 248, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager.cc", "XlaDebugInfoManager::StopTracing");

  std::vector<XlaModuleEntry> modules_to_serialize;
  {
    absl::MutexLock lock(&mutex_);
    if (!tracing_active_) return;
    tracing_active_ = false;
    for (const auto& traced_module_id : active_modules_) {
      const XlaModuleEntry& active_module = traced_module_id.second;

      // Copy the instance so that we can serialize without holding the lock.
      // All instances are equivalent from the perspective of symbolization.
      // We only use the first one.
      if (!active_module.instances.empty()) {
        XlaModuleEntry e;
        e.module_id = active_module.module_id;
        e.instances.push_back(active_module.instances[0]);
        modules_to_serialize.push_back(std::move(e));
      }
    }

    // Remove all active modules which have an instance count equal to zero.
    for (auto it = active_modules_.begin(); it != active_modules_.end();) {
      auto& active_module = it->second;
      for (auto instance = active_module.instances.begin();
           instance != active_module.instances.end();) {
        if (instance->active) {
          ++instance;
        } else {
          instance = active_module.instances.erase(instance);
        }
      }

      if (active_module.instances.empty()) {
        active_modules_.erase(it++);
      } else {
        ++it;
      }
    }
  }

  if (module_debug_info) {
    module_debug_info->clear();
    for (const auto& m : modules_to_serialize) {
      XlaModuleDebugInfo info;
      info.module_id = m.module_id;
      // In real world, hlo_module and buffer_assignment will always be
      // non-nullptr. Due to the inconvenience of creation of buffer_assignment
      // object in test, we set it to nullptr and guard this for it.
      if (m.instances[0].hlo_module && m.instances[0].buffer_assignment) {
        info.hlo_proto = absl::make_unique<HloProto>(
            MakeHloProto(*m.instances[0].hlo_module));
        *info.hlo_proto->mutable_buffer_assignment() =
            *m.instances[0].buffer_assignment;
      }
      module_debug_info->emplace_back(std::move(info));
    }
  }
}

}  // namespace xla
