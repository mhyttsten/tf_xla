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
class MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc() {
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

#include <string>

#include "tensorflow/core/framework/allocator_registry.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// static
AllocatorFactoryRegistry* AllocatorFactoryRegistry::singleton() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/framework/allocator_registry.cc", "AllocatorFactoryRegistry::singleton");

  static AllocatorFactoryRegistry* singleton = new AllocatorFactoryRegistry;
  return singleton;
}

const AllocatorFactoryRegistry::FactoryEntry*
AllocatorFactoryRegistry::FindEntry(const string& name, int priority) const {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/framework/allocator_registry.cc", "AllocatorFactoryRegistry::FindEntry");

  for (auto& entry : factories_) {
    if (!name.compare(entry.name) && priority == entry.priority) {
      return &entry;
    }
  }
  return nullptr;
}

void AllocatorFactoryRegistry::Register(const char* source_file,
                                        int source_line, const string& name,
                                        int priority,
                                        AllocatorFactory* factory) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("source_file: \"" + (source_file == nullptr ? std::string("nullptr") : std::string((char*)source_file)) + "\"");
   mht_2_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/framework/allocator_registry.cc", "AllocatorFactoryRegistry::Register");

  mutex_lock l(mu_);
  CHECK(!first_alloc_made_) << "Attempt to register an AllocatorFactory "
                            << "after call to GetAllocator()";
  CHECK(!name.empty()) << "Need a valid name for Allocator";
  CHECK_GE(priority, 0) << "Priority needs to be non-negative";

  const FactoryEntry* existing = FindEntry(name, priority);
  if (existing != nullptr) {
    // Duplicate registration is a hard failure.
    LOG(FATAL) << "New registration for AllocatorFactory with name=" << name
               << " priority=" << priority << " at location " << source_file
               << ":" << source_line
               << " conflicts with previous registration at location "
               << existing->source_file << ":" << existing->source_line;
  }

  FactoryEntry entry;
  entry.source_file = source_file;
  entry.source_line = source_line;
  entry.name = name;
  entry.priority = priority;
  entry.factory.reset(factory);
  factories_.push_back(std::move(entry));
}

Allocator* AllocatorFactoryRegistry::GetAllocator() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc mht_3(mht_3_v, 249, "", "./tensorflow/core/framework/allocator_registry.cc", "AllocatorFactoryRegistry::GetAllocator");

  mutex_lock l(mu_);
  first_alloc_made_ = true;
  FactoryEntry* best_entry = nullptr;
  for (auto& entry : factories_) {
    if (best_entry == nullptr) {
      best_entry = &entry;
    } else if (entry.priority > best_entry->priority) {
      best_entry = &entry;
    }
  }
  if (best_entry) {
    if (!best_entry->allocator) {
      best_entry->allocator.reset(best_entry->factory->CreateAllocator());
    }
    return best_entry->allocator.get();
  } else {
    LOG(FATAL) << "No registered CPU AllocatorFactory";
    return nullptr;
  }
}

SubAllocator* AllocatorFactoryRegistry::GetSubAllocator(int numa_node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTcc mht_4(mht_4_v, 274, "", "./tensorflow/core/framework/allocator_registry.cc", "AllocatorFactoryRegistry::GetSubAllocator");

  mutex_lock l(mu_);
  first_alloc_made_ = true;
  FactoryEntry* best_entry = nullptr;
  for (auto& entry : factories_) {
    if (best_entry == nullptr) {
      best_entry = &entry;
    } else if (best_entry->factory->NumaEnabled()) {
      if (entry.factory->NumaEnabled() &&
          (entry.priority > best_entry->priority)) {
        best_entry = &entry;
      }
    } else {
      DCHECK(!best_entry->factory->NumaEnabled());
      if (entry.factory->NumaEnabled() ||
          (entry.priority > best_entry->priority)) {
        best_entry = &entry;
      }
    }
  }
  if (best_entry) {
    int index = 0;
    if (numa_node != port::kNUMANoAffinity) {
      CHECK_LE(numa_node, port::NUMANumNodes());
      index = 1 + numa_node;
    }
    if (best_entry->sub_allocators.size() < static_cast<size_t>(index + 1)) {
      best_entry->sub_allocators.resize(index + 1);
    }
    if (!best_entry->sub_allocators[index].get()) {
      best_entry->sub_allocators[index].reset(
          best_entry->factory->CreateSubAllocator(numa_node));
    }
    return best_entry->sub_allocators[index].get();
  } else {
    LOG(FATAL) << "No registered CPU AllocatorFactory";
    return nullptr;
  }
}

}  // namespace tensorflow
