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

// Classes to maintain a static registry of memory allocator factories.
#ifndef TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
#define TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh() {
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


#include <string>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"

namespace tensorflow {

class AllocatorFactory {
 public:
  virtual ~AllocatorFactory() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_0(mht_0_v, 201, "", "./tensorflow/core/framework/allocator_registry.h", "~AllocatorFactory");
}

  // Returns true if the factory will create a functionally different
  // SubAllocator for different (legal) values of numa_node.
  virtual bool NumaEnabled() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_1(mht_1_v, 208, "", "./tensorflow/core/framework/allocator_registry.h", "NumaEnabled");
 return false; }

  // Create an Allocator.
  virtual Allocator* CreateAllocator() = 0;

  // Create a SubAllocator. If NumaEnabled() is true, then returned SubAllocator
  // will allocate memory local to numa_node.  If numa_node == kNUMANoAffinity
  // then allocated memory is not specific to any NUMA node.
  virtual SubAllocator* CreateSubAllocator(int numa_node) = 0;
};

// ProcessState is defined in a package that cannot be a dependency of
// framework.  This definition allows us to access the one method we need.
class ProcessStateInterface {
 public:
  virtual ~ProcessStateInterface() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_2(mht_2_v, 226, "", "./tensorflow/core/framework/allocator_registry.h", "~ProcessStateInterface");
}
  virtual Allocator* GetCPUAllocator(int numa_node) = 0;
};

// A singleton registry of AllocatorFactories.
//
// Allocators should be obtained through ProcessState or cpu_allocator()
// (deprecated), not directly through this interface.  The purpose of this
// registry is to allow link-time discovery of multiple AllocatorFactories among
// which ProcessState will obtain the best fit at startup.
class AllocatorFactoryRegistry {
 public:
  AllocatorFactoryRegistry() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_3(mht_3_v, 241, "", "./tensorflow/core/framework/allocator_registry.h", "AllocatorFactoryRegistry");
}
  ~AllocatorFactoryRegistry() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_4(mht_4_v, 245, "", "./tensorflow/core/framework/allocator_registry.h", "~AllocatorFactoryRegistry");
}

  void Register(const char* source_file, int source_line, const string& name,
                int priority, AllocatorFactory* factory);

  // Returns 'best fit' Allocator.  Find the factory with the highest priority
  // and return an allocator constructed by it.  If multiple factories have
  // been registered with the same priority, picks one by unspecified criteria.
  Allocator* GetAllocator();

  // Returns 'best fit' SubAllocator.  First look for the highest priority
  // factory that is NUMA-enabled.  If none is registered, fall back to the
  // highest priority non-NUMA-enabled factory.  If NUMA-enabled, return a
  // SubAllocator specific to numa_node, otherwise return a NUMA-insensitive
  // SubAllocator.
  SubAllocator* GetSubAllocator(int numa_node);

  // Returns the singleton value.
  static AllocatorFactoryRegistry* singleton();

  ProcessStateInterface* process_state() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_5(mht_5_v, 268, "", "./tensorflow/core/framework/allocator_registry.h", "process_state");
 return process_state_; }

 protected:
  friend class ProcessState;
  ProcessStateInterface* process_state_ = nullptr;

 private:
  mutex mu_;
  bool first_alloc_made_ = false;
  struct FactoryEntry {
    const char* source_file;
    int source_line;
    string name;
    int priority;
    std::unique_ptr<AllocatorFactory> factory;
    std::unique_ptr<Allocator> allocator;
    // Index 0 corresponds to kNUMANoAffinity, other indices are (numa_node +
    // 1).
    std::vector<std::unique_ptr<SubAllocator>> sub_allocators;
  };
  std::vector<FactoryEntry> factories_ TF_GUARDED_BY(mu_);

  // Returns any FactoryEntry registered under 'name' and 'priority',
  // or 'nullptr' if none found.
  const FactoryEntry* FindEntry(const string& name, int priority) const
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(AllocatorFactoryRegistry);
};

class AllocatorFactoryRegistration {
 public:
  AllocatorFactoryRegistration(const char* file, int line, const string& name,
                               int priority, AllocatorFactory* factory) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSallocator_registryDTh mht_6(mht_6_v, 306, "", "./tensorflow/core/framework/allocator_registry.h", "AllocatorFactoryRegistration");

    AllocatorFactoryRegistry::singleton()->Register(file, line, name, priority,
                                                    factory);
  }
};

#define REGISTER_MEM_ALLOCATOR(name, priority, factory)                     \
  REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(__COUNTER__, __FILE__, __LINE__, name, \
                                     priority, factory)

#define REGISTER_MEM_ALLOCATOR_UNIQ_HELPER(ctr, file, line, name, priority, \
                                           factory)                         \
  REGISTER_MEM_ALLOCATOR_UNIQ(ctr, file, line, name, priority, factory)

#define REGISTER_MEM_ALLOCATOR_UNIQ(ctr, file, line, name, priority, factory) \
  static AllocatorFactoryRegistration allocator_factory_reg_##ctr(            \
      file, line, name, priority, new factory)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_ALLOCATOR_REGISTRY_H_
