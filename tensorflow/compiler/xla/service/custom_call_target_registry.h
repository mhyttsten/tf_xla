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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScustom_call_target_registryDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScustom_call_target_registryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScustom_call_target_registryDTh() {
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


// This file is depended on by kernels that have to build for mobile devices.
// For this reason, we avoid relying on TensorFlow and instead only use the
// standard C++ library.

#include <map>
#include <mutex>  // NOLINT
#include <string>

namespace xla {

// XLA JIT compilers use this registry to resolve symbolic CustomCall targets;
// so when using XLA as a JIT, CustomCall targets need to be registered here
// with the symbol name used in the CustomCall.
//
// The XLA:CPU ahead-of-time (AOT) compiler links using a standard offline
// linker; so when compiling in CPU AOT mode, you *also* need to make sure the
// name of the callee (presumably implemented in C++) matches up with the
// symbolic name used in the CustomCall.
//
// We maintain the registry in both the JIT and the AOT cases for simplicity,
// but we only use it when running in JIT mode.
class CustomCallTargetRegistry {
 public:
  static CustomCallTargetRegistry* Global();

  void Register(const std::string& symbol, void* address,
                const std::string& platform);
  void* Lookup(const std::string& symbol, const std::string& platform) const;

 private:
  // Maps the pair (symbol, platform) to a C function implementing a custom call
  // named `symbol` for StreamExecutor platform `platform`.
  //
  // Different platforms have different ABIs.  TODO(jlebar): Describe them!
  //
  // (We std::map rather than std::unordered_map because the STL doesn't provide
  // a default hasher for pair<std::string, std::string>, and we want to avoid
  // pulling in dependencies that might define this.)
  std::map<std::pair<std::string, std::string>, void*> registered_symbols_;
  mutable std::mutex mu_;
};

class RegisterCustomCallTarget {
 public:
  explicit RegisterCustomCallTarget(const std::string& name, void* address,
                                    const std::string& platform) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("platform: \"" + platform + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScustom_call_target_registryDTh mht_0(mht_0_v, 234, "", "./tensorflow/compiler/xla/service/custom_call_target_registry.h", "RegisterCustomCallTarget");

    CustomCallTargetRegistry::Global()->Register(name, address, platform);
  }
};

#define XLA_REGISTER_CUSTOM_CALL_CONCAT(a, b) a##b

#define XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address,   \
                                                        platform, counter) \
  static ::xla::RegisterCustomCallTarget XLA_REGISTER_CUSTOM_CALL_CONCAT(  \
      custom_call_target_register, counter)(                               \
      symbol, reinterpret_cast<void*>(address), platform)

#define XLA_REGISTER_CUSTOM_CALL_TARGET(function, platform) \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function, platform)

#define XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, platform)  \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM_HELPER(symbol, address, platform, \
                                                  __COUNTER__)

// Convenience overloads for registering custom-call targets on the CPU.
#define XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(function) \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(#function, function, "Host")

#define XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address) \
  XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, "Host")

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CUSTOM_CALL_TARGET_REGISTRY_H_
