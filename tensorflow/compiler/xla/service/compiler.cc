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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc() {
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

#include "tensorflow/compiler/xla/service/compiler.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ absl::Mutex Compiler::platform_compiler_mutex_(absl::kConstInit);

std::vector<std::unique_ptr<tensorflow::protobuf::Message>>
Compiler::ComputeBackendConfigs(const HloInstruction& hlo,
                                se::StreamExecutor* executor) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::ComputeBackendConfigs");

  CHECK(executor != nullptr);
  return {};
}

std::unique_ptr<tensorflow::protobuf::Message>
Compiler::ComputeDefaultBackendConfig(const HloInstruction& hlo,
                                      se::StreamExecutor* executor) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::ComputeDefaultBackendConfig");

  CHECK(executor != nullptr);
  return nullptr;
}

// Define a default version where metadata is not used.
StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
Compiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& options,
    std::unique_ptr<AotCompilationMetadata>* metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_2(mht_2_v, 223, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::CompileAheadOfTime");

  if (metadata != nullptr) {
    return Unimplemented(
        "Populating AotCompilationMetadata is not implemented on this "
        "compiler.");
  }
  return CompileAheadOfTime(std::move(module_group), options);
}

/* static */ std::map<se::Platform::Id, Compiler::CompilerFactory>*
Compiler::GetPlatformCompilerFactories() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_3(mht_3_v, 236, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::GetPlatformCompilerFactories");

  static auto* r = new std::map<se::Platform::Id, CompilerFactory>;
  return r;
}

/* static */
std::map<se::Platform::Id, std::unique_ptr<Compiler>>*
Compiler::GetPlatformCompilers() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_4(mht_4_v, 246, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::GetPlatformCompilers");

  static auto* r = new std::map<se::Platform::Id, std::unique_ptr<Compiler>>;
  return r;
}

/* static */ void Compiler::RegisterCompilerFactory(
    se::Platform::Id platform_id,
    std::function<std::unique_ptr<Compiler>()> compiler_factory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_5(mht_5_v, 256, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::RegisterCompilerFactory");

  absl::MutexLock lock(&platform_compiler_mutex_);
  auto* factories = GetPlatformCompilerFactories();
  CHECK(factories->find(platform_id) == factories->end())
      << "Compiler factory already registered for platform";
  (*factories)[platform_id] = std::move(compiler_factory);
}

/* static */ StatusOr<Compiler*> Compiler::GetForPlatform(
    const se::Platform* platform) {
//   std::cout << "Compiler::GetForPlatform: Entered, platform name: " << platform->Name() << std::endl;
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_6(mht_6_v, 268, "", "./tensorflow/compiler/xla/service/compiler.cc", "Compiler::GetForPlatform");

  absl::MutexLock lock(&platform_compiler_mutex_);

  auto* compilers = GetPlatformCompilers();
  // See if we already instantiated a compiler for this platform.
  {
    auto it = compilers->find(platform->id());
    if (it != compilers->end()) {
//      std::cout << "...Found through GetPlatformCompilers(), returning it" << std::endl;
      return it->second.get();
    }

    // If not, we just fall through to try to create one with a registered
    // factory.
  }

//  std::cout << "...No match in GetPlatformComilers(), can we find a factory for it?" << std::endl;
  auto* factories = GetPlatformCompilerFactories();
  auto it = factories->find(platform->id());
  if (it == factories->end()) {
//    std::cout << "... no factory found, returning error" << std::endl;
    std::string hint;
    if (platform->Name() == "Host") {
      hint =
          " (hint: try adding tensorflow/compiler/jit:xla_cpu_jit as a "
          "dependency)";
    } else if (platform->Name() == "CUDA") {
      hint =
          " (hint: try adding tensorflow/compiler/jit:xla_gpu_jit as a "
          "dependency)";
    }

    return NotFound(
        "could not find registered compiler for platform %s -- check "
        "target linkage%s",
        platform->Name(), hint);
  }

  // And then we invoke the factory, placing the result into the mapping.
//  std::cout << "... found a factory inserting in GetPlatformCompilers and returning a new Compiler" << std::endl;
  compilers->insert(std::make_pair(platform->id(), it->second()));
  return compilers->at(platform->id()).get();
}

AotCompilationOptions::AotCompilationOptions()
    : debug_options_(GetDebugOptionsFromFlags()) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTcc mht_7(mht_7_v, 312, "", "./tensorflow/compiler/xla/service/compiler.cc", "AotCompilationOptions::AotCompilationOptions");
}

}  // namespace xla
