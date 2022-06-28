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

// The compiler API is used by the XLA service to generate executables that
// run on a given platform. This is a registry and abstract interface, for
// pluggability by the various platforms.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPILER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh() {
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


#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/threadpool.h"

namespace xla {

// The following types are used for ahead of time compilation.

// Contains the object file data created as a result of ahead-of-time
// computation.
using ObjectFileData = std::vector<char>;

class Compiler;

// Abstract superclass describing the result of an ahead-of-time compilation.
class AotCompilationResult {
 public:
  AotCompilationResult(const AotCompilationResult&) = delete;
  AotCompilationResult& operator=(AotCompilationResult const&) = delete;

  virtual ~AotCompilationResult() = default;

  virtual StatusOr<std::string> SerializeAsString() const {
    return Unimplemented("SerializeAsString unimplemented.");
  }

  virtual StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      Compiler* compiler, se::StreamExecutor* executor) const {
    return Unimplemented("LoadExecutable unimplemented.");
  }

 protected:
  AotCompilationResult() = default;
};

// Abstract superclass describing options to an ahead-of-time compilation.
class AotCompilationOptions {
 public:
  AotCompilationOptions(const AotCompilationOptions&) = delete;
  AotCompilationOptions& operator=(AotCompilationOptions const&) = delete;

  explicit AotCompilationOptions(se::Platform::Id platform_id)
      : platform_id_(platform_id), debug_options_(GetDebugOptionsFromFlags()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_0(mht_0_v, 253, "", "./tensorflow/compiler/xla/service/compiler.h", "AotCompilationOptions");
}
  virtual ~AotCompilationOptions() = default;

  // Returns the ID of the platform to which these options apply.
  virtual se::Platform::Id PlatformId() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_1(mht_1_v, 260, "", "./tensorflow/compiler/xla/service/compiler.h", "PlatformId");
 return platform_id_; }

  virtual int64_t replica_count() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_2(mht_2_v, 265, "", "./tensorflow/compiler/xla/service/compiler.h", "replica_count");
 return 0; }
  virtual int64_t num_cores() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_3(mht_3_v, 269, "", "./tensorflow/compiler/xla/service/compiler.h", "num_cores");
 return 0; }
  virtual bool use_spmd_partitioning() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_4(mht_4_v, 273, "", "./tensorflow/compiler/xla/service/compiler.h", "use_spmd_partitioning");
 return false; }
  virtual bool use_auto_spmd_partitioning() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_5(mht_5_v, 277, "", "./tensorflow/compiler/xla/service/compiler.h", "use_auto_spmd_partitioning");
 return false; }
  virtual bool deduplicate_hlo() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_6(mht_6_v, 281, "", "./tensorflow/compiler/xla/service/compiler.h", "deduplicate_hlo");
 return false; }

  // Optional allocator that may be used for allocating temp space on the device
  // during compilation.
  se::DeviceMemoryAllocator* device_allocator() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_7(mht_7_v, 288, "", "./tensorflow/compiler/xla/service/compiler.h", "device_allocator");

    return device_allocator_;
  }
  void set_device_allocator(se::DeviceMemoryAllocator* device_allocator) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_8(mht_8_v, 294, "", "./tensorflow/compiler/xla/service/compiler.h", "set_device_allocator");

    device_allocator_ = device_allocator;
  }

  const DebugOptions& debug_options() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_9(mht_9_v, 301, "", "./tensorflow/compiler/xla/service/compiler.h", "debug_options");
 return debug_options_; }
  DebugOptions* mutable_debug_options() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_10(mht_10_v, 305, "", "./tensorflow/compiler/xla/service/compiler.h", "mutable_debug_options");
 return &debug_options_; }

  bool has_static_device_assignment() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_11(mht_11_v, 310, "", "./tensorflow/compiler/xla/service/compiler.h", "has_static_device_assignment");

    return static_device_assignment_.has_value();
  }
  const DeviceAssignment& static_device_assignment() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_12(mht_12_v, 316, "", "./tensorflow/compiler/xla/service/compiler.h", "static_device_assignment");

    CHECK(static_device_assignment_.has_value());
    return *static_device_assignment_;
  }
  void set_static_device_assignment(const DeviceAssignment& device_assignment) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_13(mht_13_v, 323, "", "./tensorflow/compiler/xla/service/compiler.h", "set_static_device_assignment");

    static_device_assignment_ = device_assignment;
  }

  FusionConfigCollection fusion_config_collection() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_14(mht_14_v, 330, "", "./tensorflow/compiler/xla/service/compiler.h", "fusion_config_collection");

    return fusion_config_collection_;
  }
  void set_fusion_config_collection(
      FusionConfigCollection fusion_config_collection) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_15(mht_15_v, 337, "", "./tensorflow/compiler/xla/service/compiler.h", "set_fusion_config_collection");

    fusion_config_collection_ = fusion_config_collection;
  }

  const std::vector<std::vector<bool>>& fusion_config() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_16(mht_16_v, 344, "", "./tensorflow/compiler/xla/service/compiler.h", "fusion_config");

    return fusion_config_;
  }
  void set_fusion_config(const std::vector<std::vector<bool>>& fusion_config) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_17(mht_17_v, 350, "", "./tensorflow/compiler/xla/service/compiler.h", "set_fusion_config");

    fusion_config_ = fusion_config;
  }

  se::StreamExecutor* executor() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_18(mht_18_v, 357, "", "./tensorflow/compiler/xla/service/compiler.h", "executor");
 return executor_; }
  void set_executor(se::StreamExecutor* executor) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_19(mht_19_v, 361, "", "./tensorflow/compiler/xla/service/compiler.h", "set_executor");
 executor_ = executor; }

  // Optional profile_handle and cache key may be used to trigger recompilation
  // when a compilation cache is used.
  uint64_t profile_handle() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_20(mht_20_v, 368, "", "./tensorflow/compiler/xla/service/compiler.h", "profile_handle");
 return profile_handle_; }
  void set_profile_handle(uint64_t profile_handle) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_21(mht_21_v, 372, "", "./tensorflow/compiler/xla/service/compiler.h", "set_profile_handle");

    profile_handle_ = profile_handle;
  }

  absl::string_view cache_key() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_22(mht_22_v, 379, "", "./tensorflow/compiler/xla/service/compiler.h", "cache_key");
 return cache_key_; }
  void set_cache_key(absl::string_view cache_key) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("cache_key: \"" + std::string(cache_key.data(), cache_key.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_23(mht_23_v, 384, "", "./tensorflow/compiler/xla/service/compiler.h", "set_cache_key");

    cache_key_ = std::string(cache_key);
  }

  bool run_backend_only() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_24(mht_24_v, 391, "", "./tensorflow/compiler/xla/service/compiler.h", "run_backend_only");
 return run_backend_only_; }
  void set_run_backend_only(bool run_backend_only) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_25(mht_25_v, 395, "", "./tensorflow/compiler/xla/service/compiler.h", "set_run_backend_only");

    run_backend_only_ = run_backend_only;
  }

 protected:
  AotCompilationOptions();

 private:
  se::Platform::Id platform_id_;
  se::DeviceMemoryAllocator* device_allocator_ = nullptr;
  DebugOptions debug_options_;
  absl::optional<DeviceAssignment> static_device_assignment_;
  std::vector<std::vector<bool>> fusion_config_;
  FusionConfigCollection fusion_config_collection_ =
      FusionConfigCollection::kOff;
  se::StreamExecutor* executor_ = nullptr;
  uint64_t profile_handle_ = 0;
  std::string cache_key_;
  bool run_backend_only_ = false;
};

// Abstract superclass describing metadata produced during ahead-of-time
// compilation.
class AotCompilationMetadata {
 public:
  AotCompilationMetadata(const AotCompilationMetadata&) = delete;
  AotCompilationMetadata& operator=(AotCompilationMetadata const&) = delete;
  virtual std::string ToString() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_26(mht_26_v, 425, "", "./tensorflow/compiler/xla/service/compiler.h", "ToString");
 return ""; }
  virtual ~AotCompilationMetadata() = default;

 protected:
  AotCompilationMetadata() = default;
};

// Abstract compiler interface that is subclassed for compilation on a
// particular platform.
//
// The compiler ties together high level optimization (HLO) and low level
// optimization (LLO) / codegen (CG) to generate efficient executables for the
// target platform.
//
// The platform-based compiler singletons are registered via module initializers
// in their corresponding XLA compiler libraries, and are registered via the
// RegisterCompilerFactory API below.
//
// Thread-safety: subclasses of Compiler must be thread-safe, as multiple
// XLA clients may be requesting compilation concurrently for a given
// platform.
class Compiler {
 public:
  struct CompileOptions {
    // If device_allocator is not null, the compiler may use it to allocate temp
    // space on the device for use during compilation.  For example, the
    // compiler may allocate buffers on the device and then run variants of a
    // given algorithm over those buffers, to see which variant is fastest.  Any
    // space allocated will be deallocated before the compilation returns.
    se::DeviceMemoryAllocator* device_allocator = nullptr;

    // An optional thread pool for parallel compilation.
    tensorflow::thread::ThreadPool* thread_pool = nullptr;
  };

  virtual ~Compiler() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_27(mht_27_v, 463, "", "./tensorflow/compiler/xla/service/compiler.h", "~Compiler");
}

  // Returns the ID of the platform that this compiler targets.
  virtual se::Platform::Id PlatformId() const = 0;

  // Runs Hlo passes to optimize the given Hlo module, returns the optimized
  // module.
  virtual StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) = 0;
  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      se::DeviceMemoryAllocator* device_allocator) {
    return RunHloPasses(std::move(module), executor,
                        CompileOptions{device_allocator});
  }

  // Performs scheduling and buffer assignment and returns the buffer
  // assignments.
  // The returned 'BufferAssignment' retains a pointer to the 'HloModule', so
  // the module must live at least as long as the buffer assignments.
  virtual StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      const HloModule* module) {
    return Unimplemented("This compiler does not support this method");
  }

  // Compiles the HLO module for execution on a device given by the executor,
  // and returns an executable object or an error status. No HLO passes are
  // applied to module. Generally a module should be passed through RunHloPasses
  // prior to calling this method because some HLO passes are required for
  // correctness. Takes ownership of the HLO module.
  //
  // The compiler may optionally specialize to the individual device
  // (not just type of device) indicated by the executor.
  virtual StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      const CompileOptions& options) = 0;
  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
      se::DeviceMemoryAllocator* device_allocator) {
    return RunBackend(std::move(module), executor,
                      CompileOptions{device_allocator});
  }

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  virtual StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::string& serialized_aot_result) {
    return Unimplemented("LoadAotCompilationResult unimplemented.");
  }

  // Compiles a set of HLO modules that can run in parallel, potentially
  // communicating data between the modules, and returns a corresponding
  // sequence of executable objects.
  //
  // TODO(b/68666782): Remove this method after adding support for multiple
  // modules to RunHloPasses and RunBackends.
  virtual StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      const CompileOptions& options) = 0;
  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_exec,
      se::DeviceMemoryAllocator* device_allocator) {
    return Compile(std::move(module_group), stream_exec,
                   CompileOptions{device_allocator});
  }

  // Returns the backend configurations that the backend will consider for the
  // given HLO. Returns no configurations if the backend does not support
  // configurations for the given HLO.
  //
  // The stream executor is passed in to provide information about the hardware
  // that the backend configurations would be targeting.
  virtual std::vector<std::unique_ptr<tensorflow::protobuf::Message>>
  ComputeBackendConfigs(const HloInstruction& hlo,
                        se::StreamExecutor* executor) const;

  // Returns the backend configuration that the backend chooses by default for
  // the given HLO. Returns no configuration if the backend does not support
  // configurations for the given HLO.
  //
  // The stream executor is passed in to provide information about the hardware
  // that the backend configurations would be targeting.
  virtual std::unique_ptr<tensorflow::protobuf::Message>
  ComputeDefaultBackendConfig(const HloInstruction& hlo,
                              se::StreamExecutor* executor) const;

  // Compiles the HLO module group for ahead-of-time execution.  This is
  // intended for use in static compilation.
  virtual StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) = 0;

  // Similar to CompileAheadOfTime above but AotCompilationMetadata
  // has an argument that can be populated during compilation.
  virtual StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options,
                     std::unique_ptr<AotCompilationMetadata>* metadata);

  /////
  // The Compiler class also serves as a point to register compiler objects
  // for the various platforms.

  using CompilerFactory = std::function<std::unique_ptr<Compiler>()>;

  // Registers the compiler singleton for the platform. This is assumed to
  // be a singleton, so no ownership is transferred.
  //
  // Precondition: a platform kind must not be registered more than once.
  static void RegisterCompilerFactory(se::Platform::Id platform_id,
                                      CompilerFactory compiler_factory);

  // Returns the compiler singleton pointer if it is available for the given
  // platform, or an error status if it is not.
  static StatusOr<Compiler*> GetForPlatform(const se::Platform* platform);

  // Returns a function that computes the size in bytes of the logical
  // buffer that contains a shape.
  virtual HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const = 0;

  // Returns a function that computes the size in bytes of a given
  // logical buffer.
  std::function<int64_t(const BufferValue&)> BufferSizeBytesFunction() {
    HloCostAnalysis::ShapeSizeFunction shape_size = ShapeSizeBytesFunction();
    return [shape_size](const BufferValue& buffer) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_28(mht_28_v, 593, "", "./tensorflow/compiler/xla/service/compiler.h", "lambda");

      return shape_size(buffer.shape());
    };
  }

  virtual Shape DeviceShapeRepresentation(const Shape& shape) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScompilerDTh mht_29(mht_29_v, 601, "", "./tensorflow/compiler/xla/service/compiler.h", "DeviceShapeRepresentation");

    return shape;
  }

 private:
  // Mutex that guards the platform-compiler map.
  static absl::Mutex platform_compiler_mutex_;

  // Map from platform kind to compiler factory.
  static std::map<se::Platform::Id, CompilerFactory>*
  GetPlatformCompilerFactories();

  // Map from platform kind to compiler instance, if we made one already (based
  // on the factories above).
  static std::map<se::Platform::Id, std::unique_ptr<Compiler>>*
  GetPlatformCompilers();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPILER_H_
