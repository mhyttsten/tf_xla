/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh() {
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


#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/threadpool.h"

namespace stream_executor {

// Forward-declared to avoid StreamExecutor dependency.
class DeviceMemoryAllocator;

}  // namespace stream_executor

namespace xla {

// Class containing options for building an LocalExecutable with
// LocalClient::Compile.
class ExecutableBuildOptions {
 public:
  // If set, this is the device to build the computation for. Valid
  // device_ordinal values are: 0 to # of devices - 1. These values are
  // identical to the device ordinal values used by StreamExecutor. The built
  // executable will be executable on any device equivalent to the specified
  // device as determined by Backend::devices_equivalent(). A value of -1
  // indicates this option has not been set.
  ExecutableBuildOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this specifies the layout of the result of the computation. If not
  // set, the service will chose the layout of the result. A Shape is used to
  // store the layout to accommodate tuple result shapes. A value of nullptr
  // indicates the option has not been set.
  ExecutableBuildOptions& set_result_layout(const Shape& shape_with_layout);
  const Shape* result_layout() const;

  // Expose access to the XLA debug options which will be passed to the
  // compilation process.
  bool has_debug_options() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_0(mht_0_v, 228, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "has_debug_options");
 return debug_options_.has_value(); }
  const DebugOptions& debug_options() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "debug_options");
 return *debug_options_; }
  DebugOptions* mutable_debug_options();

  // If set, this specifies an allocator that can be used to allocate temporary
  // space on the device during compilation.  For example, the compiler might
  // want to run various algorithms on the device and pick the fastest one -- it
  // might allocate buffers for use by these algorithms using this allocator.
  //
  // This does not need to be the same as the se::DeviceMemoryAllocator passed
  // when running the executable.
  ExecutableBuildOptions& set_device_allocator(
      se::DeviceMemoryAllocator* allocator);
  se::DeviceMemoryAllocator* device_allocator() const;

  // Returns a string representation of the build options, suitable for
  // debugging.
  std::string ToString() const;

  // The number of replicas of this computation that are to be executed.
  // Defaults to 1.
  int num_replicas() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_2(mht_2_v, 255, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "num_replicas");
 return num_replicas_; }
  ExecutableBuildOptions& set_num_replicas(int num_replicas);

  // The number of partitions in this computation. Defaults to 1.
  int num_partitions() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_3(mht_3_v, 262, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "num_partitions");
 return num_partitions_; }
  ExecutableBuildOptions& set_num_partitions(int num_partitions);

  // Indicates whether to use SPMD (true) or MPMD (false) partitioning when
  // num_partitions > 1 and XLA is requested to partition the input program.
  bool use_spmd_partitioning() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_4(mht_4_v, 270, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "use_spmd_partitioning");
 return use_spmd_partitioning_; }
  ExecutableBuildOptions& set_use_spmd_partitioning(bool use_spmd_partitioning);

  // Whether to automatically generate XLA shardings for SPMD partitioner.
  bool use_auto_spmd_partitioning() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_5(mht_5_v, 277, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "use_auto_spmd_partitioning");

    return use_auto_spmd_partitioning_;
  }
  ExecutableBuildOptions& set_use_auto_spmd_partitioning(
      bool use_auto_spmd_partitioning);

  bool deduplicate_hlo() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_6(mht_6_v, 286, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "deduplicate_hlo");
 return deduplicate_hlo_; }
  ExecutableBuildOptions& set_deduplicate_hlo(bool deduplicate_hlo);

  // If set, this specifies a static device assignment for the computation.
  // Otherwise, the computation will be compiled generically and can be run with
  // any device assignment compatible with the computation's replica and
  // partition counts.
  bool has_device_assignment() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_7(mht_7_v, 296, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "has_device_assignment");
 return device_assignment_.has_value(); }
  ExecutableBuildOptions& set_device_assignment(
      const DeviceAssignment& device_assignment);
  const DeviceAssignment& device_assignment() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_8(mht_8_v, 302, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "device_assignment");

    CHECK(device_assignment_.has_value());
    return device_assignment_.value();
  }

  // Whether input and output buffers are aliased if the associated parameter is
  // passed-through XLA modules without being changed.
  bool alias_passthrough_params() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_9(mht_9_v, 312, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "alias_passthrough_params");
 return alias_passthrough_params_; }
  void set_alias_passthrough_params(bool alias_passthrough_params) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_10(mht_10_v, 316, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "set_alias_passthrough_params");

    alias_passthrough_params_ = alias_passthrough_params;
  }

  bool run_backend_only() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_11(mht_11_v, 323, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "run_backend_only");
 return run_backend_only_; }
  // By default, XLA builds an executable by invoking standard compilation, i.e,
  // running Compiler::Compile, or both Compiler::RunHloPasses and
  // Compiler::RunBackend. When run_backend_only is set to true, XLA builds an
  // executable by invoking only RunBackend and skip invoking RunHloPasses,
  // which can be used to compile post-optimizations HLO modules.
  ExecutableBuildOptions& set_run_backend_only(bool run_backend_only) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_12(mht_12_v, 332, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "set_run_backend_only");

    run_backend_only_ = run_backend_only;
    return *this;
  }

  bool allow_spmd_sharding_propagation_to_output() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_13(mht_13_v, 340, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "allow_spmd_sharding_propagation_to_output");

    return allow_spmd_sharding_propagation_to_output_;
  }
  // Allows sharding propagation to propagate to the outputs. This changes the
  // output shape of the computation (which is undesirable), but it can be used
  // to allow to run partial compilation to determine what would be the output
  // sharding of a computation if XLA would be allowed to propagate the sharding
  // which can be used by higher level framework as a way to query intermediate
  // sharding of operations when multiple computation would be chained and
  // merged together.
  ExecutableBuildOptions& set_allow_spmd_sharding_propagation_to_output(
      bool allow_spmd_sharding_propagation_to_output) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_14(mht_14_v, 354, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "set_allow_spmd_sharding_propagation_to_output");

    allow_spmd_sharding_propagation_to_output_ =
        allow_spmd_sharding_propagation_to_output;
    return *this;
  }

  // Thread pool for parallel compilation.
  tensorflow::thread::ThreadPool* compile_thread_pool() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_15(mht_15_v, 364, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "compile_thread_pool");

    return compile_thread_pool_;
  }
  ExecutableBuildOptions& set_compile_thread_pool(
      tensorflow::thread::ThreadPool* compile_thread_pool) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSclientPSexecutable_build_optionsDTh mht_16(mht_16_v, 371, "", "./tensorflow/compiler/xla/client/executable_build_options.h", "set_compile_thread_pool");

    compile_thread_pool_ = compile_thread_pool;
    return *this;
  }

 private:
  int device_ordinal_ = -1;
  Shape result_layout_;
  bool result_layout_set_ = false;
  absl::optional<DebugOptions> debug_options_;
  se::DeviceMemoryAllocator* device_allocator_ = nullptr;
  int num_replicas_ = 1;
  int num_partitions_ = 1;
  bool use_spmd_partitioning_ = false;
  bool use_auto_spmd_partitioning_ = false;
  bool deduplicate_hlo_ = false;
  bool broadcast_replicated_params_ = false;
  absl::optional<DeviceAssignment> device_assignment_;
  bool alias_passthrough_params_ = false;
  bool run_backend_only_ = false;
  bool allow_spmd_sharding_propagation_to_output_ = false;
  tensorflow::thread::ThreadPool* compile_thread_pool_ = nullptr;
};

// Creates an ExecutionOptions based on a given ExecutableBuildOptions and
// ProgramShape.
ExecutionOptions CreateExecutionOptions(
    const ExecutableBuildOptions& build_options,
    const ProgramShape* program_shape);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
