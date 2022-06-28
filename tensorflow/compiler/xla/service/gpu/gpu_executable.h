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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh() {
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


#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace gpu {

// Returns whether GpuExecutable runs on TFRT (instead of thunks).
bool IsBefExecutableEnabled(const HloModuleConfig& config);

// Returns whether to create BefThunks (if the specific thunk is supported).
bool IsBefThunkEnabled(const HloModuleConfig& config);

inline bool IsBefEnabled(const HloModuleConfig& config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh mht_0(mht_0_v, 223, "", "./tensorflow/compiler/xla/service/gpu/gpu_executable.h", "IsBefEnabled");

  return IsBefExecutableEnabled(config) || IsBefThunkEnabled(config);
}

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
  struct BefBufferDeleter {
    void operator()(uint8_t* ptr) const;
    size_t size;
  };

 public:
  struct BefExecutable;

  typedef std::unique_ptr<const ThunkSchedule> OwnedThunkSchedule;
  typedef std::unique_ptr<uint8_t, BefBufferDeleter> OwnedBefBuffer;

  struct ConstantInfo {
    std::string symbol_name;
    std::vector<uint8_t> content;
    int allocation_index = -1;
  };

  struct OutputInfo {
    // Corresponding allocation index.
    int allocation_index;

    // Output is passed-through from a parameter.
    bool passthrough = false;

    // Whether this output is hinted to alias a parameter (BufferAllocation*
    // would indicate the aliased parameter), and what kind of alias it is.
    absl::optional<HloInputOutputAliasConfig::Alias> alias_config;
  };

  struct Params {
    std::string asm_text;
    std::vector<uint8_t> binary;
    GpuVersion gpu_version;
    // The GpuExecutable will either execute Thunks or a whole-program BEF
    // depending on which is supplied.
    absl::variant<OwnedThunkSchedule, OwnedBefBuffer> thunks_or_bef;
    xla::EntryFunctionAttributes entry_func_attrs;
    std::vector<ConstantInfo> constants;
    absl::flat_hash_map<ShapeIndex, OutputInfo> output_info;
    std::string module_name;
    xla::Shape output_shape;
    std::vector<BufferAllocation> allocations;
    std::unique_ptr<BufferAssignmentProto> debug_buffer_assignment = nullptr;

    // A callable that dumps out a debug string upon device OOM. It's not the
    // string itself, as the string can be huge and increase peak host memory
    // usage for the common (non-OOM) case.
    std::function<std::string()> verbose_buffer_assignment_string_dumper = [] {
      return std::string();
    };

    std::unique_ptr<HloModule> debug_module = nullptr;
  };

  // TODO(hanbinyoon): Once BEF replaces Thunks, hide this method as an
  // implementation detail of GpuExecutable.
  // Analyze the entry function to construct buffer allocation and other output
  // information. Optionally use buffer_param_offset to indicate the position of
  // buffer parameters in the entry function - in tfrt_gpu dialect, buffer
  // arguments start from the third parameter (after tfrt::Chain and GpuStream).
  static Status SetUpMlirAllocation(
      mlir::func::FuncOp func, llvm::ArrayRef<int64_t> buffer_sizes,
      std::vector<BufferAllocation>* allocations,
      absl::flat_hash_map<ShapeIndex, OutputInfo>* output_info,
      Shape* output_shape, int buffer_param_offset = 0);

  // Returns an Executable that is loaded from a BEF. This BEF must have entry
  // point information recorded by use of the tfrt::gpu::setEntryPoint()
  // function.
  static StatusOr<std::unique_ptr<Executable>> LoadFromBef(
      std::shared_ptr<HloModule> hlo_module, absl::string_view bef,
      xla::EntryFunctionAttributes entry_func_attrs, GpuVersion gpu_version);

  static StatusOr<std::unique_ptr<GpuExecutable>> Create(Params params);
  ~GpuExecutable() override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  // This should be called after set_ir_module_string.
  const std::string& ir_module_string() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh mht_1(mht_1_v, 315, "", "./tensorflow/compiler/xla/service/gpu/gpu_executable.h", "ir_module_string");
 return ir_module_string_; }

  // This should be called before ExecuteOnStream.
  void set_ir_module_string(const std::string& ir_module_string) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("ir_module_string: \"" + ir_module_string + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh mht_2(mht_2_v, 322, "", "./tensorflow/compiler/xla/service/gpu/gpu_executable.h", "set_ir_module_string");

    ir_module_string_ = ir_module_string;
  }

  // Returns the compiled code for the computation. The compiled code is PTX in
  // Cuda and unused empty string in ROCm.
  const std::string& text() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh mht_3(mht_3_v, 331, "", "./tensorflow/compiler/xla/service/gpu/gpu_executable.h", "text");
 return text_; }

  // Returns the binary stored in this GpuExecutable. The binary is cubin in
  // Cuda, and HSA code object in ROCm. It may be empty, in which case
  // compilation is left up to the GPU driver.
  const std::vector<uint8_t>& binary() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh mht_4(mht_4_v, 339, "", "./tensorflow/compiler/xla/service/gpu/gpu_executable.h", "binary");
 return binary_; }

  // ExecuteAsyncOnStream will fail if the compute capability of the stream
  // doesn't match the compute capability passed to this object's constructor.
  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  StatusOr<ScopedShapedBuffer> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      absl::Span<const ShapedBuffer* const> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  using VariantArguments = absl::variant<absl::Span<const ShapedBuffer* const>,
                                         absl::Span<ExecutionInput>>;
  StatusOr<ExecutionOutput> ExecuteAsyncOnStreamImpl(
      const ServiceExecutableRunOptions* run_options,
      VariantArguments arguments);

  absl::Span<const BufferAllocation> GetAllocations() const {
    return allocations_;
  }

  const std::vector<ConstantInfo>& constants() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSgpu_executableDTh mht_5(mht_5_v, 366, "", "./tensorflow/compiler/xla/service/gpu/gpu_executable.h", "constants");
 return constants_; }

 private:
  // Use GpuExecutable::Create() to create an instance.
  explicit GpuExecutable(Params params);

  // Constructor to use when loading a GpuExecutable from a BEF. Omits setting
  // class members that aren't used in BEF execution mode.
  GpuExecutable(std::shared_ptr<HloModule> hlo_module, GpuVersion gpu_version,
                xla::EntryFunctionAttributes entry_func_attrs,
                absl::string_view module_name, Shape xla_output_shape,
                std::vector<BufferAllocation> allocations,
                absl::flat_hash_map<ShapeIndex, OutputInfo> output_info,
                BefExecutable* bef_executable);

  // If `block_host_until_done` is false, execution will not block the host
  // until the kernels have completed. This is used as an optimization for
  // clients, such as Tensorflow, that use a single stream of execution for
  // computations, and allow host-side deallocation from the allocator before
  // GPU execution completes.
  Status ExecuteThunksOrBef(const ServiceExecutableRunOptions* run_options,
                            const BufferAllocations& buffer_allocations,
                            bool block_host_until_done);

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase>;

  // Loads the PTX or CUBIN for this executable and initializes all
  // constants that haven't already been initialized by the CUDA driver. Loaded
  // modules are owned by this executable.
  //
  // Returns a map from buffer allocation indices to device memory pointers
  // (only for allocations that contain constants).
  //
  // The returned map is cached. If the above process has already been run for
  // the given stream, it is skipped and the cached map is immediately returned
  // instead.
  StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      stream_executor::Stream* stream);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  StatusOr<BufferAllocations> GenerateBufferAllocations(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal);

  StatusOr<se::DeviceMemoryBase> BufferForAllocation(
      VariantArguments arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      const BufferAllocation& allocation,
      se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
      int64_t arg_idx);

  // The LLVM IR, in string format, of the unoptimized module generated for
  // this GpuExecutable. We save a string instead of an llvm::Module* because
  // leaving llvm::Module* in a singleton can cause the heap checker to emit
  // false positives.
  //
  // This string should be modified only before ExecuteOnStream.
  std::string ir_module_string_;

  // The compiled code for the computation.
  const std::string text_;

  // The GPU machine code for the computation, targeting GPUs at
  // compute_capability_.
  //
  // May be empty, in which case we leave compilation up to the GPU driver.
  const std::vector<uint8_t> binary_;

  // The GPU version for compute compatibility check.
  GpuVersion gpu_version_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // IrEmitter.
  OwnedThunkSchedule thunks_;

  xla::EntryFunctionAttributes entry_func_attrs_;

  std::string module_name_;

  xla::Shape output_shape_;

  // Owns the buffer data at runtime. It provides information to allocate
  // memory for every output/temp buffers.
  const std::vector<BufferAllocation> allocations_;

  std::shared_ptr<BufferAssignmentProto> debug_buffer_assignment_;
  std::function<std::string()> verbose_buffer_assignment_string_dumper_;

  absl::Mutex module_handle_mutex_;
  // Cache of module handles. Required to keep loaded modules alive until this
  // executable is destroyed.
  std::map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ ABSL_GUARDED_BY(module_handle_mutex_);
  // Cache of constant buffer allocation maps used by `ResolveConstantGlobals`.
  std::map<stream_executor::StreamExecutor*, BufferAllocToDeviceMemoryMap>
      module_globals_ ABSL_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;
  const absl::flat_hash_map<ShapeIndex, OutputInfo> output_info_;
  // Retains shared ownership of on-device constants that are managed by XLA and
  // potentially shared with other executables.
  std::vector<std::shared_ptr<se::DeviceMemoryBase>> shared_constants_;

  // Data for bef executable mode only, owned.
  BefExecutable* bef_executable_ = nullptr;

  GpuExecutable(const GpuExecutable&) = delete;
  GpuExecutable& operator=(const GpuExecutable&) = delete;
};

StatusOr<absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>>
GetOutputInfo(const HloModule& hlo_module, const BufferAssignment& assignment);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
