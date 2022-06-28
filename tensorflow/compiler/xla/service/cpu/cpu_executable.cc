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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"

#include <stdint.h>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/host/host_stream.h"

namespace xla {
namespace cpu {

static std::string ModuleUniqueName(absl::string_view module_name,
                                    const HloModule* module) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("module_name: \"" + std::string(module_name.data(), module_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "ModuleUniqueName");

  std::string unique_id;
  if (module != nullptr) {
    unique_id = absl::StrCat("module.", module->unique_id(), ".");
  }
  return absl::StrCat(unique_id, module_name);
}

CpuExecutable::CpuExecutable(
    std::unique_ptr<SimpleOrcJIT> jit,
    std::unique_ptr<const BufferAssignment> assignment,
    std::unique_ptr<HloModule> hlo_module,
    const std::string& entry_function_name,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
    std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map)
    : Executable(std::move(hlo_module), std::move(hlo_profile_printer_data),
                 std::move(hlo_profile_index_map)),
      jit_(std::move(jit)),
      assignment_(std::move(assignment)),
      module_name_(entry_function_name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("entry_function_name: \"" + entry_function_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::CpuExecutable");

  if (assignment_) {
    buffer_assignment_.reset(new BufferAssignmentProto(assignment_->ToProto()));
  }
  XlaDebugInfoManager::Get()->RegisterModule(
      ModuleUniqueName(module_name_, shared_module().get()), shared_module(),
      buffer_assignment_);

  // Resolve symbols in the constructor rather than at execution time to avoid
  // races because FindSymbol is not thread safe.
  llvm::Expected<llvm::JITEvaluatedSymbol> sym =
      jit_->FindCompiledSymbol(entry_function_name);
  // We expect to find the symbol provided with entry_function_name; otherwise
  // this is an internal error.
  CHECK(*sym) << "Symbol " << entry_function_name << " not found.";
  // getAddress can do work under the hood in the jit, so it needs to be
  // guarded by the mutex.
  compute_function_ = reinterpret_cast<ComputeFunctionType>(sym->getAddress());
  VLOG(1) << "compute_function_ at address "
          << reinterpret_cast<void*>(compute_function_);
  jit_->DoneCompiling();
}

CpuExecutable::~CpuExecutable() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_2(mht_2_v, 273, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::~CpuExecutable");

  XlaDebugInfoManager::Get()->UnregisterModule(
      ModuleUniqueName(module_name_, shared_module().get()), shared_module(),
      buffer_assignment_);
}

static StatusOr<MaybeOwningDeviceMemory> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<ExecutionInput const> arguments,
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal) {
  VLOG(3) << allocation.ToString();
  if (allocation.is_entry_computation_parameter()) {
    se::DeviceMemoryBase out = arguments[allocation.parameter_number()]
                                   .Buffer(allocation.param_shape_index())
                                   .AsDeviceMemoryBase();
    CHECK_LE(allocation.size(), out.size())
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();
    VLOG(3) << "allocation is a parameter";
    return MaybeOwningDeviceMemory{out};
  } else if (allocation.is_constant()) {
    VLOG(3) << "allocation is a constant";
    return MaybeOwningDeviceMemory{se::DeviceMemoryBase{}};
  } else if (allocation.is_thread_local()) {
    VLOG(3) << "buffer is thread-local";
    return MaybeOwningDeviceMemory{se::DeviceMemoryBase{}};
  }

  int64_t buffer_size = allocation.size();
  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory out,
                      memory_allocator->Allocate(device_ordinal, buffer_size));
  VLOG(3) << "buffer allocated " << buffer_size << " bytes [" << out->opaque()
          << "]";

  // Since the output buffer and all the temporary buffers were written into
  // by the JITed code, msan has no way of knowing their memory was
  // initialized. Mark them initialized so that msan doesn't flag loads from
  // these buffers.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->opaque(), buffer_size);
  return MaybeOwningDeviceMemory{std::move(out)};
}

StatusOr<std::vector<MaybeOwningDeviceMemory>> CpuExecutable::CreateBufferTable(
    se::DeviceMemoryAllocator* memory_allocator, int device_ordinal,
    absl::Span<ExecutionInput const> arguments) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_3(mht_3_v, 320, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::CreateBufferTable");

  std::vector<MaybeOwningDeviceMemory> buffers(
      assignment_->Allocations().size());
  VLOG(3) << "Allocating " << assignment_->Allocations().size()
          << " allocations for module " << module().name();
  for (BufferAllocation::Index i = 0; i < assignment_->Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment_->GetAllocation(i);
    TF_ASSIGN_OR_RETURN(
        buffers[i], MemoryForAllocation(allocation, arguments, memory_allocator,
                                        device_ordinal));
  }

  if (VLOG_IS_ON(3)) {
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice result_slice,
                        assignment_->GetUniqueTopLevelOutputSlice());
    VLOG(3) << "result index: " << result_slice.index();
  }
  return std::move(buffers);
}

Status CpuExecutable::ExecuteComputeFunction(
    const ExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory const> buffers,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_4(mht_4_v, 347, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::ExecuteComputeFunction");

  uint64_t start_micros = tensorflow::Env::Default()->NowMicros();

  XlaDebugInfoManager::Get()->OnModuleStart(module_name_);
  auto cleanup = absl::MakeCleanup(
      [&]() { XlaDebugInfoManager::Get()->OnModuleStop(module_name_); });

  size_t profile_counters_size =
      hlo_execution_profile ? hlo_execution_profile->profile_counters().size()
                            : 0;
  int64_t* profile_counters =
      hlo_execution_profile
          ? hlo_execution_profile->mutable_profile_counters()->data()
          : nullptr;

  // Call the computation function following the calling convention. See the
  // definition of 'ComputeFunctionType' for the details of the calling
  // convention of JITed functions.
  std::vector<void*> buffer_pointers;
  for (auto& buffer : buffers) {
    buffer_pointers.push_back(
        const_cast<void*>(buffer.AsDeviceMemoryBase().opaque()));
  }

  VLOG(3) << "Executing compute function:";
  VLOG(3) << absl::StrFormat("  Number of buffer table entries: %u",
                             buffer_pointers.size());
  auto ptr_printer = [](std::string* out, const void* p) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_5(mht_5_v, 377, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "lambda");

    absl::StrAppend(out, absl::StrFormat("%p", p));
  };
  VLOG(3) << absl::StrFormat("  Buffer table: [%s]",
                             absl::StrJoin(buffer_pointers, ", ", ptr_printer));
  VLOG(3) << absl::StrFormat("  Number of profile counters: %u",
                             profile_counters_size);
  VLOG(3) << absl::StrFormat("  Profile counters: %p", profile_counters);

  XlaCustomCallStatus status;
  // For the entry computation (like all global computations), all inputs and
  // outputs are in the buffer table, and both the result pointer and args array
  // pointers are unused (so we set them to 'nullptr').
  compute_function_(nullptr, run_options, nullptr, buffer_pointers.data(),
                    &status, profile_counters);

  uint64_t end_micros = tensorflow::Env::Default()->NowMicros();

  if (run_options->execution_profile()) {
    const double nanoseconds = (end_micros - start_micros) * 1000.0;
    run_options->execution_profile()->set_compute_time_ns(
        std::max(nanoseconds, 1.0));
    // If hlo profiling was disabled then the cycle count is left empty.
    if (hlo_execution_profile) {
      run_options->execution_profile()->set_compute_cycle_count(
          hlo_execution_profile->total_cycles_executed(
              *module().entry_computation()));
    }
  }

  absl::optional<absl::string_view> error_message =
      CustomCallStatusGetMessage(&status);
  if (error_message) {
    return InternalError("CustomCall failed: %s", *error_message);
  }

  return Status::OK();
}

StatusOr<ExecutionOutput> CpuExecutable::CreateResultShapedBuffer(
    const ServiceExecutableRunOptions* run_options,
    absl::Span<MaybeOwningDeviceMemory> buffers,
    absl::Span<ExecutionInput> arguments) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_6(mht_6_v, 422, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::CreateResultShapedBuffer");

  se::Stream* stream = run_options->stream();
  ExecutionOutput result(/*on_device_shape=*/result_shape(),
                         run_options->allocator(),
                         stream->parent()->device_ordinal());
  const HloInputOutputAliasConfig& input_output_alias =
      module().input_output_alias_config();
  HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  const Shape& root_shape = root->shape();

  // Move se::OwningDeviceMemory values which contain the array(s) of the result
  // into the respective location in ScopedShapedBuffer which is returned to the
  // caller.
  for (auto& p : result.MutableResult()->buffers()) {
    const ShapeIndex& index = p.first;
    se::DeviceMemoryBase& result_buffer = p.second;
    const HloValueSet& sources = this->GetRootValueSet().element(index);
    // The points to set is unambiguous so the set should be a
    // singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();

    // The source for this result buffer can be a nested buffer such as
    // a tuple element.
    TF_ASSIGN_OR_RETURN(
        const BufferAllocation::Slice slice,
        this->assignment_->GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();

    // TODO(cheshire): duplication with other backends.
    absl::optional<HloInputOutputAliasConfig::Alias> alias =
        input_output_alias.GetAliasedParameter(index);
    if (alias) {
      CHECK_LT(alias->parameter_number, arguments.size());
      ExecutionInput& input = arguments[alias->parameter_number];
      MaybeOwningDeviceMemory* maybe_owning_memory =
          input.MutableBuffer(alias->parameter_index);
      if (alias->must_alias() && !maybe_owning_memory->HasOwnership()) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: %s",
            alias->ToString());
      }
      if (absl::optional<se::OwningDeviceMemory> owning =
              maybe_owning_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase argument_buffer = owning->Release();
        *maybe_owning_memory = argument_buffer;
        result_buffer = argument_buffer;
        // The caller is giving us the
        // input buffer, but in case of error of the execute call, we should
        // not be releasing it as it contains valid data (for example, it is a
        // parameter which the user wants us to alias, in a gradient update
        // computation). So we store the index into the result in the aliased
        // vactor, which will be fed to the ExecutionOutput, which will be
        // using the indices to drop the addresses from its own
        // ScopedShapedBuffer result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(index);
      } else {
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size =
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(root_shape, index));
        TF_ASSIGN_OR_RETURN(
            se::OwningDeviceMemory allocated_buffer,
            run_options->allocator()->Allocate(
                stream->parent()->device_ordinal(), allocation_size));
        result_buffer = allocated_buffer.Release();
        MaybeOwningDeviceMemory& registered_buffer = buffers[buffer_index];
        CHECK_EQ(result_buffer.size(),
                 registered_buffer.AsDeviceMemoryBase().size());
        std::memcpy(/*dest=*/result_buffer.opaque(),
                    /*src=*/registered_buffer.AsDeviceMemoryBase().opaque(),
                    /*n=*/result_buffer.size());
        registered_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      MaybeOwningDeviceMemory& buffer = buffers[buffer_index];
      if (absl::optional<se::OwningDeviceMemory> owned_buffer =
              buffer.Release()) {
        result_buffer = owned_buffer->Release();
        buffer = result_buffer;
      } else {
        result_buffer = buffer.AsDeviceMemoryBase();
        result.AddAliasedIndex(index);
      }
    }
  }
  return std::move(result);
}

StatusOr<ExecutionOutput> CpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_7(mht_7_v, 525, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::ExecuteAsyncOnStream");

  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  if (hlo_module_) {
    const HloComputation* entry_comp = hlo_module_->entry_computation();
    CHECK_EQ(entry_comp->num_parameters(), arguments.size())
        << "Wrong number of arguments passed when running executable";
    for (int64_t i = 0; i < entry_comp->num_parameters(); ++i) {
      const Shape& expected_shape =
          entry_comp->parameter_instruction(i)->shape();
      const Shape& actual_shape = arguments[i].Buffers().shape();
      TF_RET_CHECK(
          ShapeUtil::DynamicShapeIsCompatible(actual_shape, expected_shape))
          << "Shape mismatch on argument " << i << ", "
          << expected_shape.ToString(/*print_layout=*/true) << " vs. "
          << actual_shape.ToString(/*print_layout=*/true);
    }
  }

  auto* host_stream = dynamic_cast<se::host::HostStream*>(
      run_options->stream()->implementation());
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  TF_ASSIGN_OR_RETURN(
      std::vector<MaybeOwningDeviceMemory> buffers,
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(buffers),
                               absl::MakeSpan(arguments)));

  // Logically we want this lambda to capture `buffers` by move, ultimately our
  // functor needs to be wrapped in an std::function, and that requires its
  // functor to be copyable.  Thus we perpetrate the hack of capturing buffers
  // "by shared pointer".
  //
  // We also need to change the types of some of the variables we capture:
  // run_options needs to change from a pointer to a value type, and arguments
  // needs to change from a Span into a vector.  We use a struct instead
  // of a lambda to make this explicit.
  struct AsyncRunTask {
    CpuExecutable* executable;
    ServiceExecutableRunOptions run_options;
    std::shared_ptr<std::vector<MaybeOwningDeviceMemory>> task_buffers;
    HloExecutionProfile* hlo_execution_profile;

    Status operator()() {
      return executable->ExecuteComputeFunction(
          &run_options.run_options(), *task_buffers, hlo_execution_profile);
    }
  };
  host_stream->EnqueueTaskWithStatus(
      AsyncRunTask{this, *run_options,
                   std::make_shared<std::vector<MaybeOwningDeviceMemory>>(
                       std::move(buffers)),
                   hlo_execution_profile});

  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

/*static*/ int64_t CpuExecutable::ShapeSizeBytes(const Shape& shape) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_8(mht_8_v, 593, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::ShapeSizeBytes");

  // On the cpu, opaques are pointers.
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*)) + metadata_size;
}

const InstructionValueSet& CpuExecutable::GetRootValueSet() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_9(mht_9_v, 609, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::GetRootValueSet");

  return assignment_->dataflow_analysis().GetInstructionValueSet(
      module().entry_computation()->root_instruction());
}

int64_t CpuExecutable::SizeOfGeneratedCodeInBytes() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_executableDTcc mht_10(mht_10_v, 617, "", "./tensorflow/compiler/xla/service/cpu/cpu_executable.cc", "CpuExecutable::SizeOfGeneratedCodeInBytes");

  return jit_->SizeOfGeneratedCodeInBytes();
}

}  // namespace cpu
}  // namespace xla
