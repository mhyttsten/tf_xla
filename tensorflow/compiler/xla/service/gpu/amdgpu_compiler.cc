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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc() {
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

#include "tensorflow/compiler/xla/service/gpu/amdgpu_compiler.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/gpu/cusolver_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_dimension_grouper.h"
#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/tree_reduction_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"

namespace xla {
namespace gpu {

namespace {

// Returns the directory containing ROCm-Device-Libs files. This function is
// called in AMDGPUCompiler's constructor, so can't return an error. But
// AMDGPUCompiler::Compile will return an error when the wanted rocdl file
// doesn't exist in the folder this function returns.
std::string GetROCDLDir(const HloModuleConfig& config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc mht_0(mht_0_v, 219, "", "./tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc", "GetROCDLDir");

  std::vector<std::string> potential_rocdl_dirs;
  const std::string datadir = config.debug_options().xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tensorflow::RocdlRoot());

  // Tries all potential ROCDL directories in the order they are inserted.
  // Returns the first directory that exists in the file system.
  for (const std::string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tensorflow::Env::Default()->IsDirectory(potential_rocdl_dir).ok()) {
      VLOG(2) << "Found ROCm-Device-Libs dir " << potential_rocdl_dir;
      return potential_rocdl_dir;
    }
    VLOG(2) << "Unable to find potential ROCm-Device-Libs dir "
            << potential_rocdl_dir;
  }

  // Last resort: maybe in the current folder.
  return ".";
}

}  // namespace

Status AMDGPUCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc", "AMDGPUCompiler::OptimizeHloConvolutionCanonicalization");

  // Convert convolutions into CustomCalls to MIOpen, then canonicalize them
  // (PadInsertion).
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantCheckerDebug<HloVerifier>(
      /*layout_sensitive=*/false,
      /*allow_mixed_precision=*/false);
  pipeline.AddPass<GpusolverRewriter>();
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<GpuConvPaddingLegalization>();

  // The conv padding/vectorization passes which we need to get rid of.  They
  // also leave behind unnecessary tuple/get-tuple-element pairs that
  // TupleSimplifier fixes.
  pipeline.AddPass<CallInliner>();
  pipeline.AddPass<TupleSimplifier>();

  // tf2xla bridge, DepthwiseConvolutionConverter and GpuConvRewriter
  // introduces reshapes and transposes that can be eliminated using
  // AlgebraicSimplifier  We run algsimp to a fixed point.
  //
  // When transposes appear in a fusion node, we can easily adjust the
  // multi-dimensional index to create the one needed for the operand. This
  // is not as easy with bitcasts, because we don't have the information
  // readily available which dimensions are permuted. In addition to that,
  // if we have a transpose and a reshape next to each other, they will both
  // be replaced by a bitcast, and we replace bitcast(bitcast) with one
  // bitcast. This leads to having to linearize and then delinearize the
  // index.
  AlgebraicSimplifierOptions options;
  options.set_replace_transpose_with_bitcast(false);
  options.set_enable_conv_operand_swap(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return Status::OK();
}

AMDGPUCompiler::AMDGPUCompiler()
    : GpuCompiler(stream_executor::rocm::kROCmPlatformId,
                  amdgpu::TargetTriple(), amdgpu::DataLayout()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc mht_2(mht_2_v, 294, "", "./tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc", "AMDGPUCompiler::AMDGPUCompiler");
}

GpuVersion AMDGPUCompiler::GetGpuVersion(se::StreamExecutor* stream_exec) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc mht_3(mht_3_v, 299, "", "./tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc", "AMDGPUCompiler::GetGpuVersion");

  return stream_exec->GetDeviceDescription().rocm_compute_capability();
}

StatusOr<std::pair<std::string, std::vector<uint8_t>>>
AMDGPUCompiler::CompileTargetBinary(const HloModuleConfig& module_config,
                                    llvm::Module* llvm_module,
                                    GpuVersion gpu_version,
                                    se::StreamExecutor* stream_exec,
                                    bool relocatable,
                                    const HloModule* debug_module) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSamdgpu_compilerDTcc mht_4(mht_4_v, 312, "", "./tensorflow/compiler/xla/service/gpu/amdgpu_compiler.cc", "AMDGPUCompiler::CompileTargetBinary");

  if (rocdl_dir_.empty()) {
    // Compute rocdl_dir_ just once and cache it in this member.
    rocdl_dir_ = GetROCDLDir(module_config);
  }

  if (relocatable) {
    return Unimplemented("relocatable target binary is not implemented");
  }

  std::vector<uint8_t> hsaco;
  {
    XLA_SCOPED_LOGGING_TIMER(
        "AMDGPUCompiler::CompileTargetBinary - CompileToHsaco");
    TF_ASSIGN_OR_RETURN(
        hsaco, amdgpu::CompileToHsaco(llvm_module, gpu_version, module_config,
                                      rocdl_dir_));
  }

  return std::pair<std::string, std::vector<uint8_t>>("", std::move(hsaco));
}

}  // namespace gpu
}  // namespace xla
