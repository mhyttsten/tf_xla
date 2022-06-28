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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/llvm_ir_runtime.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {

/* Create filtered versions of the LLVM Pass Managers to filter out some
of the expensive passes.
Profiling:
   learning/brain/google/xla/benchmarks:inception_cpu_benchmark
   learning/brain/google/xla/benchmarks:cifarnet
pointed to LICM and IndVarSimplify as the hottest passes.
LICM is known to exhibit O(n^2) time in the number of instructions.
IndVarSimplify is slow due to SCEV. If loops are emitted in canonical form,
this pass is not necessary.
Disabling these as a starting point.
*/
// TODO(b/64227304) Creating a custom pass pipeline will replace this.

namespace {
class FilteredPassManager : public llvm::legacy::PassManager {
 public:
  explicit FilteredPassManager(bool disable_expensive_passes)
      : disable_expensive_passes_(disable_expensive_passes) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc mht_0(mht_0_v, 236, "", "./tensorflow/compiler/xla/service/cpu/compiler_functor.cc", "FilteredPassManager");
}
  void add(llvm::Pass* p) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc mht_1(mht_1_v, 240, "", "./tensorflow/compiler/xla/service/cpu/compiler_functor.cc", "add");

    // Disable all the loop unroll passes in the pipeline if
    // `disable_expensive_passes_` is true (TODO: Maybe we should use
    // `builder.DisableUnrollLoops` for this legacy feature?). Disable only the
    // early loop full unroll pass, otherwise. The early loop full unroll pass
    // applies excesive unrolling in statically bounded low trip-count loops,
    // which are very common in XLA. It also creates a strong dependency on the
    // SLP vectorizer to produce all the vector code, since the loops are fully
    // unrolled. By disabling it, the Loop Vectorizer would have an opportunity
    // to vectorize the code. A later loop unroll pass will still unroll the
    // loops before SLP for those cases missed by the Loop Vectorizer.
    constexpr unsigned loop_full_unroll_pos = 0;
    if (p->getPassName().contains("Unroll loops") &&
        (disable_expensive_passes_ ||
         num_unroll_passes_ == loop_full_unroll_pos)) {
      ++num_unroll_passes_;
      delete p;
      return;
    }

    llvm::legacy::PassManager::add(p);
  }

 private:
  unsigned num_unroll_passes_ = 0;
  bool disable_expensive_passes_;
};
}  // anonymous namespace

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> CompilerFunctor::operator()(
    llvm::Module& module) {
  FilteredPassManager module_passes(disable_expensive_passes_);
  llvm::legacy::FunctionPassManager function_passes(&module);

  VLOG(2) << "IR before optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpModuleToString(module));

  if (pre_optimization_hook_) {
    pre_optimization_hook_(module);
  }

  // Add the appropriate TargetLibraryInfo and TargetTransformInfo.
  AddTargetInfoPasses(&module_passes);

  // Build up optimization pipeline.
  if (optimize_for_size_) {
    // Optimizing for size turns on -O2 level optimizations.
    //
    // TODO(b/64153864): Although the code generator supports size_level = 2 to
    // turn on more aggressive code size optimizations than size_level = 1, we
    // pass size_level = 1 because in many cases a size_level of 2 does
    // worse. Investigate why.
    AddOptimizationPasses(&module_passes, &function_passes, /*opt_level=*/2,
                          /*size_level=*/1);
  } else {
    AddOptimizationPasses(&module_passes, &function_passes,
                          /*opt_level=*/opt_level_, /*size_level=*/0);
  }

  // Run optimization passes on module.
  function_passes.doInitialization();

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  for (auto func = module.begin(); func != module.end(); ++func) {
    function_passes.run(*func);
  }
  function_passes.doFinalization();
  module_passes.run(module);

  CHECK(!llvm::verifyModule(module, &llvm::dbgs()));

  runtime::RewriteIRRuntimeFunctions(&module, fast_math_flags_);

  // Buffer for holding machine code prior to constructing the ObjectFile.
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);

  VLOG(2) << "IR after optimizations";
  XLA_VLOG_LINES(2, llvm_ir::DumpModuleToString(module));

  if (post_optimization_hook_) {
    post_optimization_hook_(module);
  }

  // Generate code.
  llvm::MCContext* mc_context;
  llvm::legacy::PassManager codegen_passes;
  target_machine_->addPassesToEmitMC(codegen_passes, mc_context, ostream);
  codegen_passes.run(module);

  std::unique_ptr<llvm::MemoryBuffer> memory_buffer(
      new llvm::SmallVectorMemoryBuffer(std::move(stream_buffer)));

  if (post_codegen_hook_) {
    llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> obj_file =
        llvm::object::ObjectFile::createObjectFile(*memory_buffer);
    if (obj_file) {
      post_codegen_hook_(*obj_file.get());
    } else {
      LOG(WARNING) << "Could convert memory buffer to object file!";
    }
  }

  return std::move(memory_buffer);
}

static std::vector<llvm::VecDesc> VectorFunctionsForTargetLibraryInfoImpl() {
  std::vector<llvm::VecDesc> result = {
      {"tanhf", runtime::kTanhV4F32SymbolName, llvm::ElementCount::getFixed(4)},
      {"llvm.tanh.f32", runtime::kTanhV4F32SymbolName,
       llvm::ElementCount::getFixed(4)},

      {"tanhf", runtime::kTanhV8F32SymbolName, llvm::ElementCount::getFixed(8)},
      {"llvm.tanh.f32", runtime::kTanhV8F32SymbolName,
       llvm::ElementCount::getFixed(8)},

      {"tanhf", runtime::kTanhV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},
      {"llvm.tanh.f32", runtime::kTanhV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},

      {"expf", runtime::kExpV4F32SymbolName, llvm::ElementCount::getFixed(4)},
      {"llvm.exp.f32", runtime::kExpV4F32SymbolName,
       llvm::ElementCount::getFixed(4)},

      {"expf", runtime::kExpV8F32SymbolName, llvm::ElementCount::getFixed(8)},
      {"llvm.exp.f32", runtime::kExpV8F32SymbolName,
       llvm::ElementCount::getFixed(8)},

      {"expf", runtime::kExpV16F32SymbolName, llvm::ElementCount::getFixed(16)},
      {"llvm.exp.f32", runtime::kExpV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},

      {"logf", runtime::kLogV4F32SymbolName, llvm::ElementCount::getFixed(4)},
      {"llvm.log.f32", runtime::kLogV4F32SymbolName,
       llvm::ElementCount::getFixed(4)},

      {"logf", runtime::kLogV8F32SymbolName, llvm::ElementCount::getFixed(8)},
      {"llvm.log.f32", runtime::kLogV8F32SymbolName,
       llvm::ElementCount::getFixed(8)},

      {"logf", runtime::kLogV16F32SymbolName, llvm::ElementCount::getFixed(16)},
      {"llvm.log.f32", runtime::kLogV16F32SymbolName,
       llvm::ElementCount::getFixed(16)},
  };
  return result;
}

void CompilerFunctor::AddTargetInfoPasses(
    llvm::legacy::PassManagerBase* passes) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc mht_2(mht_2_v, 393, "", "./tensorflow/compiler/xla/service/cpu/compiler_functor.cc", "CompilerFunctor::AddTargetInfoPasses");

  llvm::Triple target_triple(target_machine_->getTargetTriple());
  auto target_library_info_impl =
      absl::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  target_library_info_impl->addVectorizableFunctions(
      VectorFunctionsForTargetLibraryInfoImpl());

  passes->add(
      new llvm::TargetLibraryInfoWrapperPass(*target_library_info_impl));
  passes->add(createTargetTransformInfoWrapperPass(
      target_machine_->getTargetIRAnalysis()));
}

void CompilerFunctor::AddOptimizationPasses(
    llvm::legacy::PassManagerBase* module_passes,
    llvm::legacy::FunctionPassManager* function_passes, unsigned opt_level,
    unsigned size_level) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScompiler_functorDTcc mht_3(mht_3_v, 412, "", "./tensorflow/compiler/xla/service/cpu/compiler_functor.cc", "CompilerFunctor::AddOptimizationPasses");

  llvm::PassManagerBuilder builder;
  builder.OptLevel = opt_level;
  builder.SizeLevel = size_level;

  if (opt_level > 1) {
    builder.Inliner = llvm::createFunctionInliningPass();
  } else {
    // Only inline functions marked with "alwaysinline".
    builder.Inliner = llvm::createAlwaysInlinerLegacyPass();
  }

  builder.DisableUnrollLoops = opt_level == 0;
  builder.LoopVectorize = opt_level > 0 && size_level == 0;
  builder.SLPVectorize = opt_level > 1 && size_level == 0;

  builder.populateFunctionPassManager(*function_passes);
  builder.populateModulePassManager(*module_passes);
}

}  // namespace cpu
}  // namespace xla
