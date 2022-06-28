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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"

#include <stdint.h>

#include <algorithm>
#include <cstdio>
#include <list>
#include <utility>

#include "absl/memory/memory.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Host.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/orc_jit_memory_mapper.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv2d_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_conv3d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_custom_call_status.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fft.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fork_join.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_fp16.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_key_value_sort.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul_mkl.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_pow.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv2d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_conv3d.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_fft.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_topk.h"
#include "tensorflow/compiler/xla/service/cpu/windows_compatibility.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace cpu {
namespace {

llvm::SmallVector<std::string, 0> DetectMachineAttributes() {
  llvm::SmallVector<std::string, 0> result;
  llvm::StringMap<bool> host_features;
  if (llvm::sys::getHostCPUFeatures(host_features)) {
    for (auto& feature : host_features) {
      result.push_back((feature.second ? '+' : '-') +
                       std::string(feature.first()));
    }
  }
  return result;
}

}  // namespace

/*static*/ std::unique_ptr<llvm::TargetMachine>
SimpleOrcJIT::InferTargetMachineForJIT(
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOpt::Level opt_level) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_0(mht_0_v, 247, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::InferTargetMachineForJIT");

  std::unique_ptr<llvm::TargetMachine> target_machine(
      llvm::EngineBuilder()
          .setTargetOptions(target_options)
          .setOptLevel(opt_level)
          .selectTarget(
              /*TargetTriple=*/llvm::Triple(), /*MArch=*/"",
              /*MCPU=*/llvm::sys::getHostCPUName(),
              /*MAttrs=*/DetectMachineAttributes()));
  CHECK(target_machine != nullptr);
  return target_machine;
}

SimpleOrcJIT::SimpleOrcJIT(
    std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control,
    std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
    bool disable_expensive_passes, llvm::FastMathFlags fast_math_flags,
    LLVMCompiler::ModuleHook pre_optimization_hook,
    LLVMCompiler::ModuleHook post_optimization_hook,
    std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook)
    : target_machine_(InferTargetMachineForJIT(target_options, opt_level)),
      target_triple_(target_machine_->getTargetTriple()),
      data_layout_(target_machine_->createDataLayout()),
      target_process_control_(std::move(target_process_control)),
      execution_session_(std::move(execution_session)),
      object_layer_(*execution_session_,
                    []() {
                      return std::make_unique<llvm::SectionMemoryManager>(
                          orc_jit_memory_mapper::GetInstance());
                    }),
      compile_layer_(
          *execution_session_, object_layer_,
          std::make_unique<CompilerFunctor>(
              target_machine_.get(), opt_level, optimize_for_size,
              disable_expensive_passes, fast_math_flags,
              std::move(pre_optimization_hook),
              std::move(post_optimization_hook), std::move(post_codegen_hook))),
      main_jit_dylib_(&execution_session_->createBareJITDylib("<main>")),
      gdb_jit_event_listener_(
          llvm::JITEventListener::createGDBRegistrationListener()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_1(mht_1_v, 291, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::SimpleOrcJIT");

  VLOG(1) << "CPU target: " << target_machine_->getTargetCPU().str()
          << " features: " << target_machine_->getTargetFeatureString().str();

  // Materialize unknown symbols from the runtime symbol table.
  class RuntimeSymbolGenerator : public llvm::orc::DefinitionGenerator {
    SimpleOrcJIT& jit_;

   public:
    explicit RuntimeSymbolGenerator(SimpleOrcJIT& jit) : jit_(jit) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_2(mht_2_v, 303, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "RuntimeSymbolGenerator");
}
    llvm::Error tryToGenerate(
        llvm::orc::LookupState&, llvm::orc::LookupKind,
        llvm::orc::JITDylib& jit_dylib, llvm::orc::JITDylibLookupFlags,
        const llvm::orc::SymbolLookupSet& names) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "tryToGenerate");

      llvm::orc::SymbolMap new_defs;

      for (const auto& kv : names) {
        const auto& name = kv.first;
        if (llvm::JITEvaluatedSymbol symbol =
                jit_.ResolveRuntimeSymbol(*name)) {
          new_defs[name] = symbol;
        }
      }

      cantFail(jit_dylib.define(absoluteSymbols(std::move(new_defs))));
      return llvm::Error::success();
    }
  };
  main_jit_dylib_->addGenerator(
      std::make_unique<RuntimeSymbolGenerator>(*this));
  object_layer_.registerJITEventListener(*this);

  // Copied from LLJIT, required to find symbols on Windows.
  if (target_triple_.isOSBinFormatCOFF()) {
    object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
    object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
  }
}

SimpleOrcJIT::~SimpleOrcJIT() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_4(mht_4_v, 339, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::~SimpleOrcJIT");

  if (auto err = execution_session_->endSession()) {
    execution_session_->reportError(std::move(err));
  }
}

llvm::Expected<std::unique_ptr<SimpleOrcJIT>> SimpleOrcJIT::Create(
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
    bool disable_expensive_passes, llvm::FastMathFlags fast_math_flags,
    LLVMCompiler::ModuleHook pre_optimization_hook,
    LLVMCompiler::ModuleHook post_optimization_hook,
    std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_5(mht_5_v, 354, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::Create");

  auto SSP = std::make_shared<llvm::orc::SymbolStringPool>();
  auto target_process_control =
      llvm::orc::SelfExecutorProcessControl::Create(std::move(SSP));
  if (!target_process_control) {
    return target_process_control.takeError();
  }

  auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(
      std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>());
  return std::make_unique<SimpleOrcJIT>(
      std::move(*target_process_control), std::move(execution_session),
      target_options, opt_level, optimize_for_size, disable_expensive_passes,
      fast_math_flags, std::move(pre_optimization_hook),
      std::move(post_optimization_hook), std::move(post_codegen_hook));
}

llvm::JITEvaluatedSymbol SimpleOrcJIT::ResolveRuntimeSymbol(
    llvm::StringRef name) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_6(mht_6_v, 375, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::ResolveRuntimeSymbol");

  void* func_addr = nullptr;
  if (name.size() > 1 && name.front() == data_layout_.getGlobalPrefix()) {
    // On Mac OS X, 'name' may have a leading underscore prefix, even though the
    // registered name may not.
    std::string stripped_name(name.begin() + 1, name.end());
    func_addr =
        xla::CustomCallTargetRegistry::Global()->Lookup(stripped_name, "Host");
  } else {
    func_addr =
        xla::CustomCallTargetRegistry::Global()->Lookup(name.str(), "Host");
  }

  if (func_addr == nullptr) {
    LOG(ERROR)
        << "Unable to resolve runtime symbol: `" << name.str()
        << "'.  Hint: if the symbol a custom call target, make sure you've "
           "registered it with the JIT using "
           "XLA_CPU_REGISTER_CUSTOM_CALL_TARGET.";
    return nullptr;
  }
  llvm::JITEvaluatedSymbol symbol_info(reinterpret_cast<uint64_t>(func_addr),
                                       llvm::JITSymbolFlags::None);
  return symbol_info;
}

void SimpleOrcJIT::notifyObjectLoaded(
    llvm::JITEventListener::ObjectKey key,
    const llvm::object::ObjectFile& object,
    const llvm::RuntimeDyld::LoadedObjectInfo& object_info) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_7(mht_7_v, 407, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::notifyObjectLoaded");

  gdb_jit_event_listener_->notifyObjectLoaded(key, object, object_info);
  size_of_generated_code_in_bytes_ += object.getData().size();
}

void SimpleOrcJIT::notifyFreeingObject(llvm::JITEventListener::ObjectKey key) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_8(mht_8_v, 415, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::notifyFreeingObject");

  gdb_jit_event_listener_->notifyFreeingObject(key);
}

llvm::Error SimpleOrcJIT::AddModule(llvm::orc::ThreadSafeModule module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_9(mht_9_v, 422, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::AddModule");

  return compile_layer_.add(*main_jit_dylib_, std::move(module));
}

void SimpleOrcJIT::DoneCompiling() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_10(mht_10_v, 429, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::DoneCompiling");

  // The target machine takes a non-trivial amount of memory, so once we are
  // done compiling throw it away.
  target_machine_.reset();
}

llvm::Expected<llvm::JITEvaluatedSymbol> SimpleOrcJIT::FindCompiledSymbol(
    const std::string& name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_11(mht_11_v, 440, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "SimpleOrcJIT::FindCompiledSymbol");

  return execution_session_->lookup({main_jit_dylib_}, name);
}

#if defined(PLATFORM_WINDOWS)
// This function is used by compiler-generated code on windows, but it's not
// declared anywhere. The signature does not matter, we just need the address.
extern "C" void __chkstk(size_t);
#endif

namespace {
// Register some known symbols with the CustomCallTargetRegistry.
bool RegisterKnownJITSymbols() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTcc mht_12(mht_12_v, 455, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.cc", "RegisterKnownJITSymbols");

  xla::CustomCallTargetRegistry* registry =
      xla::CustomCallTargetRegistry::Global();
  registry->Register("printf", reinterpret_cast<void*>(&printf), "Host");
  registry->Register("puts", reinterpret_cast<void*>(&puts), "Host");

#define REGISTER_CPU_RUNTIME_SYMBOL(base_name)                               \
  do {                                                                       \
    auto* function_address =                                                 \
        reinterpret_cast<void*>(__xla_cpu_runtime_##base_name);              \
    registry->Register(xla::cpu::runtime::k##base_name##SymbolName,          \
                       function_address, "Host");                            \
    CHECK_EQ(absl::string_view(xla::cpu::runtime::k##base_name##SymbolName), \
             "__xla_cpu_runtime_" #base_name);                               \
  } while (false)

  REGISTER_CPU_RUNTIME_SYMBOL(AcquireInfeedBufferForDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(AcquireOutfeedBufferForPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(AllReduce);
  REGISTER_CPU_RUNTIME_SYMBOL(CollectivePermute);
  REGISTER_CPU_RUNTIME_SYMBOL(AllToAll);
  REGISTER_CPU_RUNTIME_SYMBOL(PartitionId);
  REGISTER_CPU_RUNTIME_SYMBOL(ReplicaId);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv2DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv3DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenConv3DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenFft);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulC64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulC128);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenMatMulS32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(MKLSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv2DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv2DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv3DF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedConv3DF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedFft);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF16);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF32);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulF64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulC64);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulC128);
  REGISTER_CPU_RUNTIME_SYMBOL(EigenSingleThreadedMatMulS32);
  REGISTER_CPU_RUNTIME_SYMBOL(ParallelForkJoin);
  REGISTER_CPU_RUNTIME_SYMBOL(PrintfToStderr);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseInfeedBufferAfterDequeue);
  REGISTER_CPU_RUNTIME_SYMBOL(ReleaseOutfeedBufferAfterPopulation);
  REGISTER_CPU_RUNTIME_SYMBOL(StatusIsSuccess);
  REGISTER_CPU_RUNTIME_SYMBOL(KeyValueSort);
  REGISTER_CPU_RUNTIME_SYMBOL(TopKF32);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingStart);
  REGISTER_CPU_RUNTIME_SYMBOL(TracingEnd);

  registry->Register("__gnu_f2h_ieee", reinterpret_cast<void*>(__gnu_f2h_ieee),
                     "Host");
  registry->Register("__gnu_h2f_ieee", reinterpret_cast<void*>(__gnu_h2f_ieee),
                     "Host");
  registry->Register("__truncdfhf2", reinterpret_cast<void*>(__truncdfhf2),
                     "Host");
  registry->Register("__powisf2", reinterpret_cast<void*>(__powisf2), "Host");
  registry->Register("__powidf2", reinterpret_cast<void*>(__powidf2), "Host");

#undef REGISTER_CPU_RUNTIME_SYMBOL

// Register both the f32 (float) and f64 (double) versions of a libm symbol.
// Unfortunately the double versions are overloaded on some systems, e.g.
// Mac so we need an explicit cast. This requires passing the function signature
// for that case.
#define REGISTER_LIBM_SYMBOL(name, double_sig)                                 \
  do {                                                                         \
    registry->Register(#name "f", reinterpret_cast<void*>(name##f), "Host");   \
    registry->Register(#name,                                                  \
                       reinterpret_cast<void*>(static_cast<double_sig>(name)), \
                       "Host");                                                \
  } while (false)

  REGISTER_LIBM_SYMBOL(acos, double (*)(double));
  REGISTER_LIBM_SYMBOL(acosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(asin, double (*)(double));
  REGISTER_LIBM_SYMBOL(asinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan, double (*)(double));
  REGISTER_LIBM_SYMBOL(atan2, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(atanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(cbrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(ceil, double (*)(double));
  REGISTER_LIBM_SYMBOL(copysign, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(cos, double (*)(double));
  REGISTER_LIBM_SYMBOL(cosh, double (*)(double));
  REGISTER_LIBM_SYMBOL(erf, double (*)(double));
  REGISTER_LIBM_SYMBOL(erfc, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp, double (*)(double));
  REGISTER_LIBM_SYMBOL(exp2, double (*)(double));
  REGISTER_LIBM_SYMBOL(expm1, double (*)(double));
  REGISTER_LIBM_SYMBOL(fabs, double (*)(double));
  REGISTER_LIBM_SYMBOL(fdim, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(floor, double (*)(double));
  REGISTER_LIBM_SYMBOL(fma, double (*)(double, double, double));
  REGISTER_LIBM_SYMBOL(fmax, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmin, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(fmod, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(frexp, double (*)(double, int*));
  REGISTER_LIBM_SYMBOL(hypot, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(ilogb, int (*)(double));
  REGISTER_LIBM_SYMBOL(ldexp, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(lgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(llrint, long long (*)(double));   // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(llround, long long (*)(double));  // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(log, double (*)(double));
  REGISTER_LIBM_SYMBOL(log10, double (*)(double));
  REGISTER_LIBM_SYMBOL(log1p, double (*)(double));
  REGISTER_LIBM_SYMBOL(log2, double (*)(double));
  REGISTER_LIBM_SYMBOL(logb, double (*)(double));
  REGISTER_LIBM_SYMBOL(lrint, long (*)(double));   // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(lround, long (*)(double));  // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(modf, double (*)(double, double*));
  REGISTER_LIBM_SYMBOL(nan, double (*)(const char*));
  REGISTER_LIBM_SYMBOL(nearbyint, double (*)(double));
  REGISTER_LIBM_SYMBOL(nextafter, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(nexttoward, double (*)(double, long double));
  REGISTER_LIBM_SYMBOL(pow, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remainder, double (*)(double, double));
  REGISTER_LIBM_SYMBOL(remquo, double (*)(double, double, int*));
  REGISTER_LIBM_SYMBOL(rint, double (*)(double));
  REGISTER_LIBM_SYMBOL(round, double (*)(double));
  REGISTER_LIBM_SYMBOL(scalbln,
                       double (*)(double, long));  // NOLINT(runtime/int)
  REGISTER_LIBM_SYMBOL(scalbn, double (*)(double, int));
  REGISTER_LIBM_SYMBOL(sin, double (*)(double));
#ifdef __APPLE__
  REGISTER_LIBM_SYMBOL(__sincos, void (*)(double, double*, double*));
  registry->Register("__sincosf_stret",
                     reinterpret_cast<void*>(__sincosf_stret), "Host");
  registry->Register("__sincos_stret", reinterpret_cast<void*>(__sincos_stret),
                     "Host");
#else
  REGISTER_LIBM_SYMBOL(sincos, void (*)(double, double*, double*));
#endif
  REGISTER_LIBM_SYMBOL(sinh, double (*)(double));
  REGISTER_LIBM_SYMBOL(sqrt, double (*)(double));
  REGISTER_LIBM_SYMBOL(tan, double (*)(double));
  REGISTER_LIBM_SYMBOL(tanh, double (*)(double));
  REGISTER_LIBM_SYMBOL(tgamma, double (*)(double));
  REGISTER_LIBM_SYMBOL(trunc, double (*)(double));

#undef REGISTER_LIBM_SYMBOL

  registry->Register("memcpy", reinterpret_cast<void*>(memcpy), "Host");
  registry->Register("memmove", reinterpret_cast<void*>(memmove), "Host");
  registry->Register("memset", reinterpret_cast<void*>(memset), "Host");

#ifdef __APPLE__
  registry->Register("__bzero", reinterpret_cast<void*>(bzero), "Host");
  registry->Register("bzero", reinterpret_cast<void*>(bzero), "Host");
  registry->Register("memset_pattern16",
                     reinterpret_cast<void*>(memset_pattern16), "Host");
#endif

#ifdef MEMORY_SANITIZER
  registry->Register("__msan_unpoison",
                     reinterpret_cast<void*>(__msan_unpoison), "Host");
#endif

#if defined(PLATFORM_WINDOWS)
  registry->Register("__chkstk", reinterpret_cast<void*>(__chkstk), "Host");
#endif

  return true;
}

bool unused = RegisterKnownJITSymbols();
}  // namespace

}  // namespace cpu
}  // namespace xla
