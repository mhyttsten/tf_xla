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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh() {
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


#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace cpu {

// Simplified LLVM JIT based on the new Orc API.
//
// This class wraps Orc's functionality into a single interface that only
// exposes what we need for XLA.
//
// Supports JIT-ing multiple modules but without cross-module linking.
// Implements eager compilation - the module is lowered to binary as soon as
// it's added to the JIT.
class SimpleOrcJIT : public llvm::JITEventListener {
 public:
  using ObjLayerT = llvm::orc::RTDyldObjectLinkingLayer;
  using CompileLayerT = llvm::orc::IRCompileLayer;

  // Create a new JIT, targeting the host architecture.
  //
  // {pre,post}_optimization_hook is invoked on the module before/after all
  // LLVM IR-level optimizations.  post_codegen_hook is invoked after
  // compiling to machine code.
  SimpleOrcJIT(
      std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control,
      std::unique_ptr<llvm::orc::ExecutionSession> execution_session,
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
      bool disable_expensive_passes, llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook,
      LLVMCompiler::ModuleHook post_optimization_hook,
      std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook);

  static llvm::Expected<std::unique_ptr<SimpleOrcJIT>> Create(
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOpt::Level opt_level, bool optimize_for_size,
      bool disable_expensive_passes, llvm::FastMathFlags fast_math_flags,
      LLVMCompiler::ModuleHook pre_optimization_hook,
      LLVMCompiler::ModuleHook post_optimization_hook,
      std::function<void(const llvm::object::ObjectFile&)> post_codegen_hook);

  ~SimpleOrcJIT() override;

  const llvm::DataLayout& data_layout() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh mht_0(mht_0_v, 245, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.h", "data_layout");
 return data_layout_; }

  const llvm::Triple& target_triple() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh mht_1(mht_1_v, 250, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.h", "target_triple");
 return target_triple_; }

  llvm::Error AddModule(llvm::orc::ThreadSafeModule module);

  // Discards objects we no longer need once we are done compiling.
  void DoneCompiling();

  // Get the runtime address of the compiled symbol whose name is given. Returns
  // nullptr if the symbol cannot be found.
  llvm::Expected<llvm::JITEvaluatedSymbol> FindCompiledSymbol(
      const std::string& name);

  llvm::TargetMachine* target_machine() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh mht_2(mht_2_v, 265, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.h", "target_machine");
 return target_machine_.get(); }

  // Creates an llvm::TargetMachine suitable for JITting code that will run on
  // the current machine.
  static std::unique_ptr<llvm::TargetMachine> InferTargetMachineForJIT(
      const llvm::TargetOptions& target_options,
      llvm::CodeGenOpt::Level opt_level);

  int64_t SizeOfGeneratedCodeInBytes() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPSsimple_orc_jitDTh mht_3(mht_3_v, 276, "", "./tensorflow/compiler/xla/service/cpu/simple_orc_jit.h", "SizeOfGeneratedCodeInBytes");

    return size_of_generated_code_in_bytes_;
  }

 private:
  llvm::JITEvaluatedSymbol ResolveRuntimeSymbol(llvm::StringRef name);

  void notifyObjectLoaded(
      llvm::JITEventListener::ObjectKey key,
      const llvm::object::ObjectFile& object,
      const llvm::RuntimeDyld::LoadedObjectInfo& object_info) override;
  void notifyFreeingObject(llvm::JITEventListener::ObjectKey key) override;

  std::unique_ptr<llvm::TargetMachine> target_machine_;
  llvm::Triple target_triple_;
  const llvm::DataLayout data_layout_;
  std::unique_ptr<llvm::orc::ExecutorProcessControl> target_process_control_;
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session_;
  ObjLayerT object_layer_;
  CompileLayerT compile_layer_;
  llvm::orc::JITDylib* main_jit_dylib_;
  int64_t size_of_generated_code_in_bytes_ = 0;

  // Non owning pointer to a JIT event listener that registers the JIT events
  // with an attached GDB.
  //
  // Note: we get a pointer to this event listener using
  // `createGDBRegistrationListener` which makes it look like we're supposed to
  // free this, but the function is poorly named and really just returns a
  // pointer to a static object.
  llvm::JITEventListener* gdb_jit_event_listener_;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_SIMPLE_ORC_JIT_H_
