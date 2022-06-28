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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
Status KernelSupportLibrary::ForWithStatus(
    absl::string_view name, llvm::Value* start, llvm::Value* end,
    llvm::Value* step,
    const std::function<Status(llvm::Value*, bool)>& for_body_generator) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.cc", "KernelSupportLibrary::ForWithStatus");

  return IfWithStatus(b_->CreateICmpSLT(start, end), [&]() -> Status {
    TF_RETURN_IF_ERROR(for_body_generator(start, /*is_first_iteration=*/true));
    return ForWithStatus(
        name, b_->CreateAdd(start, step), end, step,
        [&](llvm::Value* iv) { return for_body_generator(iv, false); });
  });
}

Status KernelSupportLibrary::ForWithStatus(
    absl::string_view name, llvm::Value* start, llvm::Value* end,
    llvm::Value* step, bool peel_first_iteration,
    const std::function<Status(llvm::Value*, llvm::Value*)>&
        for_body_generator) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.cc", "KernelSupportLibrary::ForWithStatus");

  if (peel_first_iteration) {
    return ForWithStatus(
        name, start, end, step, true,
        [&](llvm::Value* indvar, bool is_first_iteration) -> Status {
          return for_body_generator(indvar, b_->getInt1(is_first_iteration));
        });
  } else {
    std::unique_ptr<llvm_ir::ForLoop> loop = llvm_ir::ForLoop::EmitForLoop(
        name, start, end, step, b_,
        /*unroll_mode=*/unroll_mode_,
        /*prevent_vectorization=*/prevent_vectorization_);
    b_->SetInsertPoint(&loop->GetBodyBasicBlock()->back());
    TF_RETURN_IF_ERROR(
        for_body_generator(loop->GetIndVarValue(),
                           /*is_first_iteration=*/b_->CreateICmpEQ(
                               loop->GetIndVarValue(), start)));
    llvm_ir::SetToLastInsertPoint(loop->GetExitBasicBlock(), b_);
    return Status::OK();
  }
}

Status KernelSupportLibrary::IfWithStatus(
    absl::string_view name, llvm::Value* condition,
    const std::function<Status()>& true_block_generator,
    const std::function<Status()>& false_block_generator) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.cc", "KernelSupportLibrary::IfWithStatus");

  llvm_ir::LlvmIfData if_data =
      llvm_ir::EmitIfThenElse(condition, name, b_,
                              /*emit_else=*/false_block_generator != nullptr);
  b_->SetInsertPoint(&if_data.true_block->back());
  TF_RETURN_IF_ERROR(true_block_generator());
  if (false_block_generator != nullptr) {
    b_->SetInsertPoint(&if_data.false_block->back());
    TF_RETURN_IF_ERROR(false_block_generator());
  }
  llvm_ir::SetToLastInsertPoint(if_data.after_block, b_);
  return Status::OK();
}

void KernelSupportLibrary::EmitAndCallOutlinedKernel(
    const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
    absl::string_view kernel_name,
    KernelSupportLibrary::ArgumentVector arguments,
    const std::function<void(KernelSupportLibrary::ArgumentVector)>&
        kernel_body_generator) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSllvm_irPSkernel_support_libraryDTcc mht_3(mht_3_v, 264, "", "./tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.cc", "KernelSupportLibrary::EmitAndCallOutlinedKernel");

  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Function* function =
      module->getFunction(llvm_ir::AsStringRef(kernel_name));

  int64_t null_arg_idx = -1;
  std::vector<llvm::Value*> sanitized_args;
  sanitized_args.reserve(arguments.size());
  for (int64_t i = 0, e = arguments.size(); i < e; i++) {
    if (arguments[i]) {
      sanitized_args.push_back(arguments[i]);
    } else {
      CHECK_EQ(null_arg_idx, -1);
      null_arg_idx = i;
    }
  }

  if (!function) {
    VLOG(2) << "Generating kernel for " << kernel_name;
    std::vector<llvm::Type*> arg_types;
    std::transform(sanitized_args.begin(), sanitized_args.end(),
                   std::back_inserter(arg_types),
                   [](llvm::Value* arg) { return arg->getType(); });

    auto* function_type =
        llvm::FunctionType::get(b->getVoidTy(), arg_types, /*isVarArg=*/false);

    function = llvm_ir::CreateCpuFunction(function_type,
                                          llvm::GlobalValue::InternalLinkage,
                                          module_config, kernel_name, module);

    llvm::IRBuilder<>::InsertPointGuard guard(*b);

    auto* entry_bb =
        llvm::BasicBlock::Create(b->getContext(), "entry", function);
    auto* return_inst = llvm::ReturnInst::Create(b->getContext(),
                                                 /*retVal=*/nullptr, entry_bb);
    // Set the insert point to before return_inst.
    b->SetInsertPoint(return_inst);

    std::vector<llvm::Value*> arg_values;
    /*
     * clang on OSX doesn't like std::transform or range for loop here.
     * See https://github.com/tensorflow/tensorflow/issues/15196
     */
    for (llvm::Function::arg_iterator arg = function->arg_begin(),
                                      arg_e = function->arg_end();
         arg != arg_e; ++arg) {
      arg_values.push_back(arg);
    }
    if (null_arg_idx != -1) {
      arg_values.insert(arg_values.begin() + null_arg_idx, nullptr);
    }
    kernel_body_generator(arg_values);
  } else {
    VLOG(3) << "Re-using kernel for " << kernel_name;
  }

  b->CreateCall(function, llvm_ir::AsArrayRef(sanitized_args));
}

}  // namespace xla
