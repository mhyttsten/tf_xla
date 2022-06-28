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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc() {
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

#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

void LlvmIrGenTestBase::SetIrHook(bool match_optimized_ir) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_0(mht_0_v, 197, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::SetIrHook");

  auto llvm_compiler = GetLLVMCompiler();
  using std::placeholders::_1;

  // Add the IR inspection hook to the LLVM compiler.
  if (match_optimized_ir) {
    llvm_compiler->SetPostOptimizationHook(
        std::bind(&LlvmIrGenTestBase::IrHook, this, _1));
  } else {
    llvm_compiler->SetPreOptimizationHook(
        std::bind(&LlvmIrGenTestBase::IrHook, this, _1));
  }
}

void LlvmIrGenTestBase::ResetIrHook() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_1(mht_1_v, 214, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::ResetIrHook");

  auto llvm_compiler = GetLLVMCompiler();

  llvm_compiler->RemovePreOptimizationHook();
  llvm_compiler->RemovePostOptimizationHook();
}

void LlvmIrGenTestBase::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const std::string& pattern,
    bool match_optimized_ir) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::CompileAndVerifyIr");

  SetIrHook(match_optimized_ir);
  Status status = CompileToExecutable(std::move(hlo_module)).status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie()) << "Full IR: " << ir_;
}

void LlvmIrGenTestBase::CompileAndVerifyIr(const std::string& hlo_text,
                                           const std::string& expected_llvm_ir,
                                           bool match_optimized_ir) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("hlo_text: \"" + hlo_text + "\"");
   mht_3_v.push_back("expected_llvm_ir: \"" + expected_llvm_ir + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_3(mht_3_v, 245, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::CompileAndVerifyIr");

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  CompileAndVerifyIr(std::move(module), expected_llvm_ir, match_optimized_ir);
}

void LlvmIrGenTestBase::CompileAheadOfTimeAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const AotCompilationOptions& options,
    const std::string& pattern, bool match_optimized_ir) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_4(mht_4_v, 259, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::CompileAheadOfTimeAndVerifyIr");

  SetIrHook(match_optimized_ir);
  Status status =
      CompileToAotCompilationResult(std::move(hlo_module), options).status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie()) << "Full IR: " << ir_;
}

void LlvmIrGenTestBase::MatchOptimizedHlo(absl::string_view hlo,
                                          absl::string_view pattern,
                                          bool print_operand_shape) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("hlo: \"" + std::string(hlo.data(), hlo.size()) + "\"");
   mht_5_v.push_back("pattern: \"" + std::string(pattern.data(), pattern.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_5(mht_5_v, 278, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::MatchOptimizedHlo");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo));
  HloPrintOptions print_opts;
  print_opts.set_print_operand_shape(print_operand_shape);
  StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module->ToString(print_opts), pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

StatusOr<std::unique_ptr<HloModule>> LlvmIrGenTestBase::GetOptimizedModule(
    absl::string_view hlo) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("hlo: \"" + std::string(hlo.data(), hlo.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_6(mht_6_v, 294, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::GetOptimizedModule");

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  return backend().compiler()->RunHloPasses(
      std::move(module), backend().default_stream_executor(),
      backend().default_stream_executor()->GetAllocator());
}

StatusOr<std::unique_ptr<HloModule>> LlvmIrGenTestBase::GetOptimizedModule(
    std::unique_ptr<HloModule> hlo_module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_7(mht_7_v, 307, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::GetOptimizedModule");

  return backend().compiler()->RunHloPasses(
      std::move(hlo_module), backend().default_stream_executor(),
      backend().default_stream_executor()->GetAllocator());
}

LLVMCompiler* LlvmIrGenTestBase::GetLLVMCompiler() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_8(mht_8_v, 316, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::GetLLVMCompiler");

  return static_cast<LLVMCompiler*>(backend().compiler());
}

Status LlvmIrGenTestBase::IrHook(const llvm::Module& module) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_irgen_test_baseDTcc mht_9(mht_9_v, 323, "", "./tensorflow/compiler/xla/tests/llvm_irgen_test_base.cc", "LlvmIrGenTestBase::IrHook");

  ir_ = llvm_ir::DumpModuleToString(module);
  return Status::OK();
}

}  // namespace xla
