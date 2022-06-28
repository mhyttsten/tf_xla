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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc() {
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

#include "tensorflow/compiler/xla/service/llvm_compiler.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Creating dummy data structure needed to initialize a GpuDummyCompiler
PLATFORM_DEFINE_ID(kDummyTestId);
constexpr char kDummyTriple[] = "dummy-triple";
constexpr char kDummyLayout[] = "e";

// This class is a dummy implementation of GpuCompiler and is targeted for unit
// test only
class GpuDummyCompiler : public GpuCompiler {
 public:
  GpuDummyCompiler() : GpuCompiler(kDummyTestId, kDummyTriple, kDummyLayout) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "GpuDummyCompiler");
}

  Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "OptimizeHloConvolutionCanonicalization");

    return Status::OK();
  }

  Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      se::DeviceMemoryAllocator* device_allocator) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "OptimizeHloPostLayoutAssignment");

    return Status::OK();
  }

  GpuVersion GetGpuVersion(se::StreamExecutor*) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "GetGpuVersion");

    return se::CudaComputeCapability{0, 0};
  }

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      GpuVersion gpu_version, se::StreamExecutor* stream_exec, bool relocatable,
      const HloModule* debug_module) {
    std::vector<uint8_t> compiled_results;
    return std::pair<std::string, std::vector<uint8_t>>(
        "", std::move(compiled_results));
  }
};
}  // namespace gpu

namespace {

class LLVMCompilerTest : public ::testing::Test {
 public:
  void SetUp() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_4(mht_4_v, 256, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "SetUp");

    Platform* platform = FindPlatform();
    ASSERT_NE(platform, nullptr);

    BackendOptions backend_options;
    backend_options.set_platform(platform);
    StatusOr<std::unique_ptr<Backend>> backend_or_status =
        Backend::CreateBackend(backend_options);
    ASSERT_IS_OK(backend_or_status.status());
    backend_ = backend_or_status.ConsumeValueOrDie();
  }

  ~LLVMCompilerTest() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_5(mht_5_v, 271, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "~LLVMCompilerTest");
}

 protected:
  using Platform = se::Platform;

  explicit LLVMCompilerTest(std::string platform_name)
      : platform_name_(std::move(platform_name)) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("platform_name: \"" + platform_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_6(mht_6_v, 281, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "LLVMCompilerTest");
}

  void TestCompilerHooks(LLVMCompiler* compiler) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_7(mht_7_v, 286, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "TestCompilerHooks");

    int pre_opt_hook_call_count = 0;
    int post_opt_hook_call_count = 0;

    auto pre_opt_hook = [&pre_opt_hook_call_count](const llvm::Module&) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_8(mht_8_v, 293, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "lambda");

      ++pre_opt_hook_call_count;
      return Status::OK();
    };
    auto post_opt_hook = [&post_opt_hook_call_count](const llvm::Module&) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_9(mht_9_v, 300, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "lambda");

      ++post_opt_hook_call_count;
      return Status::OK();
    };

    // Create HLO module, and run the compiler.
    auto builder = HloComputation::Builder(TestName());
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

    auto hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(builder.Build());

    compiler->SetPreOptimizationHook(pre_opt_hook);
    compiler->SetPostOptimizationHook(post_opt_hook);

    ASSERT_TRUE(compiler
                    ->RunBackend(std::move(hlo_module),
                                 backend_->default_stream_executor(),
                                 /*device_allocator=*/nullptr)
                    .ok());

    // Test that hooks were called.
    EXPECT_EQ(1, pre_opt_hook_call_count);
    EXPECT_EQ(1, post_opt_hook_call_count);
  }

  void TestMultiModuleCompilation(LLVMCompiler* compiler) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_10(mht_10_v, 330, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "TestMultiModuleCompilation");

    HloComputation::Builder builder(TestName());
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

    std::unique_ptr<HloModule> hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(builder.Build());

    auto module_group = absl::make_unique<HloModuleGroup>("test_module_group");
    module_group->push_back(hlo_module->Clone());
    module_group->push_back(std::move(hlo_module));

    std::vector<std::vector<se::StreamExecutor*>> executors;
    executors.push_back({backend_->default_stream_executor()});
    executors.push_back({backend_->default_stream_executor()});

    EXPECT_IS_OK(compiler->Compile(std::move(module_group),
                                   std::move(executors),
                                   /*device_allocator=*/nullptr));
  }

 private:
  Platform* FindPlatform() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_11(mht_11_v, 355, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "FindPlatform");

    auto status_or_platform = PlatformUtil::GetPlatform(platform_name_);
    return status_or_platform.ok() ? status_or_platform.ValueOrDie() : nullptr;
  }

  std::string platform_name_;
  std::unique_ptr<Backend> backend_;

  static std::string TestName() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_12(mht_12_v, 366, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "TestName");

    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  std::unique_ptr<HloModule> CreateNewVerifiedModule() {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    return absl::make_unique<VerifiedHloModule>(
        TestName(), config, /*verifier_layout_sensitive=*/false,
        /*allow_mixed_precision_in_hlo_verifier=*/true,
        backend_->compiler()->ShapeSizeBytesFunction());
  }
};

class CpuCompilerTest : public LLVMCompilerTest {
 public:
  CpuCompilerTest() : LLVMCompilerTest("Host") {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_13(mht_13_v, 385, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "CpuCompilerTest");
}
};

class GpuCompilerTest : public LLVMCompilerTest {
 public:
  GpuCompilerTest() : LLVMCompilerTest("GPU") {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSllvm_compiler_testDTcc mht_14(mht_14_v, 393, "", "./tensorflow/compiler/xla/tests/llvm_compiler_test.cc", "GpuCompilerTest");
}
};

TEST_F(CpuCompilerTest, HooksTest) {
  cpu::CpuCompiler compiler;
  TestCompilerHooks(&compiler);
}

TEST_F(GpuCompilerTest, HooksTest) {
  gpu::GpuDummyCompiler compiler;
  TestCompilerHooks(&compiler);
}

TEST_F(CpuCompilerTest, CpuMultiModuleCompilation) {
  cpu::CpuCompiler compiler;
  TestMultiModuleCompilation(&compiler);
}

TEST_F(GpuCompilerTest, GpuMultModuleCompilation) {
  gpu::GpuDummyCompiler compiler;
  TestMultiModuleCompilation(&compiler);
}
}  // namespace
}  // namespace xla
