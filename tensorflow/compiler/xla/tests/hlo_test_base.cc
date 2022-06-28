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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc() {
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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

namespace {

using absl::optional;
using absl::string_view;

constexpr char kInterpreter[] = "interpreter";

bool ProgramShapesEqual(const ProgramShape& lhs, const ProgramShape& rhs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "ProgramShapesEqual");

  if (lhs.parameters_size() != rhs.parameters_size()) {
    return false;
  }
  for (int i = 0; i < lhs.parameters_size(); i++) {
    if (!ShapeUtil::Equal(lhs.parameters(i), rhs.parameters(i))) {
      return false;
    }
  }
  return ShapeUtil::Equal(lhs.result(), rhs.result());
}

ProgramShape GetProgramShapeWithLayout(const HloModule& module) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_1(mht_1_v, 233, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "GetProgramShapeWithLayout");

  ProgramShape program_shape;
  const auto* entry = module.entry_computation();
  for (const auto* param : entry->parameter_instructions()) {
    *program_shape.add_parameters() = param->shape();
    *program_shape.add_parameter_names() = param->name();
  }
  *program_shape.mutable_result() = entry->root_instruction()->shape();
  return program_shape;
}

}  // namespace

HloTestBase::HloTestBase(bool verifier_layout_sensitive,
                         bool allow_mixed_precision_in_hlo_verifier,
                         std::function<bool(const HloInstruction*)>
                             instruction_can_change_layout_func)
    : HloTestBase(GetTestPlatform(), GetReferencePlatform(),
                  verifier_layout_sensitive,
                  allow_mixed_precision_in_hlo_verifier,
                  instruction_can_change_layout_func) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_2(mht_2_v, 256, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::HloTestBase");
}

HloTestBase::HloTestBase(se::Platform* test_platform,
                         se::Platform* reference_platform,
                         bool verifier_layout_sensitive,
                         bool allow_mixed_precision_in_hlo_verifier,
                         std::function<bool(const HloInstruction*)>
                             instruction_can_change_layout_func)
    : test_runner_(test_platform),
      reference_runner_(reference_platform),
      verifier_layout_sensitive_(verifier_layout_sensitive),
      allow_mixed_precision_in_hlo_verifier_(
          allow_mixed_precision_in_hlo_verifier) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_3(mht_3_v, 271, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::HloTestBase");

  hlo_verifier_ = absl::make_unique<HloVerifier>(
      /*layout_sensitive=*/verifier_layout_sensitive,
      /*allow_mixed_precision=*/allow_mixed_precision_in_hlo_verifier,
      instruction_can_change_layout_func);
}

/*static*/ se::Platform* HloTestBase::GetReferencePlatform() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::GetReferencePlatform");

  auto result = PlatformUtil::GetPlatform(kInterpreter);
  TF_CHECK_OK(result.status()) << "could not get interpreter platform";
  return result.ValueOrDie();
}

/*static*/ se::Platform* HloTestBase::GetTestPlatform() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_5(mht_5_v, 290, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::GetTestPlatform");

  auto result = PlatformUtil::GetDefaultPlatform();
  TF_CHECK_OK(result.status()) << "could not get test platform";
  return result.ValueOrDie();
}

std::unique_ptr<HloModule> HloTestBase::CreateNewUnverifiedModule(
    const std::string& name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_6(mht_6_v, 301, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::CreateNewUnverifiedModule");

  return absl::make_unique<HloModule>(name, GetModuleConfigForTest());
}

std::unique_ptr<VerifiedHloModule> HloTestBase::CreateNewVerifiedModule(
    const std::string& name, int64_t replica_count) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_7(mht_7_v, 310, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::CreateNewVerifiedModule");

  return absl::make_unique<VerifiedHloModule>(
      name, GetModuleConfigForTest(replica_count), verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_,
      backend().compiler()->ShapeSizeBytesFunction());
}

StatusOr<std::unique_ptr<VerifiedHloModule>>
HloTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                                          int64_t replica_count,
                                          int64_t num_partitions) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("hlo_text: \"" + std::string(hlo_text.data(), hlo_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_8(mht_8_v, 324, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ParseAndReturnVerifiedModule");

  return ParseAndReturnVerifiedModule(
      hlo_text, GetModuleConfigForTest(replica_count, num_partitions));
}

StatusOr<std::unique_ptr<VerifiedHloModule>>
HloTestBase::ParseAndReturnVerifiedModule(absl::string_view hlo_text,
                                          const HloModuleConfig& config) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("hlo_text: \"" + std::string(hlo_text.data(), hlo_text.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_9(mht_9_v, 335, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ParseAndReturnVerifiedModule");

  auto module = absl::make_unique<VerifiedHloModule>(
      TestName(), config, verifier_layout_sensitive_,
      allow_mixed_precision_in_hlo_verifier_,
      backend().compiler()->ShapeSizeBytesFunction());
  TF_RETURN_IF_ERROR(module->ParseHloStringAndVerifyModule(hlo_text));
  return std::move(module);
}

/* static */
StatusOr<bool> HloTestBase::RunHloPass(HloPassInterface* hlo_pass,
                                       HloModule* module) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_10(mht_10_v, 349, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunHloPass");

  const std::string module_str_before_run =
      module->ToProto().ShortDebugString();
  const auto status_or = hlo_pass->Run(module);
  if (status_or.status().ok()) {
    const std::string module_str_after_run =
        module->ToProto().ShortDebugString();
    if (!status_or.ValueOrDie()) {
      // Check that the proto remains same.
      EXPECT_EQ(module_str_after_run, module_str_before_run);
    }
  }
  return status_or;
}

/* static */
PrecisionConfig HloTestBase::DefaultPrecisionConfig(int operands) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_11(mht_11_v, 368, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::DefaultPrecisionConfig");

  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      operands, PrecisionConfig::DEFAULT);
  return precision_config;
}

void HloTestBase::SetAotFastMathDebugOptions(DebugOptions* options) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_12(mht_12_v, 378, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::SetAotFastMathDebugOptions");

  options->set_xla_cpu_enable_fast_math(true);
  options->set_xla_gpu_enable_fast_min_max(true);
  options->set_xla_cpu_enable_fast_min_max(true);
  options->set_xla_cpu_fast_math_honor_nans(false);
  options->set_xla_cpu_fast_math_honor_infs(false);
  options->set_xla_cpu_fast_math_honor_functions(false);
  options->set_xla_cpu_fast_math_honor_division(false);
}

DebugOptions HloTestBase::GetDebugOptionsForTest() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_13(mht_13_v, 391, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::GetDebugOptionsForTest");

  auto debug_options = GetDebugOptionsFromFlags();
  // TODO(b/38354253): Change tests to use Parameters instead of Constants.
  debug_options.add_xla_disable_hlo_passes("constant_folding");
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  debug_options.set_xla_hlo_evaluator_use_fast_path(true);
  return debug_options;
}

StatusOr<Literal> HloTestBase::Execute(std::unique_ptr<HloModule> module,
                                       absl::Span<Literal* const> arguments) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_14(mht_14_v, 404, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::Execute");

  return test_runner_.Execute(std::move(module), arguments);
}

Literal HloTestBase::ExecuteNoHloPasses(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_15(mht_15_v, 412, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ExecuteNoHloPasses");

  return test_runner_
      .Execute(std::move(module), arguments,
               /*run_hlo_passes=*/false)
      .ValueOrDie();
}

Literal HloTestBase::ExecuteAndTransfer(std::unique_ptr<HloModule> module,
                                        absl::Span<Literal* const> arguments) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_16(mht_16_v, 423, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ExecuteAndTransfer");

  return test_runner_.Execute(std::move(module), arguments).ValueOrDie();
}

StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    int64_t num_replicas, bool use_threads, bool run_hlo_passes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_17(mht_17_v, 432, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ExecuteReplicated");

  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  for (auto argument : arguments) {
    options.arguments.push_back(argument);
  }
  return test_runner_.ExecuteReplicated(std::move(module), options);
}

StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::unique_ptr<HloModule> module, absl::Span<Literal* const> arguments,
    int64_t num_replicas, DeviceAssignment* device_assignment,
    bool run_hlo_passes, bool use_threads) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_18(mht_18_v, 449, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ExecuteReplicated");

  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = use_threads;
  for (auto argument : arguments) {
    options.arguments.push_back(argument);
  }
  return test_runner_.ExecuteReplicated(std::move(module), options,
                                        device_assignment);
}

StatusOr<std::vector<Literal>> HloTestBase::ExecuteReplicated(
    std::function<Executable*(int64_t)> executable_provider,
    std::function<int64_t(int64_t)> argument_count_provider,
    std::function<const Literal*(int64_t, int64_t)> argument_provider,
    int64_t num_replicas, bool run_hlo_passes) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_19(mht_19_v, 468, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::ExecuteReplicated");

  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  return test_runner_.ExecuteReplicated(
      executable_provider, argument_count_provider, argument_provider, options);
}

StatusOr<std::unique_ptr<HloModule>> HloTestBase::MakeReferenceModule(
    const HloModule& test_module,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_20(mht_20_v, 482, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::MakeReferenceModule");

  std::unique_ptr<HloModule> reference_module = test_module.Clone();
  const auto& program_shape = GetProgramShapeWithLayout(test_module);

  if (reference_preprocessor != nullptr) {
    reference_preprocessor(reference_module.get());
    if (!ProgramShapesEqual(program_shape,
                            GetProgramShapeWithLayout(*reference_module))) {
      return InvalidArgument(
          "reference preprocessor must not modify the program shape");
    }
  }
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(reference_module.get()).status());
  return std::move(reference_module);
}

StatusOr<::testing::AssertionResult> HloTestBase::RunAndCompareInternal(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error, bool run_hlo_passes,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_21(mht_21_v, 505, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareInternal");

  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module.get()).status());
  TF_ASSIGN_OR_RETURN(auto reference_module,
                      MakeReferenceModule(*module, reference_preprocessor));

  // Execute on two backends.
  TF_ASSIGN_OR_RETURN(
      auto test,
      test_runner_.Execute(std::move(module), arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(auto reference,
                      reference_runner_.Execute(std::move(reference_module),
                                                arguments, run_hlo_passes));
  if (reference.IsAll(0)) {
    LOG(WARNING) << "Reference value is only zeros.";
  }

  return LiteralTestUtil::NearOrEqual(/*expected=*/reference, /*actual=*/test,
                                      error);
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_22(mht_22_v, 532, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompare");

  auto result =
      RunAndCompareInternal(std::move(module), arguments, error,
                            /*run_hlo_passes=*/true, reference_preprocessor);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.ValueOrDie();
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_23(mht_23_v, 549, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareNoHloPasses");

  auto result =
      RunAndCompareInternal(std::move(module), arguments, error,
                            /*run_hlo_passes=*/false, reference_preprocessor);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.ValueOrDie();
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    std::unique_ptr<HloModule> module, const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_24(mht_24_v, 564, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompare");

  auto fake_arguments = MakeFakeArguments(module.get()).ConsumeValueOrDie();

  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompare(std::move(module), fake_argument_ptrs, error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    std::unique_ptr<HloModule> module, const optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_25(mht_25_v, 581, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareNoHloPasses");

  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompareNoHloPasses(std::move(module), fake_argument_ptrs, error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::Run(std::unique_ptr<HloModule> module,
                                            bool run_hlo_passes) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_26(mht_26_v, 597, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::Run");

  const auto fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
  const auto change = hlo_verifier_->Run(module.get());
  if (!change.ok()) {
    return ::testing::AssertionFailure() << change.status();
  }

  const auto output =
      test_runner_.Execute(std::move(module), fake_arguments, run_hlo_passes);
  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().error_message();
}

::testing::AssertionResult HloTestBase::RunAndCompare(
    string_view hlo_string, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_27(mht_27_v, 618, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompare");

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompare(module_or_status.ConsumeValueOrDie(), error,
                       reference_preprocessor);
}

StatusOr<::testing::AssertionResult>
HloTestBase::RunAndCompareTwoModulesInternal(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const absl::Span<Literal* const> arguments,
    const absl::optional<ErrorSpec>& error, bool run_hlo_passes) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_28(mht_28_v, 636, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareTwoModulesInternal");

  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module_0.get()).status());
  TF_RETURN_IF_ERROR(hlo_verifier_->Run(module_1.get()).status());

  // Execute the two modules.
  TF_ASSIGN_OR_RETURN(
      auto test_0,
      test_runner_.Execute(std::move(module_0), arguments, run_hlo_passes));
  TF_ASSIGN_OR_RETURN(
      auto test_1,
      test_runner_.Execute(std::move(module_1), arguments, run_hlo_passes));

  return LiteralTestUtil::NearOrEqual(/*expected=*/test_0, /*actual=*/test_1,
                                      error);
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const absl::Span<Literal* const> arguments,
    const optional<ErrorSpec>& error) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_29(mht_29_v, 658, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareTwoModules");

  auto result = RunAndCompareTwoModulesInternal(
      std::move(module_0), std::move(module_1), arguments, error,
      /*run_hlo_passes=*/true);
  if (!result.ok()) {
    return ::testing::AssertionFailure() << result.status();
  }
  return result.ValueOrDie();
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    std::unique_ptr<HloModule> module_0, std::unique_ptr<HloModule> module_1,
    const optional<ErrorSpec>& error) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_30(mht_30_v, 673, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareTwoModules");

  const auto params_0 = module_0->entry_computation()->parameter_instructions();
  const auto params_1 = module_1->entry_computation()->parameter_instructions();
  for (int i = 0; i < params_0.size(); ++i) {
    const HloModuleConfig& module_config_0 = module_0->config();
    const Shape& param_shape_0 =
        (module_config_0.has_entry_computation_layout() &&
         module_config_0.entry_computation_layout()
             .parameter_layout(i)
             .shape()
             .is_static())
            ? module_config_0.entry_computation_layout()
                  .parameter_layout(i)
                  .shape()
            : params_0[i]->shape();

    const HloModuleConfig& module_config_1 = module_1->config();
    const Shape& param_shape_1 =
        (module_config_1.has_entry_computation_layout() &&
         module_config_1.entry_computation_layout()
             .parameter_layout(i)
             .shape()
             .is_static())
            ? module_config_1.entry_computation_layout()
                  .parameter_layout(i)
                  .shape()
            : params_1[i]->shape();

    if (!ShapeUtil::Equal(param_shape_0, param_shape_1)) {
      return ::testing::AssertionFailure()
             << "Error : mismatching parameter shapes: "
             << param_shape_0.ToString() << " Vs. " << param_shape_1.ToString();
    }
  }

  auto fake_arguments = MakeFakeArguments(module_0.get()).ConsumeValueOrDie();

  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  return RunAndCompareTwoModules(std::move(module_0), std::move(module_1),
                                 fake_argument_ptrs, error);
}

::testing::AssertionResult HloTestBase::RunAndCompareTwoModules(
    string_view hlo_string_module_0, string_view hlo_string_module_1,
    const absl::optional<ErrorSpec>& error) {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("hlo_string_module_0: \"" + std::string(hlo_string_module_0.data(), hlo_string_module_0.size()) + "\"");
   mht_31_v.push_back("hlo_string_module_1: \"" + std::string(hlo_string_module_1.data(), hlo_string_module_1.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_31(mht_31_v, 726, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareTwoModules");

  auto module_0_or_status = ParseAndReturnVerifiedModule(hlo_string_module_0);
  if (!module_0_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_0_or_status.status().ToString();
  }

  auto module_1_or_status = ParseAndReturnVerifiedModule(hlo_string_module_1);
  if (!module_1_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_1_or_status.status().ToString();
  }
  return RunAndCompareTwoModules(module_0_or_status.ConsumeValueOrDie(),
                                 module_1_or_status.ConsumeValueOrDie(), error);
}

::testing::AssertionResult HloTestBase::Run(
    string_view hlo_string, bool run_hlo_passes, ExecutionProfile* profile,
    const tensorflow::protobuf::Message* backend_config) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_32(mht_32_v, 750, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::Run");

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }

  std::unique_ptr<HloModule> module = std::move(module_or_status.ValueOrDie());
  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (profile != nullptr) {
    // We have to enable HLO profiling since otherwise currently the
    // ExecutionProfile is not correct.
    //
    // TODO(b/119432044): Fix collection of the ExecutionProfile
    // so that this is not necessary.
    HloModuleConfig config = module->config();
    DebugOptions debug_options = config.debug_options();
    debug_options.set_xla_hlo_profile(true);
    config.set_debug_options(debug_options);
    module->set_config(config);
  }

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        module->entry_computation()->root_instruction();
    Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.error_message();
  }

  auto output = test_runner_.Execute(std::move(module), fake_argument_ptrs,
                                     /*run_hlo_passes=*/run_hlo_passes,
                                     /*profile=*/profile);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().error_message();
}

::testing::AssertionResult HloTestBase::RunReplicated(
    string_view hlo_string, bool run_hlo_passes, int64_t num_replicas,
    const tensorflow::protobuf::Message* backend_config) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_33(mht_33_v, 803, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunReplicated");

  auto module_or_status =
      ParseAndReturnVerifiedModule(hlo_string, num_replicas);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }

  std::unique_ptr<HloModule> module = std::move(module_or_status.ValueOrDie());
  const auto& fake_arguments =
      MakeFakeArguments(module.get()).ConsumeValueOrDie();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  if (backend_config) {
    // Set backend configuration if it is given.
    HloInstruction* instruction =
        module->entry_computation()->root_instruction();
    Status s = instruction->set_backend_config(*backend_config);
    return s.ok() ? ::testing::AssertionSuccess()
                  : ::testing::AssertionFailure() << s.error_message();
  }

  HloRunner::ReplicatedExecuteOptions options;
  options.num_replicas = num_replicas;
  options.run_hlo_passes = run_hlo_passes;
  options.use_threads = true;
  for (auto argument : fake_argument_ptrs) {
    options.arguments.push_back(argument);
  }
  auto output = test_runner_.ExecuteReplicated(std::move(module), options);

  return output.ok()
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << output.status().error_message();
}

::testing::AssertionResult HloTestBase::RunMultipleTimes(
    string_view hlo_string, bool run_hlo_passes,
    std::vector<ExecutionProfile>* profiles,
    const tensorflow::protobuf::Message* backend_config,
    bool assert_determinism) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_34(mht_34_v, 851, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunMultipleTimes");

  int n = profiles->size();
  std::vector<std::vector<Literal*>> fake_argument_ptrs(n);
  std::vector<std::vector<Literal>> fake_arguments(n);
  std::vector<std::unique_ptr<Executable>> executables(n);

  for (int i = 0; i < n; ++i) {
    auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
    if (!module_or_status.ok()) {
      return ::testing::AssertionFailure()
             << "Error while parsing HLO text format: "
             << module_or_status.status().ToString();
    }
    std::unique_ptr<HloModule> module =
        std::move(module_or_status.ValueOrDie());

    fake_arguments[i] = MakeFakeArguments(module.get()).ConsumeValueOrDie();

    if (profiles != nullptr) {
      // We have to enable HLO profiling since otherwise currently the
      // ExecutionProfile is not correct.
      //
      // TODO(b/119432044): Fix collection of the ExecutionProfile
      // so that this is not necessary.
      HloModuleConfig config = module->config();
      DebugOptions debug_options = config.debug_options();
      debug_options.set_xla_hlo_profile(true);
      config.set_debug_options(debug_options);
      module->set_config(config);
    }

    if (backend_config) {
      // Set backend configuration if it is given.
      HloInstruction* instruction =
          module->entry_computation()->root_instruction();
      Status s = instruction->set_backend_config(*backend_config);
      return s.ok() ? ::testing::AssertionSuccess()
                    : ::testing::AssertionFailure() << s.error_message();
    }

    auto executable =
        test_runner_.CreateExecutable(std::move(module), run_hlo_passes);
    if (!executable.ok()) {
      return ::testing::AssertionFailure()
             << executable.status().error_message();
    }
    executables[i] = std::move(executable.ValueOrDie());
  }

  absl::optional<Literal> canonical_output;
  for (int i = 0; i < n; ++i) {
    StatusOr<Literal> output = test_runner_.ExecuteWithExecutable(
        executables[i].get(), fake_arguments[i],
        /*profile=*/&((*profiles)[i]));
    if (!output.ok()) {
      return ::testing::AssertionFailure() << output.status().error_message();
    }

    if (assert_determinism) {
      if (!canonical_output.has_value()) {
        canonical_output = output.ConsumeValueOrDie();
      } else {
        if (*canonical_output != output.ValueOrDie()) {
          return ::testing::AssertionFailure()
                 << "Successive runs have returned different results: "
                 << *canonical_output << " vs. " << output.ValueOrDie();
        }
      }
    }
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HloTestBase::RunAndCompareFromFile(
    const std::string& filename, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_35(mht_35_v, 931, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareFromFile");

  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompare(module_or_status.ConsumeValueOrDie(), error,
                       reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPasses(
    string_view hlo_string, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_36(mht_36_v, 948, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareNoHloPasses");

  auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "Error while parsing HLO text format: "
           << module_or_status.status().ToString();
  }
  return RunAndCompareNoHloPasses(module_or_status.ConsumeValueOrDie(), error,
                                  reference_preprocessor);
}

::testing::AssertionResult HloTestBase::RunAndCompareNoHloPassesFromFile(
    const std::string& filename, const absl::optional<ErrorSpec>& error,
    const std::function<void(HloModule*)>& reference_preprocessor) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_37(mht_37_v, 965, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::RunAndCompareNoHloPassesFromFile");

  auto module_or_status =
      HloRunner::ReadModuleFromHloTextFile(filename, GetDebugOptionsForTest());
  if (!module_or_status.ok()) {
    return ::testing::AssertionFailure()
           << "failed reading hlo module from file";
  }
  return RunAndCompareNoHloPasses(module_or_status.ConsumeValueOrDie(), error,
                                  reference_preprocessor);
}

HloComputation* HloTestBase::FindComputation(HloModule* module,
                                             absl::string_view name) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_38(mht_38_v, 981, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::FindComputation");

  auto computations = module->computations();
  auto it = absl::c_find_if(
      computations, [&](HloComputation* c) { return c->name() == name; });
  if (it == computations.end()) {
    return nullptr;
  }
  return *it;
}

HloInstruction* HloTestBase::FindInstruction(HloModule* module,
                                             absl::string_view name) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_39(mht_39_v, 996, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::FindInstruction");

  for (const HloComputation* c : module->computations()) {
    auto instructions = c->instructions();
    auto it = absl::c_find_if(
        instructions, [&](HloInstruction* i) { return i->name() == name; });
    if (it != instructions.end()) {
      return *it;
    }
  }
  return nullptr;
}

HloInstruction* HloTestBase::FindInstruction(HloModule* module,
                                             HloOpcode opcode) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_40(mht_40_v, 1012, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::FindInstruction");

  for (const HloComputation* c : module->computations()) {
    auto instructions = c->instructions();
    auto it = absl::c_find_if(
        instructions, [&](HloInstruction* i) { return i->opcode() == opcode; });
    if (it != instructions.end()) {
      return *it;
    }
  }
  return nullptr;
}

Backend& HloTestBase::backend() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_41(mht_41_v, 1027, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::backend");
 return test_runner_.backend(); }

/* static */
std::string HloTestBase::TestName() {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPShlo_test_baseDTcc mht_42(mht_42_v, 1033, "", "./tensorflow/compiler/xla/tests/hlo_test_base.cc", "HloTestBase::TestName");

  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

}  // namespace xla
