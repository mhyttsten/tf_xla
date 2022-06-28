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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc() {
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

#include "tensorflow/compiler/xla/service/interpreter/compiler.h"

#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/interpreter/executable.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace interpreter {

namespace {

// Handles custom_call ops during evaluation by routing them through the global
// CPU registry used by other CPU-based backends.
StatusOr<Literal> HandleEvaluatorCustomCall(
    HloInstruction* custom_call, absl::Span<const Literal*> operands) {
  // Find the target C function in the global registry.
  auto* registry = CustomCallTargetRegistry::Global();
  void* target_fn = registry->Lookup(custom_call->custom_call_target(), "Host");
  if (!target_fn) {
    return NotFound("Custom call target '%s' was not registered",
                    custom_call->custom_call_target());
  }

  // Populate pointers to operand and output literal data.
  std::vector<const void*> operand_data;
  operand_data.reserve(operands.size());
  for (const auto* literal : operands) {
    operand_data.push_back(literal->untyped_data());
  }
  auto output = Literal::CreateFromShape(custom_call->shape());
  void* output_data = output.untyped_data();

  // Call the target function matching the C ABI used by the CPU backends.
  auto* typed_fn = reinterpret_cast<void (*)(void*, const void**)>(target_fn);
  (*typed_fn)(output_data, operand_data.data());

  return std::move(output);
}

}  // namespace

Status InterpreterCompiler::RunHloOptimization(HloModule* hlo_module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_0(mht_0_v, 250, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::RunHloOptimization");

  HloPassPipeline pipeline("Interpreter");

  pipeline.AddPass<DynamicIndexSplitter>();
  pipeline.AddPass<CholeskyExpander>();
  pipeline.AddPass<QrExpander>();
  pipeline.AddPass<EighExpander>();
  pipeline.AddPass<ComparisonExpander>();
  pipeline.AddPass<TriangularSolveExpander>();
  pipeline.AddPass<LayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());

  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<HloModule>> InterpreterCompiler::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* /*stream_exec*/,
    const CompileOptions& /*options*/) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_1(mht_1_v, 270, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::RunHloPasses");

  VLOG(1) << "Run hlo passes on graph " << hlo_module->name();
  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));
  return std::move(hlo_module);
}

StatusOr<std::unique_ptr<Executable>> InterpreterCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& /*options*/) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_2(mht_2_v, 281, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::RunBackend");

  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Run backend " << hlo_module->name();

  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(hlo_module.get()));

  auto evaluator = absl::make_unique<HloEvaluator>();
  evaluator->set_use_fast_path(
      hlo_module->config().debug_options().xla_hlo_evaluator_use_fast_path());
  evaluator->set_custom_call_handler(HandleEvaluatorCustomCall);

  // Create executable from only the Hlo module.
  std::unique_ptr<Executable> executable =
      absl::make_unique<InterpreterExecutable>(
          std::move(hlo_module), std::move(evaluator),
          std::move(dynamic_dimension_inference));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> InterpreterCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_exec,
    const CompileOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_3(mht_3_v, 309, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::Compile");

  if (module_group->empty()) {
    return std::vector<std::unique_ptr<Executable>>();
  }
  if (module_group->size() > 1) {
    return tensorflow::errors::Unimplemented(
        "Compilation of multiple HLO modules is not supported on Interpreter.");
  }
  if (stream_exec.size() != 1 || stream_exec[0].size() != 1) {
    return tensorflow::errors::Unimplemented(
        "Unexpected number of StreamExecutor's.");
  }
  auto hlo_modules = module_group->ConsumeModules();
  TF_ASSIGN_OR_RETURN(auto module, RunHloPasses(std::move(hlo_modules[0]),
                                                stream_exec[0][0], options));
  TF_ASSIGN_OR_RETURN(auto executable, RunBackend(std::move(module),
                                                  stream_exec[0][0], options));
  std::vector<std::unique_ptr<Executable>> ret;
  ret.push_back(std::move(executable));
  return std::move(ret);
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
InterpreterCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModuleGroup> module_group,
    const AotCompilationOptions& aot_options) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_4(mht_4_v, 337, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::CompileAheadOfTime");

  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on Interpreter");
}

se::Platform::Id InterpreterCompiler::PlatformId() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_5(mht_5_v, 345, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::PlatformId");

  return se::interpreter::kXlaInterpreterPlatformId;
}

HloCostAnalysis::ShapeSizeFunction InterpreterCompiler::ShapeSizeBytesFunction()
    const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_6(mht_6_v, 353, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InterpreterCompiler::ShapeSizeBytesFunction");

  return InterpreterExecutable::ShapeSizeBytes;
}

static bool InitModule() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPScompilerDTcc mht_7(mht_7_v, 360, "", "./tensorflow/compiler/xla/service/interpreter/compiler.cc", "InitModule");

  xla::Compiler::RegisterCompilerFactory(
      se::interpreter::kXlaInterpreterPlatformId, []() {
        return absl::make_unique<xla::interpreter::InterpreterCompiler>();
      });
  xla::ComputationPlacer::RegisterComputationPlacer(
      se::interpreter::kXlaInterpreterPlatformId,
      []() { return absl::make_unique<xla::ComputationPlacer>(); });
  return true;
}

static bool module_initialized = InitModule();

}  // namespace interpreter
}  // namespace xla
