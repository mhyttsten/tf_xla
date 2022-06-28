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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/run_hlo_module.h"

#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/testing.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/error_spec.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"
#include "tensorflow/compiler/xla/tools/prepare_reference_module.h"
#include "tensorflow/compiler/xla/tools/run_hlo_module.pb.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"

namespace xla {
namespace {

// Writes the given literal to a file in the test temporary directory.
void WriteLiteralToTempFile(const LiteralSlice& literal,
                            const std::string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc mht_0(mht_0_v, 222, "", "./tensorflow/compiler/xla/tools/run_hlo_module.cc", "WriteLiteralToTempFile");

  // Bazel likes for tests to write "debugging outputs" like these to
  // TEST_UNDECLARED_OUTPUTS_DIR.  This plays well with tools that inspect test
  // results, especially when they're run on remote machines.
  auto* env = tensorflow::Env::Default();
  std::string binary_filename;
  std::string text_filename;
  std::string outdir;
  if (tensorflow::io::GetTestUndeclaredOutputsDir(&outdir)) {
    std::string filename = tensorflow::io::JoinPath(
        outdir, absl::StrFormat("tempfile-%d-%s", env->NowMicros(), name));
    binary_filename = absl::StrCat(filename, ".pb");
    text_filename = absl::StrCat(filename, ".txt");
  } else {
    binary_filename =
        tensorflow::io::GetTempFilename(absl::StrCat(name, ".pb"));
    text_filename = tensorflow::io::GetTempFilename(absl::StrCat(name, ".txt"));
  }

  TF_CHECK_OK(
      tensorflow::WriteBinaryProto(env, binary_filename, literal.ToProto()));
  TF_CHECK_OK(
      tensorflow::WriteStringToFile(env, text_filename, literal.ToString()));
  LOG(ERROR) << "wrote Literal to " << name << " binary: " << binary_filename
             << " text: " << text_filename;
}

// Callback helper that dumps literals to temporary files in the event of a
// miscomparison.
void OnMiscompare(const LiteralSlice& expected, const LiteralSlice& actual,
                  const LiteralSlice& mismatches,
                  const ShapeIndex& /*shape_index*/) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc mht_1(mht_1_v, 256, "", "./tensorflow/compiler/xla/tools/run_hlo_module.cc", "OnMiscompare");

  LOG(INFO) << "expected: " << ShapeUtil::HumanString(expected.shape()) << " "
            << literal_comparison::ToStringTruncated(expected);
  LOG(INFO) << "actual:   " << ShapeUtil::HumanString(actual.shape()) << " "
            << literal_comparison::ToStringTruncated(actual);
  LOG(INFO) << "Dumping literals to temp files...";
  WriteLiteralToTempFile(expected, "expected");
  WriteLiteralToTempFile(actual, "actual");
  WriteLiteralToTempFile(mismatches, "mismatches");
}

Literal ExecuteWithRunner(std::unique_ptr<HloModule> module,
                          absl::Span<const Literal> args,
                          HloRunnerInterface* runner, bool run_hlo_passes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc mht_2(mht_2_v, 272, "", "./tensorflow/compiler/xla/tools/run_hlo_module.cc", "ExecuteWithRunner");

  TF_QCHECK_OK(VerifyHloModule(module.get(), /*layout_sensitive=*/false,
                               /*allow_mixed_precision=*/true))
      << " (on " << runner->Name() << ")";

  std::cerr << "Running HLO module with runner " << runner->Name() << "...\n";
  XLA_VLOG_LINES(1, module->ToString());
  const auto start = std::chrono::high_resolution_clock::now();
  auto result_status = runner->Execute(std::move(module), args, run_hlo_passes);
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cerr << "... compiled and ran in " << diff.count() << "s.\n";

  TF_QCHECK_OK(result_status.status())
      << "Failed to execute on " << runner->Name() << "\n";

  return result_status.ConsumeValueOrDie();
}
}  // namespace

Status RunAndCompare(
    std::unique_ptr<HloModule> test_module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc mht_3(mht_3_v, 302, "", "./tensorflow/compiler/xla/tools/run_hlo_module.cc", "RunAndCompare");

  if (!config_modifier_hook) {
    config_modifier_hook = [](HloModuleConfig* config) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc mht_4(mht_4_v, 307, "", "./tensorflow/compiler/xla/tools/run_hlo_module.cc", "lambda");

      config->set_seed(42);
    };
  }

  if (options.flatten_control_flow) {
    HloControlFlowFlattening control_flow_flattening(
        HloControlFlowFlattening::Options{/*while_execution_count=*/1});
    TF_RETURN_IF_ERROR(control_flow_flattening.Run(test_module.get()).status());
  }

  const HloModuleProto test_module_proto = test_module->ToProto();

  std::vector<Literal> args = MakeFakeArguments(test_module.get(), engine,
                                                options.use_large_float_range)
                                  .ConsumeValueOrDie();
  // Use provided input literals as arguments, if any.
  if (iteration_literals_proto != nullptr &&
      iteration_literals_proto->arguments_size() != 0) {
    if (iteration_literals_proto->arguments_size() != args.size()) {
      return xla::InvalidArgument(
          "Failed to use input literals as arguments; mismatched "
          "number of expected arguments.");
    } else {
      for (int i = 0; i < args.size(); ++i) {
        if (!literal_comparison::EqualShapes(
                 xla::Shape(args[i].shape()),
                 xla::Shape(iteration_literals_proto->arguments(i).shape()))
                 .ok()) {
          return xla::InvalidArgument(
              "Failed to use input literals for argument %d "
              "because of a shape mismatch.",
              i);
        }
        TF_ASSIGN_OR_RETURN(args[i],
                            xla::Literal::CreateFromProto(
                                iteration_literals_proto->arguments(i)));
      }
    }
  }
  if (options.print_literals) {
    for (int i = 0; i < args.size(); ++i) {
      std::cout << "\n** Argument " << i << " **\n"
                << args[i].ToString() << "\n";
    }
  }
  if (iteration_literals_proto != nullptr &&
      iteration_literals_proto->arguments_size() == 0) {
    for (int i = 0; i < args.size(); ++i) {
      *iteration_literals_proto->add_arguments() = args[i].ToProto();
    }
  }

  std::unique_ptr<HloModule> reference_module;
  if (reference_runner != nullptr) {
    // PrepareReferenceModule needs to know the *test* runner, in order to
    // properly match the test runner's numerics.
    reference_module =
        PrepareReferenceModule(*test_module, test_runner, config_modifier_hook,
                               reference_module_modifier_hook)
            .ConsumeValueOrDie();
  }

  Literal test_result = ExecuteWithRunner(
      std::move(test_module), args, test_runner, options.run_test_hlo_passes);
  if (options.print_literals) {
    std::cout << "\n** Result with test runner " << test_runner->Name()
              << " **\n"
              << test_result.ToString() << "\n";
  }
  if (iteration_literals_proto != nullptr) {
    LiteralProto test_result_proto = test_result.ToProto();
    iteration_literals_proto->mutable_result()->Swap(&test_result_proto);
  }

  if (reference_module == nullptr) {
    std::cerr << "Skipping reference runner\n";
    return Status::OK();
  }

  Literal reference_result =
      ExecuteWithRunner(std::move(reference_module), args, reference_runner,
                        options.run_reference_hlo_passes);

  if (options.print_literals) {
    std::cout << "\n** Result with reference runner "
              << reference_runner->Name() << " **\n"
              << reference_result.ToString() << "\n";
  }
  if (iteration_literals_proto != nullptr) {
    LiteralProto reference_result_proto = reference_result.ToProto();
    iteration_literals_proto->mutable_reference_result()->Swap(
        &reference_result_proto);
  }
  ErrorSpec error_spec(static_cast<float>(options.abs_error_bound),
                       static_cast<float>(options.rel_error_bound));
  return literal_comparison::Near(/*expected=*/reference_result,
                                  /*actual=*/test_result,
                                  /*error=*/error_spec,
                                  /*detailed_message=*/true, &OnMiscompare);
}

Status RunAndCompare(
    const std::string& hlo_filename, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, std::minstd_rand0* engine,
    const RunHloModuleOptions& options,
    xla::RunHloModuleIterationLiterals* iteration_literals_proto,
    std::function<Status(const HloModule&, HloRunnerInterface*, HloModule*)>
        reference_module_modifier_hook,
    std::function<void(HloModuleConfig*)> config_modifier_hook) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("hlo_filename: \"" + hlo_filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSrun_hlo_moduleDTcc mht_5(mht_5_v, 420, "", "./tensorflow/compiler/xla/tools/run_hlo_module.cc", "RunAndCompare");

  std::unique_ptr<HloModule> test_module =
      LoadModuleFromFile(hlo_filename, hlo_module_loader_details::Config(),
                         options.input_format, config_modifier_hook)
          .ValueOrDie();
  return RunAndCompare(std::move(test_module), test_runner, reference_runner,
                       engine, options, iteration_literals_proto,
                       reference_module_modifier_hook, config_modifier_hook);
}
}  // namespace xla
